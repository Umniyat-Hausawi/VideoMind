import os
import json
import re
import time
import difflib
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator

load_dotenv()


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")
REPORT_MODEL         = "claude-sonnet-4-6"
PERIODS_SUMMARY_MODEL = "claude-haiku-4-5-20251001"  # cheaper model for the intermediate
                                                       # periods-summarization step
MAX_TOKENS           = 4000

# Retry settings for transient API failures — same policy as audio_layer
# and visual_layer: one extra attempt after a short delay.
RETRY_MAX_ATTEMPTS  = 2
RETRY_DELAY_SECONDS = 1

# Minimum acceptable report length now scales with report_mode and how
# much content the video actually has (see _get_min_report_length) —
# a fixed 200-char floor made a 10-second TikTok summary and a full
# lecture analysis look equally "too short" or "acceptable", which
# isn't a meaningful comparison for either one.
SUMMARY_MIN_LENGTHS  = {"tiny": 100, "small": 200, "normal": 250}
ANALYSIS_MIN_LENGTHS = {"tiny": 150, "small": 350, "normal": 500}


# ──────────────────────────────────────────────
# Report Schema + Tool Definition (structured output)
# ──────────────────────────────────────────────

class ReportSchema(BaseModel):
    """
    Strict schema for the report data. Both fields default to "" so a
    genuinely missing field is caught by the "both empty" check right
    after validation, same as the old .get(key, "") behavior — this
    schema's real job is catching wrong TYPES and giving a precise error,
    not enforcing presence on its own.
    """
    report_ar: str = ""
    report_en: str = ""

    @field_validator("report_ar", "report_en", mode="before")
    @classmethod
    def _coerce_to_str(cls, value):
        if value is None:
            return ""
        return str(value)


# Tool-use / function-calling definition — avoids asking Claude to
# hand-write escaped JSON inside a text block. Forcing a tool call
# guarantees valid, correctly-typed structured
# input at the API level itself, so _best_effort_language_split below
# becomes a rare defense-in-depth fallback rather than the primary
# recovery path.
REPORT_TOOL = {
    "name": "submit_report",
    "description": (
        "Submit the completed bilingual (Arabic + English) video analysis report. "
        "Call this exactly once, with both fields fully written out."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "report_ar": {
                "type": "string",
                "description": "The complete Arabic report, as plain text (use real newlines, no escaping needed)."
            },
            "report_en": {
                "type": "string",
                "description": "The complete English report, as plain text (use real newlines, no escaping needed)."
            },
        },
        "required": ["report_ar", "report_en"],
    },
}


# ──────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────

def process_report(fusion_result: dict, report_mode: str = "analysis") -> dict:
    """
    Main entry point for the Report Layer.

    VideoMind is scoped to social media video analysis (see README.md /
    LIMITATIONS.md) — YouTube, TikTok, X, and direct file upload (Instagram
    links are detected and rejected early, see LIMITATIONS.md), including
    longer-form YouTube content with an educational or podcast-style
    tone. There's no separate content_type selection: the same report
    instructions apply across that whole range, since the audio/visual
    fusion signal that drives the report is identical regardless of
    subject matter.

    Args:
        fusion_result : output from fusion_layer
        report_mode   : "analysis" | "summary"

    Returns:
        {
            "report_ar"           : str,
            "report_en"           : str,
            "validation_warnings" : list[str],
            "status"              : "success" | "error",
            "message"             : str
        }
    """
    if fusion_result.get("status") == "error":
        return _error_result(
            f"Fusion layer failed: {fusion_result.get('message')} "
            f"| فشلت طبقة الدمج: {fusion_result.get('message')}"
        )

    if report_mode not in ["analysis", "summary"]:
        report_mode = "analysis"

    summary          = fusion_result.get("summary", {})
    aligned_segments = fusion_result.get("aligned_segments", [])
    total_segments   = summary.get("total_segments", 0)

    # ── Periods-based summarization for long videos ──
    # For short/medium videos, the flat full_text_preview (capped at
    # 5000 chars in fusion_layer) already covers the whole transcript —
    # no need for the extra API call. For long videos, that same 5000-char
    # cap would only cover the early portion, biasing the report toward
    # the beginning. _build_periods_summary condenses the ENTIRE video
    # (split into periods, each period summarized) into something that
    # fits the prompt, at the cost of one extra (cheap, Haiku) API call.
    periods_summary_text = None
    num_periods = _get_period_count_for_report(total_segments)
    if num_periods > 0:
        print(
            f"Video has {total_segments} segments (>60) — using periods-based "
            f"summarization ({num_periods} periods) for full-video report coverage"
        )
        periods_summary_text = _build_periods_summary(aligned_segments, num_periods)
        if periods_summary_text:
            print("Periods summarization succeeded — report will cover the full video timeline")
        else:
            print("Periods summarization returned no result — falling back to flat preview")

    prompt = _build_prompt(summary, aligned_segments, report_mode, periods_summary_text)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"Generating {report_mode} report")

    try:
        response = _retry_call(
            client.messages.create,
            model       = REPORT_MODEL,
            max_tokens  = MAX_TOKENS,
            tools       = [REPORT_TOOL],
            tool_choice = {"type": "tool", "name": "submit_report"},
            messages    = [{"role": "user", "content": prompt}]
        )

        report_ar, report_en, parse_warning = _parse_report_response(response)

        validation_warnings = _validate_report(report_ar, report_en, report_mode, total_segments)
        if parse_warning:
            validation_warnings.insert(0, parse_warning)

        # ── Deterministic grounding check ──
        # No extra API call — pure string comparison against the
        # structured summary data this report was built from. Catches
        # obvious contradictions (sentiment, confidence, untraceable
        # quotes); does NOT verify interpretive sections, which have no
        # checkable ground truth. See LIMITATIONS.md.
        grounding_warnings = _run_grounding_check(report_ar, report_en, summary)
        validation_warnings.extend(grounding_warnings)

        print("Report generated successfully")

        return {
            "report_ar"           : report_ar,
            "report_en"           : report_en,
            "validation_warnings" : validation_warnings,
            "status"              : "success",
            "message"             : (
                f"{report_mode} report generated "
                f"| تم إنشاء تقرير {report_mode}"
            )
        }

    except Exception as e:
        return _error_result(
            f"Claude API error after {RETRY_MAX_ATTEMPTS} attempt(s): {str(e)} "
            f"| خطأ بـ Claude API بعد {RETRY_MAX_ATTEMPTS} محاولة"
        )


# ──────────────────────────────────────────────
# Retry Helper
# ──────────────────────────────────────────────

def _retry_call(func, *args, **kwargs):
    """
    Call `func(*args, **kwargs)`, retrying once (RETRY_MAX_ATTEMPTS total
    attempts) after a short delay if it raises an exception. Same policy
    as audio_layer / visual_layer — most failures here are transient.
    """
    last_error = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < RETRY_MAX_ATTEMPTS:
                print(
                    f"WARNING: API call failed (attempt {attempt}/{RETRY_MAX_ATTEMPTS}) "
                    f"— retrying in {RETRY_DELAY_SECONDS}s: {str(e)}"
                )
                time.sleep(RETRY_DELAY_SECONDS)

    raise last_error


# ──────────────────────────────────────────────
# Periods-Based Summarization (long-video coverage)
# ──────────────────────────────────────────────

def _get_period_count_for_report(total_segments: int) -> int:
    """
    Decide how many periods to split a long video's transcript into for
    report generation, or 0 to skip periods-based summarization entirely.

    Why a segment-count threshold instead of a duration threshold?
    total_segments is already available here without needing to thread
    video duration through fusion_layer's summary — and it's a reasonable
    proxy: a video with ≤60 segments almost always fits comfortably
    within the flat 5000-char full_text_preview already used for
    short/medium videos, so there's no coverage problem to solve and no
    reason to spend an extra API call.

        Why can these numbers be more generous than visual_layer's frame caps?
    Raising the frame cap directly multiplies Claude Vision API calls
    (one call per frame) — a real, linear cost increase. Raising the
    period count here does NOT multiply API calls: _summarize_periods
    still makes exactly one Haiku call regardless of how many periods it's
    asked to summarize in that single prompt. More periods mostly just
    means shorter, more focused periods (less content crammed into each
    one, so less gets dropped when a period's summary is limited to a
    few sentences) — a real coverage win with only a modest token-size
    cost, not a per-period API-call cost.

        total_segments <= 60   → 0   (flat preview is already enough)
        total_segments <= 120  → 8   periods
        total_segments <= 250  → 12  periods
        total_segments >  250  → 15  periods
    """
    if total_segments <= 60:
        return 0
    elif total_segments <= 120:
        return 8
    elif total_segments <= 250:
        return 12
    else:
        return 15


def _split_segments_into_period_texts(aligned_segments: list, num_periods: int) -> list[str]:
    """
    Split aligned_segments into num_periods roughly-equal, chronologically
    ordered chunks (aligned_segments is already in chronological order),
    and join each chunk's segment text into one raw text blob per period.

    Each period's raw text is capped at 2000 chars before being sent to
    _summarize_periods — this is the RAW input to that summarization
    call, not the final report content, so a generous cap here is fine;
    it's the condensed OUTPUT of _summarize_periods that actually ends up
    in the report prompt.
    """
    if not aligned_segments:
        return []

    total = len(aligned_segments)
    chunk_size = max(1, -(-total // num_periods))  # ceil division

    period_texts = []
    for i in range(0, total, chunk_size):
        chunk = aligned_segments[i:i + chunk_size]
        text = " ".join(seg.get("text", "") for seg in chunk).strip()
        if text:
            period_texts.append(text[:2000])

    return period_texts


def _summarize_periods(period_texts: list[str]) -> list[str] | None:
    """
    One extra Claude call (Haiku — cheap, this is a condensation task, not
    the final report writing) that summarizes each time period's raw
    segment text into 2-3 concise sentences, so the final report prompt
    can reflect the ENTIRE video's content instead of only whatever fits
    in a flat 5000-char preview of the beginning.

    Returns None on any failure (API error, unparseable response, wrong
    array length) — callers must treat that as "periods summarization
    unavailable" and fall back to the flat preview, not as a fatal error
    for the whole report.
    """
    numbered_periods = "\n\n".join(
        f"[Period {i + 1}]\n{text}" for i, text in enumerate(period_texts)
    )

    prompt = f"""You will be given {len(period_texts)} time periods of raw transcript text from one video, in chronological order.

For EACH period, write a concise summary of what was said, in English, regardless of the transcript's
original language. Default to 2-3 sentences, but if a period covers multiple distinct topics or
events (e.g. several unrelated news items discussed back-to-back), use up to 5-6 sentences instead —
mention every distinct topic by name at least briefly, rather than compressing several topics into one
vague sentence that drops most of them. It's fine for periods to vary in summary length depending on
how much distinct content they actually contain.

{numbered_periods}

Respond with ONLY a JSON array of exactly {len(period_texts)} strings (one summary per period, in the same order), no other text before or after it, no markdown code fences.
Example shape: ["summary of period 1", "summary of period 2", ...]"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = _retry_call(
            client.messages.create,
            model      = PERIODS_SUMMARY_MODEL,
            max_tokens = 1500,
            messages   = [{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        parsed = json.loads(text)

        if isinstance(parsed, list) and len(parsed) == len(period_texts) and all(isinstance(s, str) for s in parsed):
            return parsed

        print(
            f"WARNING: periods summarization returned unexpected shape "
            f"(expected list of {len(period_texts)} strings) — falling back to flat preview"
        )
        return None

    except Exception as e:
        print(f"WARNING: periods summarization call failed — falling back to flat preview: {str(e)}")
        return None


def _build_periods_summary(aligned_segments: list, num_periods: int) -> str | None:
    """
    Orchestrates period splitting + summarization, and formats the result
    for direct inclusion in the report prompt. Returns None (triggering
    the flat-preview fallback in _build_prompt) if anything along the way
    didn't produce usable summaries.
    """
    period_texts = _split_segments_into_period_texts(aligned_segments, num_periods)
    if not period_texts:
        return None

    summaries = _summarize_periods(period_texts)
    if not summaries:
        return None

    return "\n\n".join(f"[Period {i + 1}] {s}" for i, s in enumerate(summaries))


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def _build_prompt(
    summary               : dict,
    aligned_segments      : list,
    report_mode           : str,
    periods_summary_text  : str | None = None
) -> str:
    """
    Build a structured prompt for Claude for social media video analysis
    (see process_report's docstring on why there's no content_type here).

    periods_summary_text: when provided (long videos only — see
    _get_period_count_for_report), this REPLACES the flat, beginning-biased
    full_text_preview with a period-by-period condensed summary spanning
    the entire video. When None (short/medium videos), the flat preview
    is used exactly as before — no behavior change for the common case.
    """
    key_moments_text = ""
    if summary.get("key_moments"):
        key_moments_text = "\nKey Moments (audio/visual conflict):\n"
        for moment in summary["key_moments"]:
            key_moments_text += f"- [{moment['time']}s] {moment['text']} ({moment['note']})\n"

    critical_text_section = ""
    if summary.get("critical_text_moments"):
        critical_text_section = (
            "\nNotable On-Screen Text Conflicts (rare/one-off text shown while audio "
            "sentiment was decisive — persistent overlays like scoreboards or watermarks "
            "have already been filtered out, so each item below is genuinely notable):\n"
        )
        for moment in summary["critical_text_moments"]:
            critical_text_section += (
                f"- [{moment['time']}s] On-screen text: \"{moment['text_on_screen']}\" "
                f"| Audio sentiment at that moment: {moment['audio_sentiment']}\n"
            )

    silent_brand_section = ""
    if summary.get("silent_brand_mentions"):
        silent_brand_section = (
            "\nBrand Logos Shown But Never Named Aloud (a recognizable brand logo/mark "
            "appeared on screen, but the speaker never said this brand's name anywhere "
            "in the transcript — worth mentioning explicitly in the report, since a "
            "transcript-only summary would miss this entirely. Note: brand names here "
            "come from visual logo recognition and may be in a different language/script "
            "than the transcript, so treat this as a strong signal, not a certainty):\n"
        )
        for mention in summary["silent_brand_mentions"]:
            silent_brand_section += f"- [{mention['time']}s] Brand shown: {mention['brand']}\n"

    sentiment_timeline_section = ""
    timeline = summary.get("sentiment_timeline", [])
    real_periods = [p for p in timeline if not p.get("is_fallback")]
    # Only worth surfacing if the tone actually moves across periods — a flat,
    # unchanging timeline has nothing meaningful to say about "change over
    # time," and forcing a mention would just be padding the report with a
    # non-observation. Comparing distinct sentiment VALUES (not periods)
    # means a video that's positive→positive→negative→positive still counts
    # as having real variation worth mentioning.
    if len({p["sentiment"] for p in real_periods}) > 1:
        sentiment_timeline_section = (
            "\nSentiment Over Time (audio tone by time period — genuine variation "
            "was detected across the video's timeline):\n"
        )
        for period in real_periods:
            sentiment_timeline_section += f"- [{period['start']}s-{period['end']}s] {period['sentiment']}\n"

    segments_text = ""
    if aligned_segments:
        segments_text = "\nSample Segments:\n"
        for seg in aligned_segments[:15]:
            visual_desc = seg.get('frame_description', '')[:120]
            segments_text += (
                f"[{seg['start']}s] {seg['text']} | emotion: {seg['frame_emotion']}"
                f"{' | visual scene: ' + visual_desc if visual_desc else ''}\n"
            )

    instructions = _get_instructions(report_mode)

    if periods_summary_text:
        transcript_section_header = (
            "## Full-Video Coverage (period-by-period condensed summary — "
            "spans the ENTIRE video's timeline, not just the beginning)"
        )
        transcript_section_body = periods_summary_text
    else:
        transcript_section_header = "## Full Transcript Preview"
        transcript_section_body   = summary.get('full_text_preview', '')

    prompt = f"""You are VideoMind, an AI system that analyzes video content from audio and visual layers.

## Analysis Data

- Language: {summary.get('language', 'unknown')}
- Report Mode: {report_mode}
- Total Segments: {summary.get('total_segments', 0)}
- Audio Sentiment: {summary.get('audio_sentiment', 'neutral')}
- Visual Sentiment: {summary.get('visual_sentiment', 'neutral')}
- Unified Sentiment: {summary.get('unified_sentiment', 'neutral')}
- Confidence: {summary.get('confidence_label', 'medium')} ({summary.get('confidence', 0)})
- Signal: {summary.get('signal', 'aligned')}
{key_moments_text}
{critical_text_section}
{silent_brand_section}
{sentiment_timeline_section}

{transcript_section_header}
{transcript_section_body}

## Aligned Segments (what was said + visual emotion at that moment)
{segments_text}

## Instructions
{instructions}

## Accuracy Guidelines
Base your report ONLY on the content provided above (transcript preview, aligned segments, sentiment data).
Do not invent facts, quotes, statistics, or events that are not present in this material.
Never infer a speaker's age category (e.g. "child" vs "adult") from how fluently or clearly they speak —
broken or halting speech can just as easily mean a non-native speaker or someone learning the language,
not a child. Only describe someone as a child if the visual frame descriptions above explicitly say so.
For interpretive sections (e.g. Possible Exam Questions, Creator Intent, Recommendations), keep your
answer grounded in what's actually said. If something is genuinely not clear from the available content,
say so explicitly (e.g. "not clear from the available content") rather than guessing or fabricating detail.
If "Notable On-Screen Text Conflicts" are provided above, weave the most significant one (if any) naturally
into the Sentiment Analysis section as a sentence or two — do not list them as a separate section, and do
not mention this instruction or the filtering process itself. If none are provided, do not mention on-screen
text conflicts at all.
If "Brand Logos Shown But Never Named Aloud" are provided above, explicitly mention this in the report —
e.g. "the video shows [brand]'s logo, though the speaker never names it directly" — since this is exactly
the kind of detail a transcript-only summary would completely miss. If none are provided, do not mention
brand logos at all.
If "Sentiment Over Time" data is provided above, weave a brief, natural mention of how the tone shifts
over the course of the video into the Sentiment Analysis section — e.g. "the tone shifts from measured
early on to more enthusiastic by the end." This section is only ever included when the data shows genuine
variation, so if it's provided, it's always worth a short mention. If it isn't provided at all, do not
mention time-based tone shifts.

## Output Format
Call the submit_report tool exactly once with two fields:
- report_ar: the complete Arabic report, as plain text
- report_en: the complete English report, as plain text
Write real newlines directly in each field — this is a structured tool call, not text you're
writing JSON into yourself, so there's no manual escaping to worry about. Do not include any
commentary outside the tool call; the tool call is your entire response.

Formatting (apply consistently, every time, no exceptions):
- Use clean Markdown: "## " for each major section heading (exactly the section names given
  in the instructions above, translated naturally), "- " for bullet points, blank lines between
  sections.
- Never use manual ASCII decoration — no repeated dashes/underscores/equals-signs as dividers
  (e.g. "───", "═══", "____"), no box-drawing borders, no manual numbering like "1." before a
  heading unless the instructions above explicitly asked for a numbered list.
  Markdown headings alone are enough visual separation.
- Never use backticks (`) or code-block formatting to emphasize a number, phrase, or fact —
  this renders as literal monospace/code styling, not emphasis, and looks broken. If something
  genuinely needs emphasis, use plain bold (**text**) instead, or simply state it plainly —
  most facts don't need any visual emphasis at all.
- Keep the same section order and heading style across both report_ar and report_en."""

    return prompt


def _get_instructions(report_mode: str) -> str:
    """
    Return instructions for the report, scoped to social media video
    analysis (includes longer-form YouTube content — educational or
    podcast-style — since the same audio/visual fusion signal applies
    regardless of subject matter; see process_report's docstring).
    """

    if report_mode == "summary":
        return """Generate a SUMMARY report — the user wants to understand what was said without watching the video.
- Main Topic: what is this video about in one sentence
- Key Points: 5-7 bullet points of the most important things said
- Notable Quotes: 2-3 direct quotes or key statements from the transcript
- Conclusion: what was the final message or takeaway"""

    return """Generate a social media content analysis with:
- What Was Said: summarize the actual spoken content in 4-5 sentences
- Creator Intent: what is the creator trying to achieve?
- Sentiment Analysis: audio vs visual breakdown
- Target Audience: who is this content for?
- Recommendations: 2-3 actionable improvements"""


# ──────────────────────────────────────────────
# Report Parser
# ──────────────────────────────────────────────

def _best_effort_language_split(raw_text: str) -> tuple[str, str] | None:
    """
    When full JSON parsing fails, try a best-effort split using the known
    structural marker between the two fields, instead of giving up and
    dumping the identical raw blob into both report_ar and report_en (which
    would surface English text even when the user only selected the
    Arabic report).

    This isn't bulletproof — a stray unescaped quote right at the field
    boundary itself would still defeat it — but it recovers cleanly in the
    common case where the break happens deeper inside one field's text,
    not at the "report_ar" / "report_en" boundary.
    """
    marker = None
    for candidate in ['", "report_en"', '","report_en"']:
        if candidate in raw_text:
            marker = candidate
            break

    if marker is None:
        return None

    ar_part, _, en_part = raw_text.partition(marker)

    ar_part = re.sub(r'^\s*\{?\s*"report_ar"\s*:\s*"', "", ar_part)
    en_part = re.sub(r'^\s*:\s*"', "", en_part)
    en_part = re.sub(r'"\s*\}?\s*$', "", en_part)

    def _unescape(text: str) -> str:
        return text.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")

    ar_part = _unescape(ar_part).strip()
    en_part = _unescape(en_part).strip()

    if not ar_part and not en_part:
        return None

    return ar_part, en_part


def _parse_report_response(response) -> tuple[str, str, str | None]:
    """
    Extract report_ar/report_en from Claude's response.

    Preferred path: read the submit_report tool_use block's `input`,
    which is already a parsed dict — guaranteed valid, correctly shaped
    JSON at the API level, since Claude's tool-use forces the model to
    produce input matching REPORT_TOOL's schema. This avoids the fragility
    of asking the model to hand-write escaped JSON inside a text block.

    Fallback path (defense-in-depth, expected to be rare): if there's no
    tool_use block at all — an unexpected API response shape, not
    something normal operation should produce — fall back to scanning
    any text block for JSON, then to _best_effort_language_split.
    """
    tool_use_block = next(
        (block for block in response.content if getattr(block, "type", None) == "tool_use"),
        None
    )

    if tool_use_block is not None:
        try:
            validated = ReportSchema.model_validate(tool_use_block.input)
        except ValidationError as ve:
            field_errors = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc']) or 'root'}: {err['msg']}"
                for err in ve.errors()
            )
            return _fallback_from_text(
                response, f"Schema validation failed on tool_use input — {field_errors}"
            )

        report_ar = validated.report_ar.strip()
        report_en = validated.report_en.strip()

        if not report_ar and not report_en:
            return _fallback_from_text(
                response, "Both report_ar and report_en were empty in tool_use input"
            )

        return report_ar, report_en, None

    return _fallback_from_text(response, "No tool_use block found in Claude's response")


def _fallback_from_text(response, reason: str) -> tuple[str, str, str]:
    """
    Legacy text-based recovery — only reached if the tool_use path above
    didn't produce a usable report. Scans for a text block, tries JSON
    parsing on it, then _best_effort_language_split, then gives up and
    returns the same raw text for both languages.
    """
    text_block = next(
        (block for block in response.content if getattr(block, "type", None) == "text"),
        None
    )
    raw_text = text_block.text.strip() if text_block else ""

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    warning = f"{reason} — falling back to text-based recovery"

    if raw_text:
        try:
            parsed = json.loads(raw_text)
            validated = ReportSchema.model_validate(parsed)
            report_ar = validated.report_ar.strip()
            report_en = validated.report_en.strip()
            if report_ar or report_en:
                warning += "; recovered via raw JSON text block"
                print(f"WARNING: {warning}")
                return report_ar, report_en, warning
        except Exception:
            pass

    split_result = _best_effort_language_split(raw_text) if raw_text else None
    if split_result:
        report_ar, report_en = split_result
        warning += "; recovered via best-effort language split"
        print(f"WARNING: {warning}")
        return report_ar, report_en, warning

    warning += "; could not recover any report text"
    print(f"WARNING: {warning}")
    fallback_text = raw_text or "(no content returned) | (لا يوجد محتوى)"
    return fallback_text, fallback_text, warning


# ──────────────────────────────────────────────
# Deterministic Grounding Check
# ──────────────────────────────────────────────

# Explicit-contradiction phrase sets. Deliberately narrow and literal —
# this is a blunt keyword check, not sentiment analysis, so it's tuned to
# only fire on unambiguous, direct claims that flatly contradict a value
# we actually computed deterministically elsewhere (unified_sentiment,
# confidence_label). It says nothing about subtler or partial disagreement.
POSITIVE_SENTIMENT_PHRASES = [
    "overall positive", "largely positive", "predominantly positive", "mostly positive",
    "بشكل عام إيجابي", "بشكل عام ايجابي", "غالباً إيجابي", "إيجابي بشكل عام",
]
NEGATIVE_SENTIMENT_PHRASES = [
    "overall negative", "largely negative", "predominantly negative", "mostly negative",
    "بشكل عام سلبي", "غالباً سلبي", "سلبي بشكل عام",
]
CONFIDENCE_HIGH_PHRASES = [
    "high confidence", "very confident", "strong confidence",
    "ثقة عالية", "ثقة كبيرة", "بثقة عالية",
]
CONFIDENCE_LOW_PHRASES = [
    "low confidence", "little confidence", "weak confidence",
    "ثقة منخفضة", "ثقة ضعيفة", "بثقة منخفضة",
]

# Only check quotes at least this long — short quoted words/phrases are
# too generic to meaningfully compare against source text and would
# produce mostly noise.
QUOTE_MIN_LENGTH_FOR_CHECK  = 15
# A genuine spoken quote is a sentence or two, not several paragraphs. If the
# text between two quote marks is implausibly long, that's a strong signal
# the report has an unclosed/mismatched quote character somewhere, and the
# regex below is capturing everything up to some unrelated, coincidental
# quote mark later in the text — not a real quoted passage. Skip anything
# that long rather than flag it as an untraceable "quote".
QUOTE_MAX_LENGTH_FOR_CHECK  = 300
QUOTE_SIMILARITY_THRESHOLD  = 0.5


def _run_grounding_check(report_ar: str, report_en: str, summary: dict) -> list[str]:
    """
    Deterministic (zero extra API calls) check for OBVIOUS contradictions
    between the generated report text and the structured data it was
    built from. See LIMITATIONS.md's "Residual Hallucination Risk" entry.

    Intentionally narrow: only compares against values we actually have a
    checkable ground truth for (unified_sentiment, confidence_label,
    source transcript/on-screen text). Interpretive sections (creator
    intent, target audience, recommendations, ...) have no ground truth
    to compare against and are NOT covered by this check — a report with
    zero grounding warnings has NOT been verified to be hallucination-free
    in those sections, only free of the specific contradictions checked
    here.
    """
    warnings: list[str] = []

    unified_sentiment = summary.get("unified_sentiment", "neutral")
    confidence_label   = summary.get("confidence_label", "medium")
    combined_text       = f"{report_ar}\n{report_en}".lower()

    # ── Sentiment contradiction ──
    if unified_sentiment == "negative" and _contains_any_phrase(combined_text, POSITIVE_SENTIMENT_PHRASES):
        warnings.append(
            "Heads up: the report describes the overall tone as positive, but the system's "
            "calculated sentiment for this video is negative — worth double-checking this "
            "part yourself. "
            "| تنبيه: التقرير يصف الأجواء العامة بأنها إيجابية، لكن النظام حسب المشاعر العامة "
            "سلبية فعلياً — يُفضّل تتأكدين من هذا الجزء بنفسك"
        )
    elif unified_sentiment == "positive" and _contains_any_phrase(combined_text, NEGATIVE_SENTIMENT_PHRASES):
        warnings.append(
            "Heads up: the report describes the overall tone as negative, but the system's "
            "calculated sentiment for this video is positive — worth double-checking this "
            "part yourself. "
            "| تنبيه: التقرير يصف الأجواء العامة بأنها سلبية، لكن النظام حسب المشاعر العامة "
            "إيجابية فعلياً — يُفضّل تتأكدين من هذا الجزء بنفسك"
        )

    # ── Confidence contradiction ──
    if confidence_label == "low" and _contains_any_phrase(combined_text, CONFIDENCE_HIGH_PHRASES):
        warnings.append(
            "Heads up: the report describes high confidence in its analysis, but the "
            "system's actual confidence score for this video is low — some parts of the "
            "analysis may be less reliable than the wording suggests. "
            "| تنبيه: التقرير يصف ثقة عالية بنتيجة التحليل، لكن مستوى الثقة الفعلي المحسوب "
            "منخفض — بعض أجزاء التقرير ممكن تكون أقل دقة مما توحي الصياغة"
        )
    elif confidence_label == "high" and _contains_any_phrase(combined_text, CONFIDENCE_LOW_PHRASES):
        warnings.append(
            "Heads up: the report describes low confidence in its analysis, but the "
            "system's actual confidence score for this video is high — the analysis is "
            "likely more reliable than the wording suggests. "
            "| تنبيه: التقرير يصف ثقة منخفضة بنتيجة التحليل، لكن مستوى الثقة الفعلي المحسوب "
            "عالٍ — التحليل غالباً أدق مما توحي الصياغة"
        )

    # ── Quoted-text traceability (best-effort) ──
    # Only check quotes from whichever report field (report_ar / report_en)
    # actually matches the transcript's original language. Comparing an
    # accurately-translated quote against a differently-worded original
    # (different language entirely) via text similarity would fail every
    # time even when the translation is perfectly faithful — that's a
    # false positive, not a real hallucination signal. If the detected
    # language isn't clearly Arabic or English, skip this specific check
    # rather than risk flooding the report with false warnings.
    detected_language = summary.get("language", "").lower()

    if "arabic" in detected_language or detected_language == "ar":
        quote_source_text = report_ar
    elif "english" in detected_language or detected_language == "en":
        quote_source_text = report_en
    else:
        quote_source_text = None

    source_parts = [summary.get("full_text_preview", "")]
    for moment in summary.get("critical_text_moments", []):
        source_parts.append(moment.get("text_on_screen", ""))
    for moment in summary.get("key_moments", []):
        source_parts.append(moment.get("text", ""))
    source_blob = " ".join(p for p in source_parts if p)

    if source_blob and quote_source_text:
        quoted_strings = re.findall(
            r'"([^"]{%d,%d})"' % (QUOTE_MIN_LENGTH_FOR_CHECK, QUOTE_MAX_LENGTH_FOR_CHECK),
            quote_source_text
        )
        for quote in quoted_strings:
            if _best_match_ratio(quote, source_blob) < QUOTE_SIMILARITY_THRESHOLD:
                preview = quote[:60]
                warnings.append(
                    f"Heads up: the report includes a quoted sentence the system couldn't "
                    f"clearly match to anything actually said in the video or shown on "
                    f"screen — it may be paraphrased or inaccurate, worth checking yourself "
                    f"(quote: \"{preview}...\"). "
                    f"| تنبيه: التقرير يحتوي على جملة مقتبسة ما قدر النظام يتأكد إنها قيلت "
                    f"فعلاً بالفيديو أو ظهرت على الشاشة — ممكن تكون إعادة صياغة أو غير دقيقة، "
                    f"يُفضّل تتأكدين منها بنفسك (الاقتباس: \"{preview}...\")"
                )
                break  # one warning is enough signal — no need to flood the list

    return warnings


def _contains_any_phrase(text: str, phrases: list[str]) -> bool:
    return any(phrase.lower() in text for phrase in phrases)


def _best_match_ratio(needle: str, haystack: str) -> float:
    """
    Best-effort substring similarity via a sliding window + difflib ratio.
    Not an efficient general-purpose algorithm, but needle is a single
    report quote (rare) and haystack is capped (full_text_preview is at
    most 5000 chars) — fine at this scale.
    """
    if not needle or not haystack:
        return 0.0

    if needle.lower() in haystack.lower():
        return 1.0

    window = len(needle)
    step   = max(1, window // 2)
    best   = 0.0

    for start in range(0, max(1, len(haystack) - window + 1), step):
        chunk = haystack[start:start + window]
        ratio = difflib.SequenceMatcher(None, needle.lower(), chunk.lower()).ratio()
        best  = max(best, ratio)

    return best


def _get_min_report_length(report_mode: str, total_segments: int) -> int:
    """
    Decide the minimum acceptable report length based on BOTH the report
    mode and how much content the video actually has.

    Why not one fixed number for every report?
    A fixed floor (e.g. 200 chars) treats a 10-second TikTok clip (maybe
    2 spoken sentences) and a full lecture the same way — but a short,
    accurate summary of a tiny clip is naturally much shorter than a
    thorough analysis of an hour of content, and that's not a quality
    problem. We use total_segments as a proxy for "how much material was
    there to report on" and scale the floor accordingly.

        total_segments <= 5   → "tiny"   video (e.g. a short clip)
        total_segments <= 20  → "small"  video
        total_segments >  20  → "normal" video

    Summary reports also get a lower floor than analysis reports, since
    a summary is expected to be more compact by design.
    """
    if total_segments <= 5:
        tier = "tiny"
    elif total_segments <= 20:
        tier = "small"
    else:
        tier = "normal"

    lengths = SUMMARY_MIN_LENGTHS if report_mode == "summary" else ANALYSIS_MIN_LENGTHS
    return lengths[tier]


def _validate_report(report_ar: str, report_en: str, report_mode: str, total_segments: int) -> list[str]:
    """
    Run structural quality checks on the generated reports and return a list
    of human-readable warnings. This does NOT block report generation — it's
    a signal for the UI / caller to surface to the user.
    """
    warnings: list[str] = []
    min_length = _get_min_report_length(report_mode, total_segments)

    if len(report_ar) < min_length:
        warnings.append(
            f"Arabic report is very short ({len(report_ar)} chars, expected at least {min_length}). "
            f"| التقرير العربي قصير جداً ({len(report_ar)} حرف، المتوقع {min_length} على الأقل)"
        )

    if len(report_en) < min_length:
        warnings.append(
            f"English report is very short ({len(report_en)} chars, expected at least {min_length}). "
            f"| التقرير الإنجليزي قصير جداً ({len(report_en)} حرف، المتوقع {min_length} على الأقل)"
        )

    if not report_ar.strip():
        warnings.append("Arabic report is empty. | التقرير العربي فاضي تماماً")

    if not report_en.strip():
        warnings.append("English report is empty. | التقرير الإنجليزي فاضي تماماً")

    has_arabic_char = any("\u0600" <= c <= "\u06FF" for c in report_ar)
    if report_ar.strip() and not has_arabic_char:
        warnings.append(
            "Arabic report does not contain any actual Arabic characters. "
            "| التقرير 'العربي' لا يحتوي على أي حرف عربي فعلي"
        )

    return warnings


# ──────────────────────────────────────────────
# Error Helper
# ──────────────────────────────────────────────

def _error_result(message: str) -> dict:
    """Return a consistent error result dictionary."""
    print(f"ERROR: {message}")
    return {
        "report_ar"           : "",
        "report_en"           : "",
        "validation_warnings" : [],
        "status"              : "error",
        "message"             : message
    }