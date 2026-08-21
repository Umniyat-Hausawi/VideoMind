import os
import time
import base64
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

VALID_SENTIMENTS = ["positive", "negative", "neutral", "mixed"]

# Retry settings for transient API failures — same policy as audio_layer:
# one extra attempt after a short delay before treating a frame as failed.
RETRY_MAX_ATTEMPTS  = 2
RETRY_DELAY_SECONDS = 1


# ──────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────

def process_visual(frames_manifest: list[dict], video_duration_seconds: float = 0.0) -> dict:
    """
    Main entry point for the Visual Layer.

    Args:
        frames_manifest         : list of {"path": str, "timestamp": float},
                                   one entry per extracted frame — this is
                                   exactly what input_layer.process_input()
                                   returns as "frames_manifest". Timestamps
                                   come from this structured data, not
                                   parsed out of the frame filename, so the
                                   pipeline doesn't depend on a fixed
                                   filename encoding staying consistent
                                   between input_layer and here.
        video_duration_seconds  : total video duration (from input_layer's
                                  "duration_seconds"), used to scale the
                                  frame analysis cap. Defaults to 0.0, which
                                  falls into the shortest tier (90 frames)
                                  when duration isn't available.

    Returns:
        {
            "frame_analyses"               : list,  # each item has "is_fallback": bool, "is_smoothed": bool
            "overall_sentiment"            : str,
            "overall_sentiment_is_fallback": bool,  # True only if EVERY analyzed frame fell back
            "status"                       : "success" | "error",
            "message"                      : str
        }
    """
    if not frames_manifest:
        return _error_result(
            "No frames provided. | لا توجد لقطات"
        )

    # Sort by timestamp (structured data), not by filename string order.
    sorted_manifest = sorted(frames_manifest, key=lambda f: f["timestamp"])

    max_frames = _get_max_frames_for_duration(video_duration_seconds)

    # Sample frames evenly up to max_frames
    # Short video  → analyze all frames (if less than max_frames)
    # Long video   → sample evenly distributed frames up to max_frames
    sampled_frames = _sample_frames(sorted_manifest, max_frames)
    print(f"Total frames: {len(sorted_manifest)} — Analyzing: {len(sampled_frames)} (cap: {max_frames})")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    frame_analyses = []
    sentiments     = []

    for i, frame_entry in enumerate(sampled_frames):
        print(f"Analyzing frame {i+1}/{len(sampled_frames)}: {Path(frame_entry['path']).name}")

        analysis = _analyze_frame(client, frame_entry)

        if analysis["status"] == "error":
            print(f"Skipping frame due to error: {analysis['message']}")
            continue

        frame_analyses.append(analysis)

    if not frame_analyses:
        return _error_result(
            "Failed to analyze any frames. | فشل تحليل كل اللقطات"
        )

    # Temporal smoothing — dampen isolated single-frame sentiment outliers
    # using neighboring frames, before computing the overall majority vote.
    frame_analyses = _apply_temporal_smoothing(frame_analyses)
    sentiments = [fa["sentiment"] for fa in frame_analyses]

    overall_sentiment   = _calculate_overall_sentiment(sentiments)
    overall_is_fallback = all(fa.get("is_fallback", False) for fa in frame_analyses)

    print(
        f"Visual analysis complete — {len(frame_analyses)} frames analyzed | "
        f"overall sentiment: {overall_sentiment} ({'fallback' if overall_is_fallback else 'model'})"
    )

    return {
        "frame_analyses"                : frame_analyses,
        "overall_sentiment"             : overall_sentiment,
        "overall_sentiment_is_fallback" : overall_is_fallback,
        "status"                        : "success",
        "message"                       : (
            f"Analyzed {len(frame_analyses)} frames "
            f"| تم تحليل {len(frame_analyses)} لقطة"
        )
    }


# ──────────────────────────────────────────────
# Retry Helper
# ──────────────────────────────────────────────

def _retry_call(func, *args, **kwargs):
    """
    Call `func(*args, **kwargs)`, retrying once (RETRY_MAX_ATTEMPTS total
    attempts) after a short delay if it raises an exception. Same policy
    and rationale as audio_layer's _retry_call — most API failures here
    are transient network/server hiccups that succeed on a second try.
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
# Helper Functions
# ──────────────────────────────────────────────

def _get_max_frames_for_duration(duration_seconds: float) -> int:
    """
    Calculate the frame analysis cap based on video duration.

    Why scale with duration instead of one fixed cap?
    A fixed 60-frame cap made sense for short clips, but for a long lecture
    or podcast it means sampling a frame every few minutes — too sparse for
    good visual context. Longer videos get a higher cap; short videos stay
    cheap and fast.

    The caps below (90/180/270) balance coverage against Claude Vision API
    cost — a denser cap gives better temporal resolution but multiplies
    Vision calls directly, so each tier is set to roughly a 1.5x margin
    over its prior tier rather than a larger jump. See input_layer.py's
    _get_frame_interval_for_duration() for the matching extraction-side
    logic (short videos use a denser extraction interval specifically to
    supply enough raw frames for their 90-frame cap).

        video <= 10 min → 90 frames
        video <= 30 min → 180 frames
        video >  30 min → 270 frames

    Note on the ">30 min" tier in practice: this function's own ceiling is
    270 frames for anything past 30 minutes, all the way up to the
    2-hour hard limit enforced in input_layer.py. But app.py auto-forces
    audio_only=True for any video over 1 hour (see app.py's "Auto-enable
    audio-only for long videos" step), which skips this function entirely.
    So in the actual running app, the 270-frame tier is only ever reached
    for videos roughly 30-60 minutes long, not the full 30 min-2 hour
    range this function's math alone would suggest.
    """
    if duration_seconds <= 10 * 60:
        return 90
    elif duration_seconds <= 30 * 60:
        return 180
    else:
        return 270


def _sample_frames(frames_manifest: list[dict], max_frames: int) -> list[dict]:
    """
    Sample frame manifest entries evenly up to max_frames.

    Why evenly distributed instead of first N frames?
    Taking the first N frames only covers the beginning of the video.
    Evenly distributed frames give coverage across the full video duration.

    Examples:
        100 entries, max=60 → take every 1.6th entry → 60 entries
        30 entries,  max=60 → take all 30 entries
    """
    if len(frames_manifest) <= max_frames:
        return frames_manifest

    step = len(frames_manifest) / max_frames
    return [frames_manifest[int(i * step)] for i in range(max_frames)]


def _analyze_frame(client, frame_entry: dict) -> dict:
    """
    Analyze a single frame using Claude Vision, with a retry on transient
    API failures.

    Uses claude-haiku — fast and cheap for simple frame descriptions.
    """
    frame_path = frame_entry["path"]
    timestamp  = frame_entry["timestamp"]

    try:
        with open(frame_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = _retry_call(
            client.messages.create,
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 300,
            messages   = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type"       : "base64",
                                "media_type" : "image/jpeg",
                                "data"       : image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyze this video frame briefly. Respond in this exact format:

DESCRIPTION: [1-2 sentences: what is happening, who is present — including
whether each person shown appears to be an adult or a child, based ONLY
on visual appearance (face, body proportions, clothing), never inferred
from how clearly or fluently anyone is speaking]
SENTIMENT: [positive / negative / neutral / mixed]
EMOTION: [main emotion visible: happy / sad / angry / excited / calm / serious / other]
TEXT_ON_SCREEN: [transcribe the visible text EXACTLY as written, in its original language — do not translate it or describe it in English. If there are multiple separate text elements, separate them with " | ". If there is truly no text visible, write 'none']
BRAND_LOGOS_VISIBLE: [name any recognizable brand logos, packaging, signage, or trademarks visible in this frame — e.g. a restaurant's logo on a bag or sign, a product's brand mark — based on the visual logo/mark itself, NOT on any text you already transcribed above. This catches brands shown visually even when no one says the brand name out loud. If there are multiple, separate them with " | ". If none are visible, write 'none']"""
                        }
                    ],
                }
            ],
        )

        raw_response_text = response.content[0].text
        parsed = _parse_frame_response(raw_response_text)

        # Validate model output. A frame is flagged as a fallback in two
        # cases: the model returned a SENTIMENT value outside the 4
        # allowed options, OR the response has no "SENTIMENT:" line at
        # all (empty/garbled/off-format). The second check matters
        # because _parse_frame_response's own default for a missing
        # field is "neutral" — a technically valid value — so checking
        # only against VALID_SENTIMENTS would let a genuinely failed
        # parse pass silently as a confident "neutral" reading, which
        # defeats the purpose of the is_fallback flag (see
        # LIMITATIONS.md's sentiment reliability entry).
        is_fallback = False
        if parsed["sentiment"] not in VALID_SENTIMENTS:
            print(
                f"WARNING: Frame sentiment fallback — model returned invalid "
                f"value '{parsed['sentiment']}' for {frame_path}"
            )
            parsed["sentiment"] = "neutral"
            is_fallback = True
        elif "SENTIMENT:" not in raw_response_text:
            print(
                f"WARNING: Frame sentiment fallback — no SENTIMENT line found "
                f"in response for {frame_path}"
            )
            is_fallback = True

        return {
            "timestamp"     : timestamp,
            "frame_path"    : frame_path,
            "description"   : parsed["description"],
            "sentiment"     : parsed["sentiment"],
            "emotion"       : parsed["emotion"],
            "text_on_screen": parsed["text_on_screen"],
            "brand_logos"   : parsed["brand_logos"],
            "is_fallback"   : is_fallback,
            "status"        : "success",
            "message"       : "OK"
        }

    except Exception as e:
        return _error_result(f"Frame analysis error: {str(e)} | خطأ بتحليل اللقطة")


def _parse_frame_response(response_text: str) -> dict:
    """Parse Claude's structured response into a dictionary."""
    result = {
        "description"    : "",
        "sentiment"      : "neutral",
        "emotion"        : "unknown",
        "text_on_screen" : "none",
        "brand_logos"    : "none"
    }

    for line in response_text.strip().split("\n"):
        if line.startswith("DESCRIPTION:"):
            result["description"] = line.replace("DESCRIPTION:", "").strip()
        elif line.startswith("SENTIMENT:"):
            result["sentiment"] = line.replace("SENTIMENT:", "").strip().lower()
        elif line.startswith("EMOTION:"):
            result["emotion"] = line.replace("EMOTION:", "").strip().lower()
        elif line.startswith("TEXT_ON_SCREEN:"):
            result["text_on_screen"] = line.replace("TEXT_ON_SCREEN:", "").strip()
        elif line.startswith("BRAND_LOGOS_VISIBLE:"):
            result["brand_logos"] = line.replace("BRAND_LOGOS_VISIBLE:", "").strip()

    return result


# ──────────────────────────────────────────────
# Temporal Smoothing
# ──────────────────────────────────────────────

# How many neighboring frames on each side to consider when checking
# whether a frame's sentiment is isolated noise. 2 on each side means a
# 5-frame window total (current + 2 before + 2 after).
SMOOTHING_WINDOW_RADIUS = 2

# A neighbor "majority" must have at least this many agreeing frames
# before it's allowed to override an isolated frame's sentiment — a
# single neighbor isn't strong enough evidence to overrule the model's
# direct read of this exact frame.
SMOOTHING_MIN_AGREEING_NEIGHBORS = 2


def _apply_temporal_smoothing(frame_analyses: list[dict]) -> list[dict]:
    """
    Dampen isolated single-frame sentiment outliers using neighboring
    frames, without erasing genuine mood shifts.

    Why this specific rule (not just "always match neighbors")?
    A real mood shift in the video shows up across 2+ consecutive frames,
    not just one. So a frame is only treated as noise — and overridden —
    when BOTH of these hold:
      1. It disagrees with every neighbor in its window (nothing nearby
         shares its sentiment)
      2. At least SMOOTHING_MIN_AGREEING_NEIGHBORS neighbors agree with
         EACH OTHER on some other sentiment (a real, consistent
         neighborhood to borrow from — not just noise on both sides)

    This mirrors the same logic already used for audio (splitting the
    transcript into multi-segment periods instead of trusting a single
    sentence), applied here across neighboring frames instead of time
    periods.

    Adds "is_smoothed": bool to every frame's dict — True only when this
    function overrode the model's original sentiment — mirroring the
    existing "is_fallback" transparency pattern, so downstream layers/UI
    can tell "the model said this" from "we adjusted this using
    neighboring frames."

    Expects frame_analyses already in chronological order (the caller,
    process_visual, analyzes sampled_frames in the manifest's sorted
    order, so this holds without needing to re-sort here).
    """
    n = len(frame_analyses)

    if n < 3:
        # Not enough frames for a meaningful neighborhood — leave as-is.
        for frame in frame_analyses:
            frame["is_smoothed"] = False
        return frame_analyses

    # Snapshot original sentiments before any overrides, so each frame's
    # neighbor-check uses the model's real original reads — not sentiments
    # already overridden by smoothing earlier in this same pass.
    original_sentiments = [f["sentiment"] for f in frame_analyses]

    for i, frame in enumerate(frame_analyses):
        # Boundary frames (near the very start/end of the video) don't have
        # a full two-sided window — e.g. the last frame only has "before"
        # neighbors, never "after". Smoothing those with one-sided context
        # risks overriding a frame that's actually the start of a genuine
        # transition we simply can't confirm yet (no future frame to check
        # against). Only frames with a complete window on both sides are
        # eligible for smoothing; edge frames are left as the model read them.
        if i < SMOOTHING_WINDOW_RADIUS or i >= n - SMOOTHING_WINDOW_RADIUS:
            frame["is_smoothed"] = False
            continue

        window_start = i - SMOOTHING_WINDOW_RADIUS
        window_end   = i + SMOOTHING_WINDOW_RADIUS + 1
        neighbor_sentiments = [
            original_sentiments[j] for j in range(window_start, window_end) if j != i
        ]

        current = original_sentiments[i]

        # Not isolated — at least one neighbor already agrees.
        if current in neighbor_sentiments:
            frame["is_smoothed"] = False
            continue

        neighbor_counts: dict[str, int] = {}
        for s in neighbor_sentiments:
            neighbor_counts[s] = neighbor_counts.get(s, 0) + 1

        if not neighbor_counts:
            frame["is_smoothed"] = False
            continue

        majority_count = max(neighbor_counts.values())
        tied_sentiments = [s for s, count in neighbor_counts.items() if count == majority_count]

        # A true tie among neighbors (e.g. 2 agreeing on "positive" vs 2
        # agreeing on "negative") is NOT treated as a clear majority to
        # borrow from, even if the tied count meets
        # SMOOTHING_MIN_AGREEING_NEIGHBORS. An exact split like that is
        # usually a real transitional moment in the video (the mood is
        # genuinely changing right around this frame), not noise — so the
        # frame is left as the model read it rather than arbitrarily
        # snapping to whichever sentiment happened to be seen first.
        if len(tied_sentiments) > 1:
            frame["is_smoothed"] = False
        elif majority_count >= SMOOTHING_MIN_AGREEING_NEIGHBORS:
            frame["sentiment"]   = tied_sentiments[0]
            frame["is_smoothed"] = True
        else:
            frame["is_smoothed"] = False

    return frame_analyses


# ──────────────────────────────────────────────
# Overall Sentiment
# ──────────────────────────────────────────────

def _calculate_overall_sentiment(sentiments: list[str]) -> str:
    """Calculate overall sentiment — simple majority vote."""
    if not sentiments:
        return "neutral"

    counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    for s in sentiments:
        if s in counts:
            counts[s] += 1
        else:
            counts["neutral"] += 1

    return max(counts, key=counts.get)


# ──────────────────────────────────────────────
# Error Helper
# ──────────────────────────────────────────────

def _error_result(message: str) -> dict:
    """Return a consistent error result dictionary."""
    print(f"ERROR: {message}")
    return {
        "frame_analyses"                : [],
        "overall_sentiment"             : "neutral",
        "overall_sentiment_is_fallback" : True,
        "status"                        : "error",
        "message"                       : message
    }