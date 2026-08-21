import os
import json
import time
import subprocess
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from observability import LANGFUSE_ENABLED

load_dotenv()

# Drop-in Langfuse-traced replacement for the OpenAI SDK (used for
# Whisper transcription below) — only when observability is actually
# enabled (see observability.py). `openai.OpenAI(...)` and every method
# on it work identically either way; the traced version just also sends
# each call's latency/cost/response to Langfuse. Falls back to the
# plain SDK if `langfuse` isn't installed, so audio_layer never hard-
# depends on it.
if LANGFUSE_ENABLED:
    try:
        from langfuse.openai import openai
    except ImportError:
        import openai
else:
    import openai


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WHISPER_MODEL     = "whisper-1"

VALID_SENTIMENTS = ["positive", "negative", "neutral", "mixed"]

# Retry settings for transient API failures (network hiccups, momentary
# server overload). One extra attempt is enough for most transient issues
# without adding too much latency if the failure is not transient.
RETRY_MAX_ATTEMPTS  = 2
RETRY_DELAY_SECONDS = 1

# Max characters of transcript text sent per time period for sentiment
# analysis.
PERIOD_TEXT_CHAR_LIMIT = 5000

# Max number of segments analyzed per sentiment-batching API call. A single
# call for a very long transcript (300+ segments) risks the JSON response
# being cut off mid-write when it exceeds max_tokens — this cap keeps every
# batch's response comfortably within budget, regardless of video length.
SEGMENTS_PER_BATCH = 200


# ──────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────

def process_audio(audio_chunks: list[str]) -> dict:
    """
    Main entry point for the Audio Layer.

    Args:
        audio_chunks : list of audio file paths (from input_layer)

    Returns:
        {
            "full_text"                    : str,
            "segments"                     : list,
            "language"                     : str,
            "overall_sentiment"            : str,   # majority sentiment across the timeline
            "overall_sentiment_is_fallback": bool,  # True only if EVERY period fell back
            "sentiment_timeline"           : list,  # [{period, start, end, sentiment, is_fallback}]
            "segment_sentiments"           : list,  # each item has "is_fallback": bool
            "status"                       : "success" | "error",
            "message"                      : str
        }
    """
    if not audio_chunks:
        return _error_result(
            "No audio chunks provided. | لا توجد قطع صوتية للمعالجة"
        )

    openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    all_segments      = []
    all_text          = []
    detected_language = "unknown"
    time_offset       = 0.0

    for i, chunk_path in enumerate(audio_chunks):
        print(f"Transcribing chunk {i+1}/{len(audio_chunks)}: {chunk_path}")

        chunk_result = _transcribe_chunk(openai_client, chunk_path, time_offset)

        if chunk_result["status"] == "error":
            return _error_result(
                f"Failed on chunk {i+1}: {chunk_result['message']} "
                f"| فشل بالقطعة الصوتية {i+1}: {chunk_result['message']}"
            )

        all_text.append(chunk_result["text"])
        all_segments.extend(chunk_result["segments"])

        if detected_language == "unknown":
            detected_language = chunk_result["language"]

        time_offset += chunk_result["duration"]

    full_text = " ".join(all_text)

    print("Analyzing text sentiment across the timeline...")
    overall_sentiment, overall_is_fallback, sentiment_timeline = _analyze_sentiment_timeline(
        anthropic_client, all_segments
    )
    segment_sentiments = _analyze_segment_sentiments_batch(anthropic_client, all_segments)

    print(
        f"Transcription complete — {len(all_segments)} segments | "
        f"overall sentiment: {overall_sentiment} | {len(sentiment_timeline)} time period(s) analyzed"
    )

    return {
        "full_text"                     : full_text,
        "segments"                      : all_segments,
        "language"                      : detected_language,
        "overall_sentiment"             : overall_sentiment,
        "overall_sentiment_is_fallback" : overall_is_fallback,
        "sentiment_timeline"            : sentiment_timeline,
        "segment_sentiments"            : segment_sentiments,
        "status"                        : "success",
        "message"                       : (
            f"Transcribed {len(audio_chunks)} chunk(s), {len(all_segments)} segments "
            f"| تم تحويل {len(audio_chunks)} قطعة صوتية، {len(all_segments)} جملة"
        )
    }


# ──────────────────────────────────────────────
# Retry Helper
# ──────────────────────────────────────────────

def _retry_call(func, *args, **kwargs):
    """
    Call `func(*args, **kwargs)`, retrying once (RETRY_MAX_ATTEMPTS total
    attempts) after a short delay if it raises an exception.

    Why retry at all? A lot of API failures are transient — a brief network
    hiccup, a momentary server overload — and succeed on a second try. This
    is NOT meant to paper over persistent failures (those still surface as
    errors after the retry budget is exhausted), just to avoid throwing away
    an entire chunk/period/frame over a one-second blip.
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

def _get_audio_duration(audio_path: str) -> float:
    """
    Get the real duration of an audio file in seconds using ffprobe.

    Why not use the last Whisper segment's end time instead?
    Whisper segments stop at the last detected speech — any trailing
    silence at the end of a chunk is not counted, which throws off the
    time_offset used to align later chunks (and therefore all timestamps
    used for visual alignment downstream).
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        print(f"WARNING: Could not read duration for '{audio_path}' via ffprobe — using 0.0")
        return 0.0


def _transcribe_chunk(client, chunk_path: str, time_offset: float) -> dict:
    """
    Transcribe a single audio chunk using OpenAI Whisper API, with a retry
    on transient failures. The file is re-opened fresh on each attempt
    (a partially-read file handle from a failed attempt can't be reused).
    """
    if not os.path.exists(chunk_path):
        return _error_result(f"Chunk file not found: {chunk_path} | ملف القطعة الصوتية غير موجود")

    last_error = None

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            with open(chunk_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model                    = WHISPER_MODEL,
                    file                     = audio_file,
                    response_format          = "verbose_json",
                    timestamp_granularities  = ["segment"]
                )

            segments = []
            for seg in response.segments:
                segments.append({
                    "start" : round(seg.start + time_offset, 2),
                    "end"   : round(seg.end   + time_offset, 2),
                    "text"  : seg.text.strip()
                })

            duration = _get_audio_duration(chunk_path)

            return {
                "text"     : response.text,
                "segments" : segments,
                "language" : response.language,
                "duration" : duration,
                "status"   : "success",
                "message"  : "OK"
            }

        except (openai.APIError, Exception) as e:
            last_error = e
            if attempt < RETRY_MAX_ATTEMPTS:
                print(
                    f"WARNING: Whisper call failed (attempt {attempt}/{RETRY_MAX_ATTEMPTS}) "
                    f"— retrying in {RETRY_DELAY_SECONDS}s: {str(e)}"
                )
                time.sleep(RETRY_DELAY_SECONDS)

    return _error_result(
        f"Whisper API error after {RETRY_MAX_ATTEMPTS} attempt(s): {str(last_error)} "
        f"| خطأ بـ Whisper بعد {RETRY_MAX_ATTEMPTS} محاولة"
    )


# ──────────────────────────────────────────────
# Sentiment Timeline (periods instead of a single overall snapshot)
# ──────────────────────────────────────────────

def _get_period_count_for_duration(span_seconds: float) -> int:
    """
    Decide how many time periods to split the transcript into, based on
    how much of the video the transcript actually spans.

    Why not a single fixed number for every video?
    A 2-minute clip doesn't need 6 separate sentiment checks — 4 is already
    more granular than the content justifies. A 90-minute lecture, on the
    other hand, easily shifts tone across its length, so more periods give
    a much more honest picture than one sentiment value for the whole thing.

        span <= 15 min → 4 periods
        span <= 45 min → 5 periods
        span >  45 min → 6 periods
    """
    if span_seconds <= 15 * 60:
        return 4
    elif span_seconds <= 45 * 60:
        return 5
    else:
        return 6


def _split_segments_into_periods(segments: list, num_periods: int) -> list[list]:
    """
    Split segments into num_periods equal-length time buckets, based on
    each segment's midpoint timestamp. Returns a list of lists (some
    buckets may be empty if segments are unevenly distributed in time).
    """
    if not segments:
        return [[] for _ in range(num_periods)]

    span_start   = segments[0]["start"]
    span_end     = segments[-1]["end"]
    total_span   = max(span_end - span_start, 0.001)  # avoid division by zero
    period_length = total_span / num_periods

    periods: list[list] = [[] for _ in range(num_periods)]

    for seg in segments:
        midpoint     = (seg["start"] + seg["end"]) / 2
        period_index = int((midpoint - span_start) / period_length)
        period_index = max(0, min(period_index, num_periods - 1))
        periods[period_index].append(seg)

    return periods


def _analyze_period_sentiment(client, period_segments: list, period_number: int,
                                period_start: float, period_end: float) -> dict:
    """
    Analyze the sentiment of a single time period's worth of transcript.
    Returns a dict describing that period, including whether the result
    is a real model output or a fallback.
    """
    combined_text = " ".join(seg["text"] for seg in period_segments).strip()

    base_entry = {
        "period" : period_number,
        "start"  : round(period_start, 1),
        "end"    : round(period_end, 1),
    }

    if not combined_text:
        # No speech fell into this time window — nothing to analyze.
        return {**base_entry, "sentiment": "neutral", "is_fallback": True}

    try:
        response = _retry_call(
            client.messages.create,
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 50,
            messages   = [{
                "role"    : "user",
                "content" : f"""Analyze the overall sentiment of this transcript segment.
Reply with ONLY one word: positive, negative, neutral, or mixed.

Transcript segment:
{combined_text[:PERIOD_TEXT_CHAR_LIMIT]}"""
            }]
        )
        sentiment = response.content[0].text.strip().lower()

        if sentiment not in VALID_SENTIMENTS:
            print(f"WARNING: Period {period_number} sentiment fallback — invalid value: '{sentiment}'")
            return {**base_entry, "sentiment": "neutral", "is_fallback": True}

        return {**base_entry, "sentiment": sentiment, "is_fallback": False}

    except Exception as e:
        print(f"WARNING: Period {period_number} sentiment fallback — API call failed: {str(e)}")
        return {**base_entry, "sentiment": "neutral", "is_fallback": True}


def _majority_sentiment(sentiments: list[str]) -> str:
    """Pick the most frequent sentiment value — simple majority vote."""
    if not sentiments:
        return "neutral"

    counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    for s in sentiments:
        if s in counts:
            counts[s] += 1
        else:
            counts["neutral"] += 1

    return max(counts, key=counts.get)


def _analyze_sentiment_timeline(client, segments: list) -> tuple[str, bool, list]:
    """
    Analyze sentiment across the transcript's timeline, split into several
    time periods (3-6 depending on how long the transcript spans) instead
    of a single value for the whole video. This captures shifts in tone
    over the course of a longer video that a single snapshot would miss.

    Why Claude for sentiment instead of a specialized model?
    Claude understands context, sarcasm, implicit emotions, and Arabic
    dialects better than dedicated sentiment models trained on formal
    text only.

    Returns:
        overall_sentiment   : str   — majority sentiment across all periods
        overall_is_fallback : bool  — True only if EVERY period fell back
                                       (i.e. we have no real signal at all)
        sentiment_timeline  : list  — [{period, start, end, sentiment, is_fallback}]
    """
    if not segments:
        return "neutral", True, []

    span_seconds = segments[-1]["end"] - segments[0]["start"]
    num_periods  = _get_period_count_for_duration(span_seconds)
    period_length = max(span_seconds / num_periods, 0.001)

    periods_of_segments = _split_segments_into_periods(segments, num_periods)

    timeline = []
    for i, period_segments in enumerate(periods_of_segments):
        period_start = segments[0]["start"] + i * period_length
        period_end   = period_start + period_length
        entry = _analyze_period_sentiment(client, period_segments, i + 1, period_start, period_end)
        timeline.append(entry)

    overall_sentiment   = _majority_sentiment([p["sentiment"] for p in timeline])
    overall_is_fallback = all(p["is_fallback"] for p in timeline)

    print(
        f"Sentiment timeline complete — {num_periods} period(s), "
        f"overall: {overall_sentiment} ({'fallback' if overall_is_fallback else 'model'})"
    )
    return overall_sentiment, overall_is_fallback, timeline


def _analyze_segment_sentiments_batch(client, segments: list) -> list:
    """
    Analyze sentiment for ALL segments, split into batches of at most
    SEGMENTS_PER_BATCH sentences per API call, instead of one call for
    everything or a sampled subset.

    Why batches instead of one call for all segments?
    A single call for a large transcript (300+ segments) can produce a
    JSON response that exceeds the model's max_tokens budget mid-write —
    the response gets cut off with an unterminated string, fails to parse,
    and the ENTIRE transcript falls back to neutral (a real failure mode
    observed in testing). Capping each call to a fixed, safe number of
    segments guarantees enough token headroom for a complete response
    every time, regardless of how long the video is — a 311-segment
    transcript becomes 2 calls instead of 1 oversized one that fails.
    """
    if not segments:
        return []

    batches = [
        segments[i:i + SEGMENTS_PER_BATCH]
        for i in range(0, len(segments), SEGMENTS_PER_BATCH)
    ]

    if len(batches) > 1:
        print(f"Splitting {len(segments)} segments into {len(batches)} batch(es) of up to {SEGMENTS_PER_BATCH} each")

    all_results = []
    for batch_number, batch in enumerate(batches, start=1):
        batch_results = _analyze_one_sentiment_batch(client, batch, batch_number, len(batches))
        all_results.extend(batch_results)

    return all_results


def _analyze_one_sentiment_batch(client, segments: list, batch_number: int, total_batches: int) -> list:
    """
    Analyze sentiment for a single batch of segments (at most
    SEGMENTS_PER_BATCH of them) in one API call. Batching keeps each
    response size-bounded so it always fits comfortably within
    max_tokens, regardless of how long the full transcript is.
    """
    numbered_text = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(segments))

    prompt = f"""Analyze the sentiment of each numbered sentence below.
Reply with ONLY a JSON array, no other text, no markdown formatting, in this exact format:
[{{"id": 1, "sentiment": "positive"}}, {{"id": 2, "sentiment": "neutral"}}]

Rules:
- "sentiment" must be exactly one of: positive, negative, neutral, mixed
- You MUST include exactly one entry for every numbered sentence below, using the same id numbers
- Do not add any text before or after the JSON array

Sentences:
{numbered_text}"""

    sentiment_by_id = {}
    batch_failed    = False
    max_tokens      = min(4096, max(200, len(segments) * 15))

    try:
        if total_batches > 1:
            print(f"Analyzing sentiment batch {batch_number}/{total_batches} ({len(segments)} segments)...")

        response = _retry_call(
            client.messages.create,
            model      = "claude-haiku-4-5-20251001",
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}]
        )

        raw_text = response.content[0].text.strip()

        # Defensively strip markdown code fences if the model adds them anyway
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        parsed = json.loads(raw_text)

        for item in parsed:
            seg_id    = item.get("id")
            sentiment = str(item.get("sentiment", "")).strip().lower()
            if sentiment not in VALID_SENTIMENTS:
                sentiment = "neutral"
            sentiment_by_id[seg_id] = sentiment

    except Exception as e:
        print(
            f"WARNING: Sentiment batch {batch_number}/{total_batches} failed — "
            f"falling back to neutral for these {len(segments)} segments: {str(e)}"
        )
        batch_failed = True

    segment_sentiments = []
    for i, seg in enumerate(segments):
        seg_id = i + 1

        if batch_failed:
            sentiment   = "neutral"
            is_fallback = True
        elif seg_id not in sentiment_by_id:
            print(f"WARNING: Segment {seg_id} missing from batch sentiment response — using neutral fallback")
            sentiment   = "neutral"
            is_fallback = True
        else:
            sentiment   = sentiment_by_id[seg_id]
            is_fallback = False

        segment_sentiments.append({
            "start"       : seg["start"],
            "end"         : seg["end"],
            "text"        : seg["text"],
            "sentiment"   : sentiment,
            "is_fallback" : is_fallback
        })

    return segment_sentiments


# ──────────────────────────────────────────────
# Error Helper
# ──────────────────────────────────────────────

def _error_result(message: str) -> dict:
    """Return a consistent error result dictionary."""
    print(f"ERROR: {message}")
    return {
        "full_text"                     : "",
        "segments"                      : [],
        "language"                      : "unknown",
        "overall_sentiment"             : "neutral",
        "overall_sentiment_is_fallback" : True,
        "sentiment_timeline"            : [],
        "segment_sentiments"            : [],
        "status"                        : "error",
        "message"                       : message
    }