import os
import difflib
from pathlib import Path


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

HIGH_CONFIDENCE   = 0.90
MEDIUM_CONFIDENCE = 0.70
LOW_CONFIDENCE    = 0.50

AUDIO_WEIGHT  = 0.65
VISUAL_WEIGHT = 0.35

# Sentiment → numeric value for weighted fusion when audio/visual disagree
SENTIMENT_VALUES = {
    "positive": 1,
    "neutral" : 0,
    "mixed"   : 0,
    "negative": -1
}

WEIGHTED_POSITIVE_THRESHOLD = 0.3
WEIGHTED_NEGATIVE_THRESHOLD = -0.3

# Max allowed time gap (seconds) between a segment and its "closest" frame
# before we consider no frame actually available for that moment
MAX_ALIGNMENT_GAP_SECONDS = 15


# ──────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────

def process_fusion(audio_result: dict, visual_result: dict) -> dict:
    """
    Main entry point for the Fusion Layer.

    Combines audio transcription + visual analysis into one unified result.

    Args:
        audio_result  : output from audio_layer
        visual_result : output from visual_layer

    Returns:
        {
            "unified_sentiment"    : str,
            "confidence"           : float,
            "confidence_label"     : str,
            "signal"               : str,
            "fusion_score"         : float | None,  # numeric weighted score when audio/visual disagreed
            "analysis_reliability" : dict,          # {"score": float, "fallback_items": int, "total_items": int}
            "aligned_segments"     : list,          # each item may now include "critical_flag": True
            "summary"              : dict,
            "status"               : "success" | "error",
            "message"              : str
        }
    """
    if audio_result.get("status") == "error":
        return _error_result(
            f"Audio layer failed: {audio_result.get('message')} "
            f"| فشلت طبقة الصوت: {audio_result.get('message')}"
        )

    if visual_result.get("status") == "error":
        return _error_result(
            f"Visual layer failed: {visual_result.get('message')} "
            f"| فشلت طبقة الصورة: {visual_result.get('message')}"
        )

    # Use real sentiment from Claude text analysis — not hardcoded neutral
    audio_sentiment  = _normalize_sentiment(audio_result.get("overall_sentiment", "neutral"))
    visual_sentiment = _normalize_sentiment(visual_result.get("overall_sentiment", "neutral"))

    unified_sentiment, confidence, signal, fusion_score = _fuse_sentiments(audio_sentiment, visual_sentiment)

    aligned_segments = _align_segments_with_frames(
        audio_result.get("segments", []),
        audio_result.get("segment_sentiments", []),
        visual_result.get("frame_analyses", [])
    )

    critical_text_moments = _extract_critical_text_moments(
        audio_result.get("segments", []),
        audio_result.get("segment_sentiments", []),
        visual_result.get("frame_analyses", [])
    )

    silent_brand_mentions = _extract_silent_brand_mentions(
        visual_result.get("frame_analyses", []),
        audio_result.get("full_text", "")
    )

    analysis_reliability = _calculate_analysis_reliability(audio_result, visual_result)

    summary = _build_summary(
        audio_result,
        visual_result,
        unified_sentiment,
        confidence,
        signal,
        critical_text_moments,
        silent_brand_mentions
    )

    print(
        f"Fusion complete — sentiment: {unified_sentiment} | confidence: {confidence} | "
        f"signal: {signal} | reliability: {analysis_reliability['score']}%"
    )

    return {
        "unified_sentiment"     : unified_sentiment,
        "confidence"            : confidence,
        "confidence_label"      : _confidence_label(confidence),
        "signal"                : signal,
        "fusion_score"          : fusion_score,
        "analysis_reliability"  : analysis_reliability,
        "aligned_segments"      : aligned_segments,
        "summary"               : summary,
        "status"                : "success",
        "message"               : (
            f"Fused {len(aligned_segments)} aligned segments "
            f"| تم دمج {len(aligned_segments)} جملة محاذاة"
        )
    }


# ──────────────────────────────────────────────
# Sentiment Fusion
# ──────────────────────────────────────────────

def _normalize_sentiment(sentiment: str) -> str:
    """Normalize sentiment to standard values."""
    sentiment = sentiment.lower().strip()
    if sentiment in ["positive", "negative", "neutral", "mixed"]:
        return sentiment
    return "neutral"


def _fuse_sentiments(audio_sentiment: str, visual_sentiment: str) -> tuple:
    """
    Fuse audio and visual sentiments into one unified result.

    Hybrid logic:
    - Both agree      → use that sentiment directly, no calculation needed
                         (high alignment score)
    - They disagree    → convert each sentiment to a number
                         (positive=+1, neutral/mixed=0, negative=-1) and
                         compute a weighted score:
                             score = audio_value * 0.65 + visual_value * 0.35
                         Then classify:
                             score >  0.3  → positive
                             score < -0.3  → negative
                             otherwise     → mixed

    Note: confidence here is an alignment score (heuristic),
    not a model-based probability score.

    Audio weight is higher (0.65) because words carry meaning more reliably
    than facial expressions — someone can smile while saying something negative.

    Returns:
        unified_sentiment : str
        confidence        : float
        signal            : str
        fusion_score       : float | None — the weighted numeric score,
                              only computed when audio and visual disagreed;
                              None when they agreed (no calculation needed).
    """
    if audio_sentiment == visual_sentiment:
        return audio_sentiment, HIGH_CONFIDENCE, "aligned", None

    audio_value  = SENTIMENT_VALUES.get(audio_sentiment, 0)
    visual_value = SENTIMENT_VALUES.get(visual_sentiment, 0)
    score        = (audio_value * AUDIO_WEIGHT) + (visual_value * VISUAL_WEIGHT)

    # Round BEFORE comparing to thresholds — floating point arithmetic can
    # produce values like 0.30000000000000004 instead of exactly 0.3, which
    # would incorrectly cross the ">" boundary and misclassify the result.
    score = round(score, 3)

    if score > WEIGHTED_POSITIVE_THRESHOLD:
        unified_sentiment = "positive"
    elif score < WEIGHTED_NEGATIVE_THRESHOLD:
        unified_sentiment = "negative"
    else:
        unified_sentiment = "mixed"

    if audio_sentiment == "neutral":
        signal = "visual_dominant"
    elif visual_sentiment == "neutral":
        signal = "audio_dominant"
    else:
        signal = "conflict"

    confidence = MEDIUM_CONFIDENCE if signal != "conflict" else LOW_CONFIDENCE

    return unified_sentiment, confidence, signal, score


def _confidence_label(confidence: float) -> str:
    """Convert alignment score to human-readable label."""
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    elif confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    else:
        return "low"


def _calculate_analysis_reliability(audio_result: dict, visual_result: dict) -> dict:
    """
    Combine the is_fallback flags from BOTH audio_layer and visual_layer
    into an overall reliability score, AND expose the audio/visual breakdown
    separately — a combined 95% could hide a visual layer that fell back
    on every single frame while audio was perfect, or vice versa. The
    combined "score" stays as the top-level number for the main dashboard
    (keeps the UI simple by default); the per-modality numbers are
    available for anyone who wants to see the breakdown.

    Why here, in fusion_layer?
    This is the only layer that already sees both audio_result and
    visual_result at once — the natural place to combine their individual
    fallback signals into one number the UI can show at a glance.

    "Fallback item" = a single sentence sentiment (audio) or a single frame
    analysis (visual) where the model's response was invalid/unavailable
    and we substituted a neutral placeholder instead of a real result.

    Returns:
        {
            "score"              : float — combined percentage (0-100),
                                    kept for backward compatibility with the
                                    existing dashboard metric
            "fallback_items"     : int   — combined fallback count
            "total_items"        : int   — combined total count
            "audio_reliability"  : dict  — {"score", "fallback_items", "total_items"} for audio only
            "visual_reliability" : dict  — {"score", "fallback_items", "total_items"} for visual only
        }
    """
    segment_sentiments = audio_result.get("segment_sentiments", [])
    frame_analyses      = visual_result.get("frame_analyses", [])

    audio_fallbacks  = sum(1 for s in segment_sentiments if s.get("is_fallback"))
    visual_fallbacks = sum(1 for f in frame_analyses if f.get("is_fallback"))

    audio_reliability  = _reliability_breakdown(audio_fallbacks, len(segment_sentiments))
    visual_reliability = _reliability_breakdown(visual_fallbacks, len(frame_analyses))

    total_items    = len(segment_sentiments) + len(frame_analyses)
    fallback_items = audio_fallbacks + visual_fallbacks

    if total_items == 0:
        # Nothing to measure (e.g. audio-only mode with no visual frames at
        # all, or an empty transcript) — treat as fully reliable rather
        # than dividing by zero or implying a problem that doesn't exist.
        score = 100.0
    else:
        score = round((total_items - fallback_items) / total_items * 100, 1)

    return {
        "score"              : score,
        "fallback_items"     : fallback_items,
        "total_items"        : total_items,
        "audio_reliability"  : audio_reliability,
        "visual_reliability" : visual_reliability,
    }


def _reliability_breakdown(fallback_count: int, total_count: int) -> dict:
    """Build a single-modality {score, fallback_items, total_items} block."""
    if total_count == 0:
        return {"score": 100.0, "fallback_items": 0, "total_items": 0}
    score = round((total_count - fallback_count) / total_count * 100, 1)
    return {"score": score, "fallback_items": fallback_count, "total_items": total_count}


# ──────────────────────────────────────────────
# Segment Alignment
# ──────────────────────────────────────────────

def _align_segments_with_frames(
    segments           : list,
    segment_sentiments : list,
    frame_analyses      : list
) -> list:
    """
    Align each audio segment with the closest video frame.

    text_on_screen is included per segment for reference, but conflicts
    are not flagged per-segment here — persistent on-screen elements
    (e.g. a sports scoreboard visible in nearly every frame) would
    trigger a "conflict" flag on almost every segment, which is noise
    rather than signal. See _extract_critical_text_moments for the
    filtered version used in the summary/report instead.
    """
    if not segments:
        return []

    # Build sentiment lookup by timestamp
    sentiment_lookup = {}
    for seg_sent in segment_sentiments:
        key = (seg_sent["start"], seg_sent["end"])
        sentiment_lookup[key] = seg_sent.get("sentiment", "neutral")

    aligned = []

    for segment in segments:
        segment_time  = (segment["start"] + segment["end"]) / 2
        closest_frame = _find_closest_frame(segment_time, frame_analyses)

        # Get real audio sentiment for this segment
        seg_key        = (segment["start"], segment["end"])
        audio_seg_sent = sentiment_lookup.get(seg_key, "neutral")

        frame_sentiment = closest_frame.get("sentiment", "neutral") if closest_frame else "neutral"
        text_on_screen  = closest_frame.get("text_on_screen", "none") if closest_frame else "none"

        aligned_segment = {
            "start"              : segment["start"],
            "end"                : segment["end"],
            "text"               : segment["text"],
            "timestamp"          : segment_time,
            "audio_sentiment"    : audio_seg_sent,
            "frame_description"  : closest_frame.get("description", "")   if closest_frame else "",
            "frame_emotion"      : closest_frame.get("emotion", "unknown") if closest_frame else "unknown",
            "frame_sentiment"    : frame_sentiment,
            "text_on_screen"     : text_on_screen
        }

        aligned.append(aligned_segment)

    return aligned


# Any single on-screen text that appears in more than this fraction of all
# analyzed frames is treated as persistent background (a scoreboard, a
# watermark, a recurring lower-third) rather than a meaningful one-off
# claim — this is what keeps a sports recap's ever-present scoreboard from
# flooding the "critical moments" list with false positives.
TEXT_REPETITION_THRESHOLD = 0.4

# difflib.SequenceMatcher ratio (0-1) above which two on-screen text
# readings are considered the same recurring element, not two different
# pieces of text. Chosen conservatively — high enough that genuinely
# different short phrases don't get merged, low enough to catch minor
# OCR phrasing drift (extra/missing punctuation, a misread character) on
# an otherwise-identical overlay. Worth re-tuning against a sample of
# real videos if it ever misbehaves (see LIMITATIONS.md).
TEXT_SIMILARITY_THRESHOLD = 0.8


def _group_similar_texts(text_counts: dict) -> dict:
    """
    Group near-duplicate on-screen text readings (e.g. minor OCR phrasing
    differences for the same scoreboard/watermark) so they count as one
    recurring element instead of several separate rare ones.

    Greedy single-pass clustering: each unique text is compared against
    existing cluster representatives (the first text seen in each
    cluster) using difflib.SequenceMatcher; if similar enough, its count
    is folded into that cluster. Every original text maps to its
    cluster's TOTAL combined count in the returned dict, so
    _extract_critical_text_moments' occurrence_ratio check picks up the
    fix automatically without any other change.

    Why greedy against representatives instead of full pairwise clustering?
    Full pairwise comparison is O(n²), but this runs on a small, bounded
    list — the number of *distinct* on-screen text strings across at most
    270 analyzed frames, almost always far fewer in practice. A
    representative-based greedy pass is simple and fast enough at this
    scale; it isn't meant to be a general-purpose clustering algorithm.
    """
    clusters: list[dict] = []  # each: {"representative": str, "members": list[str], "total": int}

    for text, count in text_counts.items():
        matched_cluster = None
        for cluster in clusters:
            similarity = difflib.SequenceMatcher(None, text, cluster["representative"]).ratio()
            if similarity >= TEXT_SIMILARITY_THRESHOLD:
                matched_cluster = cluster
                break

        if matched_cluster:
            matched_cluster["members"].append(text)
            matched_cluster["total"] += count
        else:
            clusters.append({"representative": text, "members": [text], "total": count})

    grouped_counts: dict[str, int] = {}
    for cluster in clusters:
        for member in cluster["members"]:
            grouped_counts[member] = cluster["total"]

    return grouped_counts


def _extract_critical_text_moments(
    segments           : list,
    segment_sentiments : list,
    frame_analyses      : list
) -> list:
    """
    Find genuinely notable moments where on-screen text conflicts with a
    decisive audio sentiment, filtering out on-screen text that repeats
    across a large share of the video (persistent overlays, not one-off
    claims).

    Why filter by repetition instead of just "any text + decisive audio"?
    On-screen text is often a static overlay (scoreboard, watermark, lower
    third) that's visible in nearly every frame — a naive check would flag
    almost every sentence in the video as "critical," which is noise, not
    signal. Rare or unique on-screen text (that appears briefly, once) is
    far more likely to represent a genuine, deliberate claim worth
    surfacing — e.g. a promotional statement shown once, contradicted by
    the speaker's tone at that moment.

    Near-duplicate OCR readings of the same overlay (e.g. minor phrasing
    differences the model produces for the same scoreboard) are grouped
    together via fuzzy similarity matching (_group_similar_texts) before
    the repetition check, instead of relying on exact string equality —
    so a genuinely repetitive element doesn't slip through the filter
    just because the model transcribed it slightly differently between
    frames.

    Returns a list of at most 5 entries: [{time, text_on_screen, audio_sentiment}]
    """
    if not segments or not frame_analyses:
        return []

    # Count how often each exact on-screen text string appears across all
    # analyzed frames, then fold near-duplicates together so a persistent
    # overlay with slightly inconsistent OCR readings is still recognized
    # as one repeated element.
    text_counts: dict[str, int] = {}
    for frame in frame_analyses:
        text = frame.get("text_on_screen", "none").strip()
        if text and text.lower() != "none":
            text_counts[text] = text_counts.get(text, 0) + 1

    grouped_text_counts = _group_similar_texts(text_counts)

    total_frames = len(frame_analyses)

    sentiment_lookup = {}
    for seg_sent in segment_sentiments:
        key = (seg_sent["start"], seg_sent["end"])
        sentiment_lookup[key] = seg_sent.get("sentiment", "neutral")

    candidates = []

    for segment in segments:
        segment_time  = (segment["start"] + segment["end"]) / 2
        closest_frame = _find_closest_frame(segment_time, frame_analyses)
        if not closest_frame:
            continue

        text_on_screen = closest_frame.get("text_on_screen", "none").strip()
        if not text_on_screen or text_on_screen.lower() == "none":
            continue

        seg_key        = (segment["start"], segment["end"])
        audio_seg_sent = sentiment_lookup.get(seg_key, "neutral")

        # Only decisive audio sentiment is worth flagging — a neutral/mixed
        # moment isn't a meaningful "conflict" with anything.
        if audio_seg_sent not in ["positive", "negative"]:
            continue

        # Skip on-screen text that's persistent/repetitive across the video
        # (using the fuzzy-grouped count, not the raw exact-match count)
        occurrence_ratio = grouped_text_counts.get(text_on_screen, 0) / total_frames
        if occurrence_ratio > TEXT_REPETITION_THRESHOLD:
            continue

        candidates.append({
            "time"           : round(segment_time, 1),
            "text_on_screen" : text_on_screen,
            "audio_sentiment": audio_seg_sent
        })

    return candidates[:5]


def _find_closest_frame(timestamp: float, frame_analyses: list) -> dict | None:
    """
    Find the frame with the closest timestamp to the given time.

    If the closest available frame is still more than
    MAX_ALIGNMENT_GAP_SECONDS away, treat it as "no frame available" rather
    than linking to a misleadingly distant frame.
    """
    if not frame_analyses:
        return None

    closest = min(
        frame_analyses,
        key=lambda f: abs(f.get("timestamp", 0) - timestamp)
    )

    gap = abs(closest.get("timestamp", 0) - timestamp)
    if gap > MAX_ALIGNMENT_GAP_SECONDS:
        return None

    return closest


# ──────────────────────────────────────────────
# Summary Builder
# ──────────────────────────────────────────────

def _extract_silent_brand_mentions(frame_analyses: list, full_text: str) -> list:
    """
    Detect brand logos/marks that appear visually on screen but whose name
    is never actually spoken anywhere in the transcript — e.g. a
    restaurant's logo shown on a bag or sign while the speaker describes
    the experience without ever saying the brand's name out loud.

    Why this matters: a text-only pipeline (transcript alone) would never
    see this at all — the brand only exists in the visual channel. This is
    a genuine audio/visual gap, distinct from the existing on-screen TEXT
    conflict detection (_extract_critical_text_moments), which only
    catches literal readable text, not logo/trademark recognition.

    Returns a list of at most 10 entries: [{"brand": str, "time": float}],
    each the first frame timestamp the brand was seen at.
    """
    if not frame_analyses:
        return []

    first_seen: dict[str, float] = {}
    for frame in frame_analyses:
        brands_str = frame.get("brand_logos", "none")
        if not brands_str or brands_str.strip().lower() == "none":
            continue
        for brand in brands_str.split("|"):
            brand = brand.strip()
            if brand and brand not in first_seen:
                first_seen[brand] = frame.get("timestamp", 0.0)

    if not first_seen:
        return []

    full_text_lower = full_text.lower()
    silent_brands = [
        {"brand": brand, "time": round(timestamp, 1)}
        for brand, timestamp in first_seen.items()
        if brand.lower() not in full_text_lower
    ]

    # Earliest-seen first, same "most relevant first" convention as
    # _extract_critical_text_moments.
    silent_brands.sort(key=lambda entry: entry["time"])
    return silent_brands[:10]


def _build_summary(
    audio_result           : dict,
    visual_result          : dict,
    unified_sentiment      : str,
    confidence             : float,
    signal                 : str,
    critical_text_moments  : list,
    silent_brand_mentions  : list
) -> dict:
    """
    Build a structured summary for the report layer.
    Uses real audio sentiment for key moments detection.
    """
    segments           = audio_result.get("segments", [])
    segment_sentiments = audio_result.get("segment_sentiments", [])
    full_text          = audio_result.get("full_text", "")
    sentiment_timeline = audio_result.get("sentiment_timeline", [])

    # Build sentiment lookup
    sentiment_lookup = {}
    for seg_sent in segment_sentiments:
        key = (seg_sent["start"], seg_sent["end"])
        sentiment_lookup[key] = seg_sent.get("sentiment", "neutral")

    # Extract real key moments — where audio and visual sentiment actually conflict
    key_moments = []
    for seg in segments:
        seg_time      = (seg["start"] + seg["end"]) / 2
        closest_frame = _find_closest_frame(seg_time, visual_result.get("frame_analyses", []))

        seg_key              = (seg["start"], seg["end"])
        audio_seg_sentiment  = sentiment_lookup.get(seg_key, "neutral")
        visual_seg_sentiment = closest_frame.get("sentiment", "neutral") if closest_frame else "neutral"

        # Flag any real conflict — including cases where audio is neutral but
        # visual is decisive, not just the reverse
        if audio_seg_sentiment != visual_seg_sentiment and (
            audio_seg_sentiment != "neutral" or visual_seg_sentiment != "neutral"
        ):
            key_moments.append({
                "time" : round(seg_time, 1),
                "text" : seg["text"],
                "note" : f"audio: {audio_seg_sentiment} | visual: {visual_seg_sentiment}"
            })

    return {
        "language"          : audio_result.get("language", "unknown"),
        "duration_estimate" : f"{len(segments) * 3} seconds",
        "total_segments"    : len(segments),
        "total_frames"      : len(visual_result.get("frame_analyses", [])),
        "audio_sentiment"   : audio_result.get("overall_sentiment", "neutral"),
        "visual_sentiment"  : visual_result.get("overall_sentiment", "neutral"),
        "unified_sentiment" : unified_sentiment,
        "confidence"        : confidence,
        "confidence_label"  : _confidence_label(confidence),
        "signal"            : signal,
        "alignment_note"    : "confidence is a heuristic alignment score, not a model probability",
        "key_moments"       : key_moments[:8],
        "critical_text_moments" : critical_text_moments,
        "silent_brand_mentions" : silent_brand_mentions,
        "sentiment_timeline" : sentiment_timeline,
        "full_text_preview" : full_text[:5000],
    }


# ──────────────────────────────────────────────
# Error Helper
# ──────────────────────────────────────────────

def _error_result(message: str) -> dict:
    """Return a consistent error result dictionary."""
    print(f"ERROR: {message}")
    return {
        "unified_sentiment"    : "neutral",
        "confidence"           : 0.0,
        "confidence_label"     : "low",
        "signal"               : "error",
        "fusion_score"         : None,
        "analysis_reliability" : {
            "score": 0.0, "fallback_items": 0, "total_items": 0,
            "audio_reliability"  : {"score": 0.0, "fallback_items": 0, "total_items": 0},
            "visual_reliability" : {"score": 0.0, "fallback_items": 0, "total_items": 0},
        },
        "aligned_segments"     : [],
        "summary"              : {},
        "status"               : "error",
        "message"              : message
    }