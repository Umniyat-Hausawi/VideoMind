"""
Unit tests for pure functions in layers/fusion_layer.py.

_fuse_sentiments and _find_closest_frame take plain data in and return
plain data out — no API calls, no I/O — so they're tested directly here.

_calculate_analysis_reliability returns a combined score plus a
per-modality breakdown (audio_reliability / visual_reliability) — a
combined score alone could hide a modality that fell back on every item
while the other was perfect.

_group_similar_texts fuzzy-groups near-duplicate on-screen text readings
(minor OCR phrasing drift) before the repetition-frequency filter runs.

NOTE on _fuse_sentiments' return shape:
This function returns a 4-tuple:
    (unified_sentiment, confidence, signal, fusion_score)
fusion_score is None when audio/visual agreed (no calculation was needed),
and a computed float when they disagreed. Tests unpack all 4 values.
"""

from layers.fusion_layer import (
    _fuse_sentiments,
    _find_closest_frame,
    _calculate_analysis_reliability,
    _reliability_breakdown,
    _group_similar_texts,
    _build_summary,
)


# ──────────────────────────────────────────────
# _fuse_sentiments
# ──────────────────────────────────────────────

def test_fuse_sentiments_both_agree_positive():
    """
    When audio and visual sentiment are identical, the function should
    short-circuit: return that sentiment directly with HIGH_CONFIDENCE
    and signal 'aligned', with no fusion_score computed (None).
    """
    sentiment, confidence, signal, fusion_score = _fuse_sentiments("positive", "positive")
    assert sentiment == "positive"
    assert confidence == 0.90
    assert signal == "aligned"
    assert fusion_score is None


def test_fuse_sentiments_audio_neutral_visual_dominant():
    """
    Audio is neutral (no strong signal), visual is positive.
    Expected: visual wins, MEDIUM_CONFIDENCE, signal 'visual_dominant'.
    Weighted score = (0 * 0.65) + (1 * 0.35) = 0.35.
    """
    sentiment, confidence, signal, fusion_score = _fuse_sentiments("neutral", "positive")
    assert sentiment == "positive"
    assert confidence == 0.70
    assert signal == "visual_dominant"
    assert fusion_score == 0.35


def test_fuse_sentiments_visual_neutral_audio_dominant():
    """
    Audio is negative (strong signal), visual is neutral.
    Expected: audio wins, MEDIUM_CONFIDENCE, signal 'audio_dominant'.
    Weighted score = (-1 * 0.65) + (0 * 0.35) = -0.65.
    """
    sentiment, confidence, signal, fusion_score = _fuse_sentiments("negative", "neutral")
    assert sentiment == "negative"
    assert confidence == 0.70
    assert signal == "audio_dominant"
    assert fusion_score == -0.65


def test_fuse_sentiments_conflict_uses_weighted_score_not_hardcoded():
    """
    Audio positive vs visual negative — a real conflict between two decisive
    signals. This must go through the weighted formula:
        score = (audio_value * 0.65) + (visual_value * 0.35)
              = (1 * 0.65) + (-1 * 0.35) = 0.30
    Rounding happens BEFORE the threshold comparison (a floating-point fix
    — raw Python arithmetic gives 0.30000000000000004, not exactly 0.3,
    which would incorrectly cross the '> 0.3' boundary otherwise). Rounded
    to exactly 0.3, it does not exceed the strict threshold, so it lands
    as 'mixed' with LOW_CONFIDENCE and signal 'conflict'.
    """
    sentiment, confidence, signal, fusion_score = _fuse_sentiments("positive", "negative")
    assert sentiment == "mixed"
    assert confidence == 0.50
    assert signal == "conflict"
    assert fusion_score == 0.3


def test_fuse_sentiments_both_mixed_does_not_crash():
    """
    'mixed' as input on both sides must be handled cleanly (it's a valid
    sentiment value elsewhere in the system). Since both sides are equal,
    this should hit the early-agreement branch just like any other match.
    """
    sentiment, confidence, signal, fusion_score = _fuse_sentiments("mixed", "mixed")
    assert sentiment == "mixed"
    assert confidence == 0.90
    assert signal == "aligned"
    assert fusion_score is None


# ──────────────────────────────────────────────
# _find_closest_frame
# ──────────────────────────────────────────────

def test_find_closest_frame_returns_nearest_match():
    """
    Given a normal list of frames, the function should return the one
    whose timestamp is numerically closest to the given timestamp.
    """
    frames = [
        {"timestamp": 0.0,  "sentiment": "neutral"},
        {"timestamp": 10.0, "sentiment": "positive"},
        {"timestamp": 20.0, "sentiment": "negative"},
    ]
    result = _find_closest_frame(9.0, frames)
    assert result is not None
    assert result["timestamp"] == 10.0


def test_find_closest_frame_empty_list_returns_none():
    """
    An empty frame_analyses list has nothing to match against — must
    return None rather than raising an error.
    """
    assert _find_closest_frame(15.0, []) is None


def test_find_closest_frame_too_far_returns_none():
    """
    If the nearest available frame is more than MAX_ALIGNMENT_GAP_SECONDS
    (15s) away from the timestamp, it should NOT be returned — the segment
    is considered to have no available frame, rather than being linked to
    a misleadingly distant one.
    """
    frames = [
        {"timestamp": 0.0,   "sentiment": "neutral"},
        {"timestamp": 100.0, "sentiment": "positive"},
    ]
    # Closest is 100.0, but |100 - 50| = 50s > 15s gap limit
    assert _find_closest_frame(50.0, frames) is None


def test_find_closest_frame_tie_returns_first_deterministically():
    """
    Two frames equally distant from the timestamp (5s away on each side).
    Python's min() keeps the first minimal item found, so the result must
    deterministically be the first frame in the list, not the second.
    """
    frames = [
        {"timestamp": 10.0, "sentiment": "positive", "label": "first"},
        {"timestamp": 20.0, "sentiment": "negative", "label": "second"},
    ]
    result = _find_closest_frame(15.0, frames)
    assert result is not None
    assert result["label"] == "first"


# ──────────────────────────────────────────────
# _calculate_analysis_reliability — combined score (original behavior)
# ──────────────────────────────────────────────

def test_reliability_all_items_succeeded_is_100_percent():
    """
    No is_fallback items at all across audio segments or visual frames
    → reliability should be exactly 100%, with zero fallback items counted.
    """
    audio_result  = {"segment_sentiments": [{"is_fallback": False}, {"is_fallback": False}]}
    visual_result = {"frame_analyses": [{"is_fallback": False}, {"is_fallback": False}]}

    result = _calculate_analysis_reliability(audio_result, visual_result)
    assert result["score"] == 100.0
    assert result["fallback_items"] == 0
    assert result["total_items"] == 4


def test_reliability_half_fallback_is_50_percent():
    """
    Half the combined items (audio + visual) are fallbacks → score should
    reflect exactly that ratio, not just the audio or visual side alone.
    """
    audio_result  = {"segment_sentiments": [{"is_fallback": True}, {"is_fallback": False}]}
    visual_result = {"frame_analyses": [{"is_fallback": True}, {"is_fallback": False}]}

    result = _calculate_analysis_reliability(audio_result, visual_result)
    assert result["score"] == 50.0
    assert result["fallback_items"] == 2
    assert result["total_items"] == 4


def test_reliability_audio_only_mode_no_visual_frames():
    """
    Audio-only mode means visual_result has an empty frame_analyses list.
    The function must still work correctly using audio data alone, not
    crash or misreport just because one side is empty.
    """
    audio_result  = {"segment_sentiments": [{"is_fallback": False}, {"is_fallback": True}]}
    visual_result = {"frame_analyses": []}

    result = _calculate_analysis_reliability(audio_result, visual_result)
    assert result["total_items"] == 2
    assert result["fallback_items"] == 1
    assert result["score"] == 50.0


def test_reliability_both_empty_does_not_divide_by_zero():
    """
    Edge case: both segment_sentiments and frame_analyses are empty (e.g.
    an edge-case video with no transcribable speech and no frames). This
    must not raise a ZeroDivisionError — it should report full reliability
    (100%) since there's nothing that failed, rather than implying a
    problem that doesn't exist.
    """
    audio_result  = {"segment_sentiments": []}
    visual_result = {"frame_analyses": []}

    result = _calculate_analysis_reliability(audio_result, visual_result)
    assert result["total_items"] == 0
    assert result["fallback_items"] == 0
    assert result["score"] == 100.0


# ──────────────────────────────────────────────
# _calculate_analysis_reliability — per-modality breakdown
# ──────────────────────────────────────────────
# Why split audio/visual instead of the combined score alone?
# A combined 95% could hide a visual layer that fell back on every single
# frame while audio was perfect — averaging conceals exactly the kind of
# imbalance this breakdown exists to surface.

def test_reliability_breakdown_split_scores_are_independent_of_each_other():
    audio_result  = {"segment_sentiments": [{"is_fallback": False}] * 8 + [{"is_fallback": True}] * 2}
    visual_result = {"frame_analyses": [{"is_fallback": False}] * 5 + [{"is_fallback": True}] * 5}

    result = _calculate_analysis_reliability(audio_result, visual_result)

    assert result["audio_reliability"]["score"] == 80.0
    assert result["visual_reliability"]["score"] == 50.0
    assert result["score"] == 65.0  # combined: (8+5) / (10+10) * 100


def test_reliability_breakdown_hides_nothing_when_one_modality_fully_fails():
    """
    The whole point of the split: a perfect audio layer alongside a
    completely-fallback visual layer must NOT look like a healthy
    ~90%+ combined score once you look at the breakdown.
    """
    audio_result  = {"segment_sentiments": [{"is_fallback": False}] * 20}
    visual_result = {"frame_analyses": [{"is_fallback": True}] * 20}

    result = _calculate_analysis_reliability(audio_result, visual_result)

    assert result["audio_reliability"]["score"] == 100.0
    assert result["visual_reliability"]["score"] == 0.0
    assert result["score"] == 50.0  # combined score alone would look "okay"


def test_reliability_breakdown_helper_function_directly():
    assert _reliability_breakdown(2, 10) == {"score": 80.0, "fallback_items": 2, "total_items": 10}
    assert _reliability_breakdown(0, 0) == {"score": 100.0, "fallback_items": 0, "total_items": 0}


# ──────────────────────────────────────────────
# _group_similar_texts
# ──────────────────────────────────────────────

def test_group_similar_texts_merges_near_duplicate_ocr_readings():
    """
    Near-duplicate on-screen text readings for the same persistent
    overlay (minor OCR phrasing drift between frames) must be folded
    into one combined count, so the 40%-repetition filter downstream
    still recognizes it as repetitive.
    """
    text_counts = {
        "SCORE: 3-1": 20,
        "SCORE: 3-1 ": 15,   # trailing space
        "SC0RE: 3-1": 10,    # OCR misread O as 0
    }
    grouped = _group_similar_texts(text_counts)
    assert grouped["SCORE: 3-1"] == grouped["SCORE: 3-1 "] == grouped["SC0RE: 3-1"] == 45


def test_group_similar_texts_does_not_merge_genuinely_different_text():
    text_counts = {"SCORE: 3-1": 20, "Buy Now!": 3}
    grouped = _group_similar_texts(text_counts)
    assert grouped["SCORE: 3-1"] == 20
    assert grouped["Buy Now!"] == 3


def test_group_similar_texts_empty_input_returns_empty_dict():
    assert _group_similar_texts({}) == {}


# ──────────────────────────────────────────────
# _build_summary — sentiment_timeline pass-through
# ──────────────────────────────────────────────
# audio_layer produces its own separate "sentiment_timeline" (periods of
# audio tone over time, distinct from report_layer's period mechanism of
# the same name). _build_summary must forward it into the summary dict
# untouched, since report_layer's _build_prompt reads it from there.

def test_build_summary_passes_through_sentiment_timeline():
    timeline = [
        {"period": 0, "start": 0.0, "end": 10.0, "sentiment": "neutral", "is_fallback": False},
        {"period": 1, "start": 10.0, "end": 20.0, "sentiment": "positive", "is_fallback": False},
    ]
    audio_result = {
        "segments": [],
        "segment_sentiments": [],
        "full_text": "",
        "sentiment_timeline": timeline,
    }
    visual_result = {"frame_analyses": []}

    summary = _build_summary(
        audio_result, visual_result,
        unified_sentiment="positive", confidence=0.7, signal="aligned",
        critical_text_moments=[], silent_brand_mentions=[],
    )

    assert summary["sentiment_timeline"] == timeline


def test_build_summary_missing_sentiment_timeline_defaults_to_empty_list():
    """
    Older/degraded audio_result dicts (or unexpected fallback paths) that
    don't include sentiment_timeline at all must not crash _build_summary
    — it should default to an empty list rather than raising a KeyError.
    """
    audio_result = {"segments": [], "segment_sentiments": [], "full_text": ""}
    visual_result = {"frame_analyses": []}

    summary = _build_summary(
        audio_result, visual_result,
        unified_sentiment="neutral", confidence=0.5, signal="aligned",
        critical_text_moments=[], silent_brand_mentions=[],
    )

    assert summary["sentiment_timeline"] == []