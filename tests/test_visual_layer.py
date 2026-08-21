"""
Unit tests for pure functions in layers/visual_layer.py.

_get_max_frames_for_duration and _calculate_overall_sentiment are pure —
no API calls, no I/O. _retry_call has a side effect (time.sleep between
attempts), so its test monkeypatches time.sleep to keep the test fast
while still verifying the retry actually happens.

_apply_temporal_smoothing is pure and dampens isolated single-frame
sentiment outliers using neighboring frames.
"""

from layers.visual_layer import (
    _get_max_frames_for_duration,
    _calculate_overall_sentiment,
    _retry_call,
    _apply_temporal_smoothing,
)


# ──────────────────────────────────────────────
# _get_max_frames_for_duration
# ──────────────────────────────────────────────
# NOTE: these caps (90/180/270) must match _get_max_frames_for_duration's
# actual current values exactly — if the caps change, update these tests too.

def test_max_frames_short_video_is_90():
    """A video of 10 minutes or less gets the lowest frame cap (90)."""
    assert _get_max_frames_for_duration(10 * 60) == 90


def test_max_frames_medium_video_is_180():
    """A video between 10 and 30 minutes gets the middle cap (180)."""
    assert _get_max_frames_for_duration(20 * 60) == 180


def test_max_frames_long_video_is_270():
    """
    A video longer than 30 minutes gets the highest cap (270) — this is
    also the hard ceiling for the whole system, since videos longer than
    2 hours are rejected by input_layer before reaching this point.
    """
    assert _get_max_frames_for_duration(60 * 60) == 270


def test_max_frames_boundary_values_are_exact():
    """
    Tier boundaries (10 min, 30 min) are inclusive on the lower tier —
    only values strictly ABOVE the boundary move to the next tier.
    """
    assert _get_max_frames_for_duration(10 * 60) == 90        # exactly 10 min -> still short
    assert _get_max_frames_for_duration(10 * 60 + 1) == 180   # just over -> medium
    assert _get_max_frames_for_duration(30 * 60) == 180       # exactly 30 min -> still medium
    assert _get_max_frames_for_duration(30 * 60 + 1) == 270   # just over -> long


# ──────────────────────────────────────────────
# _calculate_overall_sentiment
# ──────────────────────────────────────────────

def test_overall_sentiment_majority_vote():
    """Basic majority vote across frame sentiments."""
    result = _calculate_overall_sentiment(["positive", "positive", "negative"])
    assert result == "positive"


def test_overall_sentiment_empty_list_returns_neutral():
    """No frames analyzed at all should default to neutral, not crash."""
    assert _calculate_overall_sentiment([]) == "neutral"


# ──────────────────────────────────────────────
# _retry_call
# ──────────────────────────────────────────────

def test_retry_call_succeeds_on_second_attempt(monkeypatch):
    """
    A function that fails once and succeeds on the second call should
    have its retry attempted (RETRY_MAX_ATTEMPTS = 2), and the final
    successful result should be returned — not an exception.

    time.sleep is monkeypatched to a no-op so this test doesn't actually
    pause for the real retry delay.
    """
    import layers.visual_layer as visual_layer_module
    monkeypatch.setattr(visual_layer_module.time, "sleep", lambda seconds: None)

    call_count = {"n": 0}

    def flaky_function():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise ValueError("simulated transient failure")
        return "success"

    result = _retry_call(flaky_function)
    assert result == "success"
    assert call_count["n"] == 2


def test_retry_call_raises_after_exhausting_attempts(monkeypatch):
    """
    A function that ALWAYS fails should still raise after the retry
    budget (2 attempts) is exhausted — retry must not silently swallow
    a persistent failure.
    """
    import layers.visual_layer as visual_layer_module
    monkeypatch.setattr(visual_layer_module.time, "sleep", lambda seconds: None)

    def always_fails():
        raise ValueError("persistent failure")

    try:
        _retry_call(always_fails)
        assert False, "Expected _retry_call to raise after exhausting attempts"
    except ValueError as e:
        assert "persistent failure" in str(e)


# ──────────────────────────────────────────────
# _apply_temporal_smoothing
# ──────────────────────────────────────────────

def _make_frames(sentiments):
    return [{"sentiment": s, "is_fallback": False, "timestamp": i * 5.0} for i, s in enumerate(sentiments)]


def test_smoothing_isolated_outlier_surrounded_by_agreeing_neighbors_is_corrected():
    """
    A single frame disagreeing with every neighbor in its window, while
    at least 2 of those neighbors agree with EACH OTHER on a different
    value, is treated as noise and corrected to the neighbors' majority.
    """
    frames = _make_frames(["positive", "positive", "negative", "positive", "positive"])
    result = _apply_temporal_smoothing(frames)
    sentiments = [f["sentiment"] for f in result]
    assert sentiments[2] == "positive"
    assert result[2]["is_smoothed"] is True
    assert result[0]["is_smoothed"] is False
    assert result[1]["is_smoothed"] is False


def test_smoothing_genuine_two_frame_transition_is_left_untouched():
    """
    A real mood shift spanning 2+ consecutive frames must NOT be smoothed
    away — only single-frame isolated outliers are corrected.
    """
    frames = _make_frames(["positive", "positive", "negative", "negative", "positive"])
    result = _apply_temporal_smoothing(frames)
    assert [f["sentiment"] for f in result] == ["positive", "positive", "negative", "negative", "positive"]
    assert all(f["is_smoothed"] is False for f in result)


def test_smoothing_edge_frames_without_a_full_window_are_never_smoothed():
    """
    Frames at the very start/end of the sequence only have a one-sided
    window (no "after" or no "before" neighbors) — these must never be
    smoothed, since a one-sided window can't distinguish noise from the
    start of a genuine, unconfirmed transition.
    """
    frames = _make_frames(["positive", "positive", "negative", "negative", "positive"])
    result = _apply_temporal_smoothing(frames)
    assert result[-1]["sentiment"] == "positive"
    assert result[-1]["is_smoothed"] is False


def test_smoothing_too_few_frames_for_any_window_changes_nothing():
    """Fewer frames than the smoothing window needs -> no changes at all."""
    frames = _make_frames(["positive", "negative"])
    result = _apply_temporal_smoothing(frames)
    assert [f["sentiment"] for f in result] == ["positive", "negative"]
    assert all(f["is_smoothed"] is False for f in result)


def test_smoothing_single_disagreeing_neighbor_is_not_enough_to_override():
    """
    The middle frame disagrees with all 4 neighbors, but no single
    neighbor VALUE repeats twice among them (no majority) — a single
    dissenting neighbor isn't strong enough evidence to override the
    model's direct read of this frame.
    """
    frames = _make_frames(["a", "b", "mixed", "c", "d"])
    result = _apply_temporal_smoothing(frames)
    assert result[2]["sentiment"] == "mixed"
    assert result[2]["is_smoothed"] is False


def test_smoothing_exact_tie_among_neighbors_is_left_uncorrected():
    """
    The middle frame's 4 neighbors split exactly 2-vs-2 between two
    different sentiments (both meet SMOOTHING_MIN_AGREEING_NEIGHBORS on
    their own). An exact tie like this is treated as a genuine
    transitional moment, not noise to correct — the frame must be left
    as the model read it rather than arbitrarily snapping to whichever
    sentiment the dict iteration happened to encounter first.
    """
    frames = _make_frames(["positive", "positive", "mixed", "negative", "negative"])
    result = _apply_temporal_smoothing(frames)
    assert result[2]["sentiment"] == "mixed"
    assert result[2]["is_smoothed"] is False