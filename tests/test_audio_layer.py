"""
Unit tests for pure functions in layers/audio_layer.py.

_get_period_count_for_duration, _split_segments_into_periods, and
_majority_sentiment are all pure — no API calls, no I/O — so they're
tested directly here. They power the sentiment_timeline feature
(splitting a transcript into 4-6 time periods instead of one overall
sentiment snapshot).
"""

from layers.audio_layer import (
    _get_period_count_for_duration,
    _split_segments_into_periods,
    _majority_sentiment,
)


def test_period_count_short_video_gets_4_periods():
    assert _get_period_count_for_duration(15 * 60) == 4


def test_period_count_medium_video_gets_5_periods():
    assert _get_period_count_for_duration(30 * 60) == 5


def test_period_count_long_video_gets_6_periods():
    assert _get_period_count_for_duration(90 * 60) == 6


def test_period_count_boundary_values_are_exact():
    assert _get_period_count_for_duration(15 * 60) == 4
    assert _get_period_count_for_duration(15 * 60 + 1) == 5
    assert _get_period_count_for_duration(45 * 60) == 5
    assert _get_period_count_for_duration(45 * 60 + 1) == 6


def test_split_segments_preserves_all_segments():
    segments = [
        {"start": 0.0,  "end": 5.0,  "text": "a"},
        {"start": 10.0, "end": 15.0, "text": "b"},
        {"start": 20.0, "end": 25.0, "text": "c"},
        {"start": 30.0, "end": 35.0, "text": "d"},
    ]
    periods = _split_segments_into_periods(segments, 4)
    assert len(periods) == 4
    assert sum(len(p) for p in periods) == len(segments)


def test_split_segments_empty_list_returns_empty_periods():
    periods = _split_segments_into_periods([], 4)
    assert len(periods) == 4
    assert all(p == [] for p in periods)


def test_split_segments_single_segment_goes_in_first_period():
    segments = [{"start": 5.0, "end": 5.0, "text": "only one"}]
    periods = _split_segments_into_periods(segments, 4)
    assert sum(len(p) for p in periods) == 1
    assert len(periods[0]) == 1


def test_split_segments_last_segment_does_not_overflow_period_index():
    segments = [
        {"start": 0.0,  "end": 10.0, "text": "first"},
        {"start": 90.0, "end": 100.0, "text": "last"},
    ]
    periods = _split_segments_into_periods(segments, 4)
    assert periods[-1] and periods[-1][-1]["text"] == "last"


def test_majority_sentiment_picks_most_frequent():
    assert _majority_sentiment(["positive", "positive", "negative"]) == "positive"


def test_majority_sentiment_empty_list_returns_neutral():
    assert _majority_sentiment([]) == "neutral"


def test_majority_sentiment_unknown_value_counts_as_neutral():
    result = _majority_sentiment(["positive", "some_unexpected_value"])
    assert result in ["positive", "neutral"]