"""
Unit tests for pure functions in layers/report_layer.py.

_get_min_report_length is pure — no API calls, no I/O — so it's tested
directly here. It scales the minimum acceptable report length by
report_mode and how much transcript material the video actually has
(total_segments).

_run_grounding_check is pure (zero API calls) — a deterministic check
comparing generated report text against the structured summary data it
was built from, flagging sentiment/confidence contradictions and
untraceable quoted passages.

_summarize_periods / _build_periods_summary make one Claude API call
each, mocked here the same way audio_layer/visual_layer's API-calling
functions are tested in test_mocked_api_calls.py.

_get_period_count_for_report is pure — decides how many periods a long
video's transcript gets split into (or 0 to skip periods-based
summarization for shorter videos).
"""

from unittest.mock import MagicMock, patch

from layers.report_layer import (
    _get_min_report_length,
    _validate_report,
    _run_grounding_check,
    _summarize_periods,
    _build_periods_summary,
    _get_period_count_for_report,
    _build_prompt,
)


# ──────────────────────────────────────────────
# _get_min_report_length
# ──────────────────────────────────────────────

def test_min_length_tiny_video_summary_is_lowest_tier():
    """
    A very short video (e.g. a 10-second TikTok clip, 2 segments) in
    summary mode should get the lowest threshold — a short summary of a
    short clip is naturally short, not a quality failure.
    """
    assert _get_min_report_length("summary", 2) == 100


def test_min_length_tiny_video_analysis_is_higher_than_summary():
    """
    Even for a tiny video, an 'analysis' report is expected to be more
    thorough than a 'summary' — so its minimum should still be higher
    than the summary minimum at the same content-size tier.
    """
    tiny_summary  = _get_min_report_length("summary", 2)
    tiny_analysis = _get_min_report_length("analysis", 2)
    assert tiny_analysis > tiny_summary
    assert tiny_analysis == 150


def test_min_length_normal_video_analysis_matches_original_default():
    """
    A video with plenty of transcript material (> 20 segments) getting a
    full analysis report should land on the same 500-char floor the
    system used before this became variable — the common case shouldn't
    have quietly gotten stricter or looser.
    """
    assert _get_min_report_length("analysis", 100) == 500


def test_min_length_tier_boundaries_are_exact():
    """
    The tier boundaries (5 segments and 20 segments) must be inclusive on
    the lower tier — exactly 5 segments is still 'tiny', exactly 20 is
    still 'small', with only values ABOVE the boundary moving up a tier.
    """
    assert _get_min_report_length("summary", 5)  == 100   # exactly 5 -> still tiny
    assert _get_min_report_length("summary", 6)  == 200   # just over -> small
    assert _get_min_report_length("summary", 20) == 200   # exactly 20 -> still small
    assert _get_min_report_length("summary", 21) == 250   # just over -> normal


# ──────────────────────────────────────────────
# _validate_report (integration check with the variable threshold)
# ──────────────────────────────────────────────

def test_validate_report_uses_scaled_threshold_not_fixed_one():
    """
    A report just over 100 chars (which would FAIL against the OLD fixed
    200-char minimum) should PASS length validation for a tiny video in
    summary mode (threshold = 100), proving _validate_report is actually
    using the scaled threshold rather than a hardcoded constant.
    """
    short_but_valid_ar = "هذا تقرير قصير جداً بس يغطي فيديو قصير جداً بشكل كافي ومناسب لحجمه، ويكفي لتلخيص المحتوى المتاح بدقة معقولة دون إطالة غير ضرورية."
    short_but_valid_en = "This is a very short report that adequately covers a very short video, and is sufficient to summarize the available content reasonably well."

    warnings = _validate_report(short_but_valid_ar, short_but_valid_en, "summary", 2)

    length_warnings = [w for w in warnings if "short" in w.lower()]
    assert length_warnings == [], f"Expected no length warnings, got: {length_warnings}"


def test_validate_report_still_flags_genuinely_short_report():
    """
    Even with the lowered threshold for tiny videos, a report that's
    truly too short (well under even the lowest tier) should still be
    flagged.
    """
    warnings = _validate_report("قصير جداً", "too short", "summary", 2)
    length_warnings = [w for w in warnings if "short" in w.lower()]
    assert len(length_warnings) > 0


# ──────────────────────────────────────────────
# _run_grounding_check
# ──────────────────────────────────────────────

def _make_summary(**overrides):
    base = {
        "unified_sentiment": "negative",
        "confidence_label": "medium",
        "language": "ARABIC",
        "full_text_preview": "",
        "critical_text_moments": [],
        "key_moments": [],
    }
    base.update(overrides)
    return base


def test_grounding_check_sentiment_contradiction_is_flagged():
    summary = _make_summary(unified_sentiment="negative")
    warnings = _run_grounding_check("النبرة كانت إيجابي بشكل عام", "", summary)
    assert len(warnings) == 1
    assert "positive" in warnings[0]


def test_grounding_check_matching_sentiment_produces_no_warning():
    summary = _make_summary(unified_sentiment="negative")
    warnings = _run_grounding_check("النبرة كانت سلبية بشكل عام", "", summary)
    assert len(warnings) == 0


def test_grounding_check_confidence_contradiction_is_flagged():
    summary = _make_summary(confidence_label="low")
    warnings = _run_grounding_check("", "The analysis shows high confidence", summary)
    assert len(warnings) == 1


def test_grounding_check_untraceable_quote_is_flagged():
    summary = _make_summary(
        unified_sentiment="neutral",
        language="ARABIC",
        full_text_preview="هذا هو النص الأصلي الموجود فعلاً بالترانسكربت الحقيقي للفيديو",
    )
    # Quote must be in report_ar to match summary["language"] — see the
    # language-aware test below for why report_en is deliberately skipped.
    report_ar = 'قال المتحدث: "هذا اقتباس مختلق بالكامل وما قيل إطلاقاً بالفيديو الأصلي"'
    warnings = _run_grounding_check(report_ar, "", summary)
    assert len(warnings) == 1
    assert "quoted" in warnings[0].lower()


def test_grounding_check_quote_in_wrong_language_is_not_falsely_flagged():
    """
    Arabic transcript, but the quote lives in the English report as an
    accurate translation — must not be flagged just because the two
    scripts don't match character-for-character. Comparing a translated
    quote against a differently-scripted original would otherwise fail
    every time even when the translation is perfectly faithful.
    """
    summary = _make_summary(
        unified_sentiment="neutral",
        full_text_preview="فريق سيطر على أوروبا بس هذا الفريق اسمه إسرائيل",
    )
    report_ar = 'قال: "فريق سيطر على أوروبا بس هذا الفريق اسمه إسرائيل"'
    report_en = 'He said: "A team that dominated Europe, and this team is called Israel"'
    warnings = _run_grounding_check(report_ar, report_en, summary)
    assert len(warnings) == 0


def test_grounding_check_implausibly_long_quote_is_ignored():
    """
    A stray unclosed quote character can make the regex capture an
    entire paragraph as a "quote" — this is a quote-extraction artifact,
    not a real quoted passage, and must be skipped rather than flagged.
    """
    summary = _make_summary(unified_sentiment="positive", full_text_preview="نص قصير")
    long_fake_quote = "التي " + ("كلمة " * 100) + '"'
    warnings = _run_grounding_check(long_fake_quote, "", summary)
    assert len(warnings) == 0


# ──────────────────────────────────────────────
# _get_period_count_for_report
# ──────────────────────────────────────────────

def test_period_count_short_transcript_skips_periods_summarization():
    """
    <=60 segments already fits comfortably in the flat 5000-char
    preview — no need for the extra API call.
    """
    assert _get_period_count_for_report(30) == 0
    assert _get_period_count_for_report(60) == 0


def test_period_count_tiers_scale_with_transcript_length():
    assert _get_period_count_for_report(61) == 8
    assert _get_period_count_for_report(120) == 8
    assert _get_period_count_for_report(121) == 12
    assert _get_period_count_for_report(250) == 12
    assert _get_period_count_for_report(251) == 15


# ──────────────────────────────────────────────
# _summarize_periods / _build_periods_summary (mocked API)
# ──────────────────────────────────────────────

def test_summarize_periods_parses_valid_json_array():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='["summary one", "summary two"]')]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    with patch("layers.report_layer.anthropic.Anthropic", return_value=fake_client):
        result = _summarize_periods(["period 1 text", "period 2 text"])

    assert result == ["summary one", "summary two"]


def test_summarize_periods_returns_none_on_wrong_length():
    """
    If the model returns a different number of summaries than periods
    requested, the mismatch can't be trusted to map to the right period
    — treated as a failure, triggering the flat-preview fallback.
    """
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='["only one summary"]')]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    with patch("layers.report_layer.anthropic.Anthropic", return_value=fake_client):
        result = _summarize_periods(["period 1", "period 2"])

    assert result is None


def test_summarize_periods_returns_none_on_api_failure():
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = ConnectionError("down")

    with patch("layers.report_layer.anthropic.Anthropic", return_value=fake_client), \
         patch("layers.report_layer.time.sleep", return_value=None):
        result = _summarize_periods(["period 1"])

    assert result is None


def test_build_periods_summary_falls_back_gracefully_on_empty_segments():
    """
    No segments to split into periods (edge case) — must return None
    (triggering the flat-preview fallback in _build_prompt) rather than
    raising an error.
    """
    result = _build_periods_summary([], num_periods=8)
    assert result is None


# ──────────────────────────────────────────────
# _build_prompt — sentiment_timeline section (2.15)
# ──────────────────────────────────────────────
# The "Sentiment Over Time" section should only ever appear when the
# timeline shows genuine variation across NON-fallback periods — a flat
# or missing timeline must not pad the prompt with a non-observation.

def test_build_prompt_includes_sentiment_timeline_when_variation_exists():
    summary = _make_summary(sentiment_timeline=[
        {"period": 0, "start": 0.0,  "end": 10.0, "sentiment": "neutral",  "is_fallback": False},
        {"period": 1, "start": 10.0, "end": 20.0, "sentiment": "positive", "is_fallback": False},
    ])
    prompt = _build_prompt(summary, [], "summary")
    assert "Sentiment Over Time" in prompt
    assert "[0.0s-10.0s] neutral" in prompt
    assert "[10.0s-20.0s] positive" in prompt


def test_build_prompt_omits_sentiment_timeline_when_flat():
    """
    Every non-fallback period reports the same sentiment — no real
    change over time to report, so the section must not appear at all.
    """
    summary = _make_summary(sentiment_timeline=[
        {"period": 0, "start": 0.0,  "end": 10.0, "sentiment": "positive", "is_fallback": False},
        {"period": 1, "start": 10.0, "end": 20.0, "sentiment": "positive", "is_fallback": False},
    ])
    prompt = _build_prompt(summary, [], "summary")
    assert "audio tone by time period" not in prompt


def test_build_prompt_omits_sentiment_timeline_when_missing():
    """
    No sentiment_timeline key at all (e.g. an older/degraded summary) —
    _build_prompt must not crash, and must simply omit the section.
    """
    summary = _make_summary()
    prompt = _build_prompt(summary, [], "summary")
    assert "audio tone by time period" not in prompt


def test_build_prompt_ignores_fallback_periods_for_variation_check():
    """
    A fallback period (Claude call failed for that period, so its
    'sentiment' is a placeholder, not a real reading) must not count
    toward the variation check — otherwise a fallback placeholder could
    either falsely trigger the section or falsely mask real variation.
    Here the only two REAL periods are both 'negative' (flat), while a
    fallback period claims 'positive' — the section must still be
    omitted, and the fallback period itself must not be printed.
    """
    summary = _make_summary(sentiment_timeline=[
        {"period": 0, "start": 0.0,  "end": 10.0, "sentiment": "negative", "is_fallback": False},
        {"period": 1, "start": 10.0, "end": 20.0, "sentiment": "positive", "is_fallback": True},
        {"period": 2, "start": 20.0, "end": 30.0, "sentiment": "negative", "is_fallback": False},
    ])
    prompt = _build_prompt(summary, [], "summary")
    assert "audio tone by time period" not in prompt