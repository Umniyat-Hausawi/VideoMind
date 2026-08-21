"""
Automated mocked tests for functions that call external APIs.

Why mocked tests specifically, alongside the existing 42 pure-function
unit tests?
The existing test suite covers pure functions (fusion math, duration
tiering, period splitting, ...) that need no network access. But
_transcribe_chunk (audio_layer), _analyze_frame (visual_layer), and
process_report (report_layer) are exactly the functions that talk to
Whisper / Claude Vision / Claude Sonnet — the ones that were untested
until now, since testing them "for real" would cost money and require
network access, an API key, and real audio/image/video data on every
test run.

Mocking the client object lets us test the SURROUNDING logic (response
parsing, retry behavior, fallback-on-invalid-response, timestamp offset
math, tool_use extraction) deterministically and for free, without
touching a real API.

process_report is the highest-stakes function of the three: it's the
one that produces the actual bilingual report the user reads, via a
forced submit_report tool_use call. Unlike _transcribe_chunk and
_analyze_frame (which take the API client as a parameter), process_report
constructs its own `anthropic.Anthropic()` client internally — so it's
mocked by patching the class at the module level
(`{REPORT_LAYER_MODULE}.anthropic.Anthropic`) rather than by injecting a
fake client argument. See TestProcessReport* below.

IMPORTANT — "mock drift" risk:
These mocks encode our CURRENT understanding of the OpenAI Whisper and
Anthropic Claude response shapes. If either SDK's response object shape
changes upstream, these tests will keep passing (the mock still "returns"
what we told it to) even though real API calls would now fail or behave
differently — the tests would be validating our parsing logic against a
now-stale assumption, not against reality. Re-check these mock shapes
against the real SDK's response objects periodically (e.g. after any
`pip install --upgrade openai anthropic`), not just when a test fails.
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from layers.audio_layer  import _transcribe_chunk
    from layers.visual_layer import _analyze_frame
    AUDIO_LAYER_MODULE = "layers.audio_layer"
except ImportError:
    from audio_layer  import _transcribe_chunk
    from visual_layer import _analyze_frame
    AUDIO_LAYER_MODULE = "audio_layer"

try:
    from layers.report_layer import process_report
    REPORT_LAYER_MODULE = "layers.report_layer"
except ImportError:
    from report_layer import process_report
    REPORT_LAYER_MODULE = "report_layer"


@pytest.fixture
def dummy_audio_chunk(tmp_path):
    chunk_path = tmp_path / "chunk_000.wav"
    chunk_path.write_bytes(b"\x00" * 100)
    return str(chunk_path)


@pytest.fixture
def dummy_frame_image(tmp_path):
    frame_path = tmp_path / "frame_00005.00s.jpg"
    frame_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    return str(frame_path)


@pytest.fixture(autouse=True)
def no_real_sleep():
    with patch("time.sleep", return_value=None):
        yield


@pytest.fixture
def minimal_fusion_result():
    """
    A small, valid fusion_layer output for feeding into process_report.

    total_segments is deliberately kept at 2 (well under the >60 threshold
    in _get_period_count_for_report), so process_report never enters the
    periods-based summarization branch — that branch makes its OWN
    separate API call (_build_periods_summary → _summarize_periods) which
    would otherwise need a second, independent mock. Keeping segments low
    lets these tests focus purely on process_report's own client call and
    response-parsing logic, without coupling to that other code path
    (already covered separately in test_report_layer.py).
    """
    return {
        "status": "success",
        "summary": {
            "unified_sentiment": "positive",
            "confidence_label": "high",
            "confidence": 0.90,
            "signal": "aligned",
            "audio_sentiment": "positive",
            "visual_sentiment": "positive",
            "language": "ARABIC",
            "full_text_preview": "هذا نص تجريبي قصير يمثل معاينة الترانسكربت الكاملة لفيديو قصير.",
            "critical_text_moments": [],
            "key_moments": [],
            "silent_brand_mentions": [],
            "total_segments": 2,
        },
        "aligned_segments": [
            {
                "start": 0.0,
                "text": "مرحباً بكم في هذا الفيديو",
                "frame_emotion": "happy",
                "frame_description": "شخص يبتسم أمام الكاميرا",
            },
            {
                "start": 5.0,
                "text": "أتمنى أن يعجبكم المحتوى",
                "frame_emotion": "happy",
                "frame_description": "نفس الشخص يتحدث بحماس",
            },
        ],
    }


def _make_fake_whisper_response(text, segments, language="ar"):
    response = MagicMock()
    response.text = text
    response.language = language
    response.segments = [
        MagicMock(start=start, end=end, text=seg_text)
        for start, end, seg_text in segments
    ]
    return response


class TestTranscribeChunkSuccess:
    def test_success_applies_time_offset_correctly(self, dummy_audio_chunk):
        fake_response = _make_fake_whisper_response(
            text="مرحبا بالجميع",
            segments=[(0.0, 2.5, "مرحبا"), (2.5, 5.0, "بالجميع")],
        )
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_response

        with patch(f"{AUDIO_LAYER_MODULE}._get_audio_duration", return_value=5.0):
            result = _transcribe_chunk(fake_client, dummy_audio_chunk, time_offset=600.0)

        assert result["status"] == "success"
        assert result["segments"] == [
            {"start": 600.0, "end": 602.5, "text": "مرحبا"},
            {"start": 602.5, "end": 605.0, "text": "بالجميع"},
        ]
        assert result["language"] == "ar"

    def test_success_with_zero_offset(self, dummy_audio_chunk):
        fake_response = _make_fake_whisper_response(
            text="hello", segments=[(1.0, 3.0, "hello")], language="en"
        )
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_response

        with patch(f"{AUDIO_LAYER_MODULE}._get_audio_duration", return_value=3.0):
            result = _transcribe_chunk(fake_client, dummy_audio_chunk, time_offset=0.0)

        assert result["status"] == "success"
        assert result["segments"] == [{"start": 1.0, "end": 3.0, "text": "hello"}]


class TestTranscribeChunkEmpty:
    def test_empty_transcription_is_not_an_error(self, dummy_audio_chunk):
        fake_response = _make_fake_whisper_response(text="", segments=[])
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_response

        with patch(f"{AUDIO_LAYER_MODULE}._get_audio_duration", return_value=5.0):
            result = _transcribe_chunk(fake_client, dummy_audio_chunk, time_offset=0.0)

        assert result["status"] == "success"
        assert result["segments"] == []
        assert result["text"] == ""


class TestTranscribeChunkFailure:
    def test_api_error_on_every_attempt_returns_error_status(self, dummy_audio_chunk):
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.side_effect = ConnectionError("network unreachable")

        result = _transcribe_chunk(fake_client, dummy_audio_chunk, time_offset=0.0)

        assert result["status"] == "error"
        assert "network unreachable" in result["message"]

    def test_retries_once_then_succeeds(self, dummy_audio_chunk):
        fake_response = _make_fake_whisper_response(text="ok", segments=[(0.0, 1.0, "ok")])
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.side_effect = [
            TimeoutError("temporary timeout"),
            fake_response,
        ]

        with patch(f"{AUDIO_LAYER_MODULE}._get_audio_duration", return_value=1.0):
            result = _transcribe_chunk(fake_client, dummy_audio_chunk, time_offset=0.0)

        assert result["status"] == "success"
        assert fake_client.audio.transcriptions.create.call_count == 2

    def test_missing_chunk_file_fails_fast_without_calling_api(self):
        fake_client = MagicMock()
        result = _transcribe_chunk(fake_client, "/nonexistent/path/chunk.wav", time_offset=0.0)
        assert result["status"] == "error"
        fake_client.audio.transcriptions.create.assert_not_called()


def _make_fake_claude_response(response_text):
    response = MagicMock()
    text_block = MagicMock()
    text_block.text = response_text
    response.content = [text_block]
    return response


class TestAnalyzeFrameSuccess:
    def test_valid_response_parses_all_fields(self, dummy_frame_image):
        fake_response = _make_fake_claude_response(
            "DESCRIPTION: A person speaking to camera\n"
            "SENTIMENT: positive\n"
            "EMOTION: happy\n"
            "TEXT_ON_SCREEN: SALE 50% OFF"
        )
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        entry = {"path": dummy_frame_image, "timestamp": 45.0}
        result = _analyze_frame(fake_client, entry)

        assert result["status"] == "success"
        assert result["is_fallback"] is False
        assert result["sentiment"] == "positive"
        assert result["emotion"] == "happy"
        assert result["text_on_screen"] == "SALE 50% OFF"
        assert result["timestamp"] == 45.0


class TestAnalyzeFrameInvalidResponse:
    def test_invalid_sentiment_value_falls_back_to_neutral(self, dummy_frame_image):
        fake_response = _make_fake_claude_response(
            "DESCRIPTION: Unclear scene\n"
            "SENTIMENT: somewhat_okay_ish\n"
            "EMOTION: unknown\n"
            "TEXT_ON_SCREEN: none"
        )
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        entry = {"path": dummy_frame_image, "timestamp": 10.0}
        result = _analyze_frame(fake_client, entry)

        assert result["status"] == "success"
        assert result["sentiment"] == "neutral"
        assert result["is_fallback"] is True

    def test_empty_response_falls_back_gracefully(self, dummy_frame_image):
        fake_response = _make_fake_claude_response("")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        entry = {"path": dummy_frame_image, "timestamp": 0.0}
        result = _analyze_frame(fake_client, entry)

        assert result["status"] == "success"
        assert result["sentiment"] == "neutral"
        assert result["is_fallback"] is True
        assert result["text_on_screen"] == "none"


class TestAnalyzeFrameFailure:
    def test_api_error_on_every_attempt_returns_error_status(self, dummy_frame_image):
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = ConnectionError("Claude API unreachable")

        entry = {"path": dummy_frame_image, "timestamp": 0.0}
        result = _analyze_frame(fake_client, entry)

        assert result["status"] == "error"
        assert "خطأ بتحليل اللقطة" in result["message"] or "Frame analysis error" in result["message"]

    def test_retries_once_then_succeeds(self, dummy_frame_image):
        fake_response = _make_fake_claude_response(
            "DESCRIPTION: Recovered on retry\n"
            "SENTIMENT: neutral\n"
            "EMOTION: calm\n"
            "TEXT_ON_SCREEN: none"
        )
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = [
            TimeoutError("transient"),
            fake_response,
        ]

        entry = {"path": dummy_frame_image, "timestamp": 0.0}
        result = _analyze_frame(fake_client, entry)

        assert result["status"] == "success"
        assert fake_client.messages.create.call_count == 2


# ──────────────────────────────────────────────
# process_report (report_layer) — the highest-stakes untested function
# before this addition: the call that produces the actual bilingual
# report text the user reads, via a forced submit_report tool_use call.
#
# Unlike _transcribe_chunk / _analyze_frame, process_report builds its
# own `anthropic.Anthropic()` client internally rather than taking one
# as a parameter — so instead of passing a fake_client in, these tests
# patch the class itself at the module level:
#     patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client)
# which is the same pattern test_report_layer.py already uses for
# _summarize_periods.
# ──────────────────────────────────────────────

# Sample report bodies long enough to clear _get_min_report_length's
# "tiny" tier threshold (100 chars, since minimal_fusion_result's
# total_segments=2 falls in that tier for report_mode="summary") — so
# these tests exercise tool_use extraction specifically, without also
# tripping an unrelated length validation_warning that would muddy the
# assertions.
SAMPLE_REPORT_AR = (
    "هذا تقرير تجريبي كامل يغطي محتوى الفيديو القصير المستخدم في هذا الاختبار "
    "بشكل كافٍ ومناسب لحجمه، ويوضح النبرة العامة الإيجابية التي ظهرت في الصوت والصورة معاً."
)
SAMPLE_REPORT_EN = (
    "This is a complete sample report covering the short test video used here, "
    "sufficiently and appropriately for its size, and it reflects the overall "
    "positive tone observed in both the audio and visual signals."
)


def _make_fake_report_tool_use_response(report_ar, report_en):
    """
    Simulates Claude's response to a forced submit_report tool call: one
    content block of type 'tool_use' whose `.input` is already a parsed
    dict (guaranteed valid, correctly-shaped JSON at the API level —
    this is the whole point of tool-use over hand-written JSON in text).
    """
    response = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"report_ar": report_ar, "report_en": report_en}
    response.content = [tool_block]
    return response


def _make_fake_report_text_only_response(raw_text):
    """
    Simulates an unexpected API response shape with no tool_use block at
    all — only a plain text block. This is the case _parse_report_response
    falls back on (_fallback_from_text), expected to be rare in normal
    operation but exercised here as defense-in-depth.
    """
    response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = raw_text
    response.content = [text_block]
    return response


class TestProcessReportSuccess:
    def test_success_extracts_report_via_tool_use(self, minimal_fusion_result):
        fake_response = _make_fake_report_tool_use_response(SAMPLE_REPORT_AR, SAMPLE_REPORT_EN)
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        with patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client):
            result = process_report(minimal_fusion_result, report_mode="summary")

        assert result["status"] == "success"
        assert result["report_ar"] == SAMPLE_REPORT_AR
        assert result["report_en"] == SAMPLE_REPORT_EN

    def test_success_passes_tool_choice_forcing_submit_report(self, minimal_fusion_result):
        """
        Confirms process_report actually forces the tool call (tool_choice
        pinned to submit_report) rather than merely offering the tool as
        optional — this is the guarantee the whole tool-use design relies
        on for structured output, so it's worth asserting directly on the
        call arguments, not just on the parsed result.
        """
        fake_response = _make_fake_report_tool_use_response(SAMPLE_REPORT_AR, SAMPLE_REPORT_EN)
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        with patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client):
            process_report(minimal_fusion_result, report_mode="summary")

        _, call_kwargs = fake_client.messages.create.call_args
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "submit_report"}
        assert call_kwargs["tools"][0]["name"] == "submit_report"


class TestProcessReportFallbackRecovery:
    def test_missing_tool_use_recovers_via_raw_json_text_block(self, minimal_fusion_result):
        """
        No tool_use block at all — an unexpected response shape, not
        something normal operation should produce. _parse_report_response
        must fall through to _fallback_from_text, which here successfully
        recovers by JSON-parsing the raw text block. The report should
        still come back usable, with a warning flagging that recovery
        path was needed (defense-in-depth working as intended, not
        silently masking the unexpected shape).
        """
        raw_json = f'{{"report_ar": "{SAMPLE_REPORT_AR}", "report_en": "{SAMPLE_REPORT_EN}"}}'
        fake_response = _make_fake_report_text_only_response(raw_json)
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        with patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client):
            result = process_report(minimal_fusion_result, report_mode="summary")

        assert result["status"] == "success"
        assert result["report_ar"] == SAMPLE_REPORT_AR
        assert result["report_en"] == SAMPLE_REPORT_EN
        assert any("tool_use" in w or "recovered" in w for w in result["validation_warnings"])

    def test_unrecoverable_response_still_returns_success_with_warning(self, minimal_fusion_result):
        """
        No tool_use block AND the text block isn't valid JSON or
        splittable — the worst-case fallback. process_report must not
        crash; it returns whatever raw text it has (or a placeholder) for
        both languages, with a warning explaining nothing could be
        recovered, rather than raising an unhandled exception up to the
        UI layer.
        """
        fake_response = _make_fake_report_text_only_response("not json and no language marker at all")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response

        with patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client):
            result = process_report(minimal_fusion_result, report_mode="summary")

        assert result["status"] == "success"
        assert "could not recover" in " ".join(result["validation_warnings"])


class TestProcessReportFailure:
    def test_api_error_on_every_attempt_returns_error_status(self, minimal_fusion_result):
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = ConnectionError("Claude API unreachable")

        with patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client):
            result = process_report(minimal_fusion_result, report_mode="summary")

        assert result["status"] == "error"
        assert "Claude API error" in result["message"] or "خطأ" in result["message"]

    def test_retries_once_then_succeeds(self, minimal_fusion_result):
        fake_response = _make_fake_report_tool_use_response(SAMPLE_REPORT_AR, SAMPLE_REPORT_EN)
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = [
            TimeoutError("transient"),
            fake_response,
        ]

        with patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client):
            result = process_report(minimal_fusion_result, report_mode="summary")

        assert result["status"] == "success"
        assert fake_client.messages.create.call_count == 2


class TestProcessReportUpstreamFailure:
    def test_fusion_layer_error_short_circuits_without_calling_api(self):
        """
        If fusion_layer itself already failed, process_report must return
        its own error immediately (chained failure message) WITHOUT ever
        constructing a client or making an API call — no point spending a
        Claude Sonnet call on a report built from data that doesn't exist.
        """
        failed_fusion_result = {"status": "error", "message": "audio layer aborted"}
        fake_client = MagicMock()

        with patch(f"{REPORT_LAYER_MODULE}.anthropic.Anthropic", return_value=fake_client):
            result = process_report(failed_fusion_result, report_mode="summary")

        assert result["status"] == "error"
        assert "Fusion layer failed" in result["message"]
        fake_client.messages.create.assert_not_called()