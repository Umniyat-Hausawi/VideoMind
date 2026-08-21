"""
One end-to-end integration test running the full analysis chain — audio +
visual → fusion → report — with mocks only at the external API boundary
(Whisper, Claude Vision, Claude Sonnet). Everything in between (fusion math,
prompt building, data shapes passed between layers) runs as real code.

Why this test exists, specifically:
Every other test file in this suite tests one layer in isolation (either a
pure function, or one API-calling function with a mocked client). That's
the right default — it's fast and pinpoints failures precisely. But it
can't catch a bug where layer A computes something correctly and layer B
never actually reads it, or reads it under the wrong key, or expects a
different shape than what A produced. That exact class of bug has already
happened in this project: `frame_description` was computed correctly in
fusion_layer but never made it into report_layer's prompt (fixed by wiring
it into `_build_prompt`'s `segments_text`), and `process_visual`'s
signature changed from a directory path to a structured `frames_manifest`
without every caller being updated in lockstep. Unit tests on either side
of a seam like that can be green while the seam itself is broken.

This test is intentionally narrow in scope (one happy-path run, not a
matrix of scenarios) — the per-layer test files already cover edge cases,
retries, and failure paths exhaustively. This one is a tripwire for the
data actually flowing correctly end to end, not a replacement for those.

Two specific regression guards are asserted at the end, both drawn directly
from real bugs found during manual review (see the project's bug-hunting
notes): a brand seen only visually (never spoken) must surface as a silent
brand mention out of fusion_layer, and a frame's visual description must
actually reach the text sent to Claude for report generation.
"""

from unittest.mock import MagicMock, patch

from layers.audio_layer import process_audio
from layers.visual_layer import process_visual
from layers.fusion_layer import process_fusion
from layers.report_layer import process_report


SAMPLE_REPORT_AR = (
    "هذا تقرير تجريبي كامل يغطي محتوى الفيديو القصير المستخدم بهذا اختبار التكامل "
    "بشكل كافٍ ومناسب لحجمه، ويوضح النبرة العامة الإيجابية التي ظهرت بالصوت والصورة معاً."
)
SAMPLE_REPORT_EN = (
    "This is a complete sample report covering the short test video used in this "
    "integration test, sufficiently and appropriately for its size, and it reflects "
    "the overall positive tone observed in both the audio and visual signals."
)


def _fake_whisper_response():
    response = MagicMock()
    response.text = "مرحباً بكم في هذا الفيديو الرائع"
    response.language = "ar"
    response.segments = [
        MagicMock(start=0.0, end=2.5, text="مرحباً بكم"),
        MagicMock(start=2.5, end=5.0, text="في هذا الفيديو الرائع"),
    ]
    return response


def _audio_anthropic_side_effect(**kwargs):
    """
    audio_layer's Anthropic client is reused for two different prompt
    shapes (single-period sentiment, and batched JSON-array sentiment) —
    branch on the prompt content to return the right shape for each,
    same as a real model would respond differently to each instruction.
    """
    prompt = kwargs["messages"][0]["content"]
    response = MagicMock()
    if "Analyze the overall sentiment of this transcript segment" in prompt:
        response.content = [MagicMock(text="positive")]
    else:
        response.content = [MagicMock(
            text='[{"id": 1, "sentiment": "positive"}, {"id": 2, "sentiment": "positive"}]'
        )]
    return response


def _fake_vision_response(description: str, emotion: str, brand_logos: str) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=(
        f"DESCRIPTION: {description}\n"
        f"SENTIMENT: positive\n"
        f"EMOTION: {emotion}\n"
        f"TEXT_ON_SCREEN: none\n"
        f"BRAND_LOGOS_VISIBLE: {brand_logos}"
    ))]
    return response


def _fake_report_tool_use_response(report_ar: str, report_en: str) -> MagicMock:
    response = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"report_ar": report_ar, "report_en": report_en}
    response.content = [tool_block]
    return response


def test_full_pipeline_data_flows_correctly(tmp_path, monkeypatch):
    # ── Fabricate inputs, as if input_layer already ran ──
    chunk_path = tmp_path / "chunk_000.wav"
    chunk_path.write_bytes(b"\x00" * 100)
    audio_chunks = [str(chunk_path)]

    frame1 = tmp_path / "frame_00000.00s.jpg"
    frame1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    frame2 = tmp_path / "frame_00005.00s.jpg"
    frame2.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    frames_manifest = [
        {"path": str(frame1), "timestamp": 0.0},
        {"path": str(frame2), "timestamp": 5.0},
    ]

    # ── Mock every external API boundary ──
    fake_openai_client = MagicMock()
    fake_openai_client.audio.transcriptions.create.return_value = _fake_whisper_response()

    fake_audio_anthropic_client = MagicMock()
    fake_audio_anthropic_client.messages.create.side_effect = _audio_anthropic_side_effect

    fake_visual_anthropic_client = MagicMock()
    fake_visual_anthropic_client.messages.create.side_effect = [
        _fake_vision_response(
            "An adult smiling at the camera, holding a cup with a visible brand logo",
            "happy",
            "Acme Coffee",
        ),
        _fake_vision_response(
            "The same adult gesturing enthusiastically while speaking",
            "excited",
            "none",
        ),
    ]

    fake_report_anthropic_client = MagicMock()
    fake_report_anthropic_client.messages.create.return_value = _fake_report_tool_use_response(
        SAMPLE_REPORT_AR, SAMPLE_REPORT_EN
    )

    monkeypatch.setattr("layers.audio_layer._get_audio_duration", lambda path: 5.0)
    monkeypatch.setattr("layers.audio_layer.time.sleep", lambda s: None)
    monkeypatch.setattr("layers.visual_layer.time.sleep", lambda s: None)
    monkeypatch.setattr("layers.report_layer.time.sleep", lambda s: None)

    # ── Audio layer ──
    with patch("layers.audio_layer.openai.OpenAI", return_value=fake_openai_client), \
         patch("layers.audio_layer.anthropic.Anthropic", return_value=fake_audio_anthropic_client):
        audio_result = process_audio(audio_chunks)

    assert audio_result["status"] == "success"
    assert len(audio_result["segments"]) == 2

    # ── Visual layer ──
    with patch("layers.visual_layer.anthropic.Anthropic", return_value=fake_visual_anthropic_client):
        visual_result = process_visual(frames_manifest, video_duration_seconds=10.0)

    assert visual_result["status"] == "success"
    assert len(visual_result["frame_analyses"]) == 2

    # ── Fusion layer (real code, no mocking — pure computation) ──
    fusion_result = process_fusion(audio_result, visual_result)
    assert fusion_result["status"] == "success"

    # Regression guard #1: a brand seen only visually (never spoken in the
    # transcript) must surface as a silent brand mention. This is the
    # feature's whole reason for existing — losing this silently would
    # defeat the point without any single-layer test noticing, since
    # audio_layer and visual_layer each did their own job correctly in
    # isolation; only fusion's cross-referencing step could catch this.
    silent_brands = fusion_result["summary"]["silent_brand_mentions"]
    assert any(b["brand"] == "Acme Coffee" for b in silent_brands)

    # ── Report layer ──
    with patch("layers.report_layer.anthropic.Anthropic", return_value=fake_report_anthropic_client):
        report_result = process_report(fusion_result, report_mode="summary")

    assert report_result["status"] == "success"
    assert report_result["report_ar"] == SAMPLE_REPORT_AR
    assert report_result["report_en"] == SAMPLE_REPORT_EN

    # Regression guard #2: a frame's visual scene description must actually
    # reach the prompt sent to Claude for report generation. This is
    # exactly the class of bug already found once in this project (a real
    # frame description existed in fusion_result but never reached
    # report_layer's prompt) — a unit test on report_layer alone, with a
    # hand-built fusion_result fixture, would not have caught it, because
    # the fixture would already have whatever shape the test author
    # assumed was correct.
    _, call_kwargs = fake_report_anthropic_client.messages.create.call_args
    sent_prompt = call_kwargs["messages"][0]["content"]
    assert (
        "smiling at the camera" in sent_prompt
        or "gesturing enthusiastically" in sent_prompt
    ), "frame_description did not reach the report-generation prompt"