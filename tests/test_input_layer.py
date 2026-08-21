"""
Unit tests for pure functions in layers/input_layer.py.

_clean_url is a pure function (no API calls, no filesystem, no
subprocess), tested here directly.

_get_frame_interval_for_duration decides the FFmpeg frame extraction
interval (3s for short videos, 5s otherwise) based on video duration, so
short videos extract enough raw frames to fill the 90-frame analysis cap
(see visual_layer.py's _get_max_frames_for_duration).

process_input's platform-blocked download error classification (mocked
subprocess.run) is also covered below — see that section's docstring for
why it exists. Everything else in input_layer.py that touches yt-dlp,
ffmpeg, or the filesystem beyond that one path needs integration-style
testing with mocks, which is out of scope for this pass.
"""

from unittest.mock import MagicMock, patch

import layers.input_layer as input_layer
from layers.input_layer import _clean_url, _get_frame_interval_for_duration, process_input


# ──────────────────────────────────────────────
# _clean_url
# ──────────────────────────────────────────────

def test_clean_url_youtu_be_strips_all_query_params():
    """
    youtu.be short links are not the 'youtube.com/watch' case, so ALL query
    params (including tracking ones like si=) must be stripped entirely.
    """
    url = "https://youtu.be/abc123?si=xyz"
    assert _clean_url(url) == "https://youtu.be/abc123"


def test_clean_url_youtube_watch_keeps_only_v_param():
    """
    youtube.com/watch links must KEEP the v= param (required to identify
    the video) while dropping every other tracking param (feature=share
    here, but the same applies to utm_*, si, etc.).
    """
    url = "https://youtube.com/watch?v=abc123&feature=share"
    assert _clean_url(url) == "https://youtube.com/watch?v=abc123"


def test_clean_url_no_params_returns_unchanged():
    """
    A URL with no query string at all should pass through unchanged —
    there's nothing to strip.
    """
    url = "https://youtu.be/abc123"
    assert _clean_url(url) == "https://youtu.be/abc123"


def test_clean_url_local_file_path_returns_unchanged():
    """
    A local file path (no 'http' in the string) is not a URL at all, so
    _clean_url must return it completely untouched.
    """
    path = "C:\\videos\\test.mp4"
    assert _clean_url(path) == "C:\\videos\\test.mp4"


# ──────────────────────────────────────────────
# _get_frame_interval_for_duration
# ──────────────────────────────────────────────

def test_frame_interval_short_video_uses_denser_3_second_interval():
    """
    A <=10-min video needs a denser extraction interval (3s) so there
    are enough raw frames to fill the raised 90-frame analysis cap — at
    the default 5s interval, a 10-min video would only yield 120 raw
    frames, barely above (and in some cases below) that cap.
    """
    assert _get_frame_interval_for_duration(10 * 60) == 3


def test_frame_interval_longer_video_uses_default_5_second_interval():
    """
    Medium/long videos already extract far more raw frames than their
    cap needs at the default 5s interval, so there's no reason to
    extract more densely there — it would only cost extra FFmpeg time
    with zero effect on final analyzed coverage.
    """
    assert _get_frame_interval_for_duration(20 * 60) == 5
    assert _get_frame_interval_for_duration(60 * 60) == 5


def test_frame_interval_boundary_is_exact():
    """Exactly 10 minutes is still the 'short' tier (3s interval)."""
    assert _get_frame_interval_for_duration(10 * 60) == 3
    assert _get_frame_interval_for_duration(10 * 60 + 1) == 5


# ──────────────────────────────────────────────
# process_input — platform-blocked download error classification
# ──────────────────────────────────────────────
# Found via live testing: a real yt-dlp 403 (YouTube's anti-bot measures
# refusing the download request — a temporary, network/time-dependent
# condition, not a broken pipeline) surfaced as the same generic "Failed
# to prepare video" message as every other failure. Someone trying the
# deployed app who hits this could easily read it as "this project
# doesn't work" rather than "this platform is rate-limiting requests
# right now" — so a 403 specifically gets a distinct, honest message
# pointing at an alternative (TikTok/X) instead. Every other failure
# reason must keep falling back to the generic message, not be
# incorrectly swept into this specific, confirmed category.

def _mock_failed_subprocess(stderr_text):
    result = MagicMock()
    result.returncode = 1
    result.stderr = stderr_text
    return result


def test_process_input_403_gets_platform_blocked_message(tmp_path):
    fake_result = _mock_failed_subprocess(
        "ERROR: unable to download video data: HTTP Error 403: Forbidden"
    )
    with patch("layers.input_layer.subprocess.run", return_value=fake_result), \
         patch("layers.input_layer._check_required_api_keys", return_value=None):
        result = process_input(
            "https://www.youtube.com/watch?v=abc123",
            output_dir=str(tmp_path / "dl_403"),
        )

    assert result["status"] == "error"
    assert "temporarily blocking" in result["message"]
    assert "TikTok" in result["message"]


def test_process_input_other_failure_keeps_generic_message(tmp_path):
    """
    A non-403 failure (bad URL, private video, no internet, etc.) must NOT
    be misclassified as "platform blocked" — that would misinform the user
    about a cause we haven't actually confirmed.
    """
    fake_result = _mock_failed_subprocess("ERROR: Video unavailable")
    with patch("layers.input_layer.subprocess.run", return_value=fake_result), \
         patch("layers.input_layer._check_required_api_keys", return_value=None):
        result = process_input(
            "https://www.youtube.com/watch?v=abc123",
            output_dir=str(tmp_path / "dl_generic"),
        )

    assert result["status"] == "error"
    assert result["message"] == "Failed to prepare video. | فشل تجهيز الفيديو"


# ──────────────────────────────────────────────
# ensure_yt_dlp_updated — best-effort self-update guard
# ──────────────────────────────────────────────
# The module-level flag is reset at the top of each test below, since pytest
# runs this whole suite in one process and the flag is meant to persist for
# the life of a real process (that's the entire point — see the function's
# docstring). Without resetting it here, only the first test to touch it
# would ever see the "not yet attempted" state.

def test_ensure_yt_dlp_updated_runs_pip_upgrade_once():
    input_layer._yt_dlp_update_attempted = False
    fake_result = MagicMock(returncode=0, stderr="")
    with patch("layers.input_layer.subprocess.run", return_value=fake_result) as mock_run:
        input_layer.ensure_yt_dlp_updated()
        input_layer.ensure_yt_dlp_updated()  # second call, same process

    # Only the FIRST call may actually invoke pip — the module-level guard
    # exists specifically so this never re-runs (and re-hits the network)
    # on every download attempt within one process's lifetime.
    assert mock_run.call_count == 1
    assert mock_run.call_args[0][0] == ["pip", "install", "--upgrade", "--quiet", "yt-dlp"]


def test_ensure_yt_dlp_updated_swallows_subprocess_exception():
    """
    A failed upgrade (no internet, PyPI unreachable) must never raise — this
    is a best-effort background step, not something a user's analysis
    should ever fail because of.
    """
    input_layer._yt_dlp_update_attempted = False
    with patch("layers.input_layer.subprocess.run", side_effect=Exception("network unreachable")):
        input_layer.ensure_yt_dlp_updated()  # must not raise


def test_ensure_yt_dlp_updated_swallows_nonzero_returncode():
    input_layer._yt_dlp_update_attempted = False
    fake_result = MagicMock(returncode=1, stderr="Could not find a version that satisfies...")
    with patch("layers.input_layer.subprocess.run", return_value=fake_result):
        input_layer.ensure_yt_dlp_updated()  # must not raise