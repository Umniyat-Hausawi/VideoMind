import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

SUPPORTED_PLATFORMS     = ["youtube.com", "youtu.be", "tiktok.com", "twitter.com", "x.com"]
# Instagram is deliberately NOT in this list. Every Instagram download attempt
# fails consistently (yt-dlp needs a logged-in session this app doesn't have —
# see the explicit instagram.com check below), so it isn't a "supported"
# platform in any real sense. It's still detected and rejected with a clear,
# actionable message rather than falling through to the generic "unsupported
# source" error.
SUPPORTED_FILE_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

FRAME_INTERVAL_SECONDS        = 5      # default extraction interval (medium/long videos)
SHORT_VIDEO_FRAME_INTERVAL    = 3      # denser interval for short videos — see
                                        # _get_frame_interval_for_duration() for why
SHORT_VIDEO_THRESHOLD_SECONDS = 10 * 60
MAX_DURATION_SECONDS          = 2 * 60 * 60   # 2 hours
CHUNK_DURATION_SECONDS        = 10 * 60


# ──────────────────────────────────────────────
# yt-dlp self-update (best-effort, once per process)
# ──────────────────────────────────────────────

_yt_dlp_update_attempted = False


def ensure_yt_dlp_updated() -> None:
    """
    Best-effort self-update for yt-dlp, meant to be called once at app
    startup (see app.py).

    yt-dlp ships its own `-U`/`--update` self-update flag, but it explicitly
    refuses to run when yt-dlp was installed via pip (our case, via
    requirements.txt) — it detects the install method and tells you to
    update through your package manager instead. So keeping it current
    after deployment means running `pip install --upgrade yt-dlp` ourselves.

    Why this matters: YouTube's anti-bot measures change frequently, and
    yt-dlp ships near-weekly releases to keep up — an out-of-date yt-dlp is
    one of the two realistic causes behind a YouTube 403 (the other being
    the hosting platform's IP range being rate-limited, which no amount of
    updating fixes — see Limitations.md). requirements.txt intentionally
    leaves yt-dlp unpinned, so every fresh container build already picks up
    the latest release; this function extends that to every process start
    *between* deploys too — including a Hugging Face Space waking up from
    sleep — without needing a manual redeploy.

    Deliberately fails silently, same pattern as observability.py: no
    internet at startup, PyPI unreachable, or the upgrade command itself
    failing must never crash or block the app — worst case, it just keeps
    running whatever yt-dlp version was already installed. Runs at most
    once per process (guarded by the module-level flag below), not on every
    download attempt, so it never adds latency beyond the first call.
    """
    global _yt_dlp_update_attempted
    if _yt_dlp_update_attempted:
        return
    _yt_dlp_update_attempted = True

    try:
        result = subprocess.run(
            ["pip", "install", "--upgrade", "--quiet", "yt-dlp"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print("yt-dlp self-update check completed (pip install --upgrade yt-dlp).")
        else:
            print(f"WARNING: yt-dlp self-update failed — continuing with existing version: {result.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: yt-dlp self-update check failed — continuing with existing version: {e}")


# ──────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────

def process_input(source: str, output_dir: str = "videomind_output") -> dict:
    """
    Main entry point for the Input Layer.

    Args:
        source     : URL from social media or local video file path
        output_dir : Directory to save audio and frames

    Returns:
        {
            "audio_path"       : str,
            "audio_chunks"     : list[str],
            "frames_dir"       : str,
            "frames_manifest"  : list[dict],  # [{"path": str, "timestamp": float}, ...]
                                               # structured frame data; visual_layer reads
                                               # timestamps from here, not from filenames
            "duration_seconds" : float,
            "source_type"      : "url" | "file",
            "warnings"         : list[str],   # non-fatal issues (e.g. failed audio chunks)
            "status"           : "success" | "error",
            "message"          : str
        }
    """
    # Fail fast — check required API keys before doing any expensive work
    key_error = _check_required_api_keys()
    if key_error:
        return _error_result(key_error)

    # Clear previous output to avoid stale data from old videos
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Clean URL before processing
    source = _clean_url(source)

    source_type = _detect_source_type(source)

    # Instagram requires a logged-in session (cookies) that yt-dlp doesn't
    # have here — every download attempt fails with the same underlying
    # error regardless of the specific post. Rather than burn time on a
    # download attempt that's known to fail, and surface yt-dlp's generic
    # "empty media response" error, catch it explicitly here with a clear
    # message pointing to the working alternative (direct file upload).
    if "instagram.com" in source.lower():
        return _error_result(
            "Instagram links currently require a logged-in session that this app doesn't "
            "have, so downloads fail consistently regardless of the specific post. Please "
            "download the video yourself and use \"Upload File\" instead. "
            "| روابط Instagram تحتاج حالياً جلسة تسجيل دخول غير متوفرة بهذا التطبيق، فالتحميل "
            "يفشل بشكل ثابت بغض النظر عن الرابط. الرجاء تحميل الفيديو بنفسك واستخدام "
            "\"رفع ملف\" بدلاً من الرابط"
        )

    download_error_reason = None

    if source_type == "url":
        video_path, download_error_reason = _download_video(source, output_dir)
    elif source_type == "file":
        video_path = _validate_local_file(source)
    else:
        return _error_result(
            "Unsupported source. Please provide a URL or a video file. "
            "| مصدر غير مدعوم. الرجاء إدخال رابط أو رفع ملف فيديو"
        )

    if not video_path:
        if download_error_reason == "platform_blocked":
            # A specific, confirmed cause (yt-dlp's request was actively
            # refused with 403) — distinct from the generic case below,
            # which covers everything we *haven't* diagnosed (bad URL, no
            # internet, private/removed video, etc.). Worth naming
            # explicitly so this doesn't read as a broken pipeline to
            # someone trying the app — it's a known, temporary,
            # platform-side restriction that varies by network and time.
            return _error_result(
                "This platform is temporarily blocking automated download requests for "
                "this video (a known, temporary restriction — common with YouTube's "
                "anti-bot measures, and it can vary by network and time of day, not a "
                "problem with this app). Please try a different video, or use TikTok/X "
                "for this demo instead. "
                "| هذه المنصة تحجب مؤقتاً طلبات التحميل الآلي لهذا الفيديو (قيد مؤقت "
                "معروف — شائع مع إجراءات يوتيوب لمكافحة الروبوتات، ويختلف حسب الشبكة "
                "والوقت، وليس عطلاً بهذا التطبيق). جربي فيديو آخر، أو استخدمي تيك توك/X "
                "لهذا العرض التوضيحي"
            )
        return _error_result("Failed to prepare video. | فشل تجهيز الفيديو")

    duration = _get_video_duration(video_path)
    if duration > MAX_DURATION_SECONDS:
        return _error_result(
            f"Video duration ({duration/60:.1f} min) exceeds the {MAX_DURATION_SECONDS/60:.0f}-minute limit. "
            f"| مدة الفيديو ({duration/60:.1f} دقيقة) تتجاوز الحد الأقصى ({MAX_DURATION_SECONDS/60:.0f} دقيقة)"
        )

    audio_path = _extract_audio(video_path, output_dir)
    if not audio_path:
        return _error_result("Failed to extract audio. | فشل استخراج الصوت")

    audio_chunks, chunk_warnings = _chunk_audio(audio_path, output_dir, duration)

    frame_interval = _get_frame_interval_for_duration(duration)
    frames_dir, frames_manifest = _extract_frames(video_path, output_dir, frame_interval)
    if not frames_dir:
        return _error_result("Failed to extract frames. | فشل استخراج اللقطات")

    message = (
        f"Processed successfully — {duration/60:.1f} min, {len(audio_chunks)} chunk(s) "
        f"| تمت المعالجة بنجاح — {duration/60:.1f} دقيقة، {len(audio_chunks)} قطعة"
    )
    if chunk_warnings:
        message += (
            f" (with {len(chunk_warnings)} warning(s) — see 'warnings') "
            f"| (مع {len(chunk_warnings)} تحذير — راجعي 'warnings')"
        )

    return {
        "audio_path"       : audio_path,
        "audio_chunks"     : audio_chunks,
        "frames_dir"       : frames_dir,
        "frames_manifest"  : frames_manifest,
        "duration_seconds" : duration,
        "source_type"      : source_type,
        "warnings"         : chunk_warnings,
        "status"           : "success",
        "message"          : message
    }


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def _check_required_api_keys() -> str | None:
    """
    Fail-fast check for required API keys.

    Why check here instead of letting audio_layer / visual_layer fail later?
    Downloading + extracting audio/frames for a long video can take minutes.
    Better to fail in under a second if a key is missing than to burn that
    time and fail deep in the middle of the pipeline.
    """
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if missing:
        return (
            f"Missing required API key(s): {', '.join(missing)}. Please set them in your .env file. "
            f"| مفتاح (مفاتيح) API ناقصة: {', '.join(missing)}. الرجاء إضافتها بملف .env"
        )
    return None


def _clean_url(source: str) -> str:
    """
    Clean URL from tracking parameters using urllib.parse.

    Rule:
    - YouTube "watch" links (youtube.com/watch?v=...) keep ONLY the `v` param —
      it's required to identify the video, everything else (si, utm_source, ...) is dropped.
    - Every other URL (youtu.be short links, tiktok, instagram, twitter/x, ...)
      has its entire query string stripped — none of them need query params
      for yt-dlp to work, and tracking params can cause 403 errors.

    Examples:
        https://www.youtube.com/watch?v=abc123&si=xyz  → https://www.youtube.com/watch?v=abc123
        https://youtu.be/abc123?si=xyz                 → https://youtu.be/abc123
    """
    source = source.strip()

    if "http" not in source:
        return source

    parsed = urlparse(source)
    netloc_lower = parsed.netloc.lower()
    path_lower   = parsed.path.lower().rstrip("/")

    is_youtube_watch = "youtube.com" in netloc_lower and path_lower.endswith("watch")

    if is_youtube_watch:
        query_params = parse_qs(parsed.query)
        if "v" in query_params:
            new_query = urlencode({"v": query_params["v"][0]})
        else:
            new_query = ""
    else:
        new_query = ""

    cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ""))
    return cleaned


def _detect_source_type(source: str) -> str:
    """Detect whether the source is a URL or a local file."""
    source_lower = source.lower()

    for platform in SUPPORTED_PLATFORMS:
        if platform in source_lower:
            return "url"

    if os.path.exists(source):
        ext = Path(source).suffix.lower()
        if ext in SUPPORTED_FILE_EXTENSIONS:
            return "file"

    return "unknown"


def _download_video(url: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Download video at lowest quality using yt-dlp.
    Low quality is intentional — we only need frames and audio, not HD video.

    Returns (video_path, error_reason). error_reason is None on success, and
    is only ever set to a *specific* known category (currently just
    "platform_blocked") — every other failure leaves it None so the caller
    falls back to the generic message rather than guessing at a cause we
    haven't actually confirmed.
    """
    video_output = os.path.join(output_dir, "input_video.mp4")
    is_youtube   = "youtube.com" in url.lower() or "youtu.be" in url.lower()

    command = [
        "yt-dlp",
        "--format", "worst[ext=mp4]+bestaudio/worst",
        "--merge-output-format", "mp4",
        "--output", video_output,
        "--no-playlist",
        "--max-filesize", "500m",
        url
    ]

    try:
        print(f"Downloading video from: {url}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"yt-dlp error: {result.stderr}")
            stderr_lower = result.stderr.lower()

            # YouTube's anti-bot blocking shows up in several different
            # error shapes depending on which check tripped — a plain HTTP
            # 403, an HTTP 429 (rate limit), or the newer "sign in to
            # confirm you're not a bot" challenge text. All of these are
            # the same underlying temporary, platform-side restriction,
            # not a broken download pipeline — worth distinguishing from
            # every other failure mode (bad URL, no internet, etc.) so the
            # UI can say something more useful than a generic "failed"
            # message.
            BLOCKED_SIGNATURES = [
                "http error 403", "403: forbidden",
                "http error 429", "429: too many requests",
                "sign in to confirm", "not a bot",
            ]
            # Failure modes that are genuinely NOT a platform block — a
            # video that's actually gone/private/invalid shouldn't be
            # mislabeled as "temporarily blocked" just because it's a
            # YouTube URL.
            NOT_BLOCKED_SIGNATURES = [
                "video unavailable", "private video",
                "this video is not available", "has been removed",
                "is not a valid url",
            ]

            is_blocked_error = any(sig in stderr_lower for sig in BLOCKED_SIGNATURES)
            is_not_blocked   = any(sig in stderr_lower for sig in NOT_BLOCKED_SIGNATURES)

            # For YouTube specifically: unless the error clearly points to
            # something else (private/removed/invalid), treat any download
            # failure as the anti-bot block — in practice this covers the
            # error shapes YouTube introduces over time that we haven't
            # seen/matched yet, without waiting for another silent
            # "Failed to prepare video." report.
            if is_blocked_error or (is_youtube and not is_not_blocked):
                return None, "platform_blocked"
            return None, None

        if os.path.exists(video_output):
            print(f"Download complete: {video_output}")
            return video_output, None

    except subprocess.TimeoutExpired:
        print("Download timed out (5 minutes)")
    except FileNotFoundError:
        print("yt-dlp not found. Run: pip install yt-dlp")

    return None, None


def _validate_local_file(file_path: str) -> str | None:
    """Validate that the local file exists and has a supported extension."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_FILE_EXTENSIONS:
        print(f"Unsupported file extension: {ext}")
        return None

    print(f"Local file ready: {file_path}")
    return file_path


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        print(f"Video duration: {duration/60:.1f} minutes")
        return duration
    except (ValueError, subprocess.SubprocessError):
        print("Could not read duration — using default value")
        return 0.0


def _extract_audio(video_path: str, output_dir: str) -> str | None:
    """
    Extract audio as .wav using FFmpeg.
    16kHz mono PCM — optimal format for Whisper.
    """
    audio_output = os.path.join(output_dir, "audio.wav")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_output
    ]

    try:
        print("Extracting audio...")
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and os.path.exists(audio_output):
            size_mb = os.path.getsize(audio_output) / (1024 * 1024)
            print(f"Audio extracted: {size_mb:.1f} MB")
            return audio_output
        else:
            print(f"FFmpeg error: {result.stderr[-300:]}")

    except FileNotFoundError:
        print("FFmpeg not found.")
    except subprocess.TimeoutExpired:
        print("Audio extraction timed out.")

    return None


def _chunk_audio(audio_path: str, output_dir: str, duration: float) -> tuple[list[str], list[str]]:
    """
    Split audio into chunks if longer than CHUNK_DURATION_SECONDS.
    Short videos are returned as a single-item list with no splitting.

    Returns:
        (chunks, warnings) — warnings is a list of human-readable messages
        for any chunk that failed to be created, plus a summary warning if
        the final chunk count is lower than expected.
    """
    chunks_dir = os.path.join(output_dir, "audio_chunks")
    Path(chunks_dir).mkdir(exist_ok=True)
    warnings: list[str] = []

    if duration <= CHUNK_DURATION_SECONDS:
        print(f"Audio is short ({duration/60:.1f} min) — no chunking needed")
        return [audio_path], warnings

    num_chunks = int(duration / CHUNK_DURATION_SECONDS) + 1
    chunks     = []

    for i in range(num_chunks):
        start_time = i * CHUNK_DURATION_SECONDS
        chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}.wav")

        command = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(CHUNK_DURATION_SECONDS),
            "-y",
            chunk_path
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(chunk_path):
            chunks.append(chunk_path)
        else:
            error_tail = result.stderr[-200:] if result.stderr else "unknown error"
            warning_msg = (
                f"Failed to create audio chunk {i+1}/{num_chunks} (start={start_time}s): {error_tail} "
                f"| فشل إنشاء القطعة الصوتية {i+1}/{num_chunks} (تبدأ من {start_time} ثانية)"
            )
            print(f"WARNING: {warning_msg}")
            warnings.append(warning_msg)

    if len(chunks) < num_chunks:
        summary_warning = (
            f"Only {len(chunks)}/{num_chunks} audio chunks created successfully — "
            f"transcription may be incomplete. "
            f"| نجحت {len(chunks)}/{num_chunks} قطعة صوتية بس — النص المحوّل ممكن يكون ناقص"
        )
        print(f"WARNING: {summary_warning}")
        warnings.append(summary_warning)

    print(f"Audio split into {len(chunks)} chunks")
    return chunks, warnings


def _get_frame_interval_for_duration(duration_seconds: float) -> int:
    """
    Choose the FFmpeg frame extraction interval based on video duration.

    Only short videos need a denser interval. For a ≤10-min video, the
    default 5s interval yields only 120 raw frames — barely above (and in
    some cases below) visual_layer's analysis cap for that tier (90
    frames, see visual_layer.py's _get_max_frames_for_duration), which
    leaves _sample_frames() no real room to distribute evenly.

    Medium/long videos don't need this: at the default 5s interval, a
    30-min video already yields 360 raw frames against a 180-frame cap,
    and a 60-min video yields 720 against a 270-frame cap — both already
    far above their target, so extracting more densely there would only
    add FFmpeg decode time and temp disk usage with zero effect on final
    analyzed coverage (the downstream cap discards the extra frames
    either way).
    """
    if duration_seconds <= SHORT_VIDEO_THRESHOLD_SECONDS:
        return SHORT_VIDEO_FRAME_INTERVAL
    return FRAME_INTERVAL_SECONDS


def _extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: int = FRAME_INTERVAL_SECONDS
) -> tuple[str | None, list[dict]]:
    """
    Extract one frame every `interval_seconds` using a direct FFmpeg filter
    (fps=1/N) instead of reading every frame sequentially with OpenCV.

    Why FFmpeg fps filter instead of OpenCV frame-by-frame reading?
    OpenCV had to decode and read every single frame of the video just to
    discard almost all of them (a 1-hour video at 25fps means reading ~90,000
    frames to keep ~720). FFmpeg's fps filter decodes and outputs only the
    frames we actually need, directly at the decoder level.

    Returns (frames_dir, frames_manifest):
        frames_dir      : str | None — the folder frames were saved to
        frames_manifest : list[dict] — [{"path": str, "timestamp": float}, ...]
                           in chronological order. Timestamps are computed
                           directly here (index * interval_seconds) while
                           we already know the real interval used — this
                           is the single source of truth for frame
                           timing. The filename is still given a
                           timestamp-based name for human-readability when
                           browsing the output folder, but nothing
                           downstream needs to parse it back out anymore;
                           visual_layer reads timestamps from this
                           manifest instead.
    """
    frames_dir      = os.path.join(output_dir, "frames")
    frames_dir_path = Path(frames_dir)
    frames_dir_path.mkdir(exist_ok=True)

    output_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
    fps_expression  = f"fps=1/{interval_seconds}"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", fps_expression,
        "-qscale:v", "2",
        "-y",
        output_pattern
    ]

    try:
        print(f"Extracting frames via FFmpeg (every {interval_seconds} seconds)...")
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"FFmpeg frame extraction error: {result.stderr[-300:]}")
            return None, []

        extracted_files = sorted(frames_dir_path.glob("frame_*.jpg"))

        frames_manifest = []
        for index, frame_file in enumerate(extracted_files):
            timestamp_seconds = index * interval_seconds

            # Filename still encodes the timestamp — purely for a human
            # browsing the output folder to make sense of it. The
            # manifest entry below (not this filename) is what the rest
            # of the pipeline actually relies on.
            new_name = frames_dir_path / f"frame_{timestamp_seconds:08.2f}s.jpg"
            frame_file.rename(new_name)

            frames_manifest.append({
                "path"      : str(new_name),
                "timestamp" : float(timestamp_seconds),
            })

        print(f"Saved {len(extracted_files)} frames to: {frames_dir}")
        return frames_dir, frames_manifest

    except FileNotFoundError:
        print("FFmpeg not found.")
    except subprocess.TimeoutExpired:
        print("Frame extraction timed out.")

    return None, []


# ──────────────────────────────────────────────
# Error Helper
# ──────────────────────────────────────────────

def _error_result(message: str) -> dict:
    """Return a consistent error result dictionary."""
    print(f"ERROR: {message}")
    return {
        "audio_path"       : None,
        "audio_chunks"     : [],
        "frames_dir"       : None,
        "frames_manifest"  : [],
        "duration_seconds" : 0,
        "source_type"      : "unknown",
        "warnings"         : [],
        "status"           : "error",
        "message"          : message
    }