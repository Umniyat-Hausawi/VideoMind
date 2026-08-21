import streamlit as st
import sys
import os
import uuid
import shutil
import hashlib
import tempfile
import pandas as pd
import altair as alt
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

sys.path.append(".")

# MUST run before importing observability: observability.py reads
# LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY from os.environ at IMPORT TIME
# (LANGFUSE_ENABLED is set once, at module load). Every other file in this
# project that needs .env (audio_layer.py, visual_layer.py, report_layer.py)
# calls load_dotenv() itself, but those imports all happen AFTER
# observability's — so without this line here, os.getenv() sees nothing,
# LANGFUSE_ENABLED silently evaluates to False, and init_observability()
# returns False with no error/warning printed at all (that branch has no
# print statement) — tracing never turns on, no matter how many times the
# app is restarted. Root cause of the "Waiting for first trace" issue.
load_dotenv()

from observability import init_observability, flush_observability
init_observability()  # opt-in Langfuse tracing — no-op unless LANGFUSE_* env vars are set

from layers.input_layer  import process_input, ensure_yt_dlp_updated
ensure_yt_dlp_updated()  # best-effort yt-dlp self-update at process start — see input_layer.py
from layers.audio_layer  import process_audio
from layers.visual_layer import process_visual
from layers.fusion_layer import process_fusion
from layers.report_layer import process_report


# ──────────────────────────────────────────────
# Partial Content Hash
# ──────────────────────────────────────────────

# Bytes read from the start AND from the end of the file (so this many
# bytes total, twice).
PARTIAL_HASH_CHUNK_BYTES = 65536  # 64KB


def _partial_file_hash(uploaded_file) -> str:
    """
    Build a content-based identifier for an uploaded file using a partial
    hash (first + last PARTIAL_HASH_CHUNK_BYTES, plus total size) instead
    of just filename + filesize.

    Why? Two different files can share the same name and even the exact
    same size (e.g. two clips independently trimmed to an identical
    duration) — a filename+filesize identifier would collide and wrongly
    serve cached results from a completely different video. Hashing the
    ENTIRE file avoids that, but costs real time on a large video upload
    just to build a cache key. Hashing the first and last chunk (plus
    total size) is a middle ground: two different videos are extremely
    unlikely to share the same first 64KB AND last 64KB AND exact byte
    size, at a fraction of the cost of a full-file hash.

    Streamlit's UploadedFile supports seek()/read() like a normal file
    object, and is reset back to position 0 before returning so the
    later `uploaded_file.read()` (when actually writing the temp file)
    isn't affected.
    """
    uploaded_file.seek(0)
    start_bytes = uploaded_file.read(PARTIAL_HASH_CHUNK_BYTES)

    uploaded_file.seek(0, os.SEEK_END)
    total_size = uploaded_file.tell()

    end_read_start = max(0, total_size - PARTIAL_HASH_CHUNK_BYTES)
    uploaded_file.seek(end_read_start)
    end_bytes = uploaded_file.read(PARTIAL_HASH_CHUNK_BYTES)

    uploaded_file.seek(0)  # reset — the analyze block still needs to read() this file

    hasher = hashlib.sha256()
    hasher.update(start_bytes)
    hasher.update(end_bytes)
    hasher.update(str(total_size).encode())
    return hasher.hexdigest()[:16]  # 16 hex chars is plenty of collision resistance for a cache key


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title = "VideoMind",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .vm-header {
        background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }

    .vm-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .vm-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        margin-top: 0.4rem;
        line-height: 1.6;
    }

    .vm-section {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .vm-step {
        border-left: 4px solid #7c3aed;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 1.2rem;
        margin: 0.3rem 0;
        font-size: 0.92rem;
        background: #ede9fe;
        color: #1c1e21;
        font-weight: 500;
    }

    /* Larger, more readable font for all markdown-rendered content —
       this covers the final bilingual report, which was too small to
       read comfortably at Streamlit's default size. */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-size: 1.15rem;
        line-height: 1.9;
    }

    [data-testid="stMarkdownContainer"] h1 { font-size: 1.9rem; }
    [data-testid="stMarkdownContainer"] h2 { font-size: 1.5rem; }
    [data-testid="stMarkdownContainer"] h3 { font-size: 1.25rem; }

    .vm-cache-note {
        border-left: 4px solid #16a34a;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1.1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        background: #f0fdf4;
        color: #14532d;
    }

    .stProgress > div > div > div > div {
        background: #7c3aed !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown("""
<div class="vm-header">
    <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
        <span style="font-size:2.2rem;">🎬</span>
        <h1 class="vm-title">VideoMind</h1>
    </div>
    <p class="vm-subtitle">
        نظام تحليل الفيديو الذكي &nbsp;·&nbsp; Multimodal Video Intelligence System
    </p>
    <p class="vm-subtitle" style="font-size:0.85rem; margin-top:0.2rem;">
        حلل أي فيديو صوتاً وصورةً — واحصل على تقرير ذكي بلغتين
        &nbsp;·&nbsp;
        Analyze any video from audio + visual — get a smart bilingual report
    </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Input Section
# ──────────────────────────────────────────────

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<p class="vm-section">📥 مصدر الفيديو | Video Source</p>', unsafe_allow_html=True)

    source_type = st.radio(
        "نوع المدخل",
        ["🔗 رابط | URL (YouTube / TikTok / X)", "📁 رفع ملف | Upload File"],
        horizontal=True,
        label_visibility="collapsed"
    )

    uploaded_file = None
    video_source  = None

    if "رابط" in source_type:
        video_source = st.text_input(
            "رابط الفيديو",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )
    else:
        # The uploaded file stays in memory here — the actual temp file is
        # created (and later deleted) inside the analyze block, using
        # tempfile.NamedTemporaryFile with a random name.
        uploaded_file = st.file_uploader(
            "ارفع ملف الفيديو | Upload video file",
            type=["mp4", "mov", "avi", "mkv"],
            label_visibility="collapsed"
        )

with col2:
    st.markdown('<p class="vm-section">⚙️ خيارات التحليل | Analysis Options</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        report_type = st.radio(
            "نوع التقرير | Report Type",
            ["📊 تحليل | Analysis", "📝 ملخص | Summary"],
            index=0
        )
    with c2:
        report_language = st.radio(
            "لغة التقرير | Language",
            ["العربية", "English", "كلاهما | Both"],
            index=2
        )

    audio_only = st.checkbox("🎙️ تحليل صوتي فقط | Audio only (faster)")

st.divider()


# ──────────────────────────────────────────────
# Analyze Button
# ──────────────────────────────────────────────

analyze_clicked = st.button(
    "🚀 تحليل الفيديو | Analyze",
    type="primary",
    width="stretch"
)

if analyze_clicked:
    if not video_source and uploaded_file is None:
        st.error("❌ الرجاء إدخال رابط أو رفع ملف | Please provide a URL or upload a file")
    else:
        progress = st.progress(0, text="جاري التحضير... | Preparing...")
        step_box = st.empty()

        def show_step(msg: str, pct: int):
            progress.progress(pct, text=msg)
            step_box.markdown(
                f'<div class="vm-step">{msg}</div>',
                unsafe_allow_html=True
            )

        # ── Build a cache key identifying "this exact video + this exact
        # audio_only setting" — used to decide whether we can skip
        # input/audio/visual/fusion and just regenerate the report. ──
        if uploaded_file is not None:
            source_identifier = f"upload:{_partial_file_hash(uploaded_file)}"
        else:
            source_identifier = f"url:{video_source}"

        cache_key = (source_identifier, audio_only)

        cached = st.session_state.get("videomind_cache")
        cache_hit = cached is not None and cached.get("cache_key") == cache_key

        temp_video_path = None
        session_output_dir = None

        try:
            if cache_hit:
                # ── Reuse previously computed input/audio/visual/fusion
                # results for this exact video — only report generation
                # depends on report_mode/language, so that's the only
                # step that needs to run again. ──
                input_result  = cached["input_result"]
                audio_result  = cached["audio_result"]
                visual_result = cached["visual_result"]
                fusion_result = cached["fusion_result"]

                st.markdown(
                    '<div class="vm-cache-note">♻️ نستخدم نتائج التحليل المحفوظة لنفس الفيديو '
                    '— جاري إنشاء التقرير بس | Reusing cached analysis for this video '
                    '— only regenerating the report</div>',
                    unsafe_allow_html=True
                )
                show_step("📊 جاري إنشاء التقرير... | Generating report...", 70)

            else:
                # Unique output folder per analysis run — avoids collisions
                # between concurrent/successive analyses. Deleted in the
                # finally block below once results are safely cached, since
                # cached report regeneration only needs the in-memory data
                # (segments, sentiments, frame analyses), never the raw
                # video/audio/frame files on disk again.
                session_output_dir = f"videomind_output_{uuid.uuid4().hex[:8]}"

                # ── Prepare the actual video source for this run ──
                if uploaded_file is not None:
                    original_ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_video_path = tmp_file.name
                    actual_source = temp_video_path
                else:
                    actual_source = video_source

                show_step("⬇️ جاري تحميل الفيديو... | Downloading...", 10)
                input_result = process_input(actual_source, output_dir=session_output_dir)

                if input_result["status"] == "error":
                    progress.empty(); step_box.empty()
                    st.error(f"❌ {input_result['message']}")
                    st.stop()

                show_step(f"✅ تم تحميل الفيديو — {input_result['duration_seconds']/60:.1f} دقيقة | min", 20)

                # ── Auto-enable audio-only for long videos ──
                if input_result["duration_seconds"] > 3600 and not audio_only:
                    audio_only = True
                    cache_key = (source_identifier, audio_only)  # keep cache key in sync
                    st.warning(
                        "⏱️ الفيديو أطول من ساعة، سيتم تحليل الصوت فقط لتوفير الوقت "
                        "| Video is longer than 1 hour — analyzing audio only to save time"
                    )

                # ── Run audio + visual analysis in parallel (independent layers) ──
                if audio_only:
                    show_step("🎵 جاري تحويل الصوت إلى نص... | Transcribing...", 40)
                else:
                    show_step("🎵👁️ جاري تحويل الصوت وتحليل المشاهد بالتوازي... | Transcribing & analyzing frames in parallel...", 40)

                # ThreadPoolExecutor (not ProcessPoolExecutor) is the right tool here:
                # both process_audio and process_visual spend nearly all their time
                # waiting on network I/O (Whisper / Claude API responses), not doing
                # local CPU work. Threads are released during that wait (Python's GIL
                # isn't held on I/O), so two threads genuinely run concurrently for
                # this workload. A process pool would add real process-startup
                # overhead (spawning a new interpreter, re-importing modules) without
                # any concurrency benefit an I/O-bound task doesn't already get from
                # threads.
                with ThreadPoolExecutor(max_workers=2) as executor:
                    audio_future = executor.submit(process_audio, input_result["audio_chunks"])

                    visual_future = None
                    if not audio_only:
                        visual_future = executor.submit(
                            process_visual,
                            input_result["frames_manifest"],
                            input_result["duration_seconds"]
                        )

                    # ── Audio failure aborts the whole analysis ──
                    # Audio is the dominant signal (0.65 weight vs 0.35 for
                    # visual, see fusion_layer) — a broken audio layer means
                    # there's no reliable basis for a report at all, so we
                    # surface the error immediately rather than continuing
                    # with a meaningless visual-only result.
                    try:
                        audio_result = audio_future.result()
                    except Exception as e:
                        progress.empty(); step_box.empty()
                        st.error(
                            f"❌ فشلت طبقة الصوت بخطأ غير متوقع — تم إيقاف التحليل "
                            f"| Audio layer failed unexpectedly — analysis stopped: {str(e)}"
                        )
                        st.stop()

                    # ── Visual failure degrades gracefully to audio-only ──
                    # Unlike audio, a broken visual layer still leaves a
                    # usable analysis (audio-only) — so instead of aborting,
                    # fall back and note explicitly that visual analysis
                    # didn't complete, rather than silently pretending it
                    # succeeded with an empty result.
                    visual_incomplete_note = None
                    if visual_future is not None:
                        try:
                            visual_result = visual_future.result()
                        except Exception as e:
                            visual_result = {
                                "frame_analyses"                : [],
                                "overall_sentiment"              : "neutral",
                                "overall_sentiment_is_fallback"  : True,
                                "status"                         : "success",
                                "message"                        : "Visual layer crashed — audio-only fallback | تعطلت طبقة الصورة — تحليل صوتي فقط"
                            }
                            visual_incomplete_note = str(e)
                    else:
                        visual_result = {
                            "frame_analyses"                : [],
                            "overall_sentiment"              : "neutral",
                            "overall_sentiment_is_fallback"  : False,
                            "status"                         : "success",
                            "message"                        : "Audio only mode | وضع صوتي فقط"
                        }

                if visual_incomplete_note:
                    st.warning(
                        f"⚠️ فشل التحليل البصري بخطأ غير متوقع — تم إكمال التحليل بالصوت فقط "
                        f"| Visual analysis failed unexpectedly — continuing with audio only: "
                        f"{visual_incomplete_note}"
                    )

                if audio_result["status"] == "error":
                    progress.empty(); step_box.empty()
                    st.error(f"❌ {audio_result['message']}")
                    st.stop()

                if not audio_only:
                    if visual_result["status"] == "error":
                        progress.empty(); step_box.empty()
                        st.error(f"❌ {visual_result['message']}")
                        st.stop()

                    show_step(
                        f"✅ تم التحليل — {len(audio_result['segments'])} جملة | "
                        f"{len(visual_result['frame_analyses'])} مشهد",
                        75
                    )
                else:
                    show_step(f"✅ تم تحويل الصوت — {len(audio_result['segments'])} جملة | segments", 75)

                show_step("🔀 جاري دمج التحليل... | Fusing...", 85)
                fusion_result = process_fusion(audio_result, visual_result)

                if fusion_result["status"] == "error":
                    progress.empty(); step_box.empty()
                    st.error(f"❌ {fusion_result['message']}")
                    st.stop()

                # ── Cache these results for this exact video, so switching
                # report options later reuses them instead of recomputing. ──
                st.session_state["videomind_cache"] = {
                    "cache_key"     : cache_key,
                    "input_result"  : input_result,
                    "audio_result"  : audio_result,
                    "visual_result" : visual_result,
                    "fusion_result" : fusion_result,
                }

                show_step("📊 جاري إنشاء التقرير... | Generating report...", 92)

            report_mode   = "summary" if "ملخص" in report_type else "analysis"
            report_result = process_report(fusion_result, report_mode)

            if report_result["status"] == "error":
                progress.empty(); step_box.empty()
                st.error(f"❌ {report_result['message']}")
                st.stop()

            progress.progress(100, text="✅ اكتمل التحليل! | Analysis Complete!")
            step_box.empty()

            # Force any queued Langfuse spans out now rather than waiting on
            # the exporter's normal periodic flush — a no-op if tracing was
            # never enabled, never raises, never affects the result below.
            flush_observability()

            st.divider()
            st.markdown('<p class="vm-section">📊 نتائج التحليل | Analysis Results</p>', unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("المشاعر | Sentiment",  fusion_result["unified_sentiment"].capitalize())
            m2.metric("الثقة | Confidence",   fusion_result["confidence_label"].capitalize())
            m3.metric("الإشارة | Signal",     fusion_result["signal"].replace("_", " ").capitalize())
            m4.metric("اللغة | Language",     audio_result["language"].upper())

            reliability = fusion_result.get("analysis_reliability", {})
            reliability_score = reliability.get("score", 100.0)
            m5.metric("دقة التحليل | Reliability", f"{reliability_score:.0f}%")

            # ── Split audio/visual reliability breakdown ──
            # A combined 95% could hide a visual layer that fell back on
            # every single frame while audio was perfect (or the reverse)
            # — kept as an optional expander rather than 2 more top-level
            # metrics, so the default dashboard stays simple.
            audio_rel  = reliability.get("audio_reliability", {})
            visual_rel = reliability.get("visual_reliability", {})
            if audio_rel.get("total_items", 0) > 0 or visual_rel.get("total_items", 0) > 0:
                with st.expander("🔍 تفاصيل الموثوقية (صوت/صورة) | Reliability Breakdown"):
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.metric(
                            "موثوقية الصوت | Audio",
                            f"{audio_rel.get('score', 100.0):.0f}%",
                            help=f"{audio_rel.get('fallback_items', 0)}/{audio_rel.get('total_items', 0)} fallback items"
                        )
                    with rc2:
                        st.metric(
                            "موثوقية الصورة | Visual",
                            f"{visual_rel.get('score', 100.0):.0f}%",
                            help=f"{visual_rel.get('fallback_items', 0)}/{visual_rel.get('total_items', 0)} fallback items"
                        )

            # ── Low-reliability notice — surfaces WHY the score is low,
            # so the user knows this reflects fallback values, not a
            # judgment on the video's actual content. ──
            if reliability_score < 90 and reliability.get("total_items", 0) > 0:
                st.warning(
                    f"⚠️ {reliability['fallback_items']} من أصل {reliability['total_items']} "
                    f"عنصر تحليل اعتمد على قيم احتياطية بسبب خطأ تقني مؤقت (مو تحليل حقيقي) "
                    f"— النتيجة قد تكون أقل دقة بهذا الجزء "
                    f"| {reliability['fallback_items']} of {reliability['total_items']} analysis "
                    f"items used a fallback value due to a temporary technical issue"
                )

            # ── Validation warnings from the report layer ──
            if report_result.get("validation_warnings"):
                st.divider()
                st.markdown('<p class="vm-section">⚠️ تحذيرات جودة التقرير | Report Quality Warnings</p>', unsafe_allow_html=True)
                for warning_msg in report_result["validation_warnings"]:
                    st.warning(f"⚠️ {warning_msg}")

            # On-screen text conflicts are woven directly into the report's
            # Sentiment Analysis section by report_layer, rather than shown
            # as a separate box here — see
            # fusion_layer._extract_critical_text_moments for the
            # frequency-based filtering that keeps only genuinely notable
            # text (persistent overlays like scoreboards are excluded).

            # ── Sentiment-over-time chart ──
            # audio_layer's sentiment_timeline tracks how audio tone shifts
            # across the video's periods (see fusion_layer._build_summary,
            # which just forwards it through unchanged). Fallback periods
            # (a failed Claude call for that period) are excluded — they're
            # placeholders, not real readings. Needs at least 2 real periods
            # for a line to mean anything; a single point isn't "over time."
            timeline      = audio_result.get("sentiment_timeline", [])
            real_periods  = [p for p in timeline if not p.get("is_fallback")]
            if len(real_periods) >= 2:
                with st.expander("📈 تغيّر المشاعر عبر الوقت | Sentiment Over Time"):
                    chart_df = pd.DataFrame([
                        {
                            "minute"      : (p["start"] + p["end"]) / 2 / 60,
                            "sentiment"   : p["sentiment"],
                            "range_label" : f"{p['start']:.0f}s-{p['end']:.0f}s",
                        }
                        for p in real_periods
                    ])

                    # Vega-Lite/Altair renders the FIRST item of a nominal
                    # y-axis "sort" list at the TOP — confirmed against a
                    # live render, since this is easy to get backwards.
                    # Listed high→low so "up" on the y-axis reads as "more
                    # positive" — a plain alphabetical sort would scramble
                    # that intuition (mixed/negative/neutral/positive).
                    sentiment_order = ["positive", "neutral", "mixed", "negative"]

                    timeline_chart = alt.Chart(chart_df).mark_line(
                        point=alt.OverlayMarkDef(size=80, filled=True, color="#7c3aed"),
                        color="#7c3aed",
                        strokeWidth=2,
                    ).encode(
                        x=alt.X("minute:Q", title="الدقيقة | Minute"),
                        y=alt.Y("sentiment:N", sort=sentiment_order, title="المشاعر | Sentiment"),
                        tooltip=[
                            alt.Tooltip("range_label:N", title="الفترة | Period"),
                            alt.Tooltip("sentiment:N",   title="المشاعر | Sentiment"),
                        ],
                    ).properties(height=220)
                    # Single series, brand color reused from the app's own
                    # palette (#7c3aed) — no legend needed since the
                    # expander's own title already names what's plotted.

                    st.altair_chart(timeline_chart, width="stretch")

            st.divider()

            if "العربية" in report_language:
                st.markdown("### 📄 التقرير بالعربية")
                st.markdown(report_result["report_ar"])
            elif "English" in report_language:
                st.markdown("### 📄 English Report")
                st.markdown(report_result["report_en"])
            else:
                tab_ar, tab_en = st.tabs(["📄 التقرير بالعربية", "📄 English Report"])
                with tab_ar:
                    st.markdown(report_result["report_ar"])
                with tab_en:
                    st.markdown(report_result["report_en"])

            st.divider()

            full_report = f"## التقرير بالعربية\n\n{report_result['report_ar']}\n\n---\n\n## English Report\n\n{report_result['report_en']}"
            st.download_button(
                label     = "⬇️ تحميل التقرير | Download Report",
                data      = full_report,
                file_name = "videomind_report.md",
                mime      = "text/markdown",
                width     = "stretch"
            )

        finally:
            # Clean up the temporary uploaded video file regardless of
            # whether the analysis succeeded, failed, or was stopped early.
            # On a cache hit, temp_video_path stays None (no file was
            # created this run), so there's nothing to remove.
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            # Clean up this run's raw video/audio/frame files. Safe to do
            # unconditionally (success, failure, or early stop) because
            # cached report regeneration only ever needs the in-memory
            # results already stored in st.session_state — never these
            # files again. On a cache hit, session_output_dir stays None
            # (no new folder was created this run).
            if session_output_dir and os.path.exists(session_output_dir):
                shutil.rmtree(session_output_dir, ignore_errors=True)