# VideoMind — System Architecture

## 🏗️ High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                    │
│  ┌───────────┐ ┌────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │  Video    │ │  Analysis  │ │  Results    │ │  Session  │  │
│  │  Source   │ │  Options   │ │  Dashboard  │ │  Cache    │  │
│  └─────┬─────┘ └─────┬──────┘ └──────┬──────┘ └─────┬─────┘  │
└────────┼─────────────┼───────────────┼──────────────┼────────┘
         │              │               │              │
         ▼              ▼               ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                        Pipeline Layer                        │
│                                                                │
│  input_layer.py   audio_layer.py   visual_layer.py           │
│  fusion_layer.py  report_layer.py                             │
└─────────────────────────┬──────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┬──────────────┐
            ▼             ▼             ▼              ▼
    ┌─────────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────┐
    │   yt-dlp /   │ │  OpenAI   │ │ Anthropic │ │  Local File  │
    │   FFmpeg     │ │  Whisper  │ │  Claude   │ │    System    │
    └─────────────┘ └───────────┘ └───────────┘ └──────────────┘
```

**Scope note:** VideoMind analyzes social media video content (YouTube, TikTok, X, direct upload — Instagram is detected and rejected early, see `LIMITATIONS.md`) — including longer-form YouTube videos with an educational or podcast-style tone, since they go through the same pipeline. There is no `content_type` branching anywhere in this architecture; the report layer applies one instruction set (see section 6) regardless of subject matter. See `LIMITATIONS.md` for the reasoning behind this scope.

---

## 📦 Layer Breakdown

### 1. UI Layer — `app.py`
Streamlit-based single-page application orchestrating the full pipeline:

```
app.py
├── Video Source Selector          → URL input or file upload
├── Analysis Options               → Report depth, language, audio-only toggle
├── Analyze Button                  → Triggers pipeline execution
├── Session Cache                   → st.session_state keyed by (source, audio_only)
├── Results Dashboard               → 5 metrics + expandable audio/visual reliability breakdown
├── Report Display                  → Bilingual tabs / single-language view
└── Download Button                 → Exports full bilingual report as Markdown
```

**Temp file handling:** Uploaded files are written to `tempfile.NamedTemporaryFile` (not the project directory) and deleted in a `finally` block regardless of success, failure, or early exit.

**Upload identity (partial content hash):** the cache key for an uploaded file is built from a SHA-256 hash of its first 64KB + last 64KB + total byte size — not filename + filesize. This avoids collisions between two different videos that happen to share a name and an identical size; hashing only the head/tail (not the whole file) keeps this cheap even for large uploads.

**Parallel-execution error handling:** audio and visual analysis run concurrently via `ThreadPoolExecutor`. If the audio layer raises an unexpected exception, the analysis stops immediately with a clear error (audio is the dominant fusion signal — there's no meaningful report without it). If the visual layer raises an unexpected exception, the pipeline degrades gracefully to an audio-only result instead of crashing, with an explicit UI warning that visual analysis didn't complete.

### 1a. Observability Layer — `observability.py`
```
observability.py
├── LANGFUSE_ENABLED     → True only if both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
└── init_observability() → Instruments the Anthropic SDK via OpenTelemetry, once, at app startup
```

Opt-in tracing (latency, cost, prompt/response pairs) for every Claude call across all layers, via [Langfuse](https://langfuse.com). Entirely optional and fails silently at every stage:
- No credentials set → `init_observability()` is a no-op, nothing else changes
- Credentials set but `langfuse`/`opentelemetry-instrumentation-anthropic` not installed → one warning printed, app continues normally
- Both present → Claude calls auto-traced via OpenTelemetry instrumentation (audio/visual/report layers need zero code changes); Whisper calls are traced separately via a drop-in SDK replacement import in `audio_layer.py` (`from langfuse.openai import openai` instead of `import openai`, only when enabled)

`app.py` calls `init_observability()` once, at the very top, before any layer is imported.

### 2. Input Layer — `input_layer.py`
```
input_layer.py
├── process_input()                     → Main entry point
├── _check_required_api_keys()          → Fail-fast validation before any work begins
├── _clean_url()                         → urllib.parse-based cleaning; preserves v= on YouTube watch links
├── _detect_source_type()                → URL vs local file
├── _download_video()                    → yt-dlp, lowest quality (audio/frames only, no need for HD)
├── _get_video_duration()                → ffprobe
├── _extract_audio()                     → FFmpeg, 16kHz mono WAV (Whisper-optimal)
├── _chunk_audio()                        → Splits audio > 10 min into chunks, logs any chunk failure
├── _get_frame_interval_for_duration()   → 3s (≤10 min videos) / 5s (longer) extraction interval
└── _extract_frames()                     → FFmpeg fps filter; returns (frames_dir, frames_manifest)
```

**Structured frame timestamps:** `_extract_frames` returns a `frames_manifest` — a list of `{"path": str, "timestamp": float}` entries built directly from the known extraction interval, not parsed back out of the saved filename later. The filename still encodes the timestamp for human readability when browsing the output folder, but nothing downstream depends on that encoding surviving unchanged.

**Instagram early-exit check:** `process_input` checks for `instagram.com` in the source URL before attempting any download. `yt-dlp` requires a logged-in session (cookies) Instagram doesn't have here, and the failure is consistent across posts, not post-specific — so rather than burn time on a download attempt known to fail and surface `yt-dlp`'s generic technical error, this returns a clear bilingual message immediately, pointing to the working alternative (direct file upload). The UI's "URL" option label reflects the platforms that actually work: "YouTube / TikTok / X".

### 3. Audio Layer — `audio_layer.py`
```
audio_layer.py
├── process_audio()                        → Main entry point
├── _retry_call()                          → Generic retry wrapper (2 attempts, 1s delay)
├── _get_audio_duration()                  → ffprobe (not Whisper's last-segment timestamp)
├── _transcribe_chunk()                    → Whisper API, retried on transient failure — now covered by mocked tests
├── _get_period_count_for_duration()       → 4/5/6 periods based on transcript span
├── _split_segments_into_periods()          → Time-bucket segments for period analysis
├── _analyze_period_sentiment()             → One Claude call per period
├── _majority_sentiment()                   → Vote across periods → overall sentiment
├── _analyze_sentiment_timeline()           → Orchestrates the above
├── _analyze_segment_sentiments_batch()     → Splits segments into size-bounded batches
└── _analyze_one_sentiment_batch()          → One Claude call per batch, JSON array response
```

### 4. Visual Layer — `visual_layer.py`
```
visual_layer.py
├── process_visual()                    → Main entry point — takes frames_manifest, not a dir path
├── _retry_call()                       → Same retry policy as audio_layer
├── _get_max_frames_for_duration()      → 90/180/270 frame cap based on video length
├── _sample_frames()                    → Evenly-distributed sampling over the manifest, up to the cap
├── _analyze_frame()                    → Claude Vision call per frame — now covered by mocked tests
├── _parse_frame_response()             → Structured text → dict
├── _apply_temporal_smoothing()         → corrects isolated single-frame sentiment outliers
└── _calculate_overall_sentiment()      → Majority vote across (smoothed) frames
```

**Fallback detection covers both failure modes:** `_analyze_frame`'s fallback check verifies both that the parsed sentiment value is one of the 4 allowed options, AND that a literal `SENTIMENT:` line is present in the raw response. The second check matters because `_parse_frame_response`'s own default for a missing field is "neutral" — a technically valid value — so checking only against the valid-values list would let a genuinely empty or off-format response pass silently as a confident "neutral" reading, never flagged as a fallback.

**Brand logo detection:** each frame's prompt includes a `BRAND_LOGOS_VISIBLE` field — the model identifies any recognizable brand logos, packaging, or trademarks visible in the frame (based on the visual mark itself, separate from `TEXT_ON_SCREEN`'s literal text transcription). This exists specifically to catch a brand shown visually (a logo on a cup, a bag, a t-shirt) that's never named out loud in the audio — a case a transcript-only pipeline would miss entirely. Each frame's result now includes a `brand_logos` field alongside `text_on_screen`.

### 5. Fusion Layer — `fusion_layer.py`
```
fusion_layer.py
├── process_fusion()                        → Main entry point
├── _normalize_sentiment()                  → Guards against invalid sentiment values
├── _fuse_sentiments()                      → Weighted formula (0.65/0.35) with pre-rounding
├── _confidence_label()                     → Numeric confidence → high/medium/low
├── _align_segments_with_frames()           → Maps each transcript segment to its nearest frame
├── _find_closest_frame()                   → Timestamp matching, capped at 15s gap
├── _extract_critical_text_moments()        → Frequency-filtered on-screen text conflicts, now fuzzy-grouped
├── _group_similar_texts()                  → difflib-based clustering of near-duplicate OCR text
├── _calculate_analysis_reliability()       → combined score + separate audio/visual breakdown
├── _reliability_breakdown()                → per-modality {score, fallback_items, total_items}
├── _extract_silent_brand_mentions()        → brands shown visually but never named in the transcript
└── _build_summary()                        → Structured summary consumed by report_layer
```

**Silent brand mentions:** cross-references every distinct `brand_logos` value seen across frames against the full transcript text. A brand that appears visually but whose name never occurs anywhere in the spoken transcript is flagged as a "silent" brand mention — the exact scenario of an indirect/implicit product placement (someone describing an experience without ever naming the brand shown on screen). **Known limitation:** brand names from Vision's logo recognition may come back in a different script/language than the transcript (e.g. "Al Baik" vs "البيك") — the substring match can't catch a same-brand mention across scripts, so this can produce false positives (flagging a brand as "silent" when it was actually named, just in another language/transliteration). Treat flagged entries as a signal worth checking, not a certainty.

### 6. Report Layer — `report_layer.py`
```
report_layer.py
├── process_report()                      → Main entry point (no content_type param — see scope note above)
├── _retry_call()                         → Same retry policy as audio/visual layers
├── ReportSchema                          → Pydantic v2 schema for the report's structured output
├── REPORT_TOOL                           → Tool-use definition — forces valid structured output at the API level
├── _build_prompt()                       → Assembles summary + segments + critical moments + guidelines
├── _get_instructions()                   → ONE instruction set (analysis/summary) — no content-type matrix
├── _get_period_count_for_report()        → decides periods-based summarization threshold (>60 segments)
├── _split_segments_into_period_texts()   → chronological period splitting for long videos
├── _summarize_periods()                  → one Haiku call condensing each period
├── _build_periods_summary()              → orchestrates the two above, with graceful fallback
├── _parse_report_response()              → reads the tool_use block (replaces text-JSON parsing)
├── _fallback_from_text()                 → rare defense-in-depth path if tool_use is missing
├── _best_effort_language_split()         → Legacy recovery, now reached only via the fallback path
├── _run_grounding_check()                → deterministic contradiction check (sentiment/confidence/quotes),
│                                            language-aware (only compares quotes against the report field
│                                            matching the transcript's own language) and length-capped
│                                            (ignores implausibly long "quotes" — a sign of a mismatched
│                                            quote character, not a real quoted passage)
├── _get_min_report_length()              → Threshold scaled by report_mode × total_segments
└── _validate_report()                    → Length, emptiness, and Arabic-character validation
```

**Frame descriptions reach the prompt:** `_build_prompt`'s `segments_text` includes each aligned segment's `frame_description` (and, when present, `brand_logos`) alongside its emotion label. This gives the report-writing model actual visual grounding data for any claim about who's present in a scene, rather than requiring it to infer from transcript structure alone.

**Explicit formatting guidelines:** the prompt now includes a dedicated "Formatting" section — clean Markdown headings/bullets only, no manual ASCII dividers (`───`, `═══`), no backtick/code-block styling used for emphasis. Added after real reports showed inconsistent structure (manual dividers one run, clean headings the next) and, separately, numbers occasionally wrapped in backticks (rendering as broken monospace text in the UI). These are prompt-level instructions, not hard guarantees — the model doesn't follow them with 100% consistency every single run.

**User-facing grounding check messages:** the three warning messages (sentiment contradiction, confidence contradiction, untraceable quote) were rewritten from developer-facing technical phrasing ("Grounding check: report claims...") to plain-language explanations aimed at someone reading the report, not debugging the code.

**Periods-based summarization:** period counts scale with transcript length (61–120 segments → 8 periods, 121–250 → 12, >250 → 15), and the per-period summary length is flexible — 2–3 sentences by default, up to 5–6 for periods covering multiple distinct topics. This flexibility matters for videos with many unrelated subjects discussed back-to-back (e.g. a multi-topic news roundup): a rigid, low per-period sentence cap can compress several distinct topics into one vague sentence that effectively drops most of them.

---

## 🔄 Data Flow

### Full Analysis Flow (Cache Miss)
```
User clicks "Analyze"
     │
     ▼
Cache key built: (source_identifier, audio_only)
     │  source_identifier for uploads = partial content hash (head+tail+size),
     │  NOT filename+filesize
     ▼
Cache lookup → MISS
     │
     ▼
process_input() → download, extract audio + frames_manifest
     │
     ├── duration > 3600s? → force audio_only = True (explicit UI notice)
     │
     ▼
ThreadPoolExecutor:
     ├── process_audio()  (always runs)
     └── process_visual(frames_manifest, ...) (skipped if audio_only)
     │         (both run in parallel — independent layers)
     ▼
     ├── audio_future.result() raises  → analysis stops, error shown
     └── visual_future.result() raises → degrade to audio-only, warning shown
     ▼
process_fusion() → weighted sentiment, split audio/visual reliability,
                    fuzzy-grouped critical text moments
     │
     ▼
Results cached in st.session_state (input/audio/visual/fusion)
     │
     ▼
process_report() → periods-based summarization if >60 segments,
                    tool-use structured generation, grounding check
     │
     ▼
Results Dashboard (+ reliability breakdown expander) + Report displayed
```

### Report Regeneration Flow (Cache Hit)
```
User changes report_mode / language
     │
     ▼
User clicks "Analyze" again (same source_identifier, same audio_only)
     │
     ▼
Cache lookup → HIT
     │
     ▼
Reuse cached input/audio/visual/fusion results
     │
     ▼
process_report() only — full pipeline skipped
     │
     ▼
Results Dashboard + new report displayed (near-instant)
```

### Report Generation Flow (inside process_report)
```
fusion_result
     │
     ▼
total_segments > 60? ─── NO ──→ flat full_text_preview (existing behavior)
     │
     YES
     ▼
Split into 8-15 chronological periods (8 for 61-120 segments,
12 for 121-250, 15 beyond that)
     │
     ▼
One Haiku call: condense each period → 2-3 sentence summary
(up to 5-6 for periods covering multiple distinct topics)
     │
     ├── success → periods_summary_text used in prompt (covers full video)
     └── failure → falls back to flat full_text_preview (never blocks the report)
     │
     ▼
_build_prompt() → Claude Sonnet call with tools=[REPORT_TOOL], tool_choice forced
     │
     ▼
tool_use block present? ─── YES ──→ ReportSchema.model_validate(tool_use.input)
     │                                     │
     NO (rare/unexpected)                  ├── valid   → report_ar, report_en
     ▼                                     └── invalid → _fallback_from_text()
_fallback_from_text() ──────────────────────────┘
     │
     ▼
_run_grounding_check() → compares report text against summary's
                          unified_sentiment / confidence_label / source text
     │
     ▼
validation_warnings (parse + length/emptiness + grounding) returned together
```

---

## 🗄️ Session State Schema

```python
st.session_state["videomind_cache"] = {
    "cache_key"     : (source_identifier: str, audio_only: bool),
    "input_result"  : dict,   # from process_input(), includes frames_manifest
    "audio_result"  : dict,   # from process_audio()
    "visual_result" : dict,   # from process_visual()
    "fusion_result" : dict,   # from process_fusion()
}
```

`source_identifier` is `f"url:{video_source}"` for URL input, or `f"upload:{partial_hash}"` for uploaded files, where `partial_hash` is a SHA-256 digest of the file's first 64KB + last 64KB + total size (see `app.py`'s `_partial_file_hash`). A change in either the source or `audio_only` invalidates the cache automatically.

---

## 💰 API Cost Optimization

| Action | Model | Calls | Notes |
|---|---|---|---|
| Transcription | whisper-1 | 1 per audio chunk (≤10 min each) | — |
| Audio period sentiment | claude-haiku-4-5-20251001 | 4–6 (scaled by duration) | — |
| Audio segment sentiment | claude-haiku-4-5-20251001 | 1 per 200 segments | Batched, not per-sentence |
| Visual frame analysis | claude-haiku-4-5-20251001 | 90–270 (scaled by duration) | One call per sampled frame |
| Periods summarization | claude-haiku-4-5-20251001 | 0 or 1 | Only for videos with >60 segments |
| Report generation | claude-sonnet-4-6 | 1 | Tool-use structured call |

**Key optimizations:**
- Report regeneration on the same video costs **zero** additional Whisper/Vision/sentiment calls — only the report call re-runs
- Audio-only mode (manual or auto-triggered for videos > 1 hour) eliminates all visual-analysis calls entirely
- Every retry is capped at one extra attempt — failures don't silently multiply cost
- The grounding check adds **zero** extra API calls — it's pure string comparison

---

## 🔐 Security Considerations

- API keys stored via `.env` locally, or Hugging Face Space repository secrets in the deployed version (never committed to version control)
- Uploaded video files are written to OS-managed temporary storage (`tempfile`), never the project directory, and deleted after processing regardless of outcome
- Each analysis run uses a unique output folder (`videomind_output_<uuid>`) to avoid collisions between concurrent or successive runs
- Upload cache identity uses a partial content hash rather than filename/size, closing a (low-probability but real) collision path that could have served a cached result for the wrong video
- No authentication layer (single-user local/Space app)
- No rate limiting on outbound API calls yet — see `LIMITATIONS.md`

---

## 🚀 Deployment

VideoMind deploys to Hugging Face Spaces via the Docker SDK:
- `Dockerfile` — `python:3.11-slim` base, installs `ffmpeg` via apt, copies the `layers/` package + `app.py`, runs Streamlit on port 7860 (the port Hugging Face Spaces expects)
- `README.md` carries the required Hugging Face front-matter block (`sdk: docker`, `app_port: 7860`) at its top, above the normal project documentation
- Secrets (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are set via the Space's repository secrets, never committed