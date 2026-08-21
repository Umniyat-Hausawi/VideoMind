---
title: VideoMind
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# VideoMind — Social Media Video Intelligence System

## The Problem

The volume of social media video today is far beyond what any person — or team — can realistically watch and keep up with. This is a real operational bottleneck for companies with customer success or brand monitoring teams: they need to know when their brand shows up in social media videos and what's being said about it, but having someone manually watch and log every relevant video doesn't scale and eats significant time and effort.

## What VideoMind Does

VideoMind automates that monitoring work. Given a video, it detects whether a company is actually being discussed — including cases where a brand only appears visually (a logo on screen) without ever being mentioned by name in speech, which a transcript-only review would miss entirely. It then produces a bilingual (Arabic/English) written report covering what was actually said, the overall sentiment toward the company — positive or negative — and the reasoning behind that assessment, along with a reliability score for how confident that analysis actually is.

Under the hood, it treats audio and visual as two independent signals rather than one blended guess. It explicitly tracks where what's said and what's shown agree or contradict each other — a speaker can sound enthusiastic while nothing on screen backs that up, and that mismatch itself is part of what makes a video's message credible or not.

## Who It's For

Primarily built for brands and companies that want to monitor what's being said about them across social media video — customer success and brand monitoring teams who currently rely on manually watching videos to catch mentions and gauge sentiment. It's equally useful for content creators, analysts, and researchers who need to understand not just what was said in a video, but how consistent that message was with what was shown on screen.

> **Scope note:** VideoMind is focused on social media video analysis. Long-form educational or podcast-style YouTube videos work fine too; they go through the same pipeline, since the underlying audio/visual fusion signal doesn't depend on subject matter.

---

## 1. System Architecture

```text
Video URL / Upload
        ↓
Input Layer
(yt-dlp + FFmpeg — download, audio extraction, frame extraction
 → structured frames_manifest, not filename-encoded timestamps)
        ↓
   ┌────┴────┐
   ↓         ↓
Audio Layer   Visual Layer
(Whisper +    (Claude Vision —
 Claude —     per-frame sentiment,
 transcript,  emotion, on-screen
 sentiment    text, temporal
 timeline)    smoothing)
   └────┬────┘
        ↓  (parallel execution, ThreadPoolExecutor — audio failure aborts,
        ↓   visual failure degrades gracefully to audio-only)
Fusion Layer
(weighted sentiment fusion + split audio/visual reliability scoring
 + fuzzy-matched on-screen text conflict detection)
        ↓
Report Layer
(Claude Sonnet — tool-use structured output, periods-based summarization
 for long videos, deterministic grounding check — bilingual report)
        ↓
Streamlit UI
(results dashboard + reliability breakdown + session-cached regeneration)
```

For a detailed breakdown of every layer, function, and data flow, see [Architecture.md](Architecture.md).

---

## 2. Screenshots

### Home Interface
Video source selection (URL or direct upload), report depth, and language options — the starting point for any analysis.
![Home Interface](screenshots/home-interface.png)

---

### Results Dashboard
The five core metrics (sentiment, confidence, signal, language, reliability), with an expandable audio/visual reliability breakdown showing each modality's fallback rate separately, plus the Sentiment Over Time timeline chart tracking sentiment across the video's duration rather than a single overall score.
![Results Dashboard](screenshots/results-dashboard.png)

---

### Arabic Report — Summary Mode
A concise bilingual summary report — Main Topic, Key Points, Notable Quotes, and Conclusion — for when the full analysis depth isn't needed.
![Arabic Report](screenshots/report-arabic.png)

---

### English Report — Analysis Mode
The full-depth report, including the Sentiment Analysis section that explicitly grounds tone in both audio and visual signals — the feature that sets VideoMind apart from transcript-only summarizers.
![English Report](screenshots/report-english.png)

---

## 3. Key Features

### Input Handling
- Accepts YouTube, TikTok, and X URLs, or direct file upload (MP4/MOV/AVI/MKV) — Instagram links are detected and rejected early with a clear message, since Instagram requires a logged-in session this app doesn't have (see [LIMITATIONS.md](LIMITATIONS.md))
- URL cleaning that preserves the video-identifying parameter (`v=`) while stripping tracking parameters
- Videos longer than 2 hours are rejected outright; videos longer than 1 hour automatically switch to audio-only mode (with an explicit UI notice), to control cost and processing time
- Frame timestamps are structured data (`{"path", "timestamp"}`) built at extraction time, not re-parsed from filenames downstream

### Audio Analysis
- Full transcription via Whisper API (99 languages)
- 100% sentence-level sentiment coverage via batched Claude calls (capped per batch to guarantee response size stays within model token limits, regardless of transcript length)
- Sentiment analyzed across 4–6 time periods (scaled by video length) rather than a single value — captures tone shifts across a video's runtime

### Visual Analysis
- Frame sampling that scales with video duration (90 / 180 / 270 frames)
- Per-frame sentiment, dominant emotion, scene description, and verbatim on-screen text (in its original language)
- **Temporal smoothing:** an isolated single-frame sentiment outlier surrounded by agreeing neighbors is corrected, while genuine multi-frame mood transitions are left untouched — each corrected frame is flagged (`is_smoothed`) for transparency

### Fusion & Reliability
- Weighted sentiment fusion (audio 0.65 / visual 0.35) when the two signals disagree, with a fallback-proof rounding step to avoid floating-point misclassification
- An `analysis_reliability` score, now broken down separately for audio and visual (a combined 95% could otherwise hide a visual layer that fell back on every frame while audio was perfect)
- Frequency-filtered detection of on-screen text that conflicts with a decisive audio sentiment — persistent overlays (scoreboards, watermarks) are excluded via **fuzzy similarity matching**, so minor OCR phrasing drift doesn't let a repetitive element slip through

### Report Generation
- Bilingual (Arabic/English) report, at two depths (summary/analysis)
- **Structured tool-use output** — Claude is forced to call a defined `submit_report` tool rather than hand-writing JSON in text, guaranteeing valid structure at the API level
- **Periods-based summarization for long videos** (>60 transcript segments) — the transcript is condensed period-by-period (one extra, cheap API call) instead of relying on a flat, beginning-biased preview
- **Deterministic grounding check** — compares the generated report against the structured data it was built from, flagging sentiment/confidence contradictions and untraceable quoted text, at zero extra API cost
- Explicit anti-hallucination prompt guidance, with structural quality checks (length, emptiness, language validity) run on every generated report

### Reliability Engineering
- Automatic retry (one extra attempt) on every external API call — Whisper transcription, Claude sentiment analysis, Claude Vision, and Claude report generation
- Audio layer failure aborts the analysis with a clear error (audio is the dominant fusion signal); visual layer failure degrades gracefully to an audio-only result instead of crashing
- Session-scoped result caching: switching report depth or language for the same video regenerates only the report, not the full pipeline
- Upload cache identity uses a **partial content hash** (first + last 64KB + total size), not filename/size, avoiding cache collisions between different files that happen to share a name and size

---

## 4. Core Design Philosophy

```text
FFmpeg extracts.
Whisper transcribes.
Claude analyzes (per segment, per frame).
Python fuses, smooths, and scores reliability.
Claude Sonnet writes, via a forced structured tool call.
Python grounds the result against the data that produced it.
```

Every numeric judgment (fusion score, reliability percentage, confidence level, grounding check) is computed in Python — deterministic, testable, reproducible. Claude's role is bounded to what LLMs are actually good at: transcription-adjacent understanding, per-item classification, and narrative synthesis grounded in structured data it's explicitly given, not raw unfiltered video.

---

## 5. Technical Decisions

**Why Whisper API instead of local Whisper?**
Local Whisper on CPU takes 15+ minutes for a 5-minute video. Whisper API processes the same audio in under 30 seconds with no quality loss.

**Why Claude Vision instead of a local vision model (e.g. LLaVA)?**
A local vision model requires GPU resources and added infrastructure. Claude Vision is accessible via the existing Anthropic API key and has stronger context and on-screen-text understanding out of the box.

**Why FFmpeg instead of OpenCV for frame extraction?**
OpenCV decoded and read every single frame just to discard most of them — a 1-hour video at 25fps meant reading ~90,000 frames to keep the ~720 actually needed. FFmpeg's `fps` filter extracts only the needed frames at the decoder level, with identical output at a fraction of the CPU time on long videos.

**Why does the frame analysis cap scale with video duration?**
A fixed, low cap would mean sampling once every few minutes on a long video — too sparse for meaningful visual context. The cap scales instead (90 / 180 / 270 frames for short / medium / long videos), with a hard ceiling matching the 2-hour input limit. A denser jump (150/300/600) was considered and rejected as a 2.5–3.3x multiplier on Vision API calls for marginal coverage gain.

**Why is the frame extraction interval 3s for short videos but 5s for longer ones?**
The interval only needs to be dense enough to supply more raw frames than the analysis cap uses — anything denser than that is wasted FFmpeg work, since downstream sampling discards the excess anyway. A 10-minute video at the default 5s interval yields only 120 raw frames against a 90-frame cap — barely enough headroom — so short videos use a 3s interval instead. Medium/long videos already extract far more raw frames than their cap needs at 5s, so there's no reason to extract more densely there.

**Why structured frame timestamps instead of encoding them in the filename?**
The original design encoded each frame's timestamp in its filename (`frame_00045.00s.jpg`) and re-parsed it back out downstream — a hidden coupling where a filename format change anywhere would silently break timestamp accuracy elsewhere. Timestamps are now computed once, at extraction time, into a structured `frames_manifest` list that's passed directly to the visual layer. The filename still encodes the timestamp for human readability when browsing output folders, but nothing in the pipeline depends on that anymore.

**Why batch sentiment analysis instead of sampling every 10th segment?**
The original design analyzed roughly 10% of segments across multiple API calls. All segments are now sent in size-bounded batches (capped per call to guarantee the response fits comfortably within token limits) — 100% coverage, with the batch size chosen specifically to prevent a long transcript's JSON response from being cut off mid-write.

**Why analyze sentiment across multiple time periods instead of one value per video?**
A single sentiment value computed from an early transcript excerpt misses real tone shifts over a video's length. The transcript is split into 4–6 time periods (scaled by span), each analyzed independently, producing a `sentiment_timeline` alongside one overall majority value.

**Why periods-based summarization for the report layer on long videos?**
The same problem that motivated audio's period-based sentiment analysis applies to report generation itself: a flat, capped transcript preview biases long-video reports toward the beginning. For videos with more than 60 segments, the transcript is now split into 8–15 chronological periods (scaled by length: 8 for 61–120 segments, 12 for 121–250, 15 beyond that), each condensed by one extra (cheap, Haiku) API call, before the final report prompt is built — spanning the full video instead of just its opening portion. If that extra call fails, the system falls back to the old flat-preview behavior rather than blocking the report.

**Why tool-use (structured output) instead of asking Claude to write JSON in text?**
Asking a model to hand-write valid, correctly-escaped JSON inside a text response is inherently fragile — a single unescaped quote in the report text could break parsing. Forcing a `submit_report` tool call with a defined input schema guarantees valid, correctly-typed structure at the API level itself, not just "the model was told to be careful." The old text-based recovery path (`_best_effort_language_split`) remains as a rare defense-in-depth fallback.

**Why a deterministic grounding check on the report, separate from the LLM's own anti-hallucination instructions?**
Prompt instructions ("don't invent facts") reduce hallucination but don't guarantee it never happens. A zero-cost, pure-Python check comparing the report text against the structured data it was built from (unified sentiment, confidence label, source transcript/on-screen text) catches obvious, explicit contradictions the model might still produce — without needing another API call to verify the first one.

**Why is fused sentiment computed with a weighted formula rather than fixed rules?**
When audio and visual sentiment disagree, each is converted to a number (positive = +1, neutral/mixed = 0, negative = −1) and combined as `audio × 0.65 + visual × 0.35`. The result is rounded *before* threshold comparison, specifically to prevent floating-point artifacts (e.g. `0.30000000000000004` instead of `0.3`) from crossing a classification boundary incorrectly.

**Why is audio weighted higher than visual (0.65 vs 0.35)?**
Words carry explicit meaning. Facial expression can mislead — someone can smile while delivering bad news. Audio is treated as the more reliable signal.

**Why track `is_fallback` instead of just defaulting silently to "neutral"?**
A substituted neutral (because a model call failed) and a genuinely neutral result are otherwise indistinguishable. Every per-segment and per-frame sentiment carries an explicit fallback flag — including the case where a Vision response is empty or off-format, covered by mocked tests (see [`LIMITATIONS.md`](LIMITATIONS.md)).

**Why split reliability into separate audio/visual scores instead of one combined number?**
A combined 95% reliability score could hide a visual layer that fell back on every single frame while audio was perfect (or vice versa) — averaging conceals exactly the kind of imbalance the reliability metric exists to surface. The combined score is still shown as the primary dashboard metric (for simplicity); the per-modality breakdown is available in an expandable section.

**Why temporal smoothing for visual frames, and why require a full two-sided window?**
A single frame misread as "negative" between two "positive" neighbors would otherwise go uncorrected. Smoothing corrects a frame only when it disagrees with *every* neighbor in its window *and* at least 2 neighbors agree with each other on a different value — a genuine transition spanning 2+ consecutive frames is left alone. Frames at the very start/end of the sampled sequence (no full window available) are deliberately never smoothed, since a one-sided window can't distinguish noise from the start of a real, unconfirmed transition.

**Why fuzzy matching instead of exact-string matching for repeated on-screen text?**
On-screen text is often a persistent overlay (a scoreboard, a watermark) visible in nearly every frame. Exact-string frequency counting missed near-duplicate OCR readings of the same overlay (minor phrasing drift the model introduces between frames), letting a genuinely repetitive element slip through the 40%-repetition filter. Grouping via `difflib.SequenceMatcher` similarity (0.8 threshold) before the frequency check fixes this without needing exact text matches.

**Why a partial content hash instead of filename+filesize for the upload cache key?**
Two different uploaded files can share both a name and an exact filesize (e.g. two clips independently trimmed to the same duration) — a filename+filesize identifier would collide and serve cached results from the wrong video. Hashing the entire file avoids that but costs real time on large uploads just to build a cache key; hashing the first and last 64KB plus total size is a middle ground that's extremely unlikely to collide between genuinely different videos, at a fraction of full-file-hash cost.

**Why does an audio layer failure abort the analysis, while a visual layer failure just degrades?**
Audio carries 0.65 of the fusion weight and is the primary signal for report generation — there's no meaningful report without it. Visual analysis, while valuable, is optional by design already (audio-only mode exists) — so a visual-layer crash is treated the same way as a user choosing audio-only mode, with an explicit warning rather than a silent gap.

**Why retry failed API calls once before falling back?**
Many API failures are transient. A single extra attempt (with a short delay) recovers most of these without masking a genuinely persistent failure.

**Why cache pipeline results across report regenerations?**
Streamlit reruns the entire script on every UI interaction. Switching report depth or language for the *same* video only requires re-running the report layer — the input/audio/visual/fusion results are cached in `st.session_state`, keyed by video source, so regenerating a report is near-instant instead of repeating the full pipeline.

---

## 6. Hallucination Mitigation

VideoMind reduces hallucination risk through:

- Explicit prompt instructions to ground every report claim in the provided transcript, aligned segments, and sentiment data — and to say "not clear from the available content" rather than invent detail
- Structured tool-use output (not free-form prose asked to contain JSON), reducing formatting drift
- A deterministic grounding check comparing the generated report against structured summary data (sentiment, confidence, source text), at zero extra API cost
- Post-generation structural validation (length thresholds scaled to actual content volume, emptiness checks, language-validity checks)
- On-screen text claims surfaced to the report are pre-filtered (with fuzzy matching) for genuine novelty before ever reaching the prompt

Residual hallucination risk remains, particularly in interpretive report sections (creator intent, target audience, recommendations) that have no checkable ground truth — see [LIMITATIONS.md](LIMITATIONS.md) for exactly what is and isn't covered.

---

## 7. Reliability Engineering & Testing

- Automated unit tests (`pytest`) covering pure functions across all layers — duration-tiering logic, sentiment fusion math, segment/period splitting, reliability scoring, report-length thresholds, temporal smoothing, fuzzy text grouping, and the deterministic grounding check
- A mocked-test suite (`tests/test_mocked_api_calls.py` and `tests/test_report_layer.py`) covering the API-calling functions — `_transcribe_chunk`, `_analyze_frame`, `process_report` (success, fallback-recovery, failure, and upstream-failure short-circuit paths), and `_summarize_periods`/`_build_periods_summary` (periods-based summarization) — success, empty/malformed response, full failure, and retry-then-succeed, using mocked SDK responses (no real API calls, no cost). The audio period/batch sentiment calls (`_analyze_period_sentiment`, `_analyze_one_sentiment_batch`) remain the one API-calling path not yet covered by mocks — see [LIMITATIONS.md](LIMITATIONS.md).
- An end-to-end integration test (`tests/test_integration.py`) running the real audio→visual→fusion→report chain together, mocking only the external API clients — catches the class of bug that per-layer tests can't (one layer computing something correctly that the next layer never actually reads)
- Manual verification across multiple real videos and platforms (YouTube, TikTok, X, direct upload)
- Every fallback path is intentional and logged — no silent failure is treated as a successful result

---

## 8. Limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for a full, honest discussion of current constraints, what's been mitigated, and planned improvements.

---

## 9. Future Roadmap

- Authentication, per-user session isolation, and rate limiting for multi-user deployment
- Adaptive frame sampling around detected scene changes, rather than uniform time-based sampling
- Extend mocked test coverage to the audio period/batch sentiment API calls (`_analyze_period_sentiment`, `_analyze_one_sentiment_batch`) — the one remaining untested API-calling path
- Export to PDF and DOCX

---

## 10. Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Input | yt-dlp + FFmpeg | Video download, audio extraction, frame extraction |
| Audio | OpenAI Whisper API | Speech-to-text (99 languages) |
| Audio | Claude Haiku | Sentence-level and timeline sentiment analysis |
| Visual | Claude Vision (Haiku) | Frame description, sentiment, on-screen text |
| Fusion | Python | Weighted sentiment fusion, reliability scoring, fuzzy text grouping |
| Report | Claude Sonnet | Bilingual structured report generation (tool-use) |
| Interface | Streamlit | Web UI with session-based result caching |
| Deployment | Docker + Hugging Face Spaces | Containerized hosting |
| Validation | Pydantic v2 | Structured report schema validation |
| Testing | pytest + unittest.mock | Pure-function unit tests + mocked API-call tests |

---

## 11. AI Stack

| Component | Model | Purpose |
|---|---|---|
| Transcription | whisper-1 | Speech-to-text |
| Audio sentiment (per-segment + timeline) | claude-haiku-4-5-20251001 | Sentence and period-level sentiment |
| Visual analysis | claude-haiku-4-5-20251001 | Frame description, emotion, sentiment, on-screen text |
| Periods summarization (long videos) | claude-haiku-4-5-20251001 | Condensing transcript periods for full-video report coverage |
| Report generation | claude-sonnet-4-6 | Bilingual narrative report synthesis (tool-use) |

---

## 12. Cost Optimization

- Frame and batch caps scale with video length instead of applying one worst-case setting to every video
- Frame extraction interval only goes denser than default where actually needed (short videos) — no wasted extraction work on longer videos
- Session-cached pipeline results mean only the report layer re-runs when a user changes report options for the same video
- Audio-only mode automatically activates for videos over 1 hour, skipping visual analysis entirely
- On-screen text conflicts are pre-filtered (with fuzzy grouping) before ever reaching the report-generation prompt, avoiding wasted context tokens on repetitive overlay text
- The deterministic grounding check adds hallucination-risk mitigation at zero extra API cost
- Periods-based summarization only runs for videos actually long enough to need it (>60 segments)

---

## 13. How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/Umniyat-Hausawi/VideoMind.git
cd VideoMind
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Install FFmpeg:**
```bash
winget install ffmpeg   # Windows
brew install ffmpeg     # Mac
```

**5. Add API keys in `.env`:**
```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

> **⚠️ Safety Note:** Never commit `.env` files or API keys to version control.

**6. Run the app:**
```bash
streamlit run app.py
```

**7. (Optional) Run the test suites:**
```bash
pip install pytest
pytest tests/                    # pure-function + mocked API-call tests
```

---

## 14. Deployment (Hugging Face Spaces)

VideoMind ships with a `Dockerfile` for deployment via Hugging Face Spaces (Docker SDK). See [Architecture.md](Architecture.md) section "🚀 Deployment" for the full breakdown, and [LIMITATIONS.md](LIMITATIONS.md) for current deployment limitations.

---

## 15. Suggested Demo Flow

1. Paste a TikTok or X URL (most consistently reliable — see [LIMITATIONS.md](LIMITATIONS.md) on YouTube's occasional platform-side rate-limiting), or upload a short video
2. Select report depth and language
3. Run the analysis and review the five result metrics (sentiment, confidence, signal, language, reliability), including the expandable audio/visual reliability breakdown
4. Read the bilingual report and note how audio/visual conflicts are woven into the report, and how any grounding-check warnings appear if the report contradicts the computed data
5. Change the report depth or language on the same video and observe the cached, near-instant regeneration

---

## 16. Portfolio Value

This project demonstrates:

- Multimodal AI system design (audio + visual signal fusion, not single-modality analysis)
- Deterministic, testable numeric logic kept outside the LLM (fusion math, reliability scoring, grounding checks)
- Production-minded reliability engineering (retry logic, fallback tracking, graceful multi-modal degradation, structured tool-use output)
- Bug discovery and resolution through systematic testing — pure-function unit tests and mocked API-call tests covering retry logic, fallback tracking, and edge cases across every layer
- Prompt engineering for structured, grounded, bilingual output, including a deliberate move from prompt-only JSON to API-enforced structured output
- Deliberate scope management: recognizing when a feature (multi-content-type support, speaker diarization) added complexity without serving the project's core goal, and cutting it — not just adding features
- Honest, documented limitations rather than an overstated feature list

---

## Author

**Umniyat Hausawi**
AI Engineer | Machine Learning, Deep Learning & NLP

*Built as part of an AI engineering portfolio — following [Actionlytics](https://github.com/Umniyat-Hausawi/actionlytics-ai-copilot) (e-commerce AI copilot)*