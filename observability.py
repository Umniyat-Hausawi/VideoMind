"""
Optional Langfuse observability integration. Provides trace-level
visibility into every Claude/Whisper API call — latency, cost,
prompt/response pairs — across all five layers, complementing the
print()-based logging used elsewhere in the project.

Design principle: this is entirely OPT-IN and fails silently.
- No LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY set → LANGFUSE_ENABLED is
  False, init_observability() is a no-op, and every layer runs exactly
  as it would without this module present at all.
- Keys set but the `langfuse` / `opentelemetry-instrumentation-anthropic`
  packages aren't installed → a clear one-line warning is printed, and
  the app continues without tracing rather than crashing.
- Keys set and packages installed → Claude calls (via OpenTelemetry
  instrumentation) and Whisper calls (via a drop-in SDK replacement in
  audio_layer.py) are automatically traced to your Langfuse project.

This mirrors the project's graceful-degradation pattern elsewhere (e.g.
a visual-layer crash falling back to audio-only instead of crashing the
whole analysis) — observability is a nice-to-have layered on top, never
a hard dependency the core pipeline needs to function.
"""

import os

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))

_instrumented = False


def init_observability() -> bool:
    """
    Call once, as early as possible in the app's lifecycle (app.py's top
    level, before any layer constructs an Anthropic client). Instruments
    the Anthropic SDK globally via OpenTelemetry, so every
    `client.messages.create()` call across audio_layer, visual_layer, and
    report_layer is automatically traced — no per-call code changes
    needed in those files.

    Whisper/OpenAI tracing is handled separately, in audio_layer.py
    itself, via a drop-in import swap (see that file) rather than global
    instrumentation, since the OpenAI Python SDK integration Langfuse
    ships is a direct replacement import, not an auto-instrumentor.

    Returns True if tracing was successfully enabled, False otherwise
    (disabled by config, or enabled but failed to initialize) — callers
    can use this for an optional startup log line, but should never
    branch pipeline behavior on it.
    """
    global _instrumented

    if _instrumented:
        return True

    if not LANGFUSE_ENABLED:
        return False

    try:
        from langfuse import get_client
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        # get_client() is what actually configures the OpenTelemetry
        # TracerProvider/exporter to point at Langfuse's ingestion endpoint,
        # authenticated with LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY (and
        # LANGFUSE_BASE_URL/LANGFUSE_HOST for non-default regions). This
        # MUST run before AnthropicInstrumentor().instrument() — without
        # it, the instrumentor still emits spans, but into an unconfigured
        # default tracer that silently discards them: no error, no trace,
        # nothing in the Langfuse dashboard. (Found via live testing: traces
        # never showed up even though init reported success, because this
        # call was missing.)
        get_client()
        AnthropicInstrumentor().instrument()
        _instrumented = True
        print("Langfuse observability enabled — Claude calls will be traced.")
        return True

    except ImportError:
        print(
            "WARNING: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are set but "
            "the required packages aren't installed — observability stays "
            "disabled, the app continues normally. "
            "Run: pip install langfuse opentelemetry-instrumentation-anthropic"
        )
        return False

    except Exception as e:
        print(f"WARNING: Langfuse observability init failed — continuing without it: {e}")
        return False


def flush_observability() -> None:
    """
    Force any queued spans to send immediately, rather than waiting for the
    OTel batch exporter's normal periodic flush interval. Call this once an
    analysis run completes — cheap no-op if tracing was never enabled, and
    never raises (a flush failure must not affect the pipeline result the
    user is waiting on).
    """
    if not _instrumented:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as e:
        print(f"WARNING: Langfuse flush failed (traces may be delayed, not lost): {e}")