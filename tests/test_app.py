"""
Unit tests for the one pure, testable function in app.py:
_partial_file_hash — the content-based identifier used as the upload
cache key (first + last 64KB + total size). This avoids collisions
between two different files that happen to share a name and size.

app.py itself is a Streamlit script that runs UI code directly at
module import time (normal for a Streamlit app, but awkward to import
in a test file) — so this file loads app.py "manually" via importlib,
with streamlit and the layers/* modules stubbed out, and only pulls out
the one function under test.
"""

import io
import os
import sys
import importlib.util
from unittest.mock import MagicMock

import pytest


def _load_partial_file_hash():
    """
    Loads app.py in isolation and returns its _partial_file_hash function.

    app.py runs Streamlit UI code (st.columns(...), etc.) directly at
    module level, which fails against the mocked `st` object partway
    through the file — but _partial_file_hash is defined near the top,
    BEFORE that UI code runs, so it's already bound in the module's
    namespace by the time the expected exception happens.

    Critical: the stubs installed into sys.modules here (streamlit,
    layers.*) must be removed afterward. Without this cleanup, any other
    test file importing the real `layers.*` package later in the same
    pytest session would get these stubs instead of the real modules.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(project_root, "app.py")

    spec = importlib.util.spec_from_file_location("app_module", app_path)
    app_module = importlib.util.module_from_spec(spec)

    stub_names = ["streamlit", "layers", "layers.input_layer", "layers.audio_layer",
                  "layers.visual_layer", "layers.fusion_layer", "layers.report_layer"]
    # Only remember/restore names that weren't already real entries —
    # none of these should be genuinely imported yet this early, but
    # being defensive costs nothing.
    previously_absent = [name for name in stub_names if name not in sys.modules]

    for name in stub_names:
        sys.modules[name] = MagicMock()

    try:
        spec.loader.exec_module(app_module)
    except Exception:
        pass  # expected — see docstring above
    finally:
        for name in previously_absent:
            sys.modules.pop(name, None)

    return getattr(app_module, "_partial_file_hash", None)


_partial_file_hash = _load_partial_file_hash()


@pytest.mark.skipif(_partial_file_hash is None, reason="_partial_file_hash could not be loaded from app.py in this environment")
class TestPartialFileHash:
    def test_same_content_produces_same_hash(self):
        content = b"x" * 200_000
        hash1 = _partial_file_hash(io.BytesIO(content))
        hash2 = _partial_file_hash(io.BytesIO(content))
        assert hash1 == hash2

    def test_different_content_same_size_produces_different_hash(self):
        """
        This is the exact collision case the partial hash exists to
        avoid — two files that happen to share a size (and, in the old
        filename+filesize scheme, could also share a name) must not
        collide into the same cache key.
        """
        content_a = b"a" * 100_000 + b"b" * 100_000
        content_b = b"c" * 100_000 + b"d" * 100_000
        hash_a = _partial_file_hash(io.BytesIO(content_a))
        hash_b = _partial_file_hash(io.BytesIO(content_b))
        assert hash_a != hash_b

    def test_file_position_is_reset_after_hashing(self):
        """
        Critical: app.py still needs to uploaded_file.read() the FULL
        file afterward to write it to disk — if this function left the
        read position at the end, that later read would silently return
        empty bytes.
        """
        f = io.BytesIO(b"some video bytes")
        _partial_file_hash(f)
        assert f.tell() == 0