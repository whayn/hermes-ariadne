"""Isolate Hermes BEFORE collection imports; never read live config or credentials."""

import os
import re
import sys
import tempfile
from pathlib import Path

# The caller's TMPDIR is used, rather than the system /tmp.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="ariadne-tests-"))
os.environ["HERMES_HOME"] = str(_TEST_ROOT / "collection-home")
os.environ["TZ"] = "UTC"
for key in list(os.environ):
    if (
        re.search(r"_(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS)$", key)
        or key.startswith("HERMES_SESSION_")
        or key in {"HERMES_ACTIVE_PROFILE", "HERMES_PROFILE"}
    ):
        os.environ.pop(key, None)

# Freeze the runner-selected dependency order before collection imports host code,
# whose PM activation can replace PYTHONPATH for all subsequent child workers.
_RUNNER_PYTHONPATH = os.environ.get("PYTHONPATH")

# Only explicitly requested host source; no default-path probing.
source = os.environ.get("HERMES_SOURCE")
if source:
    # Bind the real runner-selected SDKs before host PM activation can replace
    # sys.path. Restoring PYTHONPATH below only protects fresh child processes.
    import httpx  # noqa: F401
    import openai  # noqa: F401

    sys.path.insert(0, source)

import pytest  # noqa: E402 -- isolate before any test-related imports


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    if _RUNNER_PYTHONPATH is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", _RUNNER_PYTHONPATH)
