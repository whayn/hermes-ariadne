"""The shipped plugin's runtime modules do not load the development engines (#70)."""

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# What the profile plugin's engine and provider run on.
SHIPPED = (
    "hermes_ariadne.tree_engine",
    "hermes_ariadne.memory_provider",
    "hermes_ariadne.importer",
    "hermes_ariadne.conversations",
)

# Development engines and offline tooling, deleted in #70 phase 2.
DEVELOPMENT = (
    "compaction",
    "compactor_demo",
    "demo",
    "engine",
    "install",
    "kernel",
    "live_engine",
    "mvp_engine",
    "native_compactor",
    "prepared",
    "prepared_entry",
    "receipts",
    "rolling",
    "rolling_delta",
    "rolling_engine",
    "rolling_entry",
    "rolling_migrate",
    "rolling_retrieval",
    "rolling_selection",
    "rolling_state",
    "simulation",
)


def test_shipped_modules_do_not_import_the_development_engines():
    # A fresh interpreter: this process has already imported every module. It gets
    # this process's import path (worktree src first, host source and site).
    path = [str(SRC), *(p for p in sys.path if p)]
    script = (
        "import json, sys\n"
        f"sys.path[:0] = {path!r}\n"
        f"for name in {SHIPPED!r}:\n"
        "    __import__(name)\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('hermes_ariadne.'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", script],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert set(SHIPPED) <= loaded
    forbidden = sorted(loaded & {f"hermes_ariadne.{name}" for name in DEVELOPMENT})
    assert forbidden == []
