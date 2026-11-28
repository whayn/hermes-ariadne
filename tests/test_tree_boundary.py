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


# Plugin registration with Hermes's LLM facade, as the general plugin loader runs it.
REGISTER = """
class Llm:
    def complete(self, *args, **kwargs):
        raise AssertionError("registration must not call the model")

class Ctx:
    llm = Llm()
    def __init__(self, config):
        self.config = config
    def get_config(self, key, default=None):
        return self.config.get(key, default)
    def register_memory_provider(self, provider):
        pass
    def register_context_engine(self, engine):
        return engine

from hermes_ariadne.plugin_entry import register
assert register(Ctx({"rolling": {"timeout": 30}})) is not None
"""


def _loaded(body):
    # A fresh interpreter: this process has already imported every module. It gets
    # this process's import path (worktree src first, host source and site).
    path = [str(SRC), *(p for p in sys.path if p)]
    script = (
        "import json, sys\n"
        f"sys.path[:0] = {path!r}\n"
        f"{body}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('hermes_ariadne.'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", script],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def _development(loaded):
    return sorted(loaded & {f"hermes_ariadne.{name}" for name in DEVELOPMENT})


def test_shipped_modules_do_not_import_the_development_engines():
    loaded = _loaded(f"for name in {SHIPPED!r}:\n    __import__(name)")
    assert set(SHIPPED) <= loaded
    assert _development(loaded) == []


def test_plugin_registration_does_not_import_the_development_engines():
    loaded = _loaded(REGISTER)
    assert {"hermes_ariadne.plugin_entry", "hermes_ariadne.tree_engine"} <= loaded
    assert _development(loaded) == []
