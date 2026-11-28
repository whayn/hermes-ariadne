"""Every module in the package is part of the shipped plugin (#70): no second engine."""

import ast
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

# Test tooling inside the package, not part of the plugin: the runner's host-site probe.
TOOLING = {"_host_site"}
PACKAGE = SRC / "hermes_ariadne"


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


def _modules():
    return {f"hermes_ariadne.{p.stem}" for p in PACKAGE.glob("*.py") if p.stem != "__init__"}


def test_registration_and_the_provider_use_every_module_in_the_package():
    """A module nothing ships with is a leftover engine: delete it instead."""
    loaded = _loaded(REGISTER + f"for name in {SHIPPED!r}:\n    __import__(name)")
    unused = _modules() - loaded - {f"hermes_ariadne.{name}" for name in TOOLING}
    assert unused == set()


def test_plugin_registration_registers_the_tree_engine():
    loaded = _loaded(REGISTER)
    assert {"hermes_ariadne.plugin_entry", "hermes_ariadne.tree_engine"} <= loaded


def test_every_relative_import_in_the_package_resolves():
    """Lazy imports inside functions too: a deleted module must leave no caller."""
    names = {p.stem for p in PACKAGE.glob("*.py")}
    dangling = []
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                targets = (
                    [node.module.split(".")[0]] if node.module else [a.name for a in node.names]
                )
                dangling += [f"{path.name}: .{t}" for t in targets if t not in names]
    assert dangling == []
