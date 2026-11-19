"""Combined plugin registration across the three host loader contexts."""

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("set HERMES_SOURCE for host registration contracts", allow_module_level=True)

from agent.context_engine import ContextEngine  # noqa: E402
from agent.memory_provider import MemoryProvider  # noqa: E402


def load_root(name="_ariadne_root_under_test"):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Facade:
    def complete(self, *args, **kwargs):  # never called during registration
        raise AssertionError("registration must not call the model")


class FullContext:
    """Shape of the general manager's PluginContext that registration uses."""

    def __init__(self, config=None):
        self.llm = Facade()
        self.config = config or {}
        self.engines, self.providers = [], []

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_context_engine(self, engine):
        self.engines.append(engine)

    def register_memory_provider(self, provider):
        self.providers.append(provider)


class CollectorContext:
    """Like the memory loader's collector: unknown non-register_ attributes raise."""

    def __init__(self):
        self.providers, self.engines = [], []

    def register_memory_provider(self, provider):
        self.providers.append(provider)

    def register_context_engine(self, engine):
        self.engines.append(engine)

    def __getattr__(self, attr):
        raise AttributeError(attr)


def test_root_shim_markers_and_no_classes():
    source = (ROOT / "__init__.py").read_text()
    assert "register_context_engine" in source and "register_memory_provider" in source
    tree = ast.parse(source)
    assert not [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_manifest_is_explicit_standalone(tmp_path):
    import shutil

    from hermes_cli.plugins_manifest import parse_manifest_file

    manifest = parse_manifest_file(ROOT / "plugin.yaml", ROOT, "user", "")
    assert manifest.name == "ariadne" and manifest.kind == "standalone"
    # Without the explicit kind, Hermes would route it to memory-only discovery.
    shutil.copy(ROOT / "__init__.py", tmp_path / "__init__.py")
    stripped = [
        ln for ln in (ROOT / "plugin.yaml").read_text().splitlines() if ln != "kind: standalone"
    ]
    (tmp_path / "plugin.yaml").write_text("\n".join(stripped) + "\n")
    assert parse_manifest_file(tmp_path / "plugin.yaml", tmp_path, "user", "").kind == "exclusive"


def test_full_context_registers_both():
    ctx = FullContext()
    load_root().register(ctx)
    (provider,), (engine,) = ctx.providers, ctx.engines
    assert isinstance(provider, MemoryProvider) and provider.name == "ariadne"
    assert isinstance(engine, ContextEngine) and engine.name == "ariadne"
    clone = engine.clone_for_agent()
    assert clone.name == "ariadne" and clone._llm is engine._llm


def test_collector_without_llm_registers_provider_only():
    ctx = CollectorContext()
    load_root().register(ctx)
    assert len(ctx.providers) == 1 and ctx.engines == []


def test_real_engine_directory_collector_registers_nothing():
    from plugins.context_engine import _EngineCollector

    collector = _EngineCollector(engine_name="ariadne")
    load_root().register(collector)
    assert collector.engine is None


def test_invalid_rolling_config_refuses_engine():
    with pytest.raises((TypeError, ValueError)):
        load_root().register(FullContext({"rolling": {"unknown_limit": 1}}))


def test_each_registration_builds_fresh_instances():
    a, b = FullContext(), FullContext()
    load_root().register(a)
    load_root("_ariadne_root_second").register(b)
    assert a.providers[0] is not b.providers[0]
    assert a.engines[0] is not b.engines[0]
