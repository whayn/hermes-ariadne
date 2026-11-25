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


@pytest.mark.parametrize("start_fails", [False, True])
def test_idle_task_resumes_existing_tree_without_a_chat(tmp_path, monkeypatch, start_fails):
    import os
    import sqlite3
    import threading
    from types import SimpleNamespace

    from hermes_ariadne.importer import Importer
    from hermes_ariadne.tree import Part
    from hermes_ariadne.tree_store import TreeStore

    class IdleContext(FullContext):
        def register_idle_task(self, name, callback, **schedule):
            assert name == "resume-tree"
            assert schedule == dict(interval_hours=1, min_idle_hours=0, first_run="immediate")
            self.idle = callback

    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    config = {"context": {"engine": "compressor"}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    ctx = IdleContext()
    started = threading.Event()

    class RecordingFacade(Facade):
        def complete(self, *args, **kwargs):
            started.set()
            return SimpleNamespace(text="summary")

    ctx.llm = RecordingFacade()
    Importer(home).import_batch(
        key="synthetic", title="Synthetic", messages=[{"role": "user", "content": "a" * 800}]
    )
    root = load_root()
    root.register(ctx)
    engine = ctx.engines[0]
    assert engine._shared is None and not started.is_set()
    ctx.idle(None)
    assert not started.is_set()
    config["context"]["engine"] = "ariadne"
    store = TreeStore(home)
    part = Part(0, 0)
    assert store.reserve(part, 10, f"{os.getpid()}:old", max_jobs=0, max_total_bytes=0)
    module = sys.modules[type(engine).__module__]
    if start_fails:

        def unavailable(*args):
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as patch:
            patch.setattr(module.TreeStore, "rearm_lost", unavailable)
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                ctx.idle(None)
        assert not started.is_set()
        assert module.shared_home(home).users == 0
    ctx.idle(None)
    assert started.wait(5)
    builder = module.shared_home(home).builder
    builder.stop(timeout=5)
    assert part in store.nodes()
    assert store.jobs()[part].auto_retries == 1
    assert engine._shared is None


def test_collector_without_llm_registers_provider_only():
    ctx = CollectorContext()
    load_root().register(ctx)
    assert len(ctx.providers) == 1 and ctx.engines == []


def test_real_engine_directory_collector_registers_nothing():
    from plugins.context_engine import _EngineCollector

    collector = _EngineCollector(engine_name="ariadne")
    load_root().register(collector)
    assert collector.engine is None


def test_engine_is_the_tree_engine_and_registers_the_summary_task():
    from hermes_ariadne.tree_builder import TASK

    class AuxContext(FullContext):
        def __init__(self, config=None):
            super().__init__(config)
            self.tasks = []

        def register_auxiliary_task(self, key, **kwargs):
            self.tasks.append((key, kwargs))

    ctx = AuxContext({"tree": {"high": 64000, "tail_bytes": 1024}})
    load_root().register(ctx)
    (engine,) = ctx.engines
    assert type(engine).__name__ == "TreeAriadneEngine"
    assert engine._tree_settings.high == 64000 and engine._tree_settings.tail_bytes == 1024
    assert [key for key, _ in ctx.tasks] == [TASK]
    names = [s["name"] for s in engine.get_tool_schemas()]
    assert names == ["ariadne_zoom", "ariadne_search", "ariadne_memory_status"]


@pytest.mark.parametrize("tree", [{"unknown_limit": 1}, {"high": 0}, {"jobs": True}, "x"])
def test_invalid_tree_config_refuses_engine(tree):
    with pytest.raises((TypeError, ValueError)):
        load_root().register(FullContext({"tree": tree}))


def test_invalid_rolling_config_refuses_engine():
    with pytest.raises((TypeError, ValueError)):
        load_root().register(FullContext({"rolling": {"unknown_limit": 1}}))


def test_each_registration_builds_fresh_instances():
    a, b = FullContext(), FullContext()
    load_root().register(a)
    load_root("_ariadne_root_second").register(b)
    assert a.providers[0] is not b.providers[0]
    assert a.engines[0] is not b.engines[0]
