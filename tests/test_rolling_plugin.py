"""Pure registration tests; AST-extracted host types, never native startup."""

import ast
import copy
import importlib
import importlib.util
import json
import os
import sys
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "hermes_ariadne"
HOST_SOURCE = os.environ.get("ARIADNE_PURE_HOST_SOURCE") or os.environ.get("HERMES_SOURCE")


def host_definitions(path, names, bindings):
    """Execute only named pure definitions from explicitly selected host source."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == names
    namespace = {"__name__": "pure_host_contract", **bindings}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def entry(monkeypatch):
    # No host package imports. Keep the runtime under a disposable namespace so
    # these extracted types cannot contaminate the normal native test imports.
    host = HOST_SOURCE
    if not host:
        pytest.skip("set ARIADNE_PURE_HOST_SOURCE for read-only AST contracts")
    definitions = host_definitions(
        Path(host) / "agent" / "context_engine.py",
        {"ContextEngine"},
        dict(
            ABC=ABC,
            abstractmethod=abstractmethod,
            Any=Any,
            Dict=Dict,
            List=List,
            Optional=Optional,
            copy=copy,
            json=json,
        ),
    )
    module = types.ModuleType("agent.context_engine")
    module.ContextEngine = definitions["ContextEngine"]
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.context_engine", module)
    name = "_ariadne_pure_registration"
    spec = importlib.util.spec_from_file_location(
        name, SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)]
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    try:
        yield lambda: importlib.import_module(name + ".rolling_entry")
    finally:
        for key in list(sys.modules):
            if key.startswith(name + "."):
                del sys.modules[key]


class Facade:
    def complete(self, **kwargs):
        pytest.fail("registration/construction/schema must never call the model")


class FullContext:
    plugin_id = "ariadne-rolling-plugin"

    def __init__(self, settings=None):
        self.llm = Facade()
        self.settings = settings if settings is not None else {}
        self.engine = None
        self.reads = []
        self.result = object()

    def get_config(self, key, default=None):
        self.reads.append((key, default))
        return self.settings.get(key, default)

    def register_context_engine(self, engine):
        self.engine = engine
        return self.result


def test_register_preserves_host_facade_validated_settings_and_registry_return(entry, tmp_path):
    assert (SOURCE / "rolling_entry.py").is_file(), "rolling entry is missing"
    entry = entry()
    ctx = FullContext({"rolling": {"max_logical_jobs": 3, "trigger_bytes": 1024}})
    assert entry.register(ctx) is ctx.result
    assert ctx.reads == [("rolling", {})]
    assert ctx.engine.name == "ariadne-rolling"
    assert ctx.engine._llm is ctx.llm
    assert ctx.engine._settings.max_logical_jobs == 3
    ctx.settings["rolling"]["max_logical_jobs"] = 999
    assert ctx.engine._settings.max_logical_jobs == 3
    clone = ctx.engine.clone_for_agent()
    assert clone._llm is ctx.llm
    assert clone._settings is ctx.engine._settings
    assert clone._lock is not ctx.engine._lock
    assert clone._seen is not ctx.engine._seen
    assert clone._session_db is None
    assert clone._store is None
    assert clone.get_tool_schemas()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("facade", [None, object(), types.SimpleNamespace(complete=None)])
def test_register_refuses_missing_host_llm_before_registration(entry, facade):
    ctx = FullContext()
    ctx.llm = facade
    with pytest.raises(ValueError, match="host LLM"):
        entry().register(ctx)
    assert ctx.engine is None


@pytest.mark.parametrize(
    "settings",
    [
        {"unknown": 1},
        {"max_logical_jobs": True},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": "60"},
        {"timeout": 2**31},
        {"timeout": None},
        {"timeout": 1.5},
    ],
)
def test_register_rejects_invalid_or_unknown_limits(entry, settings):
    ctx = FullContext({"rolling": settings})
    with pytest.raises((TypeError, ValueError)):
        entry().register(ctx)
    assert ctx.engine is None


@pytest.mark.parametrize("settings", [None, [], "bad", 1, True])
def test_register_rejects_non_mapping_limits(entry, settings):
    ctx = FullContext({"rolling": settings})
    with pytest.raises((TypeError, ValueError)):
        entry().register(ctx)
    assert ctx.engine is None


def test_registry_none_return_is_not_reinterpreted_as_failure(entry):
    ctx = FullContext()
    ctx.result = None
    assert entry().register(ctx) is None
    assert ctx.engine is not None


def test_limited_directory_collector_cannot_register_or_discover_fallback_class(entry):
    module = entry()

    class Collector:
        engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    collector = Collector()
    with pytest.raises((AttributeError, ValueError)):
        module.register(collector)
    assert collector.engine is None
    assert not any(isinstance(value, type) for value in vars(module).values())


def test_general_manifest_uses_native_schema_and_opt_in_identity(entry):
    from collections.abc import Mapping
    from dataclasses import fields

    assert (SOURCE / "rolling_plugin.yaml").is_file(), "general manifest is missing"
    # JSON is a strict YAML subset; no host YAML/config loader is imported.
    data = json.loads((SOURCE / "rolling_plugin.yaml").read_text())
    path = Path(HOST_SOURCE) / "hermes_cli/plugins_manifest.py"
    tree = ast.parse(path.read_text())
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SUPPORTED_MANIFEST_VERSION"
    }
    known_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and node.target.id == "_KNOWN_MANIFEST_FIELDS"
    )
    assert set(data) <= ast.literal_eval(known_node.value)
    assert data["manifest_version"] <= constants["SUPPORTED_MANIFEST_VERSION"]
    assert data["name"] == "ariadne-rolling-plugin"
    assert data["kind"] == "standalone"
    assert data["provides_tools"] == []  # engine tools, not general tools
    assert data["provides_hooks"] == []  # ContextEngine callbacks, not bus hooks
    assert data["requires_env"] == []
    ns = host_definitions(
        path,
        {"validate_config_schema"},
        dict(Mapping=Mapping, _CONFIG_SCHEMA_TYPES={"dict": (dict,)}),
    )
    schema = data["config_schema"]
    assert ns["validate_config_schema"](data["name"], schema, {"rolling": {}}) == []
    assert ns["validate_config_schema"](data["name"], schema, {"rolling": 1})
    defaults = schema["rolling"]["default"]
    ctx = FullContext({"rolling": defaults})
    entry().register(ctx)
    assert defaults == {field.name: field.default for field in fields(ctx.engine._settings)}


def test_installed_layout_cannot_shadow_engine_name_or_bypass_full_context(entry, tmp_path):
    import contextlib
    import logging

    from hermes_ariadne.install import install

    host = Path(HOST_SOURCE)
    target = install(tmp_path, rolling=True)
    lookup = host_definitions(
        host / "plugins/context_engine/__init__.py",
        {"_is_context_engine_dir", "find_engine_dir"},
        dict(
            Path=Path,
            Optional=Optional,
            _CONTEXT_ENGINE_PLUGINS_DIR=tmp_path / "bundled",
            _loader=types.SimpleNamespace(user_plugins_dir=lambda: target.parent),
        ),
    )
    assert lookup["find_engine_dir"]("ariadne-rolling") is None
    assert lookup["find_engine_dir"]("ariadne-rolling-plugin") == target
    loader = host_definitions(
        host / "plugins/plugin_loader.py",
        {"NoopPluginContext", "instance_from_module"},
        dict(Any=Any, Optional=Optional, logging=logging, contextlib=contextlib),
    )
    collector_class = host_definitions(
        host / "plugins/context_engine/__init__.py",
        {"_EngineCollector"},
        dict(_loader=types.SimpleNamespace(NoopPluginContext=loader["NoopPluginContext"])),
    )["_EngineCollector"]
    module = entry()
    collector = collector_class(engine_name="ariadne-rolling-plugin")
    # The actual directory extraction's fallback class scan also returns None.
    assert (
        loader["instance_from_module"](
            module,
            collector=collector,
            collected_attr="engine",
            base_cls=sys.modules["agent.context_engine"].ContextEngine,
            name="ariadne-rolling-plugin",
            logger=logging.getLogger(__name__),
        )
        is None
    )
    assert collector.engine is None
    assert not (tmp_path / "ariadne").exists()


def test_entry_import_construction_and_schemas_do_not_open_archive_or_call_llm(entry, monkeypatch):
    import sqlite3

    def forbidden(*args, **kwargs):
        pytest.fail("entry/construction/schema attempted archive I/O")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    module = entry()
    assert module.__package__ + ".rolling_engine" not in sys.modules
    ctx = FullContext()
    module.register(ctx)
    assert ctx.engine.get_tool_schemas()
    assert ctx.engine.clone_for_agent().get_tool_schemas()


def test_registration_exception_is_observable(entry):
    ctx = FullContext()

    def rejected(engine):
        raise RuntimeError("synthetic registry admission refusal")

    ctx.register_context_engine = rejected
    with pytest.raises(RuntimeError, match="admission refusal"):
        entry().register(ctx)
    assert ctx.engine is None
