"""Prepared engine opt-in/lifecycle tests against the actual Hermes ABC."""

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("HERMES_SOURCE required for real ContextEngine contract", allow_module_level=True)


def live_class():
    assert importlib.util.find_spec("hermes_ariadne.live_engine") is not None, (
        "prepared live engine is not implemented"
    )
    return importlib.import_module("hermes_ariadne.live_engine").PreparedAriadneEngine


def test_prepared_engine_is_separate_and_construction_has_no_io(tmp_path):
    from hermes_ariadne.engine import AriadneEngine

    cls = live_class()
    before = list(tmp_path.iterdir())
    engine = cls()
    assert engine.name == "ariadne-prepared"
    assert AriadneEngine().name == "ariadne"
    assert engine.select_context([], conversation_messages=[], budget_tokens=4096) is None
    assert engine.get_status()["selection"]["status"] == "not_initialized"
    assert engine.get_status()["selection"]["provider_fit_verified"] is False
    assert engine.clone_for_agent().name == "ariadne-prepared"
    assert list(tmp_path.iterdir()) == before
    assert not (Path(os.environ["HERMES_HOME"]) / "ariadne").exists()


def test_snapshot_state_cannot_survive_reset_rebind_or_session_end(tmp_path):
    engine = live_class()()
    engine.bind_session_state(session_id="first")
    engine._prepared = object()
    engine._home = tmp_path
    engine.on_session_reset()
    assert engine._prepared is None
    assert engine._home is None
    assert engine.get_status()["selection"]["status"] == "not_initialized"

    engine.bind_session_state(session_id="first")
    engine._prepared = object()
    engine._home = tmp_path
    engine.bind_session_state(session_id="second")
    assert engine._prepared is None
    assert engine._home is None

    engine._prepared = object()
    engine._home = tmp_path
    engine.on_session_end("second", [])
    assert engine._prepared is None
    assert engine._home is None


def test_bounded_schemas_are_static_and_do_not_change_passive_engine():
    from hermes_ariadne.engine import AriadneEngine

    engine = live_class()()
    schemas = engine.get_tool_schemas()
    assert {s["name"] for s in schemas} == {"ariadne_find", "ariadne_zoom"}
    assert {s["name"] for s in AriadneEngine().get_tool_schemas()} == {"ariadne_search"}
    assert engine._store is None


def test_prepared_directory_is_discovered_without_archive_io(tmp_path, monkeypatch):
    from agent.agent_init import _select_context_engine
    from plugins.context_engine import load_context_engine

    from hermes_ariadne.install import install

    home = tmp_path / "prepared-home"
    install(home, prepared=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    engine = load_context_engine("ariadne-prepared")
    assert engine.name == "ariadne-prepared"
    assert _select_context_engine({"context": {"engine": "ariadne-prepared"}}).name == engine.name
    assert {s["name"] for s in engine.get_tool_schemas()} == {"ariadne_find", "ariadne_zoom"}
    assert not (home / "ariadne").exists()


def engine_fixture(tmp_path, *, publish=True, **kernel_settings):
    from hermes_ariadne.kernel import MemoryKernel
    from hermes_ariadne.prepared import publish_prepared
    from hermes_ariadne.sources import Projection
    from hermes_ariadne.store import EventStore, canonical_json

    home = tmp_path / "fixture-home"
    rows = [
        {
            "message_uid": f"m{i}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"original {i} " + "λ" * 200,
        }
        for i in range(4)
    ]
    store = EventStore(home / "ariadne" / "events.sqlite3")
    store.ingest("chosen", rows, origin="hermes_db")
    projection = Projection(
        store, "chosen", [hashlib.sha256(canonical_json(row)).hexdigest() for row in rows[:2]]
    )
    kernel = MemoryKernel(projection, home / "operator.json", node_bytes=128, **kernel_settings)
    kernel.build(lambda request: "synthetic summary")
    if publish:
        publish_prepared(kernel, home=home)
    db = SimpleNamespace(get_messages=lambda session_id, **kwargs: copy.deepcopy(rows))
    engine = live_class()()
    engine.bind_session_state(db, "chosen")
    engine.on_session_start("chosen", hermes_home=home)
    conversation = copy.deepcopy(rows) + [
        {"message_uid": "current", "role": "user", "content": "current turn"}
    ]
    wire = [{"role": "system", "content": "host policy"}] + [
        {k: v for k, v in row.items() if k != "message_uid"} for row in conversation
    ]
    return engine, home, rows, conversation, wire, kernel


def select(engine, conversation, wire, **kwargs):
    return engine.select_context(
        wire,
        conversation_messages=conversation,
        incoming_message=conversation[-1],
        budget_tokens=kwargs.get("budget_tokens", 32000),
    )


def test_bound_engine_loads_and_selects_frozen_prefix(tmp_path):
    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    before = copy.deepcopy((rows, conversation, wire))
    selected = select(engine, conversation, wire)
    assert selected is not None
    assert selected[0] == wire[0]
    assert "untrusted" in selected[1]["content"]
    assert selected[2:] == wire[3:]
    assert engine.get_status()["selection"]["status"] == "selected"
    assert engine.get_status()["selection"]["provider_fit_verified"] is False
    assert (rows, conversation, wire) == before
    assert engine._home == home
    assert engine.clone_for_agent()._prepared is None


def test_evidence_tools_dispatch_read_only_with_and_without_snapshot(tmp_path):
    import base64

    from hermes_ariadne.store import canonical_json

    engine, _, rows, conversation, _, _ = engine_fixture(tmp_path, publish=False)
    before = copy.deepcopy(conversation)
    found = json.loads(
        engine.handle_tool_call("ariadne_find", {"query": "original 0"}, messages=conversation)
    )
    assert found["ok"] is True
    ref = found["results"][0]["event_ref"]
    page = json.loads(engine.handle_tool_call("ariadne_zoom", {"ref": ref, "limit": 2048}))
    assert base64.b64decode(page["page"]["data_base64"]) == canonical_json(rows[0])
    assert conversation == before
    engine.on_session_end("chosen", conversation)
    assert (
        json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))["error"]
        == "uninitialized"
    )


def test_missing_snapshot_is_frozen_until_explicit_session_start(tmp_path):
    from hermes_ariadne.prepared import publish_prepared

    engine, home, _, conversation, wire, kernel = engine_fixture(tmp_path, publish=False)
    assert select(engine, conversation, wire) is None
    assert engine.get_status()["selection"]["reason"] == "prepared_unavailable"
    publish_prepared(kernel, home=home)
    assert select(engine, conversation, wire) is None
    engine.on_session_start("chosen", hermes_home=home)
    assert select(engine, conversation, wire) is not None


@pytest.mark.parametrize("replacement", ["summary-version-and-text", "same-summary-new-settings"])
def test_republished_checkpoint_refuses_old_nodes_only_after_lifecycle_reload(
    tmp_path, replacement
):
    from hermes_ariadne.kernel import MemoryKernel
    from hermes_ariadne.prepared import publish_prepared
    from hermes_ariadne.store import canonical_json

    engine, home, rows, conversation, wire, kernel = engine_fixture(tmp_path, chat_high=25)
    selected = select(engine, conversation, wire)
    assert selected is not None
    old_memory = selected[1]["content"]
    root = json.loads(old_memory.splitlines()[1])["ref"]

    def zoom(ref, **kwargs):
        return json.loads(engine.handle_tool_call("ariadne_zoom", {"ref": ref}, **kwargs))

    expanded = zoom(root)
    assert expanded["ok"] and expanded["kind"] == "node"
    refs = [root, *(child["ref"] for child in expanded["children"])]
    old_responses = [zoom(ref) for ref in refs]
    assert all(response["ok"] for response in old_responses)
    event_ref = old_responses[1]["page"]["event_ref"]
    # An unchanged generation survives a real engine lifecycle reload.
    engine.on_session_start("chosen", hermes_home=home)
    assert [zoom(ref) for ref in refs] == old_responses
    settings = kernel._settings()
    if replacement == "summary-version-and-text":
        settings["summary_version"] = "replacement-v2"
        summary = "replacement summary"
    else:
        settings["compactor_high"] += 1
        summary = "synthetic summary"
    newer = MemoryKernel(kernel.projection, home / "replacement.json", **settings)
    newer.build(lambda _: summary)
    assert newer.projection.projection_id == kernel.projection.projection_id
    assert newer.projection.event_ids == kernel.projection.event_ids
    publish_prepared(newer, home=home)
    # Publication alone must not retarget either selection or retrieval.
    assert select(engine, conversation, wire)[1]["content"] == old_memory
    assert [zoom(ref) for ref in refs] == old_responses
    engine.on_session_start("chosen", hermes_home=home)
    assert engine._prepared.kernel.view() == newer.view()
    for ref in refs:
        response = zoom(ref, checkpoint_id="caller cannot override the attached view")
        assert not response["ok"], "old node silently rebound to a republished checkpoint"
        assert set(response) == {"trust", "ok", "error"}
        assert len(json.dumps(response, ensure_ascii=True)) < 200
    fresh = select(engine, conversation, wire)
    assert fresh is not None
    fresh_root = json.loads(fresh[1]["content"].splitlines()[1])["ref"]
    assert fresh_root != root
    assert zoom(fresh_root)["ok"]
    recovered = zoom(event_ref)
    assert recovered["ok"]
    import base64

    assert base64.b64decode(recovered["page"]["data_base64"]) == canonical_json(rows[0])


def test_archive_failure_clears_preceding_success_diagnostic(tmp_path):
    engine, _, _, conversation, wire, _ = engine_fixture(tmp_path)
    assert select(engine, conversation, wire) is not None

    def broken(*args, **kwargs):
        raise OSError("private path must not be echoed")

    engine._session_db.get_messages = broken
    assert select(engine, conversation, wire) is None
    status = engine.get_status()["selection"]
    assert status["status"] == "declined"
    assert status["reason"] == "archive_unavailable"
    assert "private path" not in json.dumps(status)


def test_rebinding_same_session_to_another_db_invalidates_snapshot(tmp_path):
    engine, _, _, conversation, wire, _ = engine_fixture(tmp_path)
    engine.bind_session_state(SimpleNamespace(get_messages=lambda *a, **k: []), "chosen")
    assert engine._prepared is None
    assert select(engine, conversation, wire) is None


def test_model_switch_invalidates_measurement_not_frozen_source(tmp_path):
    engine, _, _, conversation, wire, _ = engine_fixture(tmp_path)
    assert select(engine, conversation, wire) is not None
    frozen = engine._prepared
    engine.update_model("fixture", 1000, "", "", "", "")
    assert engine._prepared is frozen
    assert engine.get_status()["selection"]["status"] == "unmeasured"
    assert select(engine, conversation, wire, budget_tokens=1000) is None
    assert engine.get_status()["selection"]["reason"] == "over_budget"


@pytest.mark.parametrize("failure", ["unknown_budget", "revision", "active_read_failure"])
def test_real_host_uses_original_request_on_decline(tmp_path, failure):
    import logging

    from agent.conversation_loop import _apply_context_engine_selection

    engine, _, rows, conversation, wire, _ = engine_fixture(tmp_path)
    engine.context_length = 32000
    if failure == "unknown_budget":
        engine.context_length = 0
    elif failure == "revision":
        rows[0]["content"] = "revised durable source"
    else:
        original_reader = engine._session_db.get_messages

        def reader(session_id, **kwargs):
            if kwargs.get("include_inactive") is False:
                raise OSError("private failure")
            return original_reader(session_id, **kwargs)

        engine._session_db.get_messages = reader
    before = copy.deepcopy(wire)
    agent = SimpleNamespace(context_compressor=engine, session_id="chosen")
    result = _apply_context_engine_selection(
        agent,
        wire,
        conversation_messages=conversation,
        incoming_message=conversation[-1],
        logger=logging.getLogger(__name__),
    )
    assert result is wire
    assert wire == before
    assert engine.get_status()["selection"]["status"] == "declined"
    assert (
        engine.get_status()["selection"]["reason"]
        == {
            "unknown_budget": "unknown_budget",
            "revision": "unsupported_alignment",
            "active_read_failure": "selection_unavailable",
        }[failure]
    )
