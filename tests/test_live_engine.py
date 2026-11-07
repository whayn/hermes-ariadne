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


def reasoning_engine_fixture(tmp_path):
    """Native-shaped canonical TEXT, restored/list, assembled/list and SQL NULLs."""
    engine, home, rows, conversation, wire, kernel = engine_fixture(tmp_path)
    items = [
        {
            "type": "reasoning",
            "encrypted_content": "synthetic opaque : ! retained bytes",
            "id": "rs_live_14",
            "summary": [{"type": "summary_text", "text": "summary λ"}],
            "_issuer_kind": "codex_backend",
            "_issuer_model": "gpt-6.1-sol",
        }
    ]
    answer = [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": rows[3]["content"]}],
            "id": "msg_live_14",
            "phase": "final_answer",
        }
    ]
    rows[3].update(
        codex_reasoning_items=json.dumps(items),
        codex_message_items=json.dumps(answer),
        reasoning=None,
        reasoning_content=None,
    )
    conversation[3].update(
        codex_reasoning_items=copy.deepcopy(items),
        codex_message_items=copy.deepcopy(answer),
        _db_persisted=True,
    )
    wire[4].update(
        codex_reasoning_items=copy.deepcopy(items), codex_message_items=copy.deepcopy(answer)
    )
    return engine, home, rows, conversation, wire, kernel


def bind_codex(engine, **overrides):
    fields = dict(
        model="gpt-6.1-sol-900k",
        context_length=32000,
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        api_mode="codex_responses",
    )
    fields.update(overrides)
    engine.update_model(**fields)


def test_native_bound_route_enables_retained_reasoning_without_changing_inputs(tmp_path):
    engine, _, rows, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    before = copy.deepcopy((rows, conversation, wire))
    assert select(engine, conversation, wire) is None
    scope = engine._capture()
    bind_codex(engine)
    selected = select(engine, conversation, wire)
    assert selected is not None
    assert selected[0] == wire[0] and selected[2:] == wire[3:]
    assert (rows, conversation, wire) == before
    assert engine._capture() == scope  # Model labels do not refresh the frozen source.
    assert engine.get_status()["selection"]["provider_fit_verified"] is False


@pytest.mark.parametrize("change", ["same", "away_back", "unsupported"])
@pytest.mark.parametrize("raises", [False, True])
def test_route_update_during_measurement_discards_candidate_and_old_diagnostic(
    tmp_path, monkeypatch, change, raises
):
    from agent import model_metadata

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    original = model_metadata.estimate_request_tokens_rough
    armed = True
    latest = []

    def measure(messages):
        nonlocal armed
        if armed:
            armed = False
            if change != "same":
                bind_codex(engine, provider="openai")
            if change != "unsupported":
                bind_codex(engine)
                assert select(engine, conversation, wire) is not None
            latest.append(engine.get_status())
            if raises:
                raise OSError("old measurement private details")
        return original(messages)

    monkeypatch.setattr(model_metadata, "estimate_request_tokens_rough", measure)
    assert select(engine, conversation, wire) is None
    assert engine.get_status() == latest[0]


@pytest.mark.parametrize("boundary", ["observation", "active", "observe_return"])
@pytest.mark.parametrize("change", ["same", "away_back"])
@pytest.mark.parametrize("raises", [False, True])
def test_route_reentry_during_reads_preserves_latest_selection(
    tmp_path, monkeypatch, boundary, change, raises
):
    from hermes_ariadne.prepared import PreparedView

    engine, _, rows, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    selections = []
    prepared_select = PreparedView.select

    def choosing(*args, **kwargs):
        selections.append(kwargs["reasoning_policy"])
        return prepared_select(*args, **kwargs)

    monkeypatch.setattr(PreparedView, "select", choosing)
    original = engine._session_db.get_messages
    observe = engine._observe
    armed = True
    latest = []

    def update():
        nonlocal armed
        armed = False
        if change == "away_back":
            bind_codex(engine, provider="openai")
        bind_codex(engine)
        assert select(engine, conversation, wire) is not None
        latest.append(engine.get_status())

    def reader(session_id, **kwargs):
        if armed and kwargs["include_inactive"] is (boundary == "observation"):
            update()
            if raises:
                raise OSError("old read private details")
            return copy.deepcopy(rows) + ([{}] if boundary == "observation" else [])
        return original(session_id, **kwargs)

    def observed(*args, **kwargs):
        observe(*args, **kwargs)
        if armed:
            update()

    if boundary == "observe_return":
        monkeypatch.setattr(engine, "_observe", observed)
    else:
        engine._session_db.get_messages = reader
    assert select(engine, conversation, wire) is None
    assert engine.get_status() == latest[0]
    assert len(selections) == 1  # Only the new revision reaches selection/measurement.


def publish_session(engine, home, rows, session_id, summary):
    """Identical row identities across sessions deliberately cannot identify scope."""
    from hermes_ariadne.kernel import MemoryKernel
    from hermes_ariadne.prepared import publish_prepared
    from hermes_ariadne.sources import Projection
    from hermes_ariadne.store import canonical_json

    engine._store.ingest(session_id, rows, origin="hermes_db")
    projection = Projection(
        engine._store,
        session_id,
        [hashlib.sha256(canonical_json(row)).hexdigest() for row in rows[:2]],
    )
    kernel = MemoryKernel(projection, home / f"{session_id}.json", node_bytes=128)
    kernel.build(lambda _: summary)
    publish_prepared(kernel, home=home)


def test_real_host_reset_only_rebind_loads_new_session(tmp_path, monkeypatch):
    from run_agent import AIAgent

    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "unrelated-profile"))
    engine.update_from_response({"prompt_tokens": 99, "total_tokens": 101})
    agent = object.__new__(AIAgent)
    agent.context_compressor = engine
    agent._session_db = engine._session_db
    agent.session_id = "second"
    agent.reset_session_state()  # The actual /new, /resume, /branch reset-only path.
    assert engine._session_id == "second"
    selected = select(engine, conversation, wire)
    assert selected is not None
    assert "second session summary" in selected[1]["content"]
    assert engine._home == home
    assert engine.last_prompt_tokens == engine.last_total_tokens == 0
    found = json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))
    assert found["ok"] and found["results"]
    assert not (tmp_path / "unrelated-profile").exists()


def test_reentrant_archive_read_discards_old_scan_before_ingestion(tmp_path):
    engine, home, rows, _, _, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    db = engine._session_db
    marker = {"message_uid": "only-first", "role": "user", "content": "first-only marker"}
    armed = True

    def reader(session_id, **kwargs):
        nonlocal armed
        if armed and kwargs.get("include_inactive"):
            armed = False
            assert session_id == "chosen"
            engine.bind_session_state(db, "second")
            return [copy.deepcopy(marker)]
        return copy.deepcopy(rows)

    db.get_messages = reader
    result = json.loads(engine.handle_tool_call("ariadne_find", {"query": "first-only"}))
    assert not engine._store.search("first-only", session_id="second")
    assert not engine._store.search("first-only", session_id="chosen")
    from hermes_ariadne.retrieval import TRUST_LABEL

    assert result == {"trust": TRUST_LABEL, "ok": False, "error": "lifecycle_changed"}
    assert engine.get_status()["selection"]["status"] == "ready"
    fresh = json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))
    assert fresh["ok"] and fresh["results"]


def test_reentrant_active_read_cannot_return_new_sessions_summary(tmp_path):
    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    db = engine._session_db
    armed = True

    def reader(session_id, **kwargs):
        nonlocal armed
        if armed and kwargs.get("include_inactive") is False:
            armed = False
            assert session_id == "chosen"
            engine.bind_session_state(db, "second")
            # Publish a new lifetime's useful diagnostic before old selection resumes.
            assert select(engine, conversation, wire) is not None
        return copy.deepcopy(rows)

    db.get_messages = reader
    assert select(engine, conversation, wire) is None
    assert engine.get_status()["selection"]["status"] == "selected"
    fresh = select(engine, conversation, wire)
    assert "second session summary" in fresh[1]["content"]


def test_outer_start_cannot_attach_over_nested_new_lifetime(tmp_path):
    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    db = engine._session_db
    armed = True

    def reader(session_id, **kwargs):
        nonlocal armed
        if armed:
            armed = False
            assert session_id == "chosen"
            engine.bind_session_state(db, "second")
            assert select(engine, conversation, wire) is not None
        return copy.deepcopy(rows)

    db.get_messages = reader
    engine.on_session_start("chosen", hermes_home=home)
    assert engine._session_id == "second"
    assert engine._prepared.projection.session_id == "second"
    assert engine.get_status()["selection"]["status"] == "selected"
    assert "second session summary" in select(engine, conversation, wire)[1]["content"]


def test_outer_end_cannot_clear_nested_same_session_lifetime(tmp_path):
    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    armed = True

    def reader(session_id, **kwargs):
        nonlocal armed
        if armed:
            armed = False
            engine.on_session_start("chosen", hermes_home=home)
            assert select(engine, conversation, wire) is not None
        return copy.deepcopy(rows)

    engine._session_db.get_messages = reader
    engine.on_session_end("chosen", [])
    assert engine._home == home
    assert engine._prepared is not None
    assert engine.get_status()["selection"]["status"] == "selected"
    assert select(engine, conversation, wire) is not None


@pytest.mark.parametrize("boundary", ["reset", "end"])
def test_detached_lifetime_does_not_observe_until_rebind(tmp_path, monkeypatch, boundary):
    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    if boundary == "reset":
        engine.on_session_reset()
    else:
        engine.on_session_end("chosen", [])
    reads = []
    db = engine._session_db

    def reader(session_id, **kwargs):
        reads.append(session_id)
        return copy.deepcopy(rows)

    db.get_messages = reader
    engine.on_turn_complete([{"role": "user", "content": "after detach"}])
    assert reads == []
    assert select(engine, conversation, wire) is None
    assert (
        json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))["error"]
        == "uninitialized"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
    engine.bind_session_state(db, "chosen")
    assert reads == ["chosen"]
    assert engine._home == home
    assert select(engine, conversation, wire) is not None
    assert not (tmp_path / "other-profile" / "ariadne").exists()


def test_first_bind_and_static_discovery_do_not_read_or_create_archive(tmp_path, monkeypatch):
    from hermes_ariadne import store

    def unexpected(*args, **kwargs):
        pytest.fail("binding/discovery performed archive IO before first start")

    monkeypatch.setattr(store, "EventStore", unexpected)
    engine = live_class()()
    db = SimpleNamespace(get_messages=unexpected)
    engine.bind_session_state(db, "first")
    engine.bind_session_state(db, "second")
    assert len(engine.get_tool_schemas()) == 2
    assert engine._home is engine._profile_home is engine._store is None
    assert engine.select_context([], conversation_messages=[], budget_tokens=32000) is None
    assert not (Path(os.environ["HERMES_HOME"]) / "ariadne").exists()


def test_identical_binding_freezes_absence_and_loads_once(tmp_path, monkeypatch):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    engine, home, _, conversation, wire, kernel = engine_fixture(tmp_path, publish=False)
    publish_prepared(kernel, home=home)
    loads = []
    original = PreparedView.load

    def load(store, session_id, selected_home):
        loads.append((session_id, selected_home))
        return original(store, session_id, selected_home)

    monkeypatch.setattr(PreparedView, "load", load)
    db = engine._session_db
    engine.bind_session_state(db, "chosen")
    engine.update_model("fixture", 32000)
    assert select(engine, conversation, wire) is None
    assert engine._prepared is None
    assert json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))["ok"]
    assert loads == []
    engine.on_session_reset()
    engine.bind_session_state(db, "chosen")
    frozen = engine._prepared
    assert frozen is not None
    assert loads == [("chosen", home)]
    engine.bind_session_state(db, "chosen")
    assert select(engine, conversation, wire) is not None
    assert engine._prepared is frozen
    assert loads == [("chosen", home)]


def test_explicit_start_remembers_home_and_can_replace_profile(tmp_path, monkeypatch):
    engine, home, _, conversation, wire, _ = engine_fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient-profile"))
    engine.on_session_start("chosen")
    assert engine._home == home
    assert select(engine, conversation, wire) is not None
    replacement = tmp_path / "explicit-profile"
    engine.on_session_start("chosen", hermes_home=replacement)
    assert engine._home == engine._profile_home == replacement
    assert engine._prepared is None
    assert engine._store.path == replacement / "ariadne" / "events.sqlite3"
    engine.on_session_reset()
    engine.bind_session_state(engine._session_db, "chosen")
    assert engine._home == replacement
    assert not (tmp_path / "ambient-profile" / "ariadne").exists()


def test_clone_has_fresh_attachment_and_lock(tmp_path):
    engine, _, _, _, _, _ = engine_fixture(tmp_path)
    clone = engine.clone_for_agent()
    assert clone._lock is not engine._lock
    assert clone._profile_home is clone._home is clone._prepared is clone._store is None
    assert clone._session_db is clone._session_id is None
    assert clone._generation == 0
    clone.bind_session_state(SimpleNamespace(), "clone")
    assert clone._home is None
    assert engine._session_id == "chosen"


@pytest.mark.parametrize(
    "read_kind", ["archive_rows", "archive_error", "active_rows", "active_error"]
)
def test_event_barrier_fences_old_reads_and_keeps_new_lifetime_diagnostics(tmp_path, read_kind):
    import threading

    from hermes_ariadne.retrieval import TRUST_LABEL

    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    db = engine._session_db
    entered, release = threading.Event(), threading.Event()
    marker = {"message_uid": "old-thread", "role": "user", "content": "old-thread marker"}
    output, failures = [], []

    def reader(session_id, **kwargs):
        archive = read_kind.startswith("archive")
        if session_id == "chosen" and kwargs["include_inactive"] is archive:
            entered.set()
            assert release.wait(10), "test did not release old read"
            if read_kind.endswith("error"):
                raise OSError("old lifetime private error")
            return [copy.deepcopy(marker)] if archive else copy.deepcopy(rows)
        if session_id == "second" and kwargs["include_inactive"]:
            return copy.deepcopy(rows) + [{}]  # Preserve new scan's skipped count, too.
        return copy.deepcopy(rows)

    def old_request():
        try:
            if read_kind.startswith("archive"):
                output.append(
                    json.loads(engine.handle_tool_call("ariadne_find", {"query": "old-thread"}))
                )
            else:
                output.append(select(engine, conversation, wire))
        except BaseException as exc:
            failures.append(exc)

    db.get_messages = reader
    worker = threading.Thread(target=old_request, daemon=True)
    worker.start()
    try:
        assert entered.wait(10), "old read did not begin"
        # Must complete while the old callback is blocked: no lock across host IO.
        engine.bind_session_state(db, "second")
        assert select(engine, conversation, wire) is not None
        before = engine.get_status()
        assert before["archive_skipped_last_scan"] == 1
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive(), "old request did not finish"
    assert not failures
    if read_kind.startswith("archive"):
        assert output == [{"trust": TRUST_LABEL, "ok": False, "error": "lifecycle_changed"}]
    else:
        assert output == [None]
    assert engine.get_status() == before
    assert not engine._store.search("old-thread", session_id="second")
    assert not engine._store.search("old-thread", session_id="chosen")
    fresh = json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))
    assert fresh["ok"] and fresh["results"]


def test_loaded_view_cannot_publish_over_nested_start(tmp_path, monkeypatch):
    from hermes_ariadne.prepared import PreparedView

    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    original = PreparedView.load
    armed = True

    def load(store, session_id, selected_home):
        nonlocal armed
        value = original(store, session_id, selected_home)
        if armed:
            armed = False
            engine.bind_session_state(engine._session_db, "second")
            assert select(engine, conversation, wire) is not None
        return value

    monkeypatch.setattr(PreparedView, "load", load)
    engine.on_session_start("chosen", hermes_home=home)
    assert engine._prepared.projection.session_id == "second"
    assert engine.get_status()["selection"]["status"] == "selected"


def test_tool_result_is_fenced_after_retrieval_not_only_after_observation(tmp_path, monkeypatch):
    from hermes_ariadne.retrieval import TRUST_LABEL

    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    original = engine._store.search

    def search(*args, **kwargs):
        result = original(*args, **kwargs)
        engine.bind_session_state(engine._session_db, "second")
        assert select(engine, conversation, wire) is not None
        return result

    monkeypatch.setattr(engine._store, "search", search)
    result = json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))
    assert result == {"trust": TRUST_LABEL, "ok": False, "error": "lifecycle_changed"}
    assert engine.get_status()["selection"]["status"] == "selected"


def test_selection_result_is_fenced_after_prepared_selection(tmp_path, monkeypatch):
    from hermes_ariadne.prepared import PreparedView

    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    original = PreparedView.select
    armed = True

    def prepared_select(view, *args, **kwargs):
        nonlocal armed
        result = original(view, *args, **kwargs)
        if armed:
            armed = False
            engine.bind_session_state(engine._session_db, "second")
            assert select(engine, conversation, wire) is not None
        return result

    monkeypatch.setattr(PreparedView, "select", prepared_select)
    assert select(engine, conversation, wire) is None
    assert engine.get_status()["selection"]["status"] == "selected"
    assert "second session summary" in select(engine, conversation, wire)[1]["content"]


def test_initialization_error_callback_cannot_clobber_new_diagnostic(tmp_path, monkeypatch):
    import logging

    from hermes_ariadne import store

    engine, home, rows, conversation, wire, _ = engine_fixture(tmp_path)
    publish_session(engine, home, rows, "second", "second session summary")
    original = store.EventStore
    armed = True

    def archive(path):
        nonlocal armed
        if armed:
            armed = False
            raise OSError("private initialization failure")
        return original(path)

    class RebindOnError(logging.Handler):
        def emit(self, record):
            engine.bind_session_state(engine._session_db, "second")
            assert select(engine, conversation, wire) is not None

    monkeypatch.setattr(store, "EventStore", archive)
    logger = logging.getLogger("hermes_ariadne.engine")
    handler = RebindOnError()
    logger.addHandler(handler)
    try:
        engine.on_session_start("chosen", hermes_home=home)
    finally:
        logger.removeHandler(handler)
    assert engine._session_id == "second"
    assert engine.get_status()["selection"]["status"] == "selected"
    assert engine.get_status()["archive_error"] is None


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


def test_rebinding_same_session_to_another_db_reloads_but_checks_active_source(tmp_path):
    engine, _, _, conversation, wire, _ = engine_fixture(tmp_path)
    frozen = engine._prepared
    engine.bind_session_state(SimpleNamespace(get_messages=lambda *a, **k: []), "chosen")
    assert engine._prepared is not None and engine._prepared is not frozen
    assert select(engine, conversation, wire) is None


@pytest.mark.parametrize(
    "state",
    [
        "constructed",
        "reset",
        "ended",
        "missing_view",
        "missing_view_declined",
        "archive_unavailable",
        "archive_declined",
    ],
)
def test_model_update_preserves_readiness_without_prepared_view(tmp_path, monkeypatch, state):
    engine = live_class()()
    if state.startswith("archive"):
        from hermes_ariadne import store

        def broken(*args, **kwargs):
            raise OSError("private archive failure")

        monkeypatch.setattr(store, "EventStore", broken)
    if state != "constructed":
        engine.on_session_start("chosen", hermes_home=tmp_path)
    if state == "reset":
        engine.on_session_reset()
    elif state == "ended":
        engine.on_session_end("chosen", [])
    elif state.endswith("declined"):
        assert engine.select_context([]) is None

    expected = {
        "status": (
            "not_initialized"
            if state in {"constructed", "reset", "ended"}
            else "declined"
            if state.endswith("declined")
            else "unavailable"
        ),
        "provider_fit_verified": False,
        "host_fallback": "original_request_on_decline",
    }
    if state.endswith("declined"):
        expected["reason"] = "prepared_unavailable"
    before = engine.get_status()["selection"]
    assert before == expected
    scope = engine._capture()
    assert scope.prepared is None
    engine.model_thresholds = {"fixture": 0.5}

    engine.update_model("fixture", 1000)

    assert engine.get_status()["selection"] == before
    assert engine._capture() == scope
    assert engine.context_length == 1000
    assert engine.threshold_percent == 0.5
    assert engine.threshold_tokens == 500


@pytest.mark.parametrize("initial_selection", ["ready", "selected"])
def test_model_switch_invalidates_measurement_not_frozen_source(tmp_path, initial_selection):
    engine, _, _, conversation, wire, _ = engine_fixture(tmp_path)
    if initial_selection == "selected":
        assert select(engine, conversation, wire) is not None
    assert engine.get_status()["selection"]["status"] == initial_selection
    scope = engine._capture()
    frozen = engine._prepared
    engine.model_thresholds = {"fixture": 0.5}
    engine.update_model("fixture", 1000, "", "", "", "")
    assert engine._prepared is frozen
    assert engine._capture() == scope
    assert engine.context_length == 1000
    assert engine.threshold_percent == 0.5
    assert engine.threshold_tokens == 500
    assert engine.get_status()["selection"] == {
        "status": "unmeasured",
        "provider_fit_verified": False,
        "host_fallback": "original_request_on_decline",
    }
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


@pytest.mark.parametrize("field", ["model", "base_url", "provider", "api_mode"])
@pytest.mark.parametrize("value", ["", None, 7, " "])
def test_incomplete_update_revokes_preceding_reasoning_permission(tmp_path, field, value):
    engine, _, rows, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    assert select(engine, conversation, wire) is not None
    before = copy.deepcopy((rows, conversation, wire))
    bind_codex(engine, **{field: value})
    assert engine._reasoning_policy is None
    assert select(engine, conversation, wire) is None
    assert engine.get_status()["selection"]["reason"] == "unsupported_alignment"
    assert (rows, conversation, wire) == before


@pytest.mark.parametrize(
    "fields",
    [
        {"model": "gpt-6-astra"},
        {"model": "GPT-6.1-SOL"},
        {"provider": "OpenAI-Codex"},
        {"provider": "codex"},
        {"api_mode": "responses"},
        {"base_url": "https://api.x.ai/v1"},
    ],
)
def test_foreign_bound_route_or_model_declines_without_rewriting_tail(tmp_path, fields):
    engine, _, rows, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    before = copy.deepcopy((rows, conversation, wire))
    bind_codex(engine, **fields)
    assert select(engine, conversation, wire) is None
    assert engine.get_status()["selection"]["reason"] == "unsupported_alignment"
    assert (rows, conversation, wire) == before


def test_legacy_model_only_update_disables_reasoning_not_reasoning_free_selection(tmp_path):
    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path / "reasoning")
    bind_codex(engine)
    engine.update_model("gpt-6.1-sol", 32000)
    assert engine._reasoning_policy is None and select(engine, conversation, wire) is None
    plain, _, _, conversation, wire, _ = engine_fixture(tmp_path / "plain")
    assert select(plain, conversation, wire) is not None
    plain.update_model("fixture", 32000)
    assert select(plain, conversation, wire) is not None


@pytest.mark.parametrize(
    "helper",
    [
        "classify_responses_route",
        "_classify_responses_issuer",
        "_wire_model_identity",
        "responses_endpoint_quirks",
    ],
)
@pytest.mark.parametrize("failure", ["missing", "raises"])
def test_native_route_helper_failures_revoke_permission(tmp_path, monkeypatch, helper, failure):
    from agent import codex_responses_adapter as native

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    assert select(engine, conversation, wire) is not None

    def broken(*args, **kwargs):
        raise LookupError("private lookup failure")

    if failure == "missing":
        monkeypatch.delattr(native, helper)
    else:
        monkeypatch.setattr(native, helper, broken)
    bind_codex(engine)
    assert engine._reasoning_policy is None
    assert select(engine, conversation, wire) is None
    assert "private" not in json.dumps(engine.get_status())


@pytest.mark.parametrize(
    "helper,value",
    [
        ("_classify_responses_issuer", "unknown_native_issuer"),
        ("_classify_responses_issuer", "other:https://secret.invalid"),
        ("_wire_model_identity", None),
        ("_wire_model_identity", " unnormalized "),
        ("_wire_model_identity", "\ud800"),
    ],
)
def test_unknown_native_route_results_never_create_permission(tmp_path, monkeypatch, helper, value):
    from agent import codex_responses_adapter as native

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    monkeypatch.setattr(native, helper, lambda *args, **kwargs: value)
    bind_codex(engine)
    assert engine._reasoning_policy is None
    assert select(engine, conversation, wire) is None


def test_native_profile_lookup_exception_disables_reasoning(tmp_path, monkeypatch):
    import providers

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)

    def broken(*args, **kwargs):
        raise OSError("private profile lookup")

    monkeypatch.setattr(providers, "profile_for_host", broken)
    bind_codex(engine)
    assert engine._reasoning_policy is None
    assert select(engine, conversation, wire) is None


def test_native_callback_uses_actual_home_profiles_and_xai_precedence(tmp_path):
    import hermes_constants
    import providers
    from agent.codex_responses_adapter import _classify_responses_issuer, classify_responses_route

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    callback_home = tmp_path / "callback-profile"
    plugin = callback_home / "plugins" / "model-providers" / "issue14"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\nfrom providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='issue14-home-profile', api_mode='codex_responses',\n"
        "    hostnames=('issuer-conflict.invalid', 'api.x.ai'),\n"
        "    responses_issuer_kind='synthetic_profile_issuer'))\n"
    )
    # An explicit native context overrides ambient HOME and the engine's archive
    # attachment. It is the caller's route context, never a probe registry/home.
    token = hermes_constants.set_hermes_home_override(callback_home)
    try:
        assert hermes_constants.get_hermes_home() == callback_home
        for url, expected in [
            ("https://child.issuer-conflict.invalid/v1", "synthetic_profile_issuer"),
            ("https://api.x.ai/v1", "xai_responses"),
        ]:
            profile = providers.profile_for_host(url)
            assert profile is not None and profile.name == "issue14-home-profile"
            flags = classify_responses_route(SimpleNamespace(provider="openai-codex", base_url=url))
            assert _classify_responses_issuer(base_url=url, **flags._asdict()) == expected
            bind_codex(engine, base_url=url)
            assert engine._reasoning_policy is None
            assert select(engine, conversation, wire) is None
    finally:
        hermes_constants.reset_hermes_home_override(token)
    # Native per-home registry must not leak that profile into this callback.
    url = "https://child.issuer-conflict.invalid/v1"
    assert providers.profile_for_host(url) is None
    bind_codex(engine, base_url=url)
    assert select(engine, conversation, wire) is not None


def test_bound_policy_never_inspects_or_retains_credentials(tmp_path):
    import gc
    import weakref

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)

    class NeverInspect:
        def __bool__(self):
            pytest.fail("API key inspected")

        def __str__(self) -> str:
            raise AssertionError("API key inspected")

        __repr__ = __str__

    key = NeverInspect()
    ref = weakref.ref(key)
    bind_codex(
        engine, api_key=key, base_url="https://synthetic-user:synthetic-password@proxy.invalid/v1"
    )
    del key
    gc.collect()
    assert ref() is None
    assert vars(engine._reasoning_policy) == {
        "issuer_kind": "codex_backend",
        "issuer_model": "gpt-6.1-sol",
    }
    assert select(engine, conversation, wire) is not None
    retained = repr(vars(engine))
    assert all(
        secret not in retained
        for secret in ("synthetic-user", "synthetic-password", "proxy.invalid")
    )


def test_reasoning_binding_is_agent_scoped_but_clones_start_unbound(tmp_path, monkeypatch):
    from hermes_ariadne.prepared import PreparedView

    engine, _, rows, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    assert select(engine, conversation, wire) is not None
    binding = engine._reasoning_policy
    scope = engine._capture()
    frozen = engine._prepared
    clone = engine.clone_for_agent()
    assert clone._reasoning_policy is None and clone._route_revision == 0
    clone.bind_session_state(engine._session_db, "chosen")
    clone.on_session_start("chosen", hermes_home=engine._home)
    assert select(clone, conversation, wire) is None

    def no_reload(*args, **kwargs):
        pytest.fail("model update reloaded prepared source")

    with monkeypatch.context() as patcher:
        patcher.setattr(PreparedView, "load", no_reload)
        revision = engine._route_revision
        bind_codex(engine)
        assert engine._route_revision > revision
        assert engine._capture() == scope and engine._prepared is frozen
    engine.on_session_reset()
    assert engine._reasoning_policy == binding
    engine.bind_session_state(engine._session_db, "chosen")
    assert engine._capture().generation > scope.generation
    assert select(engine, conversation, wire) is not None
    replacement = SimpleNamespace(get_messages=lambda *a, **k: copy.deepcopy(rows))
    engine.bind_session_state(replacement, "chosen")
    assert engine._reasoning_policy == binding
    assert select(engine, conversation, wire) is not None


@pytest.mark.parametrize("latest_enabled", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_native_classification_reentry_cannot_overwrite_newer_binding(
    tmp_path, monkeypatch, latest_enabled, raises
):
    from agent import codex_responses_adapter as native

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    original = native.classify_responses_route
    armed = True
    latest = []

    def classify(agent):
        nonlocal armed
        result = original(agent)
        if armed:
            armed = False
            bind_codex(
                engine,
                provider="openai-codex" if latest_enabled else "openai",
                context_length=64000,
            )
            assert (select(engine, conversation, wire) is not None) is latest_enabled
            latest.append((engine._reasoning_policy, engine.get_status()))
            if raises:
                raise OSError("private old classification failure")
        return result

    monkeypatch.setattr(native, "classify_responses_route", classify)
    bind_codex(engine, model="different-model")
    assert (engine._reasoning_policy, engine.get_status()) == latest[0]
    assert engine.context_length == 64000
    assert (select(engine, conversation, wire) is not None) is latest_enabled


@pytest.mark.parametrize("boundary", ["classification", "observation", "active", "measurement"])
@pytest.mark.parametrize("change", ["same", "away_back"])
@pytest.mark.parametrize("raises", [False, True])
def test_route_event_barrier_discards_stale_work_without_clobbering_new_status(
    tmp_path, monkeypatch, boundary, change, raises
):
    import threading

    from agent import codex_responses_adapter as native
    from agent import model_metadata

    engine, _, rows, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    entered, release = threading.Event(), threading.Event()
    output, failures = [], []
    old_thread = None
    reader = engine._session_db.get_messages
    classify = native.classify_responses_route
    measure = model_metadata.estimate_request_tokens_rough
    marker = {"message_uid": "stale-route", "role": "user", "content": "stale-route marker"}

    def barrier():
        entered.set()
        assert release.wait(10), "new model update blocked behind old callback"
        if raises:
            raise OSError("private stale route callback")

    def reading(session_id, **kwargs):
        if threading.current_thread() is old_thread and boundary in {"observation", "active"}:
            if kwargs["include_inactive"] is (boundary == "observation"):
                barrier()
                if boundary == "observation":
                    return copy.deepcopy(rows) + [marker, {}]
        return reader(session_id, **kwargs)

    def classifying(agent):
        result = classify(agent)
        if boundary == "classification" and threading.current_thread() is old_thread:
            barrier()
        return result

    def measuring(messages):
        if boundary == "measurement" and threading.current_thread() is old_thread:
            barrier()
        return measure(messages)

    def old_work():
        try:
            if boundary == "classification":
                bind_codex(engine, model="different-model")
            else:
                output.append(select(engine, conversation, wire))
        except BaseException as exc:
            failures.append(exc)

    engine._session_db.get_messages = reading
    monkeypatch.setattr(native, "classify_responses_route", classifying)
    monkeypatch.setattr(model_metadata, "estimate_request_tokens_rough", measuring)
    old_thread = threading.Thread(target=old_work, daemon=True)
    old_thread.start()
    try:
        assert entered.wait(10), "old callback never reached the barrier"
        if change == "away_back":
            bind_codex(engine, provider="openai")
        bind_codex(engine)
        assert select(engine, conversation, wire) is not None
        latest = (engine._reasoning_policy, engine.get_status())
    finally:
        release.set()
        old_thread.join(10)
    assert not old_thread.is_alive() and not failures
    assert output == ([] if boundary == "classification" else [None])
    assert (engine._reasoning_policy, engine.get_status()) == latest
    assert not engine._store.search("stale-route marker", session_id="chosen")
    assert select(engine, conversation, wire) is not None


def test_initial_route_binding_precedes_session_attachment_and_freezes_source(tmp_path):
    engine, home, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    initial = live_class()()
    bind_codex(initial)
    policy = initial._reasoning_policy
    assert policy is not None and initial._home is initial._store is None
    initial.bind_session_state(engine._session_db, "chosen")
    initial.on_session_start("chosen", hermes_home=home)
    assert initial._reasoning_policy is policy
    assert select(initial, conversation, wire) is not None


@pytest.mark.parametrize("field", ["model", "base_url", "provider", "api_mode"])
def test_route_binding_requires_exact_str_not_subclasses(tmp_path, field):
    class StringSubclass(str):
        pass

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    values = dict(
        model="gpt-6.1-sol-900k",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        api_mode="codex_responses",
    )
    bind_codex(engine, **{field: StringSubclass(values[field])})
    assert engine._reasoning_policy is None
    assert select(engine, conversation, wire) is None


@pytest.mark.parametrize("failure", ["module_import", "profile_bypass"])
def test_unavailable_native_binding_context_disables_old_permission(tmp_path, monkeypatch, failure):
    import sys

    from agent import safe_worker_policy

    engine, _, _, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    if failure == "module_import":
        monkeypatch.setitem(sys.modules, "agent.codex_responses_adapter", None)
    else:
        monkeypatch.setattr(safe_worker_policy, "safe_worker_enabled", lambda: True)
    bind_codex(engine)
    assert engine._reasoning_policy is None
    assert select(engine, conversation, wire) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("_issuer_kind", "xai_responses"),
        ("_issuer_model", "gpt-6-astra"),
    ],
)
def test_explicit_binding_never_admits_foreign_retained_stamps(tmp_path, field, value):
    engine, _, rows, conversation, wire, _ = reasoning_engine_fixture(tmp_path)
    bind_codex(engine)
    for messages, index in ((rows, 3), (conversation, 3), (wire, 4)):
        row: dict = messages[index]
        items = (
            json.loads(row["codex_reasoning_items"])
            if messages is rows
            else copy.deepcopy(row["codex_reasoning_items"])
        )
        assert isinstance(items, list)
        items[0][field] = value
        row["codex_reasoning_items"] = json.dumps(items) if messages is rows else items
    before = copy.deepcopy((rows, conversation, wire))
    assert select(engine, conversation, wire) is None
    assert engine.get_status()["selection"]["reason"] == "unsupported_alignment"
    assert (rows, conversation, wire) == before


def test_model_updates_do_not_turn_evidence_tool_reads_into_route_fences(tmp_path):
    engine, _, rows, _, _, _ = engine_fixture(tmp_path)
    armed = True

    def reader(*args, **kwargs):
        nonlocal armed
        if armed:
            armed = False
            bind_codex(engine)
        return copy.deepcopy(rows)

    engine._session_db.get_messages = reader
    result = json.loads(engine.handle_tool_call("ariadne_find", {"query": "original"}))
    assert result["ok"] and result["results"]
    assert engine.get_status()["selection"]["status"] == "unmeasured"


def test_route_revision_also_fences_reasoning_free_measurement(tmp_path, monkeypatch):
    from agent import model_metadata

    engine, _, _, conversation, wire, _ = engine_fixture(tmp_path)
    original = model_metadata.estimate_request_tokens_rough
    armed = True
    latest = []

    def measure(messages):
        nonlocal armed
        if armed:
            armed = False
            engine.update_model("fixture", 32000)
            assert select(engine, conversation, wire) is not None
            latest.append(engine.get_status())
        return original(messages)

    monkeypatch.setattr(model_metadata, "estimate_request_tokens_rough", measure)
    assert select(engine, conversation, wire) is None
    assert engine.get_status() == latest[0]
