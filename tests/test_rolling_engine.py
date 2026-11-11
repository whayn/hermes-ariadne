"""Injected-facade rolling lifecycle; no agent startup, native/model calls."""

import copy
import importlib
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest
from test_rolling_selection import POLICY, assembled, exchange, restored

from hermes_ariadne import rolling

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("HERMES_SOURCE required for ContextEngine ABC", allow_module_level=True)


def engine_class():
    assert importlib.util.find_spec("hermes_ariadne.rolling_engine") is not None, (
        "rolling engine is not implemented"
    )
    return importlib.import_module("hermes_ariadne.rolling_engine").RollingAriadneEngine


class DB:
    def __init__(self):
        self.rows = []
        self.read_hook = None

    def get_messages(self, session_id, *, include_inactive):
        rows = copy.deepcopy([r for r in self.rows if r.get("session_id") == session_id])
        if self.read_hook:
            self.read_hook(include_inactive)
        return rows if include_inactive else [r for r in rows if r.get("active", 1)]


class Facade:
    def __init__(self):
        self.calls = []
        self.hook = None

    def complete(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        request = json.loads(kwargs["messages"][-1]["content"])
        if self.hook:
            self.hook()
        return SimpleNamespace(
            text=json.dumps(
                {
                    "summary": "summary " + str(len(self.calls)),
                    "covered_prefix_sha256": request["covered_prefix_sha256"],
                }
            ),
            provider="fixture",
            model="gpt-test",
            usage=None,
        )


def setup_engine(tmp_path, monkeypatch, *, settings=None, db=None, facade=None):
    cls = engine_class()
    monkeypatch.setattr(cls, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))
    facade, db = facade or Facade(), db or DB()
    engine = cls(llm=facade, settings=settings or rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    return engine, db, facade


def complete(engine, db, tag, **meta):
    engine.on_turn_complete(
        restored(db.rows),
        usage=meta.pop("usage", {}),
        turn_id=meta.pop("turn_id", tag),
        interrupted=meta.pop("interrupted", False),
        failed=meta.pop("failed", False),
        turn_exit_reason=meta.pop("turn_exit_reason", "text_response(finish_reason=stop)"),
        **meta,
    )


def select(engine, db, tag):
    history = restored(db.rows)
    incoming = next(m for m in history if m["message_uid"] == tag + "u")
    request = assembled(history)
    before = copy.deepcopy(request)
    result = engine.select_context(
        request, conversation_messages=history, incoming_message=incoming, budget_tokens=100000
    )
    assert request == before
    if result is not None:
        assert result, "a usable selection must be a nonempty replacement request"
        assert result[:2] == request[:2]
        count = engine._pinned.frontier_count
        assert result[3:] == request[2 + count :]
        result[-1]["content"] = "mutated copy"
        assert request == before
    return result


def test_automatic_two_publications_adoption_and_restart(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A")
    complete(engine, db, "A")
    assert not facade.calls
    db.rows += exchange("B")
    complete(engine, db, "B")
    first = engine._published
    assert first.frontier_count == 4
    assert engine._pinned is None
    assert len(facade.calls) == 1
    assert facade.calls[0].keys() == {"messages", "max_tokens", "timeout", "purpose"}
    assert facade.calls[0]["max_tokens"] == 2048
    assert facade.calls[0]["timeout"] == 60
    assert facade.calls[0]["purpose"] == "ariadne-rolling-preparation"
    db.rows += exchange("C", complete=False)[:1]
    assert select(engine, db, "C") is not None
    assert engine._pinned == first
    db.rows = exchange("A") + exchange("B") + exchange("C", complete=False)
    assert select(engine, db, "C") is not None
    assert engine._pinned == first
    db.rows = exchange("A") + exchange("B") + exchange("C")
    complete(engine, db, "C")
    second = engine._published
    assert second.sequence == 2 and second.frontier_count == 8
    assert engine._pinned == first
    assert len(facade.calls) == 2
    db.rows += exchange("D", complete=False)[:1]
    assert select(engine, db, "D") is not None
    assert engine._pinned == second
    expired = json.loads(
        engine.handle_tool_call("ariadne_rolling_page", {"ref": "rolling:" + first.generation_id})
    )
    assert expired["error"] == "generation_expired"
    page = json.loads(
        engine.handle_tool_call("ariadne_rolling_page", {"ref": "rolling:" + second.generation_id})
    )
    assert page["ok"] and page["total"] == 8
    old_ref = page["results"][0]["event_ref"]
    assert json.loads(engine.handle_tool_call("ariadne_zoom", {"ref": old_ref}))["ok"]
    charges = engine.get_status()["rolling"]["logical_reservations"]
    engine.on_session_end("s", [])
    restarted, _, _ = setup_engine(tmp_path, monkeypatch, db=db, facade=facade)
    assert restarted._published == second
    assert restarted.get_status()["rolling"]["logical_reservations"] == charges == 2
    assert select(restarted, db, "D") is not None
    assert len(facade.calls) == 2


def journal_snapshot(home, settings, store):
    from hermes_ariadne.rolling_state import RollingJournal

    with RollingJournal(home, "s", settings, store=store, policy=POLICY) as journal:
        target = journal.pending_target()
        return journal.report(), target, journal.inspect(target.job_id) if target else None


def test_reentrant_route_lookup_cannot_restore_obsolete_labels(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)

    def classify(model, *args):
        if model == "older":
            engine.update_model("newer", 100000, provider="new-provider")
        return None

    monkeypatch.setattr(engine, "_bound_reasoning_policy", classify)
    engine.update_model("older", 100000, provider="old-provider")
    assert engine._reasoning_policy is None
    assert engine._route_labels == ("new-provider", "newer")


def test_full_coverage_requires_all_rows_even_when_archive_has_old_versions(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    engine._observe_scope(engine._capture())
    original = db.get_messages

    def missing(session_id, *, include_inactive):
        rows = original(session_id, include_inactive=include_inactive)
        return rows[:-1] if include_inactive else rows

    monkeypatch.setattr(db, "get_messages", missing)
    complete(engine, db, "B")
    assert not facade.calls
    assert (
        journal_snapshot(tmp_path, engine._settings, engine._store)[0]["logical_reservations"] == 0
    )


def test_discovery_clone_and_initial_bind_are_io_free(tmp_path, monkeypatch):
    facade = Facade()
    engine = engine_class()(llm=facade)
    before = sorted(tmp_path.rglob("*"))
    schemas = engine.get_tool_schemas()
    engine.bind_session_state(DB(), "s")
    clone = engine.clone_for_agent()
    assert clone._llm is facade and clone._settings == engine._settings
    assert clone._session_db is None and clone._profile_home is None
    assert clone._seen is not engine._seen
    assert {s["name"] for s in schemas} == {"ariadne_find", "ariadne_zoom", "ariadne_rolling_page"}
    assert engine.get_status()["selection"]["status"] == "not_initialized"
    assert sorted(tmp_path.rglob("*")) == before
    assert not facade.calls


@pytest.mark.parametrize(
    "metadata",
    [
        {"usage": None},
        {"interrupted": True},
        {"interrupted": None},
        {"failed": True},
        {"failed": None},
        {"turn_id": ""},
        {"turn_id": None},
        {"turn_exit_reason": "text_response(finish_reason=length)"},
        {"turn_exit_reason": "text_response(finish_reason=None)"},
    ],
)
def test_unsuccessful_callback_never_reserves(tmp_path, monkeypatch, metadata):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B", **metadata)
    assert not facade.calls
    assert (
        journal_snapshot(tmp_path, engine._settings, engine._store)[0]["logical_reservations"] == 0
    )


@pytest.mark.parametrize("missing", ["user", "tool", "assistant", "partial", "payload"])
def test_missing_persistence_skips_entire_callback(tmp_path, monkeypatch, missing):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    observed = exchange("A") + exchange("B")
    db.rows = copy.deepcopy(observed)
    if missing == "partial":
        db.rows[-1]["tool_calls"] = [{"id": "unfinished"}]
    elif missing == "payload":
        db.rows[-1]["content"] = "not the persisted response"
    else:
        index = next(i for i in range(4, 8) if db.rows[i]["role"] == missing)
        del db.rows[index]
    engine.on_turn_complete(
        restored(observed),
        usage={},
        interrupted=False,
        failed=False,
        turn_id="B",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert not facade.calls
    assert (
        journal_snapshot(tmp_path, engine._settings, engine._store)[0]["logical_reservations"] == 0
    )


@pytest.mark.parametrize("failure", ["exception", "invalid", "echo", "oversize"])
def test_failed_or_uncertain_job_keeps_prior_view_and_never_retries(tmp_path, monkeypatch, failure):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    first = engine._published
    db.rows += exchange("C", complete=False)[:1]
    assert select(engine, db, "C") is not None
    original = facade.complete

    def fail(**kwargs):
        result = original(**kwargs)
        if failure == "exception":
            raise TimeoutError("unknown dispatch outcome")
        if failure == "invalid":
            result.text = "not json"
        elif failure == "echo":
            result.text = json.dumps({"summary": "bad echo", "covered_prefix_sha256": "0" * 64})
        else:
            result.text = json.dumps(
                {
                    "summary": "x" * 4097,
                    "covered_prefix_sha256": json.loads(result.text)["covered_prefix_sha256"],
                }
            )
        return result

    monkeypatch.setattr(facade, "complete", fail)
    db.rows = exchange("A") + exchange("B") + exchange("C")
    complete(engine, db, "C")
    complete(engine, db, "C")
    complete(engine, db, "different-callback")
    assert len(facade.calls) == 2
    assert engine._published == engine._pinned == first
    report, target, job = journal_snapshot(tmp_path, engine._settings, engine._store)
    assert report["logical_reservations"] == 2
    assert report["cumulative_request_bytes"] == sum(
        len(rolling.encode_json(c["messages"], 65536)) for c in facade.calls
    )
    assert job["state"] == ("uncertain" if failure == "exception" else "failed")
    engine.on_session_end("s", [])
    restarted, _, _ = setup_engine(tmp_path, monkeypatch, db=db, facade=facade)
    assert restarted._published == first
    assert restarted.get_status()["rolling"]["logical_reservations"] == 2
    complete(restarted, db, "after-restart")
    assert len(facade.calls) == 2


@pytest.mark.parametrize("binding", [None, {}, "opaque-remote-placeholder"])
def test_unsupported_binding_never_initializes_or_spends(tmp_path, binding):
    facade = Facade()
    engine = engine_class()(llm=facade)
    engine.bind_session_state(binding, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    engine.on_turn_complete(
        [],
        usage={},
        interrupted=False,
        failed=False,
        turn_id="T",
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert engine._store is None
    assert not (tmp_path / "ariadne").exists()
    assert engine.select_context([]) is None
    assert not facade.calls


def test_reset_rebind_uses_captured_home_not_ambient_profile(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    original_store = engine._store
    engine.on_session_reset()
    other_home = tmp_path / "wrong-home"
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    engine.bind_session_state(db, "other")
    assert engine._home == tmp_path
    assert engine._session_id == "other" and engine._pinned is None
    store = engine._store
    engine.bind_session_state(db, "other")
    assert engine._store is store and store.path == original_store.path
    assert not other_home.exists()
    assert not facade.calls


def test_ready_pending_target_publishes_without_another_reservation(tmp_path, monkeypatch):
    from hermes_ariadne.rolling_state import RollingJournal

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    engine._observe_scope(engine._capture())
    target = engine._new_target(engine._capture(), db.rows, None)
    with RollingJournal(
        tmp_path, "s", engine._settings, store=engine._store, policy=POLICY
    ) as journal:
        journal.freeze_target(target)
        assert journal.reserve(target.job_id, target.request_bytes, lambda: None, lambda: None)
        journal.record_ready(
            target.job_id,
            "durable ready result",
            {"provider": "fixture", "model": "gpt-test"},
            None,
        )
    engine.on_session_end("s", [])
    restarted, _, _ = setup_engine(tmp_path, monkeypatch, db=db, facade=facade)
    assert restarted._published is None
    db.rows += exchange("C")
    complete(restarted, db, "C")
    assert restarted._published is not None
    assert restarted._published.summary_text == "durable ready result"
    assert restarted._published.frontier_count == 4
    assert restarted.get_status()["rolling"]["logical_reservations"] == 1
    assert not facade.calls


def test_frozen_planned_target_wins_over_newly_appended_exchanges(tmp_path, monkeypatch):
    from hermes_ariadne.rolling_state import RollingJournal

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    engine._observe_scope(engine._capture())
    target = engine._new_target(engine._capture(), db.rows, None)
    with RollingJournal(
        tmp_path, "s", engine._settings, store=engine._store, policy=POLICY
    ) as journal:
        journal.freeze_target(target)
    db.rows += exchange("C")
    complete(engine, db, "C")
    assert len(facade.calls) == 1
    assert engine._published.frontier_count == 4
    assert engine._published.job_id == target.job_id
    complete(engine, db, "C")
    assert len(facade.calls) == 1


@pytest.mark.parametrize(
    "change", ["source-edit", "rewind", "same-route", "away-back", "reset-rebind"]
)
def test_facade_reentry_revokes_publication_but_keeps_charge(tmp_path, monkeypatch, change):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    store = engine._store

    def change_source_or_capture():
        if change == "source-edit":
            db.rows[0]["content"] = "edited canonical source"
        elif change == "rewind":
            db.rows = []
        elif change == "reset-rebind":
            engine.on_session_reset()
            engine.bind_session_state(DB(), "other")
        else:
            if change == "away-back":
                engine.update_model("other", 100000, provider="fixture")
            engine.update_model("gpt-test", 100000, provider="fixture")

    facade.hook = change_source_or_capture
    complete(engine, db, "B")
    assert engine._published is None
    assert len(facade.calls) == 1
    report, target, job = journal_snapshot(tmp_path, engine._settings, store)
    assert report["logical_reservations"] == 1
    assert report["cumulative_request_bytes"] == len(target.request_bytes)
    assert job["state"] in {"failed", "uncertain"}
    for event, row in zip(target.event_ids, exchange("A")):
        assert store.get_event(event, session_id="s") == row
        assert store.get_event(event, session_id="other") is None
    complete(engine, db, "new-callback")
    assert len(facade.calls) == 1


@pytest.mark.parametrize("transition", ["rebind", "route"])
@pytest.mark.parametrize("read_raises", [False, True])
def test_event_barrier_host_read_never_ingests_stale_rows(
    tmp_path, monkeypatch, transition, read_raises
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    engine._observe_scope(engine._capture())
    store = engine._store
    db.rows += exchange("C", complete=False)[:1]
    entered, release = Event(), Event()

    def barrier(inactive):
        if inactive:
            entered.set()
            assert release.wait(5), "test barrier timed out"
            if read_raises:
                raise OSError("stale read failure")

    db.read_hook = barrier
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(select, engine, db, "C")
        try:
            assert entered.wait(5), "host read was not reached"
            if transition == "rebind":
                engine.on_session_reset()
                engine.bind_session_state(DB(), "other")
            else:
                engine.update_model("gpt-test", 100000, provider="fixture")
            status = copy.deepcopy(engine.get_status())
        finally:
            release.set()
        assert future.result(timeout=5) is None
    assert engine.get_status() == status
    event = rolling.sha256(
        __import__("hermes_ariadne.store", fromlist=["canonical_json"]).canonical_json(db.rows[-1])
    )
    assert store.get_event(event, session_id="s") is None
    assert store.get_event(event, session_id="other") is None
    assert not facade.calls


def test_source_edit_after_ready_commit_still_refuses_publication(tmp_path, monkeypatch):
    from hermes_ariadne.rolling_state import RollingJournal

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    original = RollingJournal.record_ready

    def edit_after_ready(journal, *args):
        original(journal, *args)
        db.rows[0]["content"] = "changed before publication"

    monkeypatch.setattr(RollingJournal, "record_ready", edit_after_ready)
    complete(engine, db, "B")
    assert len(facade.calls) == 1
    assert engine._published is None
    report, target, job = journal_snapshot(tmp_path, engine._settings, engine._store)
    assert report["logical_reservations"] == 1 and job["state"] == "ready"


def test_explicit_no_view_pin_survives_publication_until_next_incoming(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B", complete=False)[:1]
    assert select(engine, db, "B") is None
    assert engine._anchor[0] == "Bu" and engine._pinned is None
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    first = engine._published
    db.rows = exchange("A") + exchange("B", complete=False)
    assert select(engine, db, "B") is None
    assert engine._pinned is None
    db.rows = exchange("A") + exchange("B") + exchange("C", complete=False)[:1]
    assert select(engine, db, "C") is not None
    assert engine._pinned == first
    db.rows[-1]["content"] = "edited incoming payload with same uid"
    assert select(engine, db, "C") is None
    assert engine._pinned == first and engine._anchor[0] == "Cu"
    assert len(facade.calls) == 1


@pytest.mark.parametrize(
    "settings, expected",
    [
        (rolling.RollingSettings(trigger_bytes=100000), 0),
        (rolling.RollingSettings(trigger_bytes=1, max_request_bytes=100), 0),
        (rolling.RollingSettings(trigger_bytes=1, max_total_request_bytes=1), 0),
        (rolling.RollingSettings(trigger_bytes=1, max_delta_exchanges=1), 1),
    ],
)
def test_bounded_earliest_delta_and_logical_limits(tmp_path, monkeypatch, settings, expected):
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)
    db.rows = exchange("A") + exchange("B") + exchange("C")
    complete(engine, db, "C")
    assert len(facade.calls) == expected
    report, target, job = journal_snapshot(tmp_path, settings, engine._store)
    assert report["logical_reservations"] == expected
    if expected:
        assert engine._published.frontier_count == 4


def test_exhaustion_preserves_previous_view_and_never_resets_campaign(tmp_path, monkeypatch):
    settings = rolling.RollingSettings(trigger_bytes=1, max_logical_jobs=1)
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    first = engine._published
    db.rows += exchange("C", complete=False)[:1]
    assert select(engine, db, "C") is not None
    db.rows = exchange("A") + exchange("B") + exchange("C")
    complete(engine, db, "C")
    assert engine._published == engine._pinned == first
    assert len(facade.calls) == 1
    assert journal_snapshot(tmp_path, settings, engine._store)[0]["logical_reservations"] == 1


@pytest.mark.parametrize("budget", [0, 1, None])
def test_budget_decline_is_none_without_input_mutation_or_model_work(tmp_path, monkeypatch, budget):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    db.rows += exchange("C", complete=False)[:1]
    history = restored(db.rows)
    request = assembled(history)
    before = copy.deepcopy(request)
    assert (
        engine.select_context(
            request,
            conversation_messages=history,
            incoming_message=history[-1],
            budget_tokens=budget,
        )
        is None
    )
    assert request == before and len(facade.calls) == 1
    assert engine.get_status()["selection"]["provider_fit_verified"] is False


def test_actual_completion_caller_clones_raw_messages_and_stop_reason(tmp_path, monkeypatch):
    import ast
    from pathlib import Path

    source = Path(os.environ["HERMES_SOURCE"]) / "agent"
    final = ast.parse((source / "turn_final_response.py").read_text())
    assignments = [
        n
        for n in ast.walk(final)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_turn_exit_reason" for t in n.targets)
        and isinstance(n.value, ast.JoinedStr)
    ]
    reasons = [
        eval(
            compile(ast.Expression(n.value), "native-stop-reason", "eval"),
            {"finish_reason": "stop"},
        )
        for n in assignments
        if "finish_reason" in ast.unparse(n.value)
    ]
    assert "text_response(finish_reason=stop)" in reasons
    tree = ast.parse((source / "conversation_loop.py").read_text())
    notify = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_notify_context_engine_turn_complete"
    )
    notify.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            notify,
        ],
        type_ignores=[],
    )
    namespace = {
        "_engine_overrides_hook": lambda *a: True,
        "_clone_message_for_send": copy.deepcopy,
    }
    exec(compile(ast.fix_missing_locations(module), "native-completion-caller", "exec"), namespace)
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    messages = restored(db.rows)
    before = copy.deepcopy(messages)
    received = []
    on_turn_complete = engine.on_turn_complete

    def mutating_callback(callback_messages, *args, **kwargs):
        # Spy that damages whatever it receives: only a real host clone keeps
        # the caller's transcript intact.
        result = on_turn_complete(callback_messages, *args, **kwargs)
        received.append(callback_messages)
        for message in callback_messages:
            message["content"] = "callback mutated"
        callback_messages.append({"role": "user", "content": "callback appended"})
        return result

    monkeypatch.setattr(engine, "on_turn_complete", mutating_callback)
    namespace["_notify_context_engine_turn_complete"](
        SimpleNamespace(context_compressor=engine, session_id="s"),
        messages,
        usage={},
        logger=SimpleNamespace(
            warning=lambda *a, **k: pytest.fail("host swallowed callback failure")
        ),
        interrupted=False,
        failed=False,
        turn_id="B",
        turn_exit_reason=reasons[0],
    )
    assert len(facade.calls) == 1 and messages == before
    assert len(received) == 1 and received[0] is not messages


def test_duplicate_turn_id_after_restart_cannot_charge_another_delta(tmp_path, monkeypatch):
    settings = rolling.RollingSettings(trigger_bytes=1, max_delta_exchanges=1)
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)
    db.rows = exchange("A") + exchange("B") + exchange("C")
    complete(engine, db, "C", turn_id="unique-turn-id")
    assert engine._published.frontier_count == 4
    engine.on_session_end("s", [])
    restarted, _, _ = setup_engine(tmp_path, monkeypatch, settings=settings, db=db, facade=facade)
    db.rows += exchange("D", complete=False)[:1]
    assert select(restarted, db, "D") is not None
    db.rows = exchange("A") + exchange("B") + exchange("C") + exchange("D")
    complete(restarted, db, "D", turn_id="unique-turn-id")
    assert len(facade.calls) == 1
    assert restarted._published.frontier_count == 4
    assert restarted.get_status()["rolling"]["logical_reservations"] == 1


def test_session_reset_clears_host_token_tracking_not_durable_charges(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    engine.update_from_response({"prompt_tokens": 123, "completion_tokens": 4, "total_tokens": 127})
    engine.on_session_reset()
    assert (
        engine.last_prompt_tokens == engine.last_completion_tokens == engine.last_total_tokens == 0
    )
    engine.bind_session_state(db, "s")
    assert engine.get_status()["rolling"]["logical_reservations"] == 1
    assert len(facade.calls) == 1


def test_facade_event_barrier_allows_reset_without_publication_or_deadlock(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    store = engine._store
    entered, release = Event(), Event()

    def barrier():
        entered.set()
        assert release.wait(5), "facade barrier timed out"

    facade.hook = barrier
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(complete, engine, db, "B")
        try:
            assert entered.wait(5), "facade was not reached"
            engine.on_session_reset()
            engine.bind_session_state(DB(), "other")
            status = copy.deepcopy(engine.get_status())
        finally:
            release.set()
        future.result(timeout=5)
    assert engine._published is None and engine.get_status() == status
    report, target, job = journal_snapshot(tmp_path, engine._settings, store)
    assert report["logical_reservations"] == 1 and job["state"] == "uncertain"
    assert len(facade.calls) == 1


@pytest.mark.parametrize("boundary", ["all-rows", "before-reserve"])
def test_callback_reentry_before_reservation_has_no_charge(tmp_path, monkeypatch, boundary):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    store = engine._store
    active_reads = 0

    def reenter(inactive):
        nonlocal active_reads
        if not inactive:
            active_reads += 1
        if (boundary == "all-rows" and inactive) or (
            boundary == "before-reserve" and active_reads == 2
        ):
            engine.on_session_reset()
            engine.bind_session_state(DB(), "other")

    db.read_hook = reenter
    complete(engine, db, "B")
    report, target, job = journal_snapshot(tmp_path, engine._settings, store)
    assert report["logical_reservations"] == 0
    assert not facade.calls and engine._published is None
    from hermes_ariadne.store import canonical_json

    for row in exchange("A") + exchange("B"):
        event = rolling.sha256(canonical_json(row))
        assert store.get_event(event, session_id="other") is None
        if boundary == "all-rows":
            assert store.get_event(event, session_id="s") is None


def test_reported_usage_is_unverified_and_publication_waits_for_adoption(tmp_path, monkeypatch):
    from hermes_ariadne.rolling_state import RollingJournal

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    original = facade.complete

    def reported(**kwargs):
        result = original(**kwargs)
        result.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        return result

    monkeypatch.setattr(facade, "complete", reported)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    first = engine._published
    db.rows += exchange("C")
    complete(engine, db, "C")
    assert engine._published == first and len(facade.calls) == 1
    status = engine.get_status()["rolling"]
    assert status["usage_verified"] is status["physical_send_cost_verified"] is False
    with RollingJournal(
        tmp_path, "s", engine._settings, store=engine._store, policy=POLICY
    ) as journal:
        usage = json.loads(journal.inspect(first.job_id)["usage_observed"])
        assert usage == {
            "kind": "reported_or_defaulted",
            "value": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def test_route_change_during_measurement_cannot_return_or_publish_old_diagnostics(
    tmp_path, monkeypatch
):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    db.rows += exchange("C", complete=False)[:1]
    assert select(engine, db, "C") is not None
    saved = []

    def measure(messages):
        engine.update_model("gpt-test", 100000, provider="fixture")
        saved.append(copy.deepcopy(engine.get_status()))
        return 1

    monkeypatch.setattr(engine, "_measure", measure)
    assert select(engine, db, "C") is None
    assert engine.get_status() == saved[0]
    assert len(facade.calls) == 1


@pytest.mark.parametrize("restart_pending", [False, True])
def test_delayed_target_records_actual_charging_callback_across_restarts(
    tmp_path, monkeypatch, restart_pending
):
    from hermes_ariadne.rolling_state import RollingJournal

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    original, reads = db.get_messages, 0

    def transient(session_id, *, include_inactive):
        nonlocal reads
        if not include_inactive:
            reads += 1
            if reads == 2:
                raise OSError("pre-reservation transient read failure")
        return original(session_id, include_inactive=include_inactive)

    monkeypatch.setattr(db, "get_messages", transient)
    complete(engine, db, "B")
    report, frozen, job = journal_snapshot(tmp_path, engine._settings, engine._store)
    assert frozen is not None and job["state"] == "planned"
    assert not facade.calls and report["logical_reservations"] == 0
    monkeypatch.setattr(db, "get_messages", original)
    if restart_pending:
        engine.on_session_end("s", [])
        engine, _, _ = setup_engine(tmp_path, monkeypatch, db=db, facade=facade)
    db.rows += exchange("C")
    complete(engine, db, "C")
    first = engine._published
    assert first.job_id == frozen.job_id
    assert json.loads(json.loads(frozen.request_bytes)[-1]["content"])["completed_turn_id"] == "B"
    engine.on_session_end("s", [])
    restarted, _, _ = setup_engine(tmp_path, monkeypatch, db=db, facade=facade)
    db.rows += exchange("D", complete=False)[:1]
    assert select(restarted, db, "D") is not None
    db.rows = exchange("A") + exchange("B") + exchange("C") + exchange("D")
    complete(restarted, db, "D", turn_id="C")
    assert len(facade.calls) == restarted.get_status()["rolling"]["logical_reservations"] == 1
    assert restarted._published == first
    with RollingJournal(
        tmp_path, "s", restarted._settings, store=restarted._store, policy=POLICY
    ) as journal:
        assert journal.inspect(frozen.job_id)["charging_callback_id"] == "C"
        assert journal.charged_callback_ids() == {"C"}
    # A genuinely new completion can advance after adoption, with no frozen identity rewrite.
    complete(restarted, db, "D")
    second = restarted._published
    assert second.sequence == 2 and len(facade.calls) == 2
    restarted.on_session_end("s", [])
    again, _, _ = setup_engine(tmp_path, monkeypatch, db=db, facade=facade)
    db.rows += exchange("E", complete=False)[:1]
    assert select(again, db, "E") is not None
    db.rows = exchange("A") + exchange("B") + exchange("C") + exchange("D") + exchange("E")
    complete(again, db, "E", turn_id="C")
    complete(again, db, "E", turn_id="D")
    assert len(facade.calls) == again.get_status()["rolling"]["logical_reservations"] == 2
    assert again._published == second
    with RollingJournal(
        tmp_path, "s", again._settings, store=again._store, policy=POLICY
    ) as journal:
        assert journal.charged_callback_ids() == {"C", "D"}


@pytest.mark.parametrize("change", ["remove-final", "edit-final", "rewind-to-prefix"])
@pytest.mark.parametrize("boundary", ["facade", "ready"])
@pytest.mark.parametrize("prior", [False, True])
def test_original_completed_tail_revalidated_before_ready_and_publication(
    tmp_path, monkeypatch, change, boundary, prior
):
    from hermes_ariadne.rolling_state import RollingJournal

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    previous = None
    if prior:
        complete(engine, db, "B")
        previous = engine._published
        db.rows += exchange("C", complete=False)[:1]
        assert select(engine, db, "C") is not None
        db.rows = exchange("A") + exchange("B") + exchange("C")
    original_rows = copy.deepcopy(db.rows)

    def mutate():
        if change == "remove-final":
            db.rows.pop()
        elif change == "edit-final":
            db.rows[-1]["content"] = "edited completed canonical tail"
        else:
            db.rows = db.rows[:-4]

    if boundary == "facade":
        facade.hook = mutate
    else:
        original = RollingJournal.record_ready

        def after_ready(journal, *args):
            original(journal, *args)
            mutate()

        monkeypatch.setattr(RollingJournal, "record_ready", after_ready)
    complete(engine, db, "C" if prior else "B")
    report, target, job = journal_snapshot(tmp_path, engine._settings, engine._store)
    assert len(facade.calls) == report["logical_reservations"] == (2 if prior else 1)
    assert engine._published == previous
    assert engine._pinned == previous
    assert job["state"] == ("failed" if boundary == "facade" else "ready")
    if boundary == "facade":
        assert job["error_code"] == "source_changed"
    from hermes_ariadne.store import canonical_json

    for row in original_rows:
        assert engine._store.get_event(rolling.sha256(canonical_json(row)), session_id="s") == row


def test_append_only_during_facade_keeps_original_complete_coverage(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    facade.hook = lambda: db.rows.extend(exchange("C", complete=False)[:1])
    complete(engine, db, "B")
    assert engine._published is not None and engine._published.frontier_count == 4
    assert len(facade.calls) == engine.get_status()["rolling"]["logical_reservations"] == 1


def native_result_types():
    """Execute only the installed host's pure dataclasses, never its runtime."""
    import ast
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Optional

    tree = ast.parse((Path(os.environ["HERMES_SOURCE"]) / "agent/plugin_llm.py").read_text())
    names = {"PluginLlmUsage", "PluginLlmCompleteResult"}
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in classes} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *classes,
        ],
        type_ignores=[],
    )
    namespace = {"dataclass": dataclass, "field": field, "Optional": Optional, "Any": Any}
    exec(compile(ast.fix_missing_locations(module), "native-result-types", "exec"), namespace)
    return namespace["PluginLlmUsage"], namespace["PluginLlmCompleteResult"]


@pytest.mark.parametrize("usage_kind", ["default", "reported", "invalid", "absent"])
def test_native_usage_dataclass_publishes_bounded_unverified_observation(
    tmp_path, monkeypatch, usage_kind
):
    from hermes_ariadne.rolling_state import RollingJournal

    usage_type, result_type = native_result_types()
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    original = facade.complete
    expected = dict(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=None,
    )
    usage = usage_type()
    if usage_kind == "reported":
        expected.update(input_tokens=12, output_tokens=3, total_tokens=15, cost_usd=0.25)
        usage = usage_type(**expected)
    elif usage_kind == "invalid":
        usage = usage_type(
            input_tokens=object(),
            output_tokens=True,
            total_tokens=-1,
            cache_read_tokens=2**100,
            cache_write_tokens=[],
            cost_usd=float("nan"),
        )
        expected = None
    elif usage_kind == "absent":
        usage, expected = None, None
    if usage is not None:
        usage.private = {"secret": object()}  # Must never serialize unknown/nested fields.

    def native(**kwargs):
        response = original(**kwargs)
        return result_type(
            text=response.text,
            provider=response.provider,
            model=response.model,
            agent_id="fixture",
            usage=usage,
            audit={"secret": object()},
        )

    monkeypatch.setattr(facade, "complete", native)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    view = engine._published
    assert view is not None
    assert len(facade.calls) == engine.get_status()["rolling"]["logical_reservations"] == 1
    assert engine.get_status()["rolling"]["usage_verified"] is False
    with RollingJournal(
        tmp_path, "s", engine._settings, store=engine._store, policy=POLICY
    ) as journal:
        job = journal.inspect(view.job_id)
        assert job["state"] == "published"
        observed = job["usage_observed"]
        assert (json.loads(observed) if observed else None) == (
            {"kind": "reported_or_defaulted", "value": expected} if expected else None
        )
    engine.on_session_end("s", [])
    restarted, _, _ = setup_engine(tmp_path, monkeypatch, db=db, facade=facade)
    assert restarted._published == view and len(facade.calls) == 1


def test_loss_of_completed_tail_before_reserve_cannot_spend_on_small_old_prefix(
    tmp_path, monkeypatch
):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    reads = 0
    original = db.get_messages

    def lost_final_row(session_id, *, include_inactive):
        nonlocal reads
        rows = original(session_id, include_inactive=include_inactive)
        if not include_inactive:
            reads += 1
            if reads >= 2:
                return rows[:-1]
        return rows

    monkeypatch.setattr(db, "get_messages", lost_final_row)
    complete(engine, db, "B")
    assert not facade.calls
    assert (
        journal_snapshot(tmp_path, engine._settings, engine._store)[0]["logical_reservations"] == 0
    )


def private_tree_snapshot(root):
    """Capture bytes and names without following an unsafe directory alias."""
    import stat

    snapshot = {}
    for path in root.rglob("*"):
        info = path.lstat()
        value = (
            os.readlink(path)
            if path.is_symlink()
            else (path.read_bytes() if stat.S_ISREG(info.st_mode) else None)
        )
        snapshot[str(path.relative_to(root))] = (info.st_mode, info.st_nlink, value)
    return snapshot


def unsafe_archive_fixture(tmp_path, unsafe):
    from hermes_ariadne.store import EventStore

    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    foreign = EventStore(outside / "events.sqlite3")
    foreign.ingest("foreign", exchange("FOREIGN"), origin="fixture")
    home = tmp_path / "explicit-home"
    home.mkdir(mode=0o700)
    ariadne = home / "ariadne"
    if unsafe == "directory-symlink":
        ariadne.symlink_to(outside, target_is_directory=True)
    else:
        ariadne.mkdir(mode=0o700)
        archive = ariadne / "events.sqlite3"
        if unsafe == "archive-symlink":
            archive.symlink_to(foreign.path)
        elif unsafe == "archive-hardlink":
            os.link(foreign.path, archive)
        else:
            foreign = EventStore(archive)
            foreign.ingest("foreign", exchange("FOREIGN"), origin="fixture")
            archive.chmod(0o644)
    return home, foreign


def assert_private_archive_refused(engine, facade):
    assert engine.get_status()["rolling"]["status"] == "unavailable"
    assert engine._store is None and engine._home is None
    assert not facade.calls


@pytest.mark.parametrize(
    "unsafe",
    ["archive-symlink", "directory-symlink", "archive-hardlink", "world-readable-archive"],
)
def test_unsafe_archive_admission_precedes_store_open_and_private_ingest(
    tmp_path, monkeypatch, unsafe
):
    from hermes_ariadne.store import canonical_json

    # Load the host's ABC/model-threshold dependencies before snapshotting engine I/O.
    module = importlib.import_module("hermes_ariadne.rolling_engine")
    importlib.import_module("agent.context_compressor")
    home, foreign = unsafe_archive_fixture(tmp_path, unsafe)
    db = DB()
    db.rows = exchange("PRIVACY-SENTINEL-A") + exchange("PRIVACY-SENTINEL-B")
    before = private_tree_snapshot(tmp_path)
    original, opened = module.EventStore, []

    def observe_open(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(module, "EventStore", observe_open)
    engine, _, facade = setup_engine(home, monkeypatch, db=db)
    assert private_tree_snapshot(tmp_path) == before
    assert opened == [], "unsafe archive must be refused before any SQLite/schema open"
    assert_private_archive_refused(engine, facade)
    for row in db.rows:
        assert foreign.get_event(rolling.sha256(canonical_json(row)), session_id="s") is None
    assert b"PRIVACY-SENTINEL" not in foreign.path.read_bytes()


def test_empty_unsafe_start_cannot_ingest_after_refusal_from_any_entrypoint(tmp_path, monkeypatch):
    home, foreign = unsafe_archive_fixture(tmp_path, "archive-symlink")
    before = private_tree_snapshot(tmp_path)
    engine, db, facade = setup_engine(home, monkeypatch)
    db.rows = exchange("PRIVACY-SENTINEL-AFTER", complete=False)[:1]
    assert select(engine, db, "PRIVACY-SENTINEL-AFTER") is None
    engine.handle_tool_call("ariadne_find", {"query": "PRIVACY-SENTINEL"})
    db.rows = exchange("PRIVACY-SENTINEL-A") + exchange("PRIVACY-SENTINEL-B")
    complete(engine, db, "PRIVACY-SENTINEL-B")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=home)
    assert private_tree_snapshot(tmp_path) == before
    assert_private_archive_refused(engine, facade)
    assert b"PRIVACY-SENTINEL" not in foreign.path.read_bytes()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_unsafe_archive_companion_refused_before_store_open(tmp_path, monkeypatch, suffix):
    home = tmp_path / "explicit-home"
    ariadne = home / "ariadne"
    ariadne.mkdir(parents=True, mode=0o700)
    home.chmod(0o700)
    external = tmp_path / "external-companion"
    external.write_bytes(b"external unchanged")
    external.chmod(0o600)
    (ariadne / ("events.sqlite3" + suffix)).symlink_to(external)
    before = private_tree_snapshot(tmp_path)
    db = DB()
    db.rows = exchange("PRIVACY-SENTINEL-A") + exchange("PRIVACY-SENTINEL-B")
    engine, _, facade = setup_engine(home, monkeypatch, db=db)
    assert private_tree_snapshot(tmp_path) == before
    assert_private_archive_refused(engine, facade)


@pytest.mark.parametrize("entrypoint", ["select", "completion", "tool"])
@pytest.mark.parametrize("boundary", ["before-read", "after-read"])
def test_archive_admission_rechecked_around_callback_reads(
    tmp_path, monkeypatch, entrypoint, boundary
):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    archive = engine._store.path
    db.rows = exchange("PRIVACY-SENTINEL-A") + exchange("PRIVACY-SENTINEL-B")
    if entrypoint == "select":
        db.rows += exchange("PRIVACY-SENTINEL-C", complete=False)[:1]
    original_bytes = archive.read_bytes()
    reads = []

    def revoke(inactive):
        reads.append(inactive)
        archive.chmod(0o644)

    db.read_hook = revoke
    if boundary == "before-read":
        archive.chmod(0o644)
    if entrypoint == "select":
        assert select(engine, db, "PRIVACY-SENTINEL-C") is None
    elif entrypoint == "completion":
        complete(engine, db, "PRIVACY-SENTINEL-B")
    else:
        engine.handle_tool_call("ariadne_find", {"query": "PRIVACY-SENTINEL"})
    assert archive.read_bytes() == original_bytes
    assert reads == ([] if boundary == "before-read" else [entrypoint != "completion"])
    assert_private_archive_refused(engine, facade)
    assert b"PRIVACY-SENTINEL" not in archive.read_bytes()


def test_safe_existing_archive_and_private_missing_home_admitted(tmp_path, monkeypatch):
    from hermes_ariadne.store import EventStore, canonical_json

    home = tmp_path / "new-private-home"
    engine, db, facade = setup_engine(home, monkeypatch)
    assert engine.get_status()["rolling"]["status"] == "ready"
    assert home.stat().st_mode & 0o077 == 0
    engine.on_session_end("s", [])
    archive = EventStore(home / "ariadne" / "events.sqlite3")
    archived = exchange("FOREIGN")
    archive.ingest("foreign", archived, origin="fixture")
    db.rows = exchange("A") + exchange("B")
    restarted, _, _ = setup_engine(home, monkeypatch, db=db, facade=facade)
    assert restarted.get_status()["rolling"]["status"] == "ready"
    for row in db.rows:
        assert archive.get_event(rolling.sha256(canonical_json(row)), session_id="s") == row
    for row in archived:
        assert archive.get_event(rolling.sha256(canonical_json(row)), session_id="foreign") == row
    assert not facade.calls
