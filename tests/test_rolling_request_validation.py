"""One explicit journal open is the request-local validation boundary."""

import json
import sqlite3
from dataclasses import replace

import pytest
from test_rolling_journal_validation import chain, count_reads

from hermes_ariadne import rolling, rolling_state


def test_open_load_and_revalidation_share_source_snapshot(tmp_path, monkeypatch):
    store, settings, views, *_ = chain(tmp_path, 6)
    reads = count_reads(monkeypatch, store)
    for _ in range(2):
        reads.clear()
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            view = journal.load_view()
            for _ in range(2):
                assert (
                    rolling.revalidate_view(
                        view,
                        store,
                        "s",
                        settings,
                        parent_loader=journal._parent_bytes,
                        _journal=journal,
                    )
                    == views[-1]
                )
            assert set(reads) == set(view.event_ids)
            assert set(reads.values()) == {1}
    assert journal._memo is None


@pytest.mark.parametrize("index", [0, 5])
@pytest.mark.parametrize("operation", ["load", "revalidate", "reopen"])
def test_cached_generation_still_rereads_ancestor_bytes(tmp_path, index, operation):
    store, settings, views, _, paths = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        value = json.loads(views[index].snapshot_bytes)
        value["summary_text"] = "forged"
        value["summary_sha256"] = rolling.sha256(b"forged")
        (paths[1] / (views[index].generation_id + ".json")).write_bytes(
            rolling.encode_json(value, settings.max_checkpoint_bytes)
        )
        with pytest.raises(ValueError):
            if operation == "load":
                journal.load_view()
            elif operation == "revalidate":
                rolling.revalidate_view(views[-1], store, "s", settings, _journal=journal)
            else:
                journal.close()
                rolling_state.RollingJournal(tmp_path, "s", settings, store=store)


def test_engine_selection_and_empty_page_read_sources_once(tmp_path, monkeypatch):
    from test_rolling_engine import complete, select, setup_engine
    from test_rolling_selection import exchange

    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    for tag in "ABCDEFG":
        db.rows += exchange(tag, complete=False)[:1]
        select(engine, db, tag)  # supported incoming adoption before completion
        db.rows += exchange(tag)[1:]
        complete(engine, db, tag)
    assert engine._published.sequence == 6
    db.rows += exchange("H", complete=False)[:1]
    reads = count_reads(monkeypatch, engine._store)
    assert select(engine, db, "H") is not None
    assert set(reads) == set(engine._published.event_ids)
    assert set(reads.values()) == {1}
    reads.clear()
    page = json.loads(
        engine.handle_tool_call(
            "ariadne_rolling_page",
            {
                "ref": "rolling:" + engine._pinned.generation_id,
                "offset": 1000,
            },
        )
    )
    assert page["ok"] and page["results"] == []
    assert set(reads) == set(engine._published.event_ids)
    assert set(reads.values()) == {1}
    assert len(facade.calls) == 6
    reads.clear()
    db.rows += exchange("H")[1:]
    complete(engine, db, "H")
    assert engine._published.sequence == 7 and len(facade.calls) == 7
    # Completion intentionally resolves every current host row once in addition
    # to once-per-open archive validation. Those host comparisons are not memoized.
    from collections import Counter

    from hermes_ariadne.store import canonical_json

    expected = Counter(rolling.sha256(canonical_json(row)) for row in db.rows)
    expected.update(engine._published.event_ids)
    assert reads == expected


@pytest.mark.parametrize("mutation", ["delete", "payload", "uid", "session"])
def test_next_open_observes_ancestor_archive_mutation(tmp_path, mutation):
    store, settings, views, *_ = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view() == views[-1]
    event = views[0].event_ids[0]
    with sqlite3.connect(store.path) as conn:
        if mutation == "payload":
            conn.execute(
                "UPDATE events SET payload_json=CAST(? AS BLOB) WHERE event_id=?",
                (b'{"content":"forged"}', event),
            )
        elif mutation == "delete":
            conn.execute("DELETE FROM observations WHERE event_id=?", (event,))
        elif mutation == "uid":
            row = json.loads(
                conn.execute(
                    "SELECT payload_json FROM events WHERE event_id=?", (event,)
                ).fetchone()[0]
            )
            row["message_uid"] = "foreign"
            conn.execute(
                "UPDATE events SET payload_json=CAST(? AS BLOB) WHERE event_id=?",
                (rolling.encode_json(row, settings.max_checkpoint_bytes), event),
            )
        else:
            conn.execute("UPDATE observations SET session_id='foreign' WHERE event_id=?", (event,))
    with pytest.raises(ValueError):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)


@pytest.mark.parametrize("field", ["store", "session_id", "policy", "settings"])
def test_foreign_revalidation_scope_refused(tmp_path, field):
    store, settings, views, *_ = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        args = dict(store=store, session_id="s", settings=settings, policy=None)
        args[field] = {
            "store": object(),
            "session_id": "foreign",
            "policy": object(),
            "settings": replace(settings, max_summary_bytes=settings.max_summary_bytes + 1),
        }[field]
        with pytest.raises(ValueError, match="scope"):
            rolling.revalidate_view(views[-1], _journal=journal, **args)


def test_failed_operation_drops_snapshot_and_rechecks_sources(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = journal._memo
        with pytest.raises(RuntimeError):
            with journal._operation():
                raise RuntimeError("boom")
        assert journal._memo is None and not journal._busy
        assert journal.load_view() == views[-1]
        assert journal._memo is not old
        with sqlite3.connect(store.path) as conn:
            conn.execute("DELETE FROM observations WHERE event_id=?", (views[0].event_ids[0],))
        with pytest.raises(RuntimeError):
            with journal._operation():
                raise RuntimeError("boom")
        with pytest.raises(ValueError, match="source"):
            journal.load_view()
    assert journal._memo is None


def test_hit_still_checks_guard_and_closed_journal(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 6)
    journal = rolling_state.RollingJournal(tmp_path, "s", settings, store=store)

    def revoked():
        raise ValueError("revoked")

    with pytest.raises(ValueError, match="revoked"):
        rolling.revalidate_view(views[-1], store, "s", settings, _journal=journal, guard=revoked)
    journal.close()
    with pytest.raises(ValueError, match="closed"):
        rolling.revalidate_view(views[-1], store, "s", settings, _journal=journal)


def test_memo_cannot_be_rebound_to_new_settings(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        changed = replace(settings, max_summary_bytes=settings.max_summary_bytes + 1)
        value = json.loads(views[0].snapshot_bytes)
        value["settings_digest"] = changed.digest
        data = rolling.encode_json(value, changed.max_checkpoint_bytes)
        with pytest.raises(ValueError, match="memo scope"):
            rolling.validate_generation(data, store, "s", changed, _memo=journal._memo)


def test_publication_recovery_and_successor_in_same_open(tmp_path, monkeypatch):
    from test_rolling_state import ready, target

    store, settings, views, _, _ = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        successor = target(store, settings, 7, views[-1])
        ready(journal, successor, "next")
        data = rolling.generation_bytes(successor, "next", sequence=7)
        marker = journal._mark_published

        def interrupted(view):
            raise RuntimeError("after manifest")

        monkeypatch.setattr(journal, "_mark_published", interrupted)
        with pytest.raises(RuntimeError, match="after manifest"):
            journal.publish_ready(successor.job_id, data)
        assert journal._memo is None and not journal._busy
        monkeypatch.setattr(journal, "_mark_published", marker)
        journal._reconcile()
        assert journal.load_view().sequence == 7
        assert journal.report()["logical_reservations"] == 7
        assert journal.inspect(successor.job_id)["state"] == "published"
        assert not journal.reserve(
            successor.job_id, successor.request_bytes, lambda: None, lambda: None
        )
        with pytest.raises(ValueError, match="immutable ready"):
            journal.record_ready(
                successor.job_id, "changed", {"provider": "test", "model": "test"}, None
            )


def test_forged_view_refusal_cleans_validation_state(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with pytest.raises(ValueError, match="exact snapshot"):
            rolling.revalidate_view(
                replace(views[-1], summary_text="forged"), store, "s", settings, _journal=journal
            )
        assert journal._memo is None and not journal._busy
        assert journal.load_view() == views[-1]


def test_load_reuses_open_validation(tmp_path, monkeypatch):
    store, settings, views, *_ = chain(tmp_path, 6)
    reads = count_reads(monkeypatch, store)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view() == views[-1]
        assert set(reads) == set(views[-1].event_ids)
        assert set(reads.values()) == {1}


_PREFLIGHT_REFUSALS = [
    "foreign_scope",
    "foreign_reader",
    "untrusted_view",
    "invalid_reserve_guard",
    "invalid_callback_id",
    "invalid_ready_text",
    "invalid_attribution",
    "invalid_attribution_label",
    "invalid_usage",
    "invalid_failure_reason",
    "invalid_uncertain",
]


def _refuse_preflight(refusal, journal, view, store, settings):
    if refusal == "foreign_scope":
        rolling.revalidate_view(view, store, "foreign", settings, _journal=journal)
    elif refusal == "foreign_reader":
        rolling.revalidate_view(
            view, store, "s", settings, _journal=journal, parent_loader=lambda _: b""
        )
    elif refusal == "untrusted_view":
        rolling.revalidate_view(object(), store, "s", settings, _journal=journal)
    elif refusal == "invalid_reserve_guard":
        journal.reserve(view.job_id, b"", None, None)
    elif refusal == "invalid_callback_id":
        journal.reserve(
            view.job_id, b"", lambda: None, lambda: None, charging_callback_id="x" * 4097
        )
    elif refusal.startswith("invalid_failure") or refusal == "invalid_uncertain":
        journal.record_failed_or_uncertain(
            view.job_id,
            "not_a_safe_code" if refusal == "invalid_failure_reason" else "facade_error",
            uncertain=1 if refusal == "invalid_uncertain" else False,
        )
    else:
        journal.record_ready(
            view.job_id,
            "x" * (settings.max_summary_bytes + 1) if refusal == "invalid_ready_text" else "ok",
            {}
            if refusal == "invalid_attribution"
            else {
                "provider": "x" * 1025 if refusal == "invalid_attribution_label" else "test",
                "model": "test",
            },
            object() if refusal == "invalid_usage" else None,
        )


@pytest.mark.parametrize("refusal", _PREFLIGHT_REFUSALS)
def test_preflight_refusal_discards_validation_without_changing_journal(tmp_path, refusal):
    store, settings, views, *_ = chain(tmp_path, 2)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        before = journal.report()
        job = journal.inspect(views[-1].job_id)
        old = journal._memo
        assert old is not None and old.accepted
        with pytest.raises(ValueError):
            _refuse_preflight(refusal, journal, views[-1], store, settings)
        assert journal._memo is None and not journal._busy
        assert journal.report() == before
        assert journal.inspect(views[-1].job_id) == job
        assert journal.load_view() == views[-1]
        assert journal._memo is not old


def test_scope_refusal_then_source_change_requires_fresh_validation(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 2)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with pytest.raises(ValueError, match="scope"):
            rolling.revalidate_view(views[-1], store, "foreign", settings, _journal=journal)
        # Mutation only AFTER failure: this must be a fresh snapshot, unlike the
        # explicitly supported reuse after a successful operation in the same open.
        with sqlite3.connect(store.path) as conn:
            conn.execute("DELETE FROM observations WHERE event_id=?", (views[0].event_ids[0],))
        with pytest.raises(ValueError, match="source"):
            journal.load_view()


@pytest.mark.parametrize("refusal", _PREFLIGHT_REFUSALS)
@pytest.mark.parametrize("caught", [False, True])
def test_nested_preflight_refusal_preserves_outer_operation_ownership(tmp_path, refusal, caught):
    store, settings, views, *_ = chain(tmp_path, 2)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        before = journal.report()
        old = journal._memo

        def outer():
            with journal._operation():
                if caught:
                    with pytest.raises(ValueError, match="reentrant"):
                        _refuse_preflight(refusal, journal, views[-1], store, settings)
                    assert journal._busy and journal._memo is old
                    assert journal._load_view() == views[-1]
                else:
                    _refuse_preflight(refusal, journal, views[-1], store, settings)

        if caught:
            outer()
            assert journal._memo is old
        else:
            with pytest.raises(ValueError, match="reentrant"):
                outer()
            assert journal._memo is None
        assert not journal._busy
        assert journal.report() == before
        assert journal.load_view() == views[-1]
