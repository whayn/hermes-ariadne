"""Owned scratch journal and immutable generation tracer bullets."""

import hashlib

import pytest
from test_rolling import rows

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.store import EventStore


def setup(tmp_path, **limits):
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    settings = rolling.RollingSettings(trigger_bytes=1, **limits)
    return store, settings


def target(store, settings, count, parent=None):
    source = rows(count)
    store.ingest("s", source, origin="canonical")
    return rolling.Target.create(
        source,
        session_id="s",
        settings=settings,
        policy=None,
        parent=parent,
        request_bytes=b'[{"content":"summarize","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )


def ready(journal, frozen, text="summary"):
    journal.freeze_target(frozen)
    assert journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
    journal.record_ready(frozen.job_id, text, {"provider": "test", "model": "test"}, None)


def test_two_immutable_successors_share_one_campaign(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        first = target(store, settings, 1)
        ready(journal, first)
        g1 = journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "summary", sequence=1)
        )
        second = target(store, settings, 2, g1)
        ready(journal, second, "next summary")
        g2 = journal.publish_ready(
            second.job_id, rolling.generation_bytes(second, "next summary", sequence=2)
        )
        assert g1.generation_id != g2.generation_id
        assert g2.event_ids[: len(g1.event_ids)] == g1.event_ids
        assert g2.parent_generation == g1.generation_id
        assert journal.report()["logical_reservations"] == 2
        assert journal.load_view().generation_id == g2.generation_id
        assert hashlib.sha256(g1.snapshot_bytes).hexdigest() == g1.generation_id
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view().snapshot_bytes == g2.snapshot_bytes
        assert journal.report()["logical_reservations"] == 2


def test_reserved_restart_is_charged_uncertain_not_retryable(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        assert journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.inspect(frozen.job_id)["state"] == "uncertain"
        assert journal.report()["logical_reservations"] == 1
        assert journal.report()["cumulative_request_bytes"] == len(frozen.request_bytes)
        with pytest.raises(ValueError):
            journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        with pytest.raises(ValueError):
            journal.freeze_target(target(store, settings, 2))


def test_planned_identity_survives_append_and_restart(tmp_path):
    store, settings = setup(tmp_path)
    frozen = target(store, settings, 1)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.freeze_target(frozen) == frozen
    store.ingest("s", rows(3), origin="canonical")
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.pending_target() == frozen
        assert journal.freeze_target(frozen) == frozen
        assert journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        assert not journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        assert journal.report()["logical_reservations"] == 1


@pytest.mark.parametrize("point", ["ready", "generation", "manifest"])
def test_ready_recovery_no_new_reservation(tmp_path, monkeypatch, point):
    store, settings = setup(tmp_path)
    atomic = rolling_state._atomic
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        data = rolling.generation_bytes(frozen, "summary", sequence=1)

        def crash_atomic(path, value):
            if path == journal.manifest_path:
                raise RuntimeError("before manifest")
            atomic(path, value)

        def crash_marker(_):
            raise RuntimeError("after manifest")

        if point != "ready":
            if point == "generation":
                monkeypatch.setattr(rolling_state, "_atomic", crash_atomic)
            else:
                monkeypatch.setattr(journal, "_mark_published", crash_marker)
            with pytest.raises(RuntimeError):
                journal.publish_ready(frozen.job_id, data)
        assert journal.inspect(frozen.job_id)["state"] == "ready"
    monkeypatch.setattr(rolling_state, "_atomic", atomic)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        if point == "manifest":
            assert journal.inspect(frozen.job_id)["state"] == "published"
        assert journal.publish_ready(frozen.job_id, data) == journal.load_view()
        assert journal.report()["logical_reservations"] == 1


@pytest.mark.parametrize(
    "mutation", ["extra_schema", "counter", "sequence", "empty", "summary", "orphan_generation"]
)
def test_corruption_never_resets(tmp_path, mutation):
    import sqlite3

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
        path, directory = journal.path, journal.generations
    if mutation == "empty":
        path.write_bytes(b"")
    elif mutation == "orphan_generation":
        p = directory / ("f" * 64 + ".json")
        p.write_bytes(b"{}")
        p.chmod(0o600)
    else:
        with sqlite3.connect(path) as conn:
            conn.execute(
                {
                    "extra_schema": "CREATE TABLE foreign_table(x)",
                    "counter": "UPDATE campaign SET logical_reservations=0",
                    "sequence": "UPDATE campaign SET next_sequence=1",
                    "summary": "UPDATE jobs SET validated_text='corrupt'",
                }[mutation]
            )
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)


def test_parent_checkpoint_read_once(tmp_path, monkeypatch):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        first = target(store, settings, 1)
        ready(journal, first)
        g1 = journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "summary", sequence=1)
        )
        second = target(store, settings, 2, g1)
        ready(journal, second)
        g2 = journal.publish_ready(
            second.job_id, rolling.generation_bytes(second, "summary", sequence=2)
        )
        original, reads = journal._parent_bytes, []

        def once(digest):
            assert digest not in reads, "checkpoint reread after validation"
            reads.append(digest)
            return original(digest)

        monkeypatch.setattr(journal, "_parent_bytes", once)
        assert journal.load_view() == g2
        assert reads == [g2.generation_id, g1.generation_id]


@pytest.mark.parametrize("limits", [{"max_logical_jobs": 1}, {"max_total_request_bytes": 40}])
def test_cumulative_bounds_across_successors(tmp_path, limits):
    store, settings = setup(tmp_path, **limits)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        first = journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
        successor = target(store, settings, 2, first)
        with pytest.raises(ValueError):
            journal.freeze_target(successor)
            journal.reserve(successor.job_id, successor.request_bytes, lambda: None, lambda: None)
        assert journal.report()["logical_reservations"] == 1


def test_writer_nonblocking_and_settings_cannot_reset(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with pytest.raises(ValueError, match="writer"):
            rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
    with pytest.raises(ValueError, match="settings_mismatch"):
        rolling_state.RollingJournal(
            tmp_path, "s", rolling.RollingSettings(trigger_bytes=2), store=store
        )
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.report()["logical_reservations"] == 1


@pytest.mark.parametrize("uncertain", [False, True])
def test_failed_targets_never_retry_or_replan(tmp_path, uncertain):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        for _ in range(2):
            journal.record_failed_or_uncertain(frozen.job_id, "facade_error", uncertain=uncertain)
        with pytest.raises(ValueError):
            journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        with pytest.raises(ValueError):
            journal.freeze_target(target(store, settings, 2))
        assert journal.report()["logical_reservations"] == 1


def test_read_capture_guard_and_reentry_prevent_write(tmp_path, monkeypatch):
    store, settings = setup(tmp_path)
    stale = False

    def guard():
        if stale:
            raise ValueError("capture changed")

    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, guard=guard) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        # A reserve in this open reads the request snapshot, not the store again.
        # Revoke during that real memo read to retain cache-hit lifecycle coverage.
        original = journal._memo.get_event

        def rebind(*args, **kwargs):
            nonlocal stale
            value = original(*args, **kwargs)
            stale = True
            return value

        monkeypatch.setattr(journal._memo, "get_event", rebind)
        with pytest.raises(ValueError, match="capture changed"):
            journal.reserve(frozen.job_id, frozen.request_bytes, guard, guard)
        stale = False
        assert journal.report()["logical_reservations"] == 0

        def reenter():
            journal.record_ready(frozen.job_id, "injected", {"provider": "x", "model": "x"}, None)

        with pytest.raises(ValueError, match="reentrant"):
            journal.reserve(frozen.job_id, frozen.request_bytes, reenter, guard)
        assert journal.inspect(frozen.job_id)["state"] == "planned"


def test_guard_required_before_initial_io(tmp_path):
    store, settings = setup(tmp_path)

    def guard():
        raise ValueError("already stale")

    with pytest.raises(ValueError, match="already stale"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store, guard=guard)
    assert not (tmp_path / "ariadne" / "rolling").exists()


def test_missing_journal_with_artifacts_never_resets(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
        path = journal.path
    path.unlink()
    with pytest.raises(ValueError, match="journal missing"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    assert not path.exists()


def test_private_single_link_and_no_symlink_artifacts(tmp_path):
    import os

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        view = journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
        path = journal.generations / (view.generation_id + ".json")
        assert path.stat().st_mode & 0o777 == 0o600
        os.link(path, tmp_path / "alias")
        with pytest.raises(ValueError, match="single-link"):
            journal.load_view()
        (tmp_path / "alias").unlink()
        path.chmod(0o644)
        with pytest.raises(ValueError, match="private"):
            journal.load_view()
        path.chmod(0o600)
        data = path.read_bytes()
        path.unlink()
        other = tmp_path / "other.json"
        other.write_bytes(data)
        path.symlink_to(other)
        with pytest.raises(ValueError, match="symlink"):
            journal.load_view()


def test_checkpoint_exact_schema_and_byte_rejections(tmp_path):
    import json

    store, settings = setup(tmp_path)
    frozen = target(store, settings, 1)
    data = rolling.generation_bytes(frozen, "summary", sequence=1)
    for patch in [
        {"sequence": True},
        {"version": True},
        {"sequence": 2},
        {"event_ids": []},
        {"message_uids": ["wrong", "wrong"]},
        {"summary_text": "changed"},
        {"summary_input_sha256": "f" * 64},
        {"policy_id": "other"},
        {"parent_generation": "f" * 64},
        {"frontier_count": 1},
        {"delta_start": 1},
        {"extra": "field"},
    ]:
        value = json.loads(data)
        value.update(patch)
        with pytest.raises(ValueError):
            rolling.validate_generation(
                rolling.encode_json(value, settings.max_checkpoint_bytes), store, "s", settings
            )
    with pytest.raises(ValueError):
        rolling.validate_generation(data + b" ", store, "s", settings)
    with pytest.raises(ValueError):
        rolling.validate_generation(data, store, "foreign", settings)


def test_sqlite_result_bound_checked_before_reading_payload(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        conn = journal._conn
        conn.execute("UPDATE jobs SET validated_text=?", ("x" * 65536,))
        conn.commit()

        class GuardedConnection:
            def execute(self, sql, *args):
                assert "SELECT * FROM jobs" not in sql, "payload loaded before length gate"
                return conn.execute(sql, *args)

        journal._conn = GuardedConnection()
        try:
            with pytest.raises(ValueError, match="bound"):
                journal.inspect(frozen.job_id)
        finally:
            journal._conn = conn


def test_exact_checkpoint_bytes_hashed_once(tmp_path, monkeypatch):
    store, settings = setup(tmp_path)
    frozen = target(store, settings, 1)
    data = rolling.generation_bytes(frozen, "summary", sequence=1)
    original, seen = rolling.sha256, []

    def once(value):
        if value == data:
            seen.append(value)
            assert len(seen) == 1
        return original(value)

    monkeypatch.setattr(rolling, "sha256", once)
    get_event = store.get_event

    def after_identity(*args, **kwargs):
        assert len(seen) == 1, "checkpoint identity not captured before dependency reads"
        return get_event(*args, **kwargs)

    monkeypatch.setattr(store, "get_event", after_identity)
    view = rolling.validate_generation(data, store, "s", settings)
    assert view.snapshot_bytes is data
    assert len(seen) == 1


def test_lineage_rejects_changed_parent_prefix(tmp_path):
    import json

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        first = target(store, settings, 1)
        ready(journal, first)
        g1 = journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "summary", sequence=1)
        )
        second = target(store, settings, 2, g1)
        data = json.loads(rolling.generation_bytes(second, "next", sequence=2))
        data["event_ids"][0], data["event_ids"][1] = data["event_ids"][1], data["event_ids"][0]
        data["prefix_sha256"] = rolling.sha256(rolling.encode_json(data["event_ids"], 65536))
        with pytest.raises(ValueError, match="lineage"):
            rolling.validate_generation(
                rolling.encode_json(data, 65536),
                store,
                "s",
                settings,
                parent_loader=lambda _: g1.snapshot_bytes,
            )


@pytest.mark.parametrize(
    "column,value",
    [
        ("attribution", b"[]"),
        ("attribution", b'{"provider":false,"model":"test"}'),
        ("usage_observed", b'{"kind":"measured_zero","value":0}'),
        ("error_code", "untrusted error payload"),
    ],
)
def test_persisted_job_observation_schema_is_strict(tmp_path, column, value):
    import sqlite3

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        path = journal.path
    with sqlite3.connect(path) as conn:
        conn.execute(f"UPDATE jobs SET {column}=?", (value,))
    with pytest.raises(ValueError):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)


def test_ready_duplicate_is_idempotent_usage_not_measured_zero(tmp_path):
    import json

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        for _ in range(2):
            journal.record_ready(
                frozen.job_id, "summary", {"provider": "test", "model": "test"}, {"tokens": 0}
            )
        assert json.loads(journal.inspect(frozen.job_id)["usage_observed"]) == {
            "kind": "reported_or_defaulted",
            "value": {"tokens": 0},
        }
        with pytest.raises(ValueError):
            journal.record_ready(
                frozen.job_id, "changed", {"provider": "test", "model": "test"}, None
            )
        assert journal.report()["logical_reservations"] == 1


def test_uncertain_error_reason_survives_repeated_restart(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        journal.record_failed_or_uncertain(frozen.job_id, "facade_error", uncertain=True)
    for _ in range(2):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            assert journal.report()["blocked_reason"] == "facade_error"
            assert journal.inspect(frozen.job_id)["error_code"] == "facade_error"


def test_target_hash_bound_before_sqlite_payload_read(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        conn = journal._conn
        conn.execute("UPDATE targets SET payload_sha256=?", ("x" * 65536,))
        conn.commit()

        class GuardedConnection:
            def execute(self, sql, *args):
                assert "SELECT payload,payload_sha256" not in sql, "unbounded target hash read"
                return conn.execute(sql, *args)

        journal._conn = GuardedConnection()
        try:
            with pytest.raises(ValueError):
                journal.inspect(frozen.job_id)
        finally:
            journal._conn = conn


def test_restart_charges_uncertainty_even_when_sources_disappear(tmp_path, monkeypatch):
    import sqlite3

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        path = journal.path
    monkeypatch.setattr(store, "get_event", lambda *a, **kw: None)
    with pytest.raises(ValueError):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT state FROM jobs").fetchone()[0] == "uncertain"
        assert conn.execute("SELECT logical_reservations FROM campaign").fetchone()[0] == 1


def test_charging_callback_identity_and_counters_commit_or_rollback_together(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)

        def after_updates():
            if journal._conn.execute("SELECT state FROM jobs").fetchone()[0] == "reserved":
                raise ValueError("revoked before commit")

        with pytest.raises(ValueError, match="revoked"):
            journal.reserve(
                frozen.job_id,
                frozen.request_bytes,
                after_updates,
                after_updates,
                charging_callback_id="actual-C",
            )
        assert journal.inspect(frozen.job_id)["state"] == "planned"
        assert journal.inspect(frozen.job_id)["charging_callback_id"] is None
        assert journal.charged_callback_ids() == set()
        assert (
            journal.report()["logical_reservations"]
            == journal.report()["cumulative_request_bytes"]
            == 0
        )
        assert journal.reserve(
            frozen.job_id,
            frozen.request_bytes,
            lambda: None,
            lambda: None,
            charging_callback_id="actual-C",
        )
        assert journal.inspect(frozen.job_id)["charging_callback_id"] == "actual-C"
    for _ in range(2):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            assert journal.charged_callback_ids() == {"actual-C"}
            assert journal.inspect(frozen.job_id)["state"] == "uncertain"
            assert journal.report()["logical_reservations"] == 1
            assert journal.report()["cumulative_request_bytes"] == len(frozen.request_bytes)


def test_callback_dedup_covers_all_charged_jobs_not_only_latest(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        parent = None
        for count, callback in [(1, "C"), (2, "D")]:
            frozen = target(store, settings, count, parent)
            journal.freeze_target(frozen)
            assert journal.reserve(
                frozen.job_id,
                frozen.request_bytes,
                lambda: None,
                lambda: None,
                charging_callback_id=callback,
            )
            journal.record_ready(
                frozen.job_id, "summary", {"provider": "test", "model": "test"}, None
            )
            parent = journal.publish_ready(
                frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=count)
            )
        third = target(store, settings, 3, parent)
        journal.freeze_target(third)
        before = journal.report()
        for callback in ("C", "D"):
            assert not journal.reserve(
                third.job_id,
                third.request_bytes,
                lambda: None,
                lambda: None,
                charging_callback_id=callback,
            )
        assert journal.report() == before
        assert journal.inspect(third.job_id)["state"] == "planned"
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.charged_callback_ids() == {"C", "D"}
        assert not journal.reserve(
            third.job_id, third.request_bytes, lambda: None, lambda: None, charging_callback_id="C"
        )
        assert journal.report()["logical_reservations"] == 2
        assert journal.reserve(
            third.job_id, third.request_bytes, lambda: None, lambda: None, charging_callback_id="E"
        )
        journal.record_failed_or_uncertain(third.job_id, "invalid_output")
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.charged_callback_ids() == {"C", "D", "E"}
        assert journal.report()["logical_reservations"] == 3


@pytest.mark.parametrize("callback", ["", False, 123, "x" * 4097, "é" * 2049])
def test_callback_identity_must_be_bounded_before_reservation(tmp_path, callback):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        with pytest.raises(ValueError):
            journal.reserve(
                frozen.job_id,
                frozen.request_bytes,
                lambda: None,
                lambda: None,
                charging_callback_id=callback,
            )
        assert journal.report()["logical_reservations"] == 0
        assert journal.charged_callback_ids() == set()


def test_unidentified_legacy_style_charge_cannot_authorize_new_callback_spending(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)  # Original S1 signature remains valid; identity is honestly unknown.
        parent = journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
        second = target(store, settings, 2, parent)
        journal.freeze_target(second)
        path = journal.path
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.charged_callback_ids() == set()
        with pytest.raises(ValueError, match="unknown charging callback"):
            journal.reserve(
                second.job_id,
                second.request_bytes,
                lambda: None,
                lambda: None,
                charging_callback_id="C",
            )
        assert journal.inspect(frozen.job_id)["charging_callback_id"] is None
        assert journal.inspect(second.job_id)["state"] == "planned"
        assert journal.report()["logical_reservations"] == 1
        assert journal.path == path


def test_v1_journal_is_rejected_without_migration_reset_or_invented_callback(tmp_path):
    import sqlite3

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        journal.freeze_target(frozen)
        journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        path = journal.path
    with sqlite3.connect(path) as conn:
        # Downgrade the owned synthetic DB to the exact original S1 schema.
        conn.execute("ALTER TABLE jobs RENAME TO newer_jobs")
        conn.execute("""CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, target_id TEXT NOT NULL UNIQUE REFERENCES targets(target_id),
            state TEXT NOT NULL, reserved_request_bytes INTEGER NOT NULL,
            validated_text TEXT, text_sha256 TEXT, attribution BLOB, usage_observed BLOB,
            error_code TEXT, generation_id TEXT
        )""")
        columns = "job_id,target_id,state,reserved_request_bytes,validated_text,text_sha256,attribution,usage_observed,error_code,generation_id"
        conn.execute(f"INSERT INTO jobs ({columns}) SELECT {columns} FROM newer_jobs")
        conn.execute("DROP TABLE newer_jobs")
        conn.execute("PRAGMA user_version=1")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported rolling journal schema"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT state FROM jobs").fetchone()[0] == "reserved"
        assert conn.execute("SELECT logical_reservations FROM campaign").fetchone()[0] == 1
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
