"""Regression evidence for storage review findings; real private journals."""

import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
from test_rolling_state import ready, setup, target

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne import rolling_migrate as migration


@pytest.mark.parametrize("with_parent", [False, True])
@pytest.mark.parametrize("after_failure", [False, True])
def test_uncached_pending_delta_returns_bound_coverage(tmp_path, with_parent, after_failure):
    store, settings = setup(tmp_path)
    parent = None
    with rolling_state.RollingJournal(
        tmp_path, "s", settings, store=store, storage_version=3
    ) as journal:
        if with_parent:
            first = journal.encode_target(target(store, settings, 1), None)
            ready(journal, first)
            parent = journal.publish_ready(
                first.job_id,
                rolling.generation_bytes(first, "summary", sequence=1, settings=settings),
            )
        expected = target(store, settings, 2 if with_parent else 1, parent)
        frozen = journal.encode_target(expected, parent)
        journal.freeze_target(frozen)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, cache=None) as journal:
        if after_failure:
            with pytest.raises(ValueError):
                journal.inspect("0" * 64)
        pending = journal.pending_target()
        assert pending is not None
        assert tuple(pending.event_ids) == tuple(expected.event_ids)
        assert tuple(pending.message_uids) == tuple(expected.message_uids)
        assert pending.request_bytes == expected.request_bytes


@pytest.mark.parametrize("action", ["migrate", "rollback"])
@pytest.mark.parametrize("damage", ["accounting", "source", "artifact"])
def test_completed_migration_retry_reaudits_selected_state(tmp_path, damage, action):
    from test_rolling_storage_legacy import inventory

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        first = target(store, settings, 1)
        ready(journal, first, "one")
        view = journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "one", sequence=1, settings=settings)
        )
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    receipt = migration.migrate(tmp_path, "s", settings, store=store)
    if action == "rollback":
        migration.rollback(tmp_path, "s", settings, store=store)
    selected = directory / "v3" if action == "migrate" else directory
    if damage == "accounting":
        with closing(sqlite3.connect(selected / "journal.sqlite3")) as connection:
            with connection:
                connection.execute("UPDATE campaign SET logical_reservations=42")
    elif damage == "source":
        with closing(sqlite3.connect(store.path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE events SET payload_json=? WHERE event_id=?",
                    (b"{}", first.event_ids[0]),
                )
    else:
        generation = (
            receipt["generations"][view.generation_id]
            if action == "migrate"
            else view.generation_id
        )
        path = selected / "generations" / (generation + ".json")
        path.write_bytes(b"{}")
    before = inventory(directory)
    with pytest.raises(ValueError):
        getattr(migration, action)(tmp_path, "s", settings, store=store)
    assert inventory(directory) == before


def test_completed_migration_retry_preserves_valid_later_activity(tmp_path):
    from test_rolling_storage_legacy import inventory

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        first = target(store, settings, 1)
        ready(journal, first, "one")
        journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "one", sequence=1, settings=settings)
        )
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    receipt = migration.migrate(tmp_path, "s", settings, store=store)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        parent = journal.load_view()
        second = journal.encode_target(target(store, settings, 2, parent), parent)
        ready(journal, second, "two")
        journal.publish_ready(
            second.job_id, rolling.generation_bytes(second, "two", sequence=2, settings=settings)
        )
    before = inventory(directory)
    assert migration.migrate(tmp_path, "s", settings, store=store) == receipt
    assert inventory(directory) == before


def test_schema_probe_connection_is_explicitly_closed(tmp_path, monkeypatch):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        path = journal.path
    connections = []
    connect = sqlite3.connect

    class Tracked(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            return super().close()

    def track(database, *args, **kwargs):
        if database == path:
            kwargs["factory"] = Tracked
        connection = connect(database, *args, **kwargs)
        if database == path:
            connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
        pass
    assert len(connections) == 2
    assert all(connection.closed for connection in connections)


def test_matching_immutable_metadata_resyncs_parent_and_propagates_failure(tmp_path, monkeypatch):
    path, value = tmp_path / "intent.json", {"version": 1}
    migration._immutable_metadata(path, value)
    calls = []
    monkeypatch.setattr(migration, "_sync", calls.append)
    migration._immutable_metadata(path, value)
    assert calls == [tmp_path]

    def denied(directory):
        raise OSError("synthetic directory sync denied")

    monkeypatch.setattr(migration, "_sync", denied)
    with pytest.raises(OSError, match="sync denied"):
        migration._immutable_metadata(path, value)


def test_rollback_streams_valid_database_larger_than_checkpoint_limit(tmp_path):
    store, settings = setup(tmp_path)
    settings = replace(settings, max_checkpoint_bytes=4096, max_logical_jobs=1)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        journal.freeze_target(target(store, settings, 1))
        directory = journal.directory
    with closing(sqlite3.connect(directory / "journal.sqlite3")) as connection:
        with connection:
            connection.execute("CREATE TABLE padding(value BLOB)")
            connection.execute("INSERT INTO padding VALUES(zeroblob(?))", (20 * 1024 * 1024,))
            connection.execute("DROP TABLE padding")
    database = directory / "journal.sqlite3"
    assert (
        database.stat().st_size
        > settings.max_checkpoint_bytes * settings.max_logical_jobs + 16 * 1024 * 1024
    )
    original = migration._file_hash(database)
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migration.migrate(tmp_path, "s", settings, store=store)
    migration.rollback(tmp_path, "s", settings, store=store)
    assert migration._file_hash(database) == original
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.schema_version == 2
        assert journal.pending_target() is not None


def test_restore_copy_failure_precedes_selector_removal(tmp_path, monkeypatch):
    import os

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        journal.freeze_target(target(store, settings, 1))
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migration.migrate(tmp_path, "s", settings, store=store)
    selector = (directory / "storage.json").read_bytes()
    fsync = os.fsync
    observed = []

    def fail_copy(fd):
        if "/.rollback-" in os.readlink(f"/proc/self/fd/{fd}"):
            observed.append(fd)
            raise OSError("synthetic restore copy fsync failure")
        return fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_copy)
        with pytest.raises(OSError, match="restore copy"):
            migration.rollback(tmp_path, "s", settings, store=store)
    assert len(observed) == 1
    assert (directory / "storage.json").read_bytes() == selector
    assert not (directory / "rollback-intent.json").exists()
    assert not list(directory.glob(".rollback-*"))
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.schema_version == 3


def test_prepared_restore_rejects_changed_snapshot_and_removes_temporary_copy(tmp_path):
    source = tmp_path / "snapshot.sqlite3"
    source.write_bytes(b"synthetic private database bytes")
    source.chmod(0o600)
    with pytest.raises(ValueError, match="database changed"):
        with migration._prepared_restore(source, tmp_path, "0" * 64):
            pytest.fail("a mismatched snapshot must never become publishable")
    assert source.read_bytes() == b"synthetic private database bytes"
    assert not list(tmp_path.glob(".rollback-*"))


@pytest.mark.parametrize("action", ["migrate", "rollback"])
@pytest.mark.parametrize("unsafe", [None, "bytes", "symlink", "hardlink", "mode", "name"])
def test_orphan_cleanup_only_removes_confirmed_private_copies(tmp_path, action, unsafe):
    import os

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        journal.freeze_target(target(store, settings, 1))
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migration.migrate(tmp_path, "s", settings, store=store)
    snapshot = directory / "legacy-snapshot" / "journal.sqlite3"
    original = snapshot.read_bytes()
    confirmed = directory / ".rollback-00000000"
    confirmed.write_bytes(original)
    confirmed.chmod(0o600)
    candidate = directory / (".rollback-zz-invalid" if unsafe == "name" else ".rollback-zzzzzzzz")
    if unsafe == "symlink":
        candidate.symlink_to(snapshot)
    elif unsafe == "hardlink":
        os.link(snapshot, candidate)
    else:
        candidate.write_bytes(b"unknown partial copy" if unsafe == "bytes" else original)
        candidate.chmod(0o644 if unsafe == "mode" else 0o600)
    selector = (directory / "storage.json").read_bytes()
    operation = getattr(migration, action)
    if unsafe is not None:
        with pytest.raises(ValueError):
            operation(tmp_path, "s", settings, store=store)
        assert confirmed.read_bytes() == original  # Validate all BEFORE deleting any.
        assert candidate.exists() or candidate.is_symlink()
        assert (directory / "storage.json").read_bytes() == selector
        assert not (directory / "rollback-intent.json").exists()
    else:
        operation(tmp_path, "s", settings, store=store)
        assert not list(directory.glob(".rollback-*"))
    assert snapshot.read_bytes() == original


@pytest.mark.parametrize("action", ["migrate", "rollback"])
def test_idempotent_completion_resyncs_and_preserves_later_legacy_activity(
    tmp_path, monkeypatch, action
):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        first = target(store, settings, 1)
        ready(journal, first, "one")
        journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "one", sequence=1, settings=settings)
        )
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    receipt = migration.migrate(tmp_path, "s", settings, store=store)
    if action == "rollback":
        migration.rollback(tmp_path, "s", settings, store=store)
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            second = target(store, settings, 2, journal.load_view())
            ready(journal, second, "two")
            journal.publish_ready(
                second.job_id,
                rolling.generation_bytes(second, "two", sequence=2, settings=settings),
            )
    before = migration.fingerprint(directory)
    operation = getattr(migration, action)

    def denied(path):
        assert path == directory
        raise OSError("synthetic completion sync denied")

    with monkeypatch.context() as patch:
        patch.setattr(migration, "_sync", denied)
        with pytest.raises(OSError, match="completion sync denied"):
            operation(tmp_path, "s", settings, store=store)
    calls = []
    monkeypatch.setattr(migration, "_sync", calls.append)
    assert operation(tmp_path, "s", settings, store=store) == receipt
    assert calls == [directory]
    assert migration.fingerprint(directory) == before
