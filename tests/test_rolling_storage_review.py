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
