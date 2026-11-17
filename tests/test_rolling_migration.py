"""Offline conversion never submits work or forgives reservations."""

import importlib
import importlib.util
import json

import pytest
from test_rolling_state import ready, setup, target
from test_rolling_storage_legacy import STATES, build_source, inspect_readonly, inventory

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.store import EventStore


def migrator():
    assert importlib.util.find_spec("hermes_ariadne.rolling_migrate") is not None, (
        "offline migration is missing"
    )
    return importlib.import_module("hermes_ariadne.rolling_migrate")


def test_published_migration_preserves_inputs_charges_and_archive(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        ready(journal, old)
        view = journal.publish_ready(
            old.job_id, rolling.generation_bytes(old, "summary", sequence=1, settings=settings)
        )
        old_report = journal.report()
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    original = (directory / "journal.sqlite3").read_bytes()
    receipt = migrator().migrate(tmp_path, "s", settings, store=store)
    assert receipt["targets"][old.target_id] != old.target_id
    assert receipt["jobs"][old.job_id] != old.job_id
    assert receipt["generations"][view.generation_id] != view.generation_id
    assert (directory / "legacy-snapshot" / "journal.sqlite3").read_bytes() == original
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.schema_version == 3
        new_view = journal.load_view()
        report = journal.report()
        assert new_view.summary_text == view.summary_text
        assert tuple(new_view.event_ids) == view.event_ids
        assert new_view.prefix_sha256 == view.prefix_sha256
        assert report["logical_reservations"] == old_report["logical_reservations"]
        assert report["cumulative_request_bytes"] == old_report["cumulative_request_bytes"]
        new_target = journal._target(receipt["targets"][old.target_id])
        assert new_target.readable_input == old.readable_input
        assert new_target.request_bytes == old.request_bytes
        assert json.loads(new_view.snapshot_bytes)["version"] == 2
    assert migrator().migrate(tmp_path, "s", settings, store=store) == receipt


@pytest.mark.parametrize("state", STATES)
def test_migration_preserves_every_legacy_lifecycle_field(tmp_path, monkeypatch, state):
    settings, old, _, directory = build_source(tmp_path, state, monkeypatch)
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    before = inspect_readonly(directory)
    old_files = inventory(directory)
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    receipt = migrator().migrate(tmp_path, "s", settings, store=store)
    after = inspect_readonly(directory / "v3")
    converted_files = inventory(directory)
    assert migrator().migrate(tmp_path, "s", settings, store=store) == receipt
    assert inventory(directory) == converted_files  # Retry audits without lifecycle recovery.
    old_job, new_job = before["jobs"][0], after["jobs"][0]
    for key in old_job:
        if key not in {"job_id", "target_id", "generation_id"}:
            assert new_job[key] == old_job[key], key
    old_campaign, new_campaign = before["campaign"][0], after["campaign"][0]
    for key in old_campaign:
        if key != "published_generation":
            assert new_campaign[key] == old_campaign[key], key
    assert inventory(directory / "legacy-snapshot") == {
        key: value for key, value in old_files.items() if key != "writer.lock"
    }
    assert receipt["source_fingerprint"] != receipt["destination_fingerprint"]
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        job = journal.inspect(receipt["jobs"][old.job_id])
        expected = (
            "uncertain"
            if state == "reserved"
            else "published"
            if state == "manifest_only"
            else "ready"
            if state == "generation_only"
            else state
        )
        assert job["state"] == expected
        assert journal.report()["logical_reservations"] == int(state != "planned")
        if state == "reserved":
            with pytest.raises(ValueError):
                journal.reserve(job["job_id"], old.request_bytes, lambda: None, lambda: None)


def test_bad_source_refuses_before_any_migration_writes(tmp_path):
    import sqlite3

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        journal.freeze_target(old)
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    with sqlite3.connect(directory / "journal.sqlite3") as conn:
        conn.execute("UPDATE campaign SET logical_reservations=42")
    before = inventory(directory)
    with pytest.raises(ValueError):
        migrator().migrate(tmp_path, "s", settings, store=store)
    assert inventory(directory) == before


@pytest.mark.parametrize("point", ["snapshot", "stage", "receipt", "intent", "fence", "selector"])
def test_process_death_never_creates_a_fresh_ledger(tmp_path, monkeypatch, point):
    import os
    import sqlite3

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        ready(journal, old)
        journal.publish_ready(
            old.job_id, rolling.generation_bytes(old, "summary", sequence=1, settings=settings)
        )
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    module = migrator()
    pid = os.fork()
    if pid == 0:
        try:
            atomic, sync = rolling_state._atomic, module._sync

            def at_write(path, data):
                atomic(path, data)
                if (point, path.name) in {
                    ("receipt", "migration.json"),
                    ("intent", "migration-intent.json"),
                    ("selector", "storage.json"),
                }:
                    os._exit(73)

            def at_sync(path):
                sync(path)
                if point == "snapshot" and path == directory / "legacy-snapshot":
                    os._exit(73)
                if point == "stage" and path == directory / "v3":
                    os._exit(73)
                if point == "fence" and path == directory:
                    with sqlite3.connect(directory / "journal.sqlite3") as conn:
                        if conn.execute("PRAGMA user_version").fetchone()[0] == module.FENCE:
                            os._exit(73)

            rolling_state._atomic, module._sync = at_write, at_sync
            module.migrate(tmp_path, "s", settings, store=store)
            os._exit(0)
        except BaseException:
            os._exit(74)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 73, "fault boundary was not reached"
    if point in {"intent", "fence"}:
        module.migrate(tmp_path, "s", settings, store=store)
    try:
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            assert journal.report()["logical_reservations"] == 1
            assert journal.load_view() is not None
    except (ValueError, FileNotFoundError):
        assert point in {"snapshot", "stage", "receipt"}
    with sqlite3.connect(directory / "journal.sqlite3") as conn:
        assert conn.execute("SELECT logical_reservations FROM campaign").fetchone()[0] == 1


@pytest.mark.parametrize("metadata", ["storage.json", "migration-intent.json"])
def test_boolean_metadata_version_is_not_an_integer_version(tmp_path, metadata):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        journal.freeze_target(old)
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migrator().migrate(tmp_path, "s", settings, store=store)
    path = directory / metadata
    value = json.loads(path.read_bytes())
    value["version"] = True
    path.write_bytes(rolling.encode_json(value, 8192))
    with pytest.raises(ValueError):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
            pass


def test_rollback_does_not_repair_missing_selector(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        journal.freeze_target(old)
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migrator().migrate(tmp_path, "s", settings, store=store)
    (directory / "storage.json").unlink()
    before = inventory(directory)
    with pytest.raises((ValueError, FileNotFoundError)):
        migrator().rollback(tmp_path, "s", settings, store=store)
    assert inventory(directory) == before


def test_resume_rejects_tampered_fenced_ledger(tmp_path):
    import sqlite3

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        journal.freeze_target(old)
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migrator().migrate(tmp_path, "s", settings, store=store)
    (directory / "storage.json").unlink()
    with sqlite3.connect(directory / "journal.sqlite3") as conn:
        conn.execute("UPDATE campaign SET logical_reservations=42")
    with pytest.raises(ValueError):
        migrator().migrate(tmp_path, "s", settings, store=store)
    assert not (directory / "storage.json").exists()


def test_unfenced_legacy_root_is_never_selected_as_migrated(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        journal.freeze_target(old)
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migrator().migrate(tmp_path, "s", settings, store=store)
    (directory / "journal.sqlite3").write_bytes(
        (directory / "legacy-snapshot/journal.sqlite3").read_bytes()
    )
    with pytest.raises(ValueError, match="fence"):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
            pass


@pytest.mark.parametrize("point", ["prepared", "database", "selector", "marker", "complete"])
def test_rollback_process_death_is_resumable_without_reset(tmp_path, point):
    import os

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        ready(journal, old)
        view = journal.publish_ready(
            old.job_id, rolling.generation_bytes(old, "summary", sequence=1, settings=settings)
        )
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    module = migrator()
    module.migrate(tmp_path, "s", settings, store=store)
    pid = os.fork()
    if pid == 0:
        try:
            atomic, sync, replace = rolling_state._atomic, module._sync, os.replace
            fsync = os.fsync

            def at_fsync(fd):
                fsync(fd)
                if point == "prepared" and "/.rollback-" in os.readlink(f"/proc/self/fd/{fd}"):
                    os._exit(73)

            def at_write(path, data):
                atomic(path, data)
                if (point, path.name) in {
                    ("marker", "rollback.json"),
                }:
                    os._exit(73)

            def at_replace(source, destination):
                replace(source, destination)
                if point == "database" and destination == directory / "journal.sqlite3":
                    os._exit(73)

            def at_sync(path):
                sync(path)
                if (
                    path == directory
                    and point == "selector"
                    and not (directory / "storage.json").exists()
                ):
                    os._exit(73)
                if (
                    path == directory
                    and point == "complete"
                    and (directory / "rollback.json").exists()
                ):
                    os._exit(73)

            rolling_state._atomic, module._sync = at_write, at_sync
            os.replace = at_replace
            os.fsync = at_fsync
            module.rollback(tmp_path, "s", settings, store=store)
            os._exit(0)
        except BaseException:
            os._exit(74)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 73
    module.rollback(tmp_path, "s", settings, store=store)
    assert not list(directory.glob(".rollback-*"))
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view() == view
        assert journal.report()["logical_reservations"] == 1


def test_rollback_restores_schema_only_after_all_metadata_is_complete(tmp_path, monkeypatch):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        journal.freeze_target(old)
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    module = migrator()
    module.migrate(tmp_path, "s", settings, store=store)
    upgrade_intent = (directory / "migration-intent.json").read_bytes()
    replace, observed = module.os.replace, []

    def guarded_restore(source, path):
        if path == directory / "journal.sqlite3":
            observed.append(path)
            assert not (directory / "storage.json").exists(), (
                "old admission before selector removal"
            )
            assert (directory / "rollback.json").is_file(), (
                "old admission before completion metadata"
            )
            assert (directory / "rollback-intent.json").is_file()
            assert (directory / "migration-intent.json").read_bytes() == upgrade_intent
        return replace(source, path)

    monkeypatch.setattr(module.os, "replace", guarded_restore)
    module.rollback(tmp_path, "s", settings, store=store)
    assert observed == [directory / "journal.sqlite3"]


def test_receipt_source_file_schema_cannot_be_rehashed_away(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        journal.freeze_target(target(store, settings, 1))
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migrator().migrate(tmp_path, "s", settings, store=store)
    value = json.loads((directory / "migration.json").read_bytes())
    value["source_files"] = {}
    data = rolling.encode_json(value, 65536)
    (directory / "migration.json").write_bytes(data)
    for name in ("migration-intent.json", "storage.json"):
        value = json.loads((directory / name).read_bytes())
        value["receipt_sha256"] = rolling.sha256(data)
        (directory / name).write_bytes(rolling.encode_json(value, 8192))
    with pytest.raises(ValueError, match="source file"):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
            pass


def test_migrated_two_generation_chain_advances_and_restarts(tmp_path):
    store, settings = setup(tmp_path)
    parent, old_ids = None, []
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        for count in (1, 2):
            frozen = target(store, settings, count, parent)
            journal.freeze_target(frozen)
            assert journal.reserve(
                frozen.job_id,
                frozen.request_bytes,
                lambda: None,
                lambda: None,
                charging_callback_id=f"callback-{count}",
            )
            journal.record_ready(
                frozen.job_id, f"summary {count}", {"provider": "t", "model": "t"}, None
            )
            parent = journal.publish_ready(
                frozen.job_id,
                rolling.generation_bytes(
                    frozen, f"summary {count}", sequence=count, settings=settings
                ),
            )
            old_ids.append(parent.generation_id)
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    receipt = migrator().migrate(tmp_path, "s", settings, store=store)
    assert set(receipt["generations"]) == set(old_ids)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        parent = journal.load_view()
        assert parent.sequence == 2
        assert journal.charged_callback_ids() == {"callback-1", "callback-2"}
        frozen = journal.encode_target(target(store, settings, 3, parent), parent)
        journal.freeze_target(frozen)
        assert not journal.reserve(
            frozen.job_id,
            frozen.request_bytes,
            lambda: None,
            lambda: None,
            charging_callback_id="callback-1",
        )
        assert journal.reserve(
            frozen.job_id,
            frozen.request_bytes,
            lambda: None,
            lambda: None,
            charging_callback_id="callback-3",
        )
        journal.record_ready(frozen.job_id, "summary 3", {"provider": "t", "model": "t"}, None)
        last = journal.publish_ready(
            frozen.job_id,
            rolling.generation_bytes(frozen, "summary 3", sequence=3, settings=settings),
        )
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view() == last
        assert journal.report()["logical_reservations"] == 3
        assert journal.charged_callback_ids() == {"callback-1", "callback-2", "callback-3"}
    with pytest.raises(ValueError, match="diverged"):
        migrator().rollback(tmp_path, "s", settings, store=store)


@pytest.mark.parametrize("unsafe", ["marker-mode", "archive-link", "journal-companion"])
def test_unsafe_migration_refuses_before_staging(tmp_path, unsafe):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        journal.freeze_target(target(store, settings, 1))
        directory = journal.directory
    marker = tmp_path / ".ariadne-offline-migration"
    marker.touch(mode=0o600)
    if unsafe == "marker-mode":
        marker.chmod(0o644)
    elif unsafe == "archive-link":
        import os

        os.link(store.path, tmp_path / "archive-alias")
    else:
        (directory / "journal.sqlite3-journal").write_bytes(b"uncheckpointed")
        (directory / "journal.sqlite3-journal").chmod(0o600)
    before = inventory(directory)
    with pytest.raises(ValueError):
        migrator().migrate(tmp_path, "s", settings, store=store)
    assert inventory(directory) == before
    assert not (directory / "v3").exists()


def test_cli_conversion_uses_an_explicit_existing_home(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    package_site = str(Path(rolling.__file__).resolve().parent.parent)
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join([package_site, child_env.get("PYTHONPATH", "")])
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        journal.freeze_target(old)
        directory = journal.directory
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_ariadne.rolling_migrate",
            "--home",
            str(tmp_path),
            "--session",
            "s",
            "--settings",
            '{"trigger_bytes":1}',
        ],
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert run.returncode == 0, run.stderr
    assert (directory / "storage.json").is_file(), run.stdout
    assert json.loads(run.stdout)["status"] == "migrated"


def test_rollback_refuses_new_charges_and_allows_unchanged_ledger(tmp_path):
    import pytest

    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        old = target(store, settings, 1)
        ready(journal, old)
        old_view = journal.publish_ready(
            old.job_id, rolling.generation_bytes(old, "summary", sequence=1, settings=settings)
        )
    (tmp_path / ".ariadne-offline-migration").touch(mode=0o600)
    migrator().migrate(tmp_path, "s", settings, store=store)
    migrator().rollback(tmp_path, "s", settings, store=store)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.schema_version == 2
        assert journal.load_view() == old_view
        assert journal.report()["logical_reservations"] == 1
    # A separate converted ledger that recovers a reservation has diverged even
    # if no further generation or provider request was ever made.
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    store2, settings2 = setup(other)
    with rolling_state.RollingJournal(other, "s", settings2, store=store2) as journal:
        pending = target(store2, settings2, 1)
        journal.freeze_target(pending)
        journal.reserve(pending.job_id, pending.request_bytes, lambda: None, lambda: None)
    (other / ".ariadne-offline-migration").touch(mode=0o600)
    migrator().migrate(other, "s", settings2, store=store2)
    with rolling_state.RollingJournal(other, "s", settings2, store=store2):
        pass
    with pytest.raises(ValueError, match="diverged"):
        migrator().rollback(other, "s", settings2, store=store2)
