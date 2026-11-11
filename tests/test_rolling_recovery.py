"""Real process-exit recovery for the rolling journal (forked child, os._exit)."""

import os
import sqlite3

import pytest
from test_rolling_state import ready, setup, target

from hermes_ariadne import rolling, rolling_state


def _die_in_child(action):
    """Run action in a forked child that hard-exits; return the child's exit code."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        try:
            action()
        finally:
            os._exit(0)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def _crash_on_sql(prefix, *, occurrence=1):
    original = rolling_state.sqlite3.connect
    seen = []

    class CrashConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.lstrip().startswith(prefix):
                seen.append(sql)
                if len(seen) == occurrence:
                    os._exit(86)
            return super().execute(sql, *args)

    rolling_state.sqlite3.connect = lambda *a, **kw: original(*a, factory=CrashConnection, **kw)


@pytest.mark.parametrize("occurrence", [1, 2])
def test_exit_during_initial_schema_creation_recovers_as_fresh(tmp_path, occurrence):
    store, settings = setup(tmp_path)

    def crash():
        _crash_on_sql("CREATE TABLE", occurrence=occurrence)
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)

    assert _die_in_child(crash) == 86
    directory = next((tmp_path / "ariadne" / "rolling").glob("*/"))
    # The journal never exists half-built under its real name.
    assert not (directory / "journal.sqlite3").exists()
    assert list(directory.glob(".rolling-*")), "crash must leave an orphaned temporary"
    for _ in range(2):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            report = journal.report()
            assert report["logical_reservations"] == 0
            assert report["cumulative_request_bytes"] == 0
            assert journal.load_view() is None
    assert not list(directory.glob(".rolling-*"))
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        view = journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
        assert journal.load_view() == view and journal.report()["logical_reservations"] == 1


def test_empty_journal_with_published_artifacts_still_refuses(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
        path = journal.path
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    assert path.read_bytes() == b""


def _published_g1_ready_g2(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        first = target(store, settings, 1)
        ready(journal, first)
        g1 = journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "summary", sequence=1)
        )
        second = target(store, settings, 2, g1)
        ready(journal, second, "next summary")
        report = journal.report()
    data = rolling.generation_bytes(second, "next summary", sequence=2)
    return store, settings, g1, second, data, report


@pytest.mark.parametrize("artifact", ["generation", "manifest"])
def test_exit_before_atomic_replace_leaves_no_poisoned_journal(tmp_path, artifact):
    store, settings, g1, second, data, report = _published_g1_ready_g2(tmp_path)

    def crash():
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            target_path = journal.generations if artifact == "generation" else journal.directory

            def crash_replace(src, dst):
                if os.path.dirname(dst) == str(target_path):
                    os._exit(87)
                os.rename(src, dst)

            rolling_state.os.replace = crash_replace
            journal.publish_ready(second.job_id, data)

    assert _die_in_child(crash) == 87
    directory = next((tmp_path / "ariadne" / "rolling").glob("*/"))
    leftovers = [p.name for p in directory.rglob(".rolling-*")]
    assert leftovers, "crash must leave an orphaned temporary file"
    for _ in range(2):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            assert journal.load_view() == g1
            assert journal.report() == report
            assert journal.inspect(second.job_id)["state"] == "ready"
    assert not list(directory.rglob(".rolling-*"))
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        g2 = journal.publish_ready(second.job_id, data)
        assert journal.load_view() == g2 and g2.parent_generation == g1.generation_id
        assert journal.report()["logical_reservations"] == report["logical_reservations"] == 2


def test_non_regular_temporary_name_is_refused_not_deleted(tmp_path):
    store, settings, g1, *_ = _published_g1_ready_g2(tmp_path)
    directory = next((tmp_path / "ariadne" / "rolling").glob("*/"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(b"keep")
    link = directory / "generations" / ".rolling-abc123"
    link.symlink_to(elsewhere)
    with pytest.raises(ValueError):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    assert link.is_symlink() and elsewhere.read_bytes() == b"keep"


@pytest.mark.parametrize("pragma", ["user_version=1", "application_id=7"])
def test_tableless_database_with_identity_header_is_not_treated_as_fresh(tmp_path, pragma):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        path = journal.path
    path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA " + pragma)
    path.chmod(0o600)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported rolling journal schema"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    assert path.read_bytes() == before


def test_truncated_journal_with_unpublished_charges_never_resets(tmp_path):
    store, settings = setup(tmp_path)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        ready(journal, target(store, settings, 1))
        assert journal.report()["logical_reservations"] == 1
        path = journal.path
    assert not any(journal.generations.iterdir()) and not journal.manifest_path.exists()
    path.write_bytes(b"")
    for _ in range(2):
        with pytest.raises(ValueError):
            rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    assert path.read_bytes() == b""
