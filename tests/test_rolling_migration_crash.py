"""Process-death probes at every Python-visible fsync in an offline conversion.

SQLite internal transaction fsyncs are covered by SQLite's transaction protocol,
not intercepted here; named tests also cover both sides of the schema fence.
"""

import os
import shutil
import sqlite3

from test_rolling_state import ready, setup, target

from hermes_ariadne import rolling, rolling_migrate, rolling_state
from hermes_ariadne.store import EventStore


def test_every_conversion_fsync_keeps_a_charged_ledger(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir(mode=0o700)
    store, settings = setup(seed)
    with rolling_state.RollingJournal(seed, "s", settings, store=store) as journal:
        frozen = target(store, settings, 1)
        ready(journal, frozen)
        journal.publish_ready(
            frozen.job_id,
            rolling.generation_bytes(frozen, "summary", sequence=1, settings=settings),
        )
        relative = journal.directory.relative_to(seed)
    (seed / ".ariadne-offline-migration").touch(mode=0o600)
    trace = tmp_path / "trace"
    shutil.copytree(seed, trace)
    trace_store = EventStore(trace / "ariadne/events.sqlite3")
    calls, real = [], os.fsync

    def observed(fd):
        real(fd)
        calls.append(fd)

    with monkeypatch.context() as context:
        context.setattr(os, "fsync", observed)
        rolling_migrate.migrate(trace, "s", settings, store=trace_store)
    assert len(calls) > 10, "the real durable conversion must have been exercised"
    for boundary in range(1, len(calls) + 1):
        home = tmp_path / f"fault-{boundary}"
        shutil.copytree(seed, home)
        store = EventStore(home / "ariadne/events.sqlite3")
        pid = os.fork()
        if pid == 0:
            count = 0

            def killed(fd):
                nonlocal count
                real(fd)
                count += 1
                if count == boundary:
                    os._exit(73)

            os.fsync = killed
            try:
                rolling_migrate.migrate(home, "s", settings, store=store)
                os._exit(0)
            except BaseException:
                os._exit(74)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 73, boundary
        try:
            with rolling_state.RollingJournal(home, "s", settings, store=store) as journal:
                assert journal.report()["logical_reservations"] == 1
                assert journal.load_view() is not None
        except (ValueError, FileNotFoundError):
            # Explicit refusal is allowed; a usable empty/reset ledger is not.
            pass
        with sqlite3.connect(home / relative / "journal.sqlite3") as conn:
            assert conn.execute("SELECT logical_reservations FROM campaign").fetchone()[0] == 1
        assert (home / "ariadne/events.sqlite3").is_file()
