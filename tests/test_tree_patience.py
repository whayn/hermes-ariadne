"""UniiChat's patient builder (#70 phase 3): failures delay a node, never stop it.

Facade errors, timeouts, invalid results and lost owners are re-asked automatically
after a backoff (#63); an idle worker wakes itself for due retries and unreadable
sources, and survives planning errors (#67); built import leaves never hold back a
live job's context (#59).
"""

import sqlite3
import threading
import time

import pytest
from test_tree_builder import CAPS, FakeLlm, Log, builder, dead_owner, drain, long, wait_for

from hermes_ariadne import profile_tree as pt
from hermes_ariadne import tree_builder
from hermes_ariadne.tree import Part
from hermes_ariadne.tree_builder import TreeBuilder, TreeSettings
from hermes_ariadne.tree_store import TreeStore


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o755)
    return home


@pytest.fixture
def log(home):
    return Log(home)


def clocked(home, llm, now, **settings):
    settings.setdefault("jobs", 4)
    return TreeBuilder(home, llm, TreeSettings(**settings), clock=lambda: now[0])


# -- backoff -------------------------------------------------------------------------------


def test_a_failed_call_is_asked_again_after_a_growing_backoff(home, log):
    log.add(long())
    failures = {1, 2}
    llm = FakeLlm(lambda n, m, k: TimeoutError("slow") if n in failures else "fine")
    now = [1000.0]
    b = clocked(home, llm, now, retry_base=30, retry_max=3600)
    drain(b)
    assert len(llm.calls) == 1
    job = log.store.jobs()[Part(0, 0)]
    assert (job.state, job.error_type) == ("uncertain", "TimeoutError")
    now[0] += 29
    drain(b)
    assert len(llm.calls) == 1  # not due yet
    now[0] += 2
    drain(b)
    assert len(llm.calls) == 2  # due after 30 s; fails again
    now[0] += 59
    drain(b)
    assert len(llm.calls) == 2  # the second delay doubled to 60 s
    now[0] += 2
    drain(b)
    assert len(llm.calls) == 3 and log.store.nodes()[Part(0, 0)] == "fine"
    assert log.store.counters()["logical_jobs"] == 3  # every re-ask is a reservation


def test_the_backoff_is_capped(home, log):
    log.add(long())
    llm = FakeLlm(lambda n, m, k: RuntimeError("down"))
    now = [0.0]
    b = clocked(home, llm, now, retry_base=10, retry_max=40)
    for _ in range(6):
        drain(b)
        now[0] += 40  # never more than retry_max between asks
    assert len(llm.calls) == 6


def test_invalid_results_and_lost_owners_are_asked_again(home, log):
    log.add(long())
    log.add(long("y"))
    log.store.reserve(Part(0, 1), 10, dead_owner(), **CAPS)  # lost with its process
    llm = FakeLlm(lambda n, m, k: b"bytes" if n == 1 else "ok")
    now = [0.0]
    b = clocked(home, llm, now, retry_base=5)
    drain(b)
    jobs = log.store.jobs()
    assert jobs[Part(0, 0)].state == "uncertain"  # a non-text answer
    assert Part(0, 1) in log.store.nodes()  # the lost one was asked again at once
    now[0] += 6
    drain(b)
    assert {Part(0, 0), Part(0, 1)} <= set(log.store.nodes())


def test_a_node_lost_again_and_again_is_still_asked(home, log):
    log.add(long())
    for _ in range(3):
        log.store.reserve(Part(0, 0), 10, dead_owner(), **CAPS)
        TreeBuilder(home, FakeLlm()).recover()
        assert log.store.rearm_due(time.time()) == [Part(0, 0)]
    llm = FakeLlm()
    drain(builder(home, llm))
    assert log.store.nodes()[Part(0, 0)] == "summary 1"


def test_an_explicit_retry_skips_the_wait_and_resets_the_backoff(home, log):
    log.add(long())
    llm = FakeLlm(lambda n, m, k: RuntimeError("down") if n < 3 else "ok")
    now = [0.0]
    b = clocked(home, llm, now, retry_base=100, retry_max=10000)
    drain(b)
    now[0] += 100
    drain(b)  # second failure: next ask in 200 s
    assert b.retry() == [Part(0, 0)]
    drain(b)
    assert log.store.nodes()[Part(0, 0)] == "ok"


def test_status_reports_retrying_jobs_and_the_next_retry_without_detail(home, log):
    log.add(long())
    llm = FakeLlm(lambda n, m, k: RuntimeError("provider said secret-detail"))
    now = [500.0]
    b = clocked(home, llm, now, retry_base=30)
    drain(b)
    now[0] += 10
    status = b.status()
    assert status["retrying"] == 1 and status["next_retry_in"] == 20
    assert status["last_error"] == "RuntimeError" and "secret-detail" not in repr(status)


def test_a_store_written_before_backoff_retries_its_blocked_jobs(home, log):
    log.add(long())
    log.store.reserve(Part(0, 0), 10, "me:x", **CAPS)
    log.store.mark(Part(0, 0), "uncertain", "TimeoutError", owner="me:x", attempt=1)
    conn = sqlite3.connect(log.store.path)  # as v0.3.0 left it: no retry schedule
    with conn:
        conn.execute("UPDATE jobs SET retry_at=NULL, failures=0")
    conn.close()
    llm = FakeLlm()
    drain(builder(home, llm))
    assert log.store.nodes()[Part(0, 0)] == "summary 1"


def test_an_outage_is_probed_once_per_backoff_not_once_per_summary(home, log):
    for i in range(10):
        log.add(long(str(i)))
    down = [True]
    llm = FakeLlm(lambda n, m, k: ConnectionError("down") if down[0] else "ok")
    now = [0.0]
    b = clocked(home, llm, now, jobs=1, retry_base=30)
    drain(b)
    assert len(llm.calls) == 2  # two failures in a row pause every new call
    now[0] += 31
    drain(b)
    assert len(llm.calls) <= 4  # probes when the retries are due, not one per summary
    down[0] = False
    now[0] += 61
    drain(b)
    assert all(Part(0, i) in log.store.nodes() for i in range(10))  # back: all of it


def test_a_single_failing_node_does_not_pause_its_lane(home, log):
    for i in range(4):
        log.add(long(str(i)))
    llm = FakeLlm(lambda n, m, k: ValueError("400") if "compress message 0" in str(m) else "ok")
    now = [0.0]
    b = clocked(home, llm, now, jobs=1, retry_base=30)
    drain(b)
    # leaf 0 keeps failing, the provider works: the other leaves are not held back
    assert all(Part(0, i) in log.store.nodes() for i in (1, 2, 3))


def test_an_import_outage_never_pauses_the_live_lane(home, log):
    log.add(long("i"), session="old", origin="import")
    llm = FakeLlm(lambda n, m, k: ConnectionError("down") if "LLLL" not in str(m) else "ok")
    now = [0.0]
    b = clocked(home, llm, now, jobs=1, retry_base=3600)
    drain(b)
    log.add(long("k"), session="old", origin="import")
    drain(b)  # two import failures in a row: the import lane pauses
    assert b.status()["paused_for"] is not None
    live = log.add(long("L"), session="now", origin="live")
    drain(b)
    assert Part(0, live) in log.store.nodes()


def test_a_leaf_built_elsewhere_does_not_keep_the_worker_spinning(home, log):
    position = log.add(long())
    b = builder(home, FakeLlm())
    b._unavailable[position] = 0.0  # read failed once, long due
    log.store.put_node(Part(0, position), "built by another process")
    plans = []
    plan = b._plan
    b._plan = lambda room: plans.append(1) or plan(room)
    b.start()
    try:
        time.sleep(1.5)
    finally:
        b.stop()
    assert len(plans) <= 3 and b._unavailable == {}


def test_a_store_error_does_not_retire_the_worker_with_retries_pending(home, log):
    log.add(long())
    b = builder(home, FakeLlm(), retry_base=1)
    reserve, failed = b._reserve, threading.Event()

    def flaky(part, size):
        if not failed.is_set():
            failed.set()
            raise sqlite3.OperationalError("database is locked")
        return reserve(part, size)

    b._reserve = flaky
    b.start()
    try:
        wait_for(lambda: Part(0, 0) in log.store.nodes(), timeout=10)
    finally:
        b.stop()


def test_a_corrupt_source_is_not_read_again_and_again(home, log, monkeypatch):
    position = log.add(long())
    reads = []

    def corrupt(pos):
        reads.append(pos)
        raise tree_builder._Unavailable("source_invalid")

    now = [0.0]
    b = clocked(home, FakeLlm(), now)
    monkeypatch.setattr(b, "_read_source", corrupt)
    for _ in range(5):
        drain(b)
        now[0] += 3600
    assert reads == [position]  # permanent: never polled again
    assert b._next_due() is None and b.status()["unreadable"] == 1


# -- settings ------------------------------------------------------------------------------


def test_retry_settings_have_defaults_bounds_and_the_old_key_is_accepted():
    s = TreeSettings()
    assert (s.retry_base, s.retry_max) == (30, 3600)
    for bad in ({"retry_base": 0}, {"retry_max": 0}, {"retry_base": 7200, "retry_max": 60}):
        try:
            TreeSettings(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    # v0.3.0 configs may set auto_reask_lost: accepted, no longer meaningful
    assert TreeSettings.from_config({"auto_reask_lost": 0}) == TreeSettings()


# -- the worker wakes itself ---------------------------------------------------------------


def test_the_worker_retries_a_failed_call_without_a_kick(home, log):
    log.add(long())
    llm = FakeLlm(lambda n, m, k: RuntimeError("down") if n == 1 else "back")
    b = builder(home, llm, retry_base=1)
    b.start()
    try:
        wait_for(lambda: Part(0, 0) in log.store.nodes(), timeout=10)
    finally:
        b.stop()
    assert len(llm.calls) == 2


def test_the_worker_reads_an_unavailable_source_again_without_a_kick(home, log, monkeypatch):
    monkeypatch.setattr(tree_builder, "UNAVAILABLE_RETRY", 0.3)
    position = log.add(long())
    archive = home / "ariadne" / "events.sqlite3"
    hidden = archive.with_name("hidden.sqlite3")
    archive.rename(hidden)
    b = builder(home, FakeLlm())
    b.start()
    try:
        wait_for(lambda: b.status()["last_error"] == "source_unavailable")
        hidden.rename(archive)
        wait_for(lambda: Part(0, position) in log.store.nodes(), timeout=10)
    finally:
        b.stop()


def test_a_planning_error_does_not_end_the_worker(home, log):
    log.add(long())
    b = builder(home, FakeLlm(), retry_base=1)
    plan, failed = b._plan, threading.Event()

    def flaky(room):
        if not failed.is_set():
            failed.set()
            raise sqlite3.OperationalError("database is locked")
        return plan(room)

    b._plan = flaky
    b.start()
    try:
        wait_for(lambda: Part(0, 0) in log.store.nodes(), timeout=10)
    finally:
        b.stop()
    assert failed.is_set()


def test_stop_ends_a_worker_waiting_for_a_retry_at_once(home, log):
    log.add(long())
    b = builder(home, FakeLlm(lambda n, m, k: RuntimeError("down")), retry_base=3600)
    b.start()
    wait_for(lambda: log.store.jobs().get(Part(0, 0)) is not None)
    wait_for(lambda: b.status()["running"] == 0)
    started = time.monotonic()
    b.stop(timeout=5)
    assert time.monotonic() - started < 2
    assert b._thread is None


# -- lanes (#59) ---------------------------------------------------------------------------


def test_built_import_leaves_never_hold_back_a_live_job(home, log):
    text = "i" * 470
    source = log.sources[log.add(text, session="old", origin="import")]
    assert pt.leaf_text_if_fits(source, 512) is not None  # fits: built without a call
    for _ in range(69):
        log.add(text, session="old", origin="import")
    live = log.add(long("L"), session="now", origin="live")
    llm = FakeLlm()
    b = builder(home, llm, jobs=1, max_jobs=1)  # one call in total
    b.step()
    assert llm.what() == [("leaf", live)]
    content = llm.calls[0][0][1]["content"]
    lines = content.split("<chat>\n", 1)[1].split("</chat>", 1)[0].splitlines()
    texts = [line.split("|", 1)[1] for line in lines]
    assert sum(len(t.encode()) for t in texts) <= TreeSettings().seen_high
    assert lines[0].startswith("7+1|") and lines[-1].startswith("69+1|")  # oldest dropped
    store = TreeStore(home)
    assert Part(0, live) in store.nodes()
