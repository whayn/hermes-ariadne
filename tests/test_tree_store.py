"""Profile tree store: admission, append-only log, nodes, jobs, views, backfill."""

import hashlib
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_ariadne import tree_store
from hermes_ariadne.private_paths import StorageRefused
from hermes_ariadne.tree import Part, View
from hermes_ariadne.tree_store import TreeStore


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o755)
    os.chmod(home, 0o755)  # Hermes creates 0755 profile homes
    return home


def eid(text):
    return hashlib.sha256(text.encode()).hexdigest()


def entry(text, role="user", date=None, restricted=False):
    return {
        "event_id": eid(text),
        "message_uid": "uid-" + text,
        "role": role,
        "date": date,
        "restricted": restricted,
    }


# -- admission ----------------------------------------------------------------------


def test_reads_never_create_the_store(home):
    store = TreeStore(home)
    assert store.count() == 0
    assert store.entries(0, 10) == []
    assert store.entry(0) is None
    assert dict(store.sessions()) == {}
    assert dict(store.nodes()) == {}
    assert store.load_view("chat") == (None, 0)
    assert not (home / "ariadne").exists()


def test_open_hermes_home_is_accepted_and_own_files_are_private(home):
    store = TreeStore(home)
    store.prepare()
    assert stat.S_IMODE(home.stat().st_mode) == 0o755
    assert stat.S_IMODE((home / "ariadne").stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "ariadne" / "tree.sqlite3").stat().st_mode) == 0o600
    assert not list((home / "ariadne").glob(".tree-init-*"))


def test_symlinked_home_is_refused(home, tmp_path):
    alias = tmp_path / "alias"
    os.symlink(home, alias)
    with pytest.raises(StorageRefused):
        TreeStore(alias).append("s", [entry("a")], origin="live")
    assert not (home / "ariadne").exists()


def test_symlinked_ariadne_directory_is_refused(home, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    os.symlink(elsewhere, home / "ariadne")
    with pytest.raises(StorageRefused):
        TreeStore(home).prepare()
    assert list(elsewhere.iterdir()) == []


def test_relative_home_is_refused():
    with pytest.raises(StorageRefused):
        TreeStore("relative/home").prepare()


def test_drifted_modes_are_tightened(home):
    TreeStore(home).append("s", [entry("a")], origin="live")
    path = home / "ariadne" / "tree.sqlite3"
    os.chmod(home / "ariadne", 0o755)
    os.chmod(path, 0o644)
    assert TreeStore(home).count() == 1
    assert stat.S_IMODE((home / "ariadne").stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_hardlinked_store_is_refused(home, tmp_path):
    TreeStore(home).prepare()
    os.link(home / "ariadne" / "tree.sqlite3", tmp_path / "alias")
    with pytest.raises(StorageRefused):
        TreeStore(home).count()


def test_foreign_application_id_is_refused(home):
    (home / "ariadne").mkdir(mode=0o700)
    path = home / "ariadne" / "tree.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA application_id=1234")
    conn.execute("PRAGMA user_version=1")  # only the application id is foreign
    conn.execute("CREATE TABLE x(y)")
    conn.commit()
    conn.close()
    os.chmod(path, 0o600)
    with pytest.raises(StorageRefused):
        TreeStore(home).count()
    with pytest.raises(StorageRefused):
        TreeStore(home).append("s", [entry("a")], origin="live")


def test_store_does_not_share_the_note_store_file(home):
    from hermes_ariadne.notes import NoteStore

    NoteStore(home).add("a fact")
    TreeStore(home).append("s", [entry("a")], origin="live")
    assert NoteStore(home).list()["total"] == 1
    assert TreeStore(home).count() == 1


# -- append-only log ----------------------------------------------------------------


def test_append_assigns_positions_from_zero_and_reads_back(home):
    store = TreeStore(home)
    positions = store.append(
        "session-a",
        [entry("hello", date="2026-11-21T10:00:00Z"), entry("hi", role="assistant")],
        origin="live",
        title="Greeting",
        source="cli",
    )
    assert positions == [0, 1]
    assert store.count() == 2
    first = store.entry(0)
    assert first.position == 0 and first.session_id == "session-a"
    assert first.event_id == eid("hello") and first.message_uid == "uid-hello"
    assert first.role == "user" and first.date == "2026-11-21T10:00:00Z"
    assert first.origin == "live" and first.restricted is False
    assert [e.position for e in store.entries(0, 5)] == [0, 1]
    assert [e.event_id for e in store.entries(1, 2)] == [eid("hi")]


def test_append_is_idempotent_per_session_and_event(home):
    first = TreeStore(home).append("s", [entry("a"), entry("b")], origin="live")
    again = TreeStore(home).append("s", [entry("b"), entry("c"), entry("a")], origin="live")
    assert first == [0, 1]
    assert again == [1, 2, 0]
    assert TreeStore(home).count() == 3
    # The same event in another conversation is another log entry.
    assert TreeStore(home).append("t", [entry("a")], origin="import") == [3]


def test_duplicates_inside_one_batch_get_one_position(home):
    assert TreeStore(home).append("s", [entry("a"), entry("a")], origin="live") == [0, 0]
    assert TreeStore(home).count() == 1


def test_sessions_legend_tracks_tag_title_and_span(home):
    store = TreeStore(home)
    store.append("s1", [entry("a")], origin="live", title="First", source="cli")
    store.append("s2", [entry("b")], origin="import", title="Imported")
    store.append("s1", [entry("c")], origin="live")  # title kept when not given
    store.append("s1", [], origin="live", title="Renamed")  # metadata-only update
    legend = store.sessions()
    assert list(legend) == ["s1", "s2"]  # ordered by first position
    s1 = legend["s1"]
    assert s1.tag == hashlib.sha256(b"s1").hexdigest()[:8]
    assert (s1.title, s1.source, s1.origin) == ("Renamed", "cli", "live")
    assert (s1.first_position, s1.last_position) == (0, 2)
    assert legend["s2"].origin == "import" and legend["s2"].source is None
    with pytest.raises(TypeError):
        legend["s3"] = s1  # read-only snapshot


def test_empty_append_for_unknown_session_creates_nothing(home):
    assert TreeStore(home).append("s", [], origin="live", title="x") == []
    assert dict(TreeStore(home).sessions()) == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"event_id": "nothex", "message_uid": "u", "role": "user"},
        {"event_id": eid("x"), "message_uid": "", "role": "user"},
        {"event_id": eid("x"), "message_uid": "u", "role": "wizard"},
        {"event_id": eid("x"), "message_uid": "u", "role": "user", "date": 3},
        {"event_id": eid("x"), "message_uid": "u", "role": "user", "restricted": "no"},
    ],
)
def test_append_rejects_invalid_entries_without_writing(home, bad):
    store = TreeStore(home)
    with pytest.raises(ValueError):
        store.append("s", [entry("ok"), bad], origin="live")
    assert store.count() == 0


def test_append_rejects_bad_session_and_origin(home):
    with pytest.raises(ValueError):
        TreeStore(home).append("", [entry("a")], origin="live")
    with pytest.raises(ValueError):
        TreeStore(home).append("s", [entry("a")], origin="elsewhere")


def test_two_processes_append_concurrently_without_gaps_or_duplicates(home, tmp_path):
    # Separate interpreters with an explicit path (like the other fixtures): a spawned
    # multiprocessing child would re-import this module and depend on the runner's state.
    TreeStore(home).prepare()
    src = Path(tree_store.__file__).resolve().parents[1]
    worker = Path(__file__).with_name("fixtures") / "tree_store_writer.py"
    start = tmp_path / "start"
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), str(src), str(home), name, "40", str(start)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name in "ab"
    ]
    start.touch()
    for proc in procs:
        _out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
    store = TreeStore(home)
    rows = store.entries(0, store.count() + 1)
    assert [row.position for row in rows] == list(range(80))
    assert len({(row.session_id, row.event_id) for row in rows}) == 80
    for name in "ab":  # each writer's own order is preserved
        mine = [row.event_id for row in rows if row.session_id.startswith(name + "-")]
        assert mine == [eid(f"{name}-{i}") for i in range(40)]
    legend = store.sessions()
    assert len(legend) == 6
    for session in legend.values():
        own = [row.position for row in rows if row.session_id == session.session_id]
        assert (session.first_position, session.last_position) == (own[0], own[-1])


# -- summary nodes ------------------------------------------------------------------


def _log(store, count):
    store.append("s", [entry(f"m{i}") for i in range(count)], origin="live")


def test_put_node_and_nodes_snapshot(home):
    store = TreeStore(home)
    _log(store, 4)
    assert store.put_node(Part(0, 1), "leaf one") is True
    assert store.put_node(Part(1, 0), "pair") is True
    assert store.put_node(Part(1, 0), "other text") is False  # built nodes never change
    nodes = store.nodes()
    assert dict(nodes) == {Part(0, 1): "leaf one", Part(1, 0): "pair"}
    with pytest.raises(TypeError):
        nodes[Part(0, 0)] = "x"


def test_put_node_refuses_parts_beyond_the_log(home):
    store = TreeStore(home)
    _log(store, 3)
    with pytest.raises(ValueError):
        store.put_node(Part(2, 0), "covers positions 0..3 but only 3 exist")
    with pytest.raises(TypeError):
        store.put_node((0, 0), "not a part")
    with pytest.raises(ValueError):
        store.put_node(Part(0, 0), 5)


# -- summary jobs -------------------------------------------------------------------


def test_reserve_is_exclusive_and_accounted(home):
    store = TreeStore(home)
    _log(store, 4)
    caps = {"max_jobs": 10, "max_total_bytes": 1000}
    assert store.reserve(Part(1, 0), 100, "builder-1", **caps) == 1
    assert store.reserve(Part(1, 0), 100, "builder-2", **caps) is False  # reserved
    assert store.counters() == {"logical_jobs": 1, "request_bytes": 100}
    job = store.jobs()[Part(1, 0)]
    assert (job.state, job.attempts, job.request_bytes, job.owner) == (
        "reserved",
        1,
        100,
        "builder-1",
    )
    store.finish(Part(1, 0), "summary", owner="builder-1", attempt=1)
    assert store.nodes()[Part(1, 0)] == "summary"
    assert store.jobs()[Part(1, 0)].state == "done"
    assert store.reserve(Part(1, 0), 100, "builder-1", **caps) is False  # done
    assert store.counters() == {"logical_jobs": 1, "request_bytes": 100}


def test_reserve_refuses_when_caps_would_be_exceeded(home):
    store = TreeStore(home)
    _log(store, 8)
    assert store.reserve(Part(0, 0), 60, "b", max_jobs=2, max_total_bytes=100) == 1
    assert store.reserve(Part(0, 1), 41, "b", max_jobs=2, max_total_bytes=100) is False
    assert store.reserve(Part(0, 1), 40, "b", max_jobs=2, max_total_bytes=100) == 1
    assert store.reserve(Part(0, 2), 0, "b", max_jobs=2, max_total_bytes=100) is False
    assert store.counters() == {"logical_jobs": 2, "request_bytes": 100}
    assert Part(0, 2) not in store.jobs()


def test_reserve_refuses_already_built_and_out_of_range_parts(home):
    store = TreeStore(home)
    _log(store, 2)
    store.put_node(Part(0, 0), "fits as is")
    caps = {"max_jobs": 10, "max_total_bytes": 1000}
    assert store.reserve(Part(0, 0), 10, "b", **caps) is False
    with pytest.raises(ValueError):
        store.reserve(Part(1, 1), 10, "b", **caps)
    with pytest.raises(ValueError):
        store.reserve(Part(0, 1), -1, "b", **caps)
    with pytest.raises(ValueError):
        store.reserve(Part(0, 1), 1, "", **caps)


def test_refused_attempt_can_be_reserved_again_and_counts_every_attempt(home):
    store = TreeStore(home)
    _log(store, 2)
    caps = {"max_jobs": 10, "max_total_bytes": 1000}
    assert store.reserve(Part(0, 0), 50, "b", **caps) == 1
    store.mark(Part(0, 0), "refused", "over_length", owner="b", attempt=1)
    assert store.jobs()[Part(0, 0)].error_type == "over_length"
    assert store.reserve(Part(0, 0), 70, "b", **caps) == 2
    job = store.jobs()[Part(0, 0)]
    assert (job.state, job.attempts, job.request_bytes) == ("reserved", 2, 120)
    assert store.counters() == {"logical_jobs": 2, "request_bytes": 120}


def test_uncertain_job_stays_blocked_until_retry(home):
    store = TreeStore(home)
    _log(store, 4)
    caps = {"max_jobs": 10, "max_total_bytes": 1000}
    assert store.reserve(Part(0, 0), 10, "b", **caps)
    assert store.reserve(Part(0, 1), 10, "b", **caps)
    store.mark(Part(0, 0), "uncertain", "TimeoutError", owner="b", attempt=1)
    store.mark(Part(0, 1), "uncertain", "ConnectionError", owner="b", attempt=1)
    assert store.reserve(Part(0, 0), 10, "b", **caps) is False  # never resubmitted
    assert store.retry([Part(0, 0)]) == [Part(0, 0)]
    assert store.jobs()[Part(0, 1)].state == "uncertain"
    assert store.reserve(Part(0, 0), 10, "b", **caps) == 2
    assert store.retry() == [Part(0, 1)]  # all uncertain jobs
    assert store.retry() == []
    assert store.counters() == {"logical_jobs": 3, "request_bytes": 30}  # never decrease


def test_mark_and_finish_require_a_reservation(home):
    store = TreeStore(home)
    _log(store, 2)
    with pytest.raises(ValueError):
        store.finish(Part(0, 0), "never reserved", owner="b", attempt=1)
    with pytest.raises(ValueError):
        store.mark(Part(0, 0), "uncertain", "X", owner="b", attempt=1)
    store.reserve(Part(0, 0), 1, "b", max_jobs=5, max_total_bytes=5)
    with pytest.raises(ValueError):
        store.mark(Part(0, 0), "done", None, owner="b", attempt=1)  # only through finish()
    store.mark(Part(0, 0), "uncertain", "X", owner="b", attempt=1)
    with pytest.raises(ValueError):
        store.finish(Part(0, 0), "late answer after an ambiguous call", owner="b", attempt=1)
    assert dict(store.nodes()) == {}


def test_stale_reservations_become_uncertain_never_free(home):
    store = TreeStore(home)
    _log(store, 4)
    caps = {"max_jobs": 10, "max_total_bytes": 1000}
    store.reserve(Part(0, 0), 1, "dead-owner", **caps)
    store.reserve(Part(0, 1), 1, "live-owner", **caps)
    assert store.stale_reserved(lambda owner: owner == "live-owner") == [Part(0, 0)]
    jobs = store.jobs()
    assert jobs[Part(0, 0)].state == "uncertain"
    assert jobs[Part(0, 0)].error_type == "owner_lost"
    assert jobs[Part(0, 1)].state == "reserved"
    assert store.reserve(Part(0, 0), 1, "live-owner", **caps) is False
    assert store.stale_reserved({"live-owner"}) == []  # a collection of live owners


def test_late_calls_from_a_replaced_reservation_are_fenced(home):
    store = TreeStore(home)
    _log(store, 2)
    caps = {"max_jobs": 10, "max_total_bytes": 1000}
    part = Part(0, 0)
    first = store.reserve(part, 10, "old", **caps)
    assert first == 1
    assert store.stale_reserved(lambda owner: False) == [part]
    assert store.retry([part]) == [part]
    second = store.reserve(part, 10, "new", **caps)
    assert second == 2
    # the original worker comes back late: refused, nothing written, new job untouched
    with pytest.raises(ValueError):
        store.finish(part, "stale answer", owner="old", attempt=first)
    with pytest.raises(ValueError):
        store.mark(part, "uncertain", "Late", owner="old", attempt=first)
    with pytest.raises(ValueError):  # right attempt, wrong owner
        store.finish(part, "x", owner="old", attempt=second)
    with pytest.raises(ValueError):  # right owner, wrong attempt
        store.mark(part, "refused", "x", owner="new", attempt=first)
    assert dict(store.nodes()) == {}
    job = store.jobs()[part]
    assert (job.state, job.owner, job.attempts) == ("reserved", "new", 2)
    store.finish(part, "fresh answer", owner="new", attempt=second)
    assert store.nodes()[part] == "fresh answer"
    assert store.jobs()[part].state == "done"


# -- views --------------------------------------------------------------------------


def test_views_round_trip_with_compare_and_swap(home):
    store = TreeStore(home)
    _log(store, 4)
    view = View((Part(1, 0), Part(0, 2)), size=12, batch=True, high=500)
    assert store.save_view("chat", view, 0) is True
    assert store.load_view("chat") == (view, 1)
    assert store.load_view("seen") == (None, 0)
    newer = View((Part(1, 0), Part(1, 1)), size=20, batch=False, high=500)
    assert store.save_view("chat", newer, 0) is False  # stale writer loses
    assert store.load_view("chat") == (view, 1)
    assert store.save_view("chat", newer, 1) is True
    assert store.load_view("chat") == (newer, 2)


def test_views_are_validated(home):
    store = TreeStore(home)
    _log(store, 2)
    with pytest.raises(ValueError):
        store.save_view("other", View(), 0)
    with pytest.raises(TypeError):
        store.save_view("chat", ((0, 0),), 0)
    with pytest.raises(ValueError):
        store.save_view("chat", View((Part(2, 0),)), 0)  # beyond the log


# -- schema version and migrations --------------------------------------------------


def test_migrations_upgrade_older_stores_and_unknown_versions_refuse(home, monkeypatch):
    assert tree_store.SCHEMA_VERSION == 2 and set(tree_store.MIGRATIONS) == {1}
    TreeStore(home).append("s", [entry("a")], origin="live")
    monkeypatch.setattr(tree_store, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(
        tree_store,
        "MIGRATIONS",
        {2: ("CREATE TABLE IF NOT EXISTS probe (x)", lambda conn: None)},
    )
    assert TreeStore(home).count() == 1
    conn = sqlite3.connect(home / "ariadne" / "tree.sqlite3")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert conn.execute("SELECT name FROM sqlite_schema WHERE name='probe'").fetchone()
    conn.execute("PRAGMA user_version=7")
    conn.commit()
    conn.close()
    with pytest.raises(StorageRefused):
        TreeStore(home).count()


# The exact v1 schema (before ``jobs.auto_retries``), as an existing dev profile has it.
V1_SCHEMA = (
    """CREATE TABLE log (
        position INTEGER PRIMARY KEY CHECK (position >= 0),
        session_id TEXT NOT NULL, message_uid TEXT NOT NULL, event_id TEXT NOT NULL,
        role TEXT NOT NULL, date TEXT, origin TEXT NOT NULL,
        restricted INTEGER NOT NULL CHECK (restricted IN (0, 1)),
        UNIQUE (session_id, event_id)
    )""",
    """CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY, tag TEXT NOT NULL UNIQUE, title TEXT, source TEXT,
        origin TEXT NOT NULL, first_position INTEGER NOT NULL,
        last_position INTEGER NOT NULL
    )""",
    """CREATE TABLE nodes (
        level INTEGER NOT NULL, idx INTEGER NOT NULL, text TEXT NOT NULL,
        bytes INTEGER NOT NULL, built_at REAL NOT NULL, PRIMARY KEY (level, idx)
    )""",
    """CREATE TABLE jobs (
        level INTEGER NOT NULL, idx INTEGER NOT NULL,
        state TEXT NOT NULL CHECK (state IN
            ('reserved', 'done', 'uncertain', 'refused', 'armed')),
        attempts INTEGER NOT NULL, request_bytes INTEGER NOT NULL, owner TEXT,
        reserved_at REAL, finished_at REAL, error_type TEXT, PRIMARY KEY (level, idx)
    )""",
    """CREATE TABLE counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)""",
    """INSERT INTO counters VALUES ('logical_jobs', 0), ('request_bytes', 0)""",
    """CREATE TABLE views (
        name TEXT PRIMARY KEY CHECK (name IN ('chat', 'seen')),
        parts_json TEXT NOT NULL, size INTEGER NOT NULL, batch INTEGER NOT NULL,
        high INTEGER NOT NULL, revision INTEGER NOT NULL
    )""",
)


def _v1_store(home):
    """A real v1 store with two log rows, an uncertain owner_lost job and a done job."""
    directory = home / "ariadne"
    directory.mkdir(mode=0o700)
    path = directory / "tree.sqlite3"
    conn = sqlite3.connect(path)
    with conn:
        for sql in V1_SCHEMA:
            conn.execute(sql)
        conn.execute(f"PRAGMA application_id={tree_store.APPLICATION_ID}")
        conn.execute("PRAGMA user_version=1")
        for position, text in enumerate("ab"):
            conn.execute(
                "INSERT INTO log VALUES(?,?,?,?,?,?,?,?)",
                (position, "s", "uid-" + text, eid(text), "user", None, "live", 0),
            )
        conn.execute("INSERT INTO jobs VALUES(0,0,'uncertain',2,200,'1:x',1.0,2.0,'owner_lost')")
        conn.execute("INSERT INTO jobs VALUES(0,1,'done',1,100,'1:x',1.0,2.0,NULL)")
        conn.execute("UPDATE counters SET value=3 WHERE name='logical_jobs'")
        conn.execute("UPDATE counters SET value=300 WHERE name='request_bytes'")
    conn.close()
    path.chmod(0o600)
    return path


def test_v1_store_migrates_in_place_to_v2_keeping_every_job(home):
    path = _v1_store(home)
    store = TreeStore(home)
    assert store.count() == 2  # a plain read migrates
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    columns = {row[1]: row for row in conn.execute("PRAGMA table_info(jobs)")}
    _, _, kind, not_null, default, _ = columns["auto_retries"]
    assert (kind, not_null, default) == ("INTEGER", 1, "0")
    conn.close()
    jobs = store.jobs()
    assert jobs[Part(0, 0)].auto_retries == 0 and jobs[Part(0, 1)].auto_retries == 0
    assert (jobs[Part(0, 0)].state, jobs[Part(0, 0)].error_type) == ("uncertain", "owner_lost")
    assert jobs[Part(0, 0)].attempts == 2 and jobs[Part(0, 0)].request_bytes == 200
    assert store.counters() == {"logical_jobs": 3, "request_bytes": 300}
    assert store.rearm_lost(1) == [Part(0, 0)]  # the migrated column is usable
    # migrating again is a no-op
    TreeStore(home).prepare()
    assert TreeStore(home).jobs()[Part(0, 0)].auto_retries == 1


def test_v1_migration_step_is_idempotent(home):
    path = _v1_store(home)
    conn = sqlite3.connect(path)
    for step in tree_store.MIGRATIONS[1] * 2:  # already applied: must not fail
        step(conn) if callable(step) else conn.execute(step)
    conn.close()
    assert TreeStore(home).jobs()[Part(0, 0)].auto_retries == 0


def test_fresh_store_has_the_v2_schema(home):
    TreeStore(home).prepare()
    conn = sqlite3.connect(home / "ariadne" / "tree.sqlite3")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert "auto_retries" in {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    conn.close()


# -- re-arming lost jobs -------------------------------------------------------------


CAPS = {"max_jobs": 100, "max_total_bytes": 10**6}


def _lost(store, n, *, other=()):
    """Log ``n`` entries; reserve every one; lose the owner of all, then mark ``other``
    parts uncertain for a different reason first."""
    _log(store, n)
    for i in range(n):
        assert store.reserve(Part(0, i), 10, "dead", **CAPS)
    for i in other:
        store.mark(Part(0, i), "uncertain", "TimeoutError", owner="dead", attempt=1)
    store.stale_reserved(lambda owner: False)


def test_rearm_lost_only_takes_owner_lost_and_counts_nothing(home):
    store = TreeStore(home)
    _lost(store, 3, other=(2,))
    before = store.counters()
    assert store.rearm_lost(1) == [Part(0, 0), Part(0, 1)]
    jobs = store.jobs()
    for i in (0, 1):
        assert jobs[Part(0, i)].state == "armed" and jobs[Part(0, i)].auto_retries == 1
        assert jobs[Part(0, i)].attempts == 1 and jobs[Part(0, i)].request_bytes == 10
    assert (jobs[Part(0, 2)].state, jobs[Part(0, 2)].auto_retries) == ("uncertain", 0)
    assert jobs[Part(0, 2)].error_type == "TimeoutError"  # facade errors stay blocked
    assert store.counters() == before  # re-arming charges nothing
    assert store.rearm_lost(1) == []  # nothing left to re-arm


def test_rearmed_job_is_reserved_again_as_a_new_charged_reservation(home):
    store = TreeStore(home)
    _lost(store, 1)
    assert store.reserve(Part(0, 0), 10, "b", **CAPS) is False  # blocked until re-armed
    store.rearm_lost(1)
    assert store.reserve(Part(0, 0), 10, "b", **CAPS) == 2
    job = store.jobs()[Part(0, 0)]
    assert (job.state, job.attempts, job.auto_retries) == ("reserved", 2, 1)
    assert store.counters() == {"logical_jobs": 2, "request_bytes": 20}


def test_rearm_lost_is_capped_per_node(home):
    store = TreeStore(home)
    _lost(store, 1)
    assert store.rearm_lost(1) == [Part(0, 0)]
    store.reserve(Part(0, 0), 10, "dead2", **CAPS)
    store.stale_reserved(lambda owner: False)  # lost a second time
    assert store.jobs()[Part(0, 0)].state == "uncertain"
    assert store.rearm_lost(1) == []  # at most once
    assert store.rearm_lost(0) == []
    assert store.rearm_lost(2) == [Part(0, 0)]  # a higher cap allows another
    assert store.jobs()[Part(0, 0)].auto_retries == 2


def test_rearm_lost_zero_disables_and_validates(home):
    store = TreeStore(home)
    _lost(store, 1)
    assert store.rearm_lost(0) == []
    assert store.jobs()[Part(0, 0)].state == "uncertain"
    for bad in (-1, True, 1.5, "1", None):
        with pytest.raises(ValueError):
            store.rearm_lost(bad)


def test_rearm_lost_never_creates_the_store(home):
    assert TreeStore(home).rearm_lost(1) == [] and not (home / "ariadne").exists()


def test_explicit_retry_is_independent_of_auto_retries(home):
    store = TreeStore(home)
    _lost(store, 1)
    store.rearm_lost(1)
    store.reserve(Part(0, 0), 10, "dead2", **CAPS)
    store.stale_reserved(lambda owner: False)
    assert store.retry() == [Part(0, 0)]  # explicit retry always works
    job = store.jobs()[Part(0, 0)]
    assert (job.state, job.auto_retries) == ("armed", 1)  # untouched by retry


# -- backfill from the exact archive ------------------------------------------------


def _archive(home):
    from hermes_ariadne.store import EventStore

    return EventStore(home / "ariadne" / "events.sqlite3")


def msg(uid, content, role="user", **extra):
    return {"message_uid": uid, "role": role, "content": content, **extra}


def _event(message):
    from hermes_ariadne.store import canonical_json

    return hashlib.sha256(canonical_json(message)).hexdigest()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_backfill_orders_by_first_observation_and_position(home):
    archive = _archive(home)
    b2, b3 = msg("b2", "later in b", timestamp=1700000000.5), msg("b3", "last in b", "assistant")
    b0, b1 = msg("b0", "first in b"), msg("b1", "second in b", "tool")
    a0 = msg("a0", "only in a", "assistant", timestamp="2026-01-02T03:04:05Z")
    archive.ingest("conv-b", [b2, b3], origin="hermes_db", start=2)  # observed first
    archive.ingest("conv-a", [a0], origin="import")
    archive.ingest("conv-b", [b0, b1], origin="hermes_db")  # earlier positions, later rowids
    path = home / "ariadne" / "events.sqlite3"
    before = _digest(path)
    store = TreeStore(home)
    outcome = store.backfill(path, titles={"conv-a": "Imported chat"})
    assert outcome == {"sessions": 2, "appended": 5, "skipped": 0}
    rows = store.entries(0, 10)
    assert [r.message_uid for r in rows] == ["b0", "b1", "b2", "b3", "a0"]
    assert [r.event_id for r in rows] == [_event(m) for m in (b0, b1, b2, b3, a0)]
    assert [r.role for r in rows] == ["user", "tool", "user", "assistant", "assistant"]
    assert [r.origin for r in rows] == ["live"] * 4 + ["import"]
    assert rows[2].date == "2023-11-14T22:13:20Z" and rows[0].date is None
    assert rows[4].date == "2026-01-02T03:04:05Z"
    legend = store.sessions()
    assert list(legend) == ["conv-b", "conv-a"]
    assert legend["conv-a"].title == "Imported chat" and legend["conv-b"].title is None
    assert _digest(path) == before  # the archive is only read


def test_backfill_is_idempotent_and_appends_only_new_events(home):
    archive = _archive(home)
    archive.ingest("conv-a", [msg("a0", "x"), msg("a1", "y")], origin="hermes_db")
    path = home / "ariadne" / "events.sqlite3"
    store = TreeStore(home)
    assert store.backfill(path)["appended"] == 2
    assert store.backfill(path)["appended"] == 0
    archive.ingest("conv-z", [msg("z0", "new chat")], origin="hermes_db")
    archive.ingest("conv-a", [msg("a2", "more")], origin="hermes_db", start=2)
    assert store.backfill(path)["appended"] == 2
    assert [r.message_uid for r in store.entries(0, 10)] == ["a0", "a1", "a2", "z0"]
    assert TreeStore(home).count() == 4


def test_backfill_skips_excluded_sessions(home):
    archive = _archive(home)
    archive.ingest("primary", [msg("p0", "hello")], origin="hermes_db")
    archive.ingest("subagent", [msg("s0", "delegated work")], origin="hermes_db")
    archive.ingest("cron", [msg("c0", "scheduled run")], origin="hermes_db")
    store = TreeStore(home)
    outcome = store.backfill(
        home / "ariadne" / "events.sqlite3", exclude_sessions=frozenset({"subagent", "cron"})
    )
    assert outcome["sessions"] == 1
    assert {r.session_id for r in store.entries(0, 10)} == {"primary"}
    assert list(store.sessions()) == ["primary"]


def test_backfill_marks_restricted_records_without_dropping_them(home):
    archive = _archive(home)
    secret = msg("r0", "do not summarize", data_use=["no_training"])
    archive.ingest("conv", [msg("ok", "fine"), secret], origin="hermes_db")
    store = TreeStore(home)
    store.backfill(home / "ariadne" / "events.sqlite3")
    assert [(r.message_uid, r.restricted) for r in store.entries(0, 5)] == [
        ("ok", False),
        ("r0", True),
    ]


def test_backfill_titles_default_to_the_notes_conversation_index(home):
    from hermes_ariadne.notes import NoteStore

    _archive(home).ingest("conv", [msg("m", "hi")], origin="import")
    NoteStore(home).upsert_conversation("conv", title="From the index", origin="import")
    store = TreeStore(home)
    store.backfill(home / "ariadne" / "events.sqlite3")
    assert store.sessions()["conv"].title == "From the index"


def test_backfill_of_a_missing_archive_writes_nothing(home):
    store = TreeStore(home)
    assert store.backfill(home / "ariadne" / "events.sqlite3") == {
        "sessions": 0,
        "appended": 0,
        "skipped": 0,
    }
    assert not (home / "ariadne" / "tree.sqlite3").exists()


def test_backfill_refuses_unsafe_archive_paths(home, tmp_path):
    _archive(home).ingest("conv", [msg("m", "hi")], origin="hermes_db")
    store = TreeStore(home)
    with pytest.raises(StorageRefused):  # not <home>/ariadne/events.sqlite3
        store.backfill(tmp_path / "events.sqlite3")
    with pytest.raises(StorageRefused):
        store.backfill(home / "ariadne" / "notes.sqlite3")
    alias = tmp_path / "alias"
    os.symlink(home, alias)
    with pytest.raises(StorageRefused):
        store.backfill(alias / "ariadne" / "events.sqlite3")
    os.link(home / "ariadne" / "events.sqlite3", tmp_path / "hardlink")
    with pytest.raises(StorageRefused):
        store.backfill(home / "ariadne" / "events.sqlite3")
    assert store.count() == 0


def test_backfill_refuses_another_profiles_archive(home, tmp_path):
    other = tmp_path / "other-home"
    other.mkdir()
    os.chmod(other, 0o700)
    from hermes_ariadne.store import EventStore

    EventStore(other / "ariadne" / "events.sqlite3").ingest(
        "foreign", [msg("f0", "someone else's chat")], origin="hermes_db"
    )
    store = TreeStore(home)
    with pytest.raises(StorageRefused):
        store.backfill(other / "ariadne" / "events.sqlite3")
    with pytest.raises(StorageRefused):  # unresolved spelling of the foreign file
        store.backfill(home / ".." / other.name / "ariadne" / "events.sqlite3")
    assert store.count() == 0


def test_backfill_accepts_the_home_archive_spelled_indirectly(home):
    _archive(home).ingest("conv", [msg("m", "hi")], origin="hermes_db")
    store = TreeStore(home)
    outcome = store.backfill(home / "ariadne" / ".." / "ariadne" / "events.sqlite3")
    assert outcome["appended"] == 1


def test_backfill_refuses_a_foreign_database(home):
    (home / "ariadne").mkdir(mode=0o700)
    path = home / "ariadne" / "events.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events (x)")
    conn.commit()
    conn.close()
    os.chmod(path, 0o600)
    with pytest.raises(StorageRefused):
        TreeStore(home).backfill(path)


def test_backfill_refuses_a_tampered_payload(home):
    _archive(home).ingest("conv", [msg("m", "hi")], origin="hermes_db")
    path = home / "ariadne" / "events.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute('UPDATE events SET payload_json=CAST(\'{"role":"user"}\' AS BLOB)')
    conn.commit()
    conn.close()
    with pytest.raises(ValueError):
        TreeStore(home).backfill(path)
    assert TreeStore(home).count() == 0


# -- lossless identifiers -----------------------------------------------------------


def test_backfill_keeps_lone_surrogate_uids_exact(home):
    archive = _archive(home)
    uid = "uid-\ud800"
    archive.ingest("conv", [msg(uid, "x"), msg("plain", "y")], origin="hook")
    store = TreeStore(home)
    assert store.backfill(home / "ariadne" / "events.sqlite3")["appended"] == 2
    assert [r.message_uid for r in store.entries(0, 10)] == [uid, "plain"]
    assert store.backfill(home / "ariadne" / "events.sqlite3")["appended"] == 0


def test_append_keeps_lone_surrogate_text_exact_and_idempotent(home):
    store = TreeStore(home)
    sid, uid, date = "sess-\udfff", "uid-\ud800", "2026-\ud800"
    item = {**entry("a"), "message_uid": uid, "date": date}
    assert store.append(sid, [item], origin="live", title="t\ud800", source="s\udc00") == [0]
    assert store.append(sid, [item], origin="live") == [0]
    row = store.entries(0, 1)[0]
    assert (row.session_id, row.message_uid, row.date) == (sid, uid, date)
    legend = TreeStore(home).sessions()
    assert list(legend) == [sid]
    assert (legend[sid].title, legend[sid].source) == ("t\ud800", "s\udc00")


def test_plain_text_rows_stay_text_and_still_read(home):
    store = TreeStore(home)
    store.append("s", [entry("a", date="2026-01-01")], origin="live", title="T", source="cli")
    conn = sqlite3.connect(home / "ariadne" / "tree.sqlite3")
    kinds = conn.execute(
        "SELECT typeof(session_id),typeof(message_uid),typeof(date) FROM log"
    ).fetchone()
    assert kinds == ("text", "text", "text")
    assert conn.execute("SELECT typeof(title),typeof(source) FROM sessions").fetchone() == (
        "text",
        "text",
    )
    conn.close()
    row = TreeStore(home).entries(0, 1)[0]
    assert (row.session_id, row.message_uid, row.date) == ("s", "uid-a", "2026-01-01")
