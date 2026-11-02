import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from hermes_ariadne.store import EventStore


def test_retries_copies_and_edits_preserve_identity_and_versions(tmp_path):
    store = EventStore(tmp_path / "archive.sqlite3")
    a = {"message_uid": "a" * 32, "role": "user", "content": "same"}
    b = {**a, "message_uid": "b" * 32}
    assert store.ingest("s", [a, b], origin="hook")["inserted"] == 2
    assert store.ingest("s", [a, b], origin="hook")["inserted"] == 0
    assert store.ingest("branch", [a], origin="hook")["inserted"] == 0
    edited = {**a, "content": "edited", "_absorbed_message_uids": [b["message_uid"]]}
    assert store.ingest("s", [edited], origin="hook")["inserted"] == 1
    assert len(store.search("same", session_id="s")) == 2
    assert store.search("edited", session_id="s")[0]["message"] == edited
    assert len(store.search("same", session_id="branch")) == 1
    assert store.search("edited", session_id="branch") == []


def test_missing_identity_is_skipped_without_mutating_messages(tmp_path):
    messages = [
        {"role": "user", "content": "not persisted"},
        {"role": "user", "content": "legacy", "message_uid": ""},
        {"role": "user", "content": "bad", "message_uid": 12},
    ]
    store = EventStore(tmp_path / "archive.sqlite3")
    assert store.ingest("s", messages, origin="hook") == {"inserted": 0, "skipped": 3}
    assert "message_uid" not in messages[0]
    assert store.search("legacy", session_id="s") == []


def test_bad_payload_rolls_back_whole_batch(tmp_path):
    store = EventStore(tmp_path / "archive.sqlite3")
    valid = {"message_uid": "a" * 32, "content": "must roll back"}
    with pytest.raises(TypeError):
        store.ingest("s", [valid, {"message_uid": "b" * 32, "bad": object()}], origin="hook")
    assert store.search("must roll back", session_id="s") == []


def test_concurrent_retries_are_idempotent(tmp_path):
    path = tmp_path / "archive.sqlite3"
    message = {"message_uid": "a" * 32, "content": "concurrent"}
    EventStore(path)

    def ingest(_):
        return EventStore(path).ingest("s", [message], origin="hook")["inserted"]

    with ThreadPoolExecutor(max_workers=4) as workers:
        assert sum(workers.map(ingest, range(12))) == 1
    assert len(EventStore(path).search("concurrent", session_id="s")) == 1


def test_concurrent_initializers_of_nonexistent_database(tmp_path):
    message = {"message_uid": "new", "content": "concurrent fresh archive"}
    with ThreadPoolExecutor(max_workers=8) as workers:
        for round_number in range(1000):
            path = tmp_path / f"fresh-{round_number}.sqlite3"
            assert not path.exists()
            start = Barrier(8)

            def initialize(_):
                start.wait(timeout=10)
                try:
                    return EventStore(path).ingest("s", [message], origin="hook")["inserted"]
                except sqlite3.OperationalError as exc:
                    exc.add_note(
                        f"SQLite {sqlite3.sqlite_version}, round {round_number}: "
                        f"{exc.sqlite_errorname} ({exc.sqlite_errorcode})"
                    )
                    raise

            assert sum(workers.map(initialize, range(8))) == 1
            assert EventStore(path).search("fresh", session_id="s")[0]["message"] == message
            with sqlite3.connect(path) as conn:
                assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
                assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_initializer_cannot_mix_metadata_from_another_initialization(tmp_path, monkeypatch):
    path = tmp_path / "archive.sqlite3"
    connect = sqlite3.connect
    interleaved = False

    class InterleavingCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            self.is_version_read = sql == "PRAGMA user_version"
            return super().execute(sql, parameters)

        def fetchone(self):
            nonlocal interleaved
            row = super().fetchone()
            if self.is_version_read and not interleaved:
                interleaved = True
                # Force a real initializer to commit between metadata reads if
                # they are not protected by the first initializer's transaction.
                if not self.connection.in_transaction:
                    EventStore(path)
            return row

    class InterleavingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            return self.cursor(factory=InterleavingCursor).execute(sql, parameters)

    def interleaving_connect(*args, **kwargs):
        return connect(*args, **kwargs, factory=InterleavingConnection)

    monkeypatch.setattr(sqlite3, "connect", interleaving_connect)
    store = EventStore(path)
    assert interleaved
    assert store.ingest("s", [{"message_uid": "new", "content": "needle"}], origin="hook") == {
        "inserted": 1,
        "skipped": 0,
    }
    assert EventStore(path).search("needle", session_id="s")[0]["message_uid"] == "new"


@pytest.mark.parametrize("contentions", [1, 3, 100])
def test_wal_transition_retries_real_sqlite_busy(tmp_path, monkeypatch, contentions):
    path = tmp_path / "archive.sqlite3"
    connect = sqlite3.connect
    busy_errors = []

    class ContendedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "PRAGMA journal_mode=WAL" and len(busy_errors) < contentions:
                # A competing initializer holds a RESERVED lock after our commit.
                # Release it only after SQLite itself reports the WAL transition
                # busy, making the contention deterministic without sleeps.
                blocker = connect(path)
                blocker.execute("BEGIN IMMEDIATE")
                super().execute("PRAGMA busy_timeout=0")
                try:
                    return super().execute(sql, parameters)
                except sqlite3.OperationalError as exc:
                    busy_errors.append(exc.sqlite_errorcode)
                    raise
                finally:
                    blocker.rollback()
                    blocker.close()
            return super().execute(sql, parameters)

    def contended_connect(*args, **kwargs):
        return connect(*args, **kwargs, factory=ContendedConnection)

    monkeypatch.setattr(sqlite3, "connect", contended_connect)
    # Advance a virtual second per real SQLITE_BUSY to exercise exhaustion
    # without adding ten seconds to the suite or relying on wall-clock timing.
    monkeypatch.setattr(
        "hermes_ariadne.store.time",
        SimpleNamespace(monotonic=lambda: len(busy_errors), sleep=lambda _: None),
    )
    if contentions > 10:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            EventStore(path)
        assert busy_errors == [sqlite3.SQLITE_BUSY] * 10
        return
    EventStore(path)
    assert busy_errors == [sqlite3.SQLITE_BUSY] * contentions
    with connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert conn.execute("PRAGMA user_version").fetchone() == (1,)


@pytest.mark.parametrize(
    "error_code,error_name",
    [
        (sqlite3.SQLITE_BUSY, "SQLITE_BUSY"),
        (sqlite3.SQLITE_BUSY_RECOVERY, "SQLITE_BUSY_RECOVERY"),
    ],
)
def test_wal_transition_retries_extended_busy_codes(tmp_path, monkeypatch, error_code, error_name):
    path = tmp_path / "archive.sqlite3"
    connect = sqlite3.connect
    attempts = []
    sleeps = []

    class RecoveryConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "PRAGMA journal_mode=WAL":
                attempts.append(sql)
                if len(attempts) == 1:
                    # Inject the documented SQLite recovery result at the driver
                    # boundary; all initialization, retry and archive IO are real.
                    exc = sqlite3.OperationalError("database is locked")
                    exc.sqlite_errorcode = error_code
                    exc.sqlite_errorname = error_name
                    raise exc
            return super().execute(sql, parameters)

    def recovery_connect(*args, **kwargs):
        return connect(*args, **kwargs, factory=RecoveryConnection)

    monkeypatch.setattr(sqlite3, "connect", recovery_connect)
    monkeypatch.setattr(
        "hermes_ariadne.store.time",
        SimpleNamespace(monotonic=lambda: 0, sleep=sleeps.append),
    )
    store = EventStore(path)
    assert len(attempts) == 2
    assert sleeps == [0.01]
    message = {"message_uid": "recovery", "content": "exact recovery sentinel"}
    assert store.ingest("s", [message], origin="hook")["inserted"] == 1
    assert store.search("sentinel", session_id="s")[0]["message"] == message
    with connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


@pytest.mark.parametrize("error_code", [sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR])
def test_wal_transition_propagates_non_busy_errors_without_retry(tmp_path, monkeypatch, error_code):
    connect = sqlite3.connect
    attempts = []
    sleeps = []

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "PRAGMA journal_mode=WAL":
                attempts.append(sql)
                exc = sqlite3.OperationalError("non-busy sentinel")
                exc.sqlite_errorcode = error_code
                raise exc
            return super().execute(sql, parameters)

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=FailingConnection)
    )
    monkeypatch.setattr(
        "hermes_ariadne.store.time",
        SimpleNamespace(monotonic=lambda: 0, sleep=sleeps.append),
    )
    with pytest.raises(sqlite3.OperationalError, match="non-busy sentinel") as caught:
        EventStore(tmp_path / "archive.sqlite3")
    assert caught.value.sqlite_errorcode == error_code
    assert len(attempts) == 1
    assert sleeps == []


def test_unknown_schema_is_refused_without_changes(tmp_path):
    path = tmp_path / "archive.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema"):
        EventStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []


def database_state(path):
    with sqlite3.connect(path) as conn:
        state = (
            conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY name"
            ).fetchall(),
            conn.execute("PRAGMA user_version").fetchone(),
            conn.execute("PRAGMA application_id").fetchone(),
            conn.execute("PRAGMA journal_mode").fetchone(),
            list(conn.iterdump()),
        )
    return state, path.read_bytes()


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("application_id", [0, 12345])
def test_foreign_database_is_refused_before_any_persistent_changes(
    tmp_path, version, application_id
):
    path = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE conversations (uid TEXT PRIMARY KEY, payload TEXT)")
        conn.execute("INSERT INTO conversations VALUES (?, ?)", ("unchanged", "café"))
        # PRAGMA assignment does not accept SQL placeholders; these are test constants.
        conn.execute(f"PRAGMA user_version={version}")
        conn.execute(f"PRAGMA application_id={application_id}")
    before = database_state(path)
    try:
        with pytest.raises(ValueError, match="schema"):
            EventStore(path)
    finally:
        assert database_state(path) == before


@pytest.mark.parametrize(
    "statements",
    [
        ["ALTER TABLE events ADD COLUMN foreign_field TEXT"],
        ["ALTER TABLE events RENAME COLUMN payload_json TO incompatible_payload"],
        [
            "DROP TABLE observations",
            (
                "CREATE TABLE observations (event_id TEXT NOT NULL, session_id TEXT NOT NULL, "
                "origin TEXT NOT NULL, position INTEGER NOT NULL, "
                "PRIMARY KEY (event_id, session_id, origin, position))"
            ),
            "CREATE INDEX observations_session ON observations(session_id)",
        ],
        [
            "DROP INDEX observations_session",
            "CREATE INDEX observations_session ON observations(origin)",
        ],
        ["DROP TABLE observations"],
        ["CREATE TABLE foreign_archive (data TEXT)"],
        [
            (
                "CREATE TRIGGER foreign_trigger AFTER INSERT ON events "
                "BEGIN DELETE FROM observations; END"
            )
        ],
    ],
    ids=[
        "extra-column",
        "wrong-column",
        "missing-fk",
        "wrong-index",
        "missing-table",
        "extra-table",
        "trigger",
    ],
)
def test_owned_version_one_incompatible_schema_is_refused_unchanged(tmp_path, statements):
    path = tmp_path / "incompatible.sqlite3"
    EventStore(path).ingest(
        "s", [{"message_uid": "untouched", "content": "sentinel"}], origin="hook"
    )
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        for statement in statements:
            conn.execute(statement)
    before = database_state(path)
    try:
        with pytest.raises(ValueError, match="schema"):
            EventStore(path)
    finally:
        assert database_state(path) == before


def test_existing_empty_version_zero_database_can_be_initialized(tmp_path):
    path = tmp_path / "empty.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=0")
    store = EventStore(path)
    store.ingest("s", [{"message_uid": "new", "content": "needle"}], origin="hook")
    assert EventStore(path).search("needle", session_id="s")[0]["message_uid"] == "new"


@pytest.mark.parametrize(
    "kwargs", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": True}, {"offset": 1.5}]
)
def test_search_rejects_invalid_bounds(tmp_path, kwargs):
    store = EventStore(tmp_path / "archive.sqlite3")
    with pytest.raises(ValueError):
        store.search("q", session_id="s", **kwargs)


def test_search_is_literal_paginated_and_preserves_tool_fields(tmp_path):
    store = EventStore(tmp_path / "archive.sqlite3")
    messages = [
        {
            "message_uid": str(i),
            "role": "tool",
            "content": "%_ café  needle",
            "tool_call_id": "repeated",
            "_tool_call_uid": str(i),
            "nested": {"null": None},
        }
        for i in range(3)
    ]
    store.ingest("s", messages, origin="hook")
    matches = store.search("%_", session_id="s", limit=2)
    tail = store.search("%_", session_id="s", limit=2, offset=2)
    assert [r["message"] for r in matches + tail] == messages
    assert store.search("' OR 1=1 --", session_id="s") == []
    assert store.search("NEEDLE", session_id="s") == []


@pytest.mark.parametrize(
    ("uid", "content", "query"),
    [
        ("ordinary", "before\ud800after", "\ud800"),
        ("uid\udfff", "needle", "needle"),
        ("uid\ud800", "before\udfffafter", "before\udfff"),
    ],
    ids=["surrogate-content", "surrogate-uid", "surrogate-both"],
)
def test_lone_surrogates_are_lossless_and_searchable(tmp_path, uid, content, query):
    path = tmp_path / "archive.sqlite3"
    message = {"message_uid": uid, "content": content, "extra": {"raw": "\ud800"}}
    store = EventStore(path)
    assert store.ingest("s", [message], origin="hook") == {"inserted": 1, "skipped": 0}
    reopened = EventStore(path)
    matches = reopened.search(query, session_id="s")
    assert len(matches) == 1
    assert matches[0]["message"] == message
    assert matches[0]["message_uid"] == uid
    assert reopened.ingest("s", [message], origin="hook")["inserted"] == 0


@pytest.mark.parametrize("query", ["café", "λ", "東京", "😀", "\ud800"])
def test_unicode_structured_content_is_searchable_without_changing_snapshot(tmp_path, query):
    path = tmp_path / "archive.sqlite3"
    message = {
        "message_uid": "structured",
        "content": [{"text": "café λ 東京 😀 \ud800", "nested": {"null": None}}],
    }
    EventStore(path).ingest("s", [message], origin="hook")
    matches = EventStore(path).search(query, session_id="s")
    assert len(matches) == 1
    assert matches[0]["message"] == message


def test_utf16_pair_is_not_silently_coalesced_with_scalar_emoji(tmp_path):
    path = tmp_path / "archive.sqlite3"
    # Distinct Python strings can otherwise serialize to the same ASCII JSON.
    pair = {"message_uid": "u", "content": "\ud83d\ude00"}
    scalar = {"message_uid": "u", "content": "😀"}
    store = EventStore(path)
    assert store.ingest("s", [pair, scalar], origin="hook")["inserted"] == 2
    hits = EventStore(path).search("", session_id="s")
    assert {hit["message"]["content"] for hit in hits} == {pair["content"], scalar["content"]}


def test_surrogate_search_does_not_alias_literal_escape_text(tmp_path):
    store = EventStore(tmp_path / "archive.sqlite3")
    messages = [
        {"message_uid": "raw", "content": "\ud800"},
        {"message_uid": "escaped", "content": r"\ud800"},
    ]
    store.ingest("s", messages, origin="hook")
    assert [r["message_uid"] for r in store.search("\ud800", session_id="s")] == ["raw"]
    assert [r["message_uid"] for r in store.search(r"\ud800", session_id="s")] == ["escaped"]


def test_exact_message_survives_reopen(tmp_path):
    path = tmp_path / "archive.sqlite3"
    message = {
        "message_uid": "a" * 32,
        "role": "user",
        "content": "  Ariadne: café\n\t<exact>  ",
        "extra": {"null": None, "parts": [1, "λ"]},
    }
    first = EventStore(path)
    assert first.ingest("session-a", [message], origin="hook")["inserted"] == 1
    matches = EventStore(path).search("café", session_id="session-a")
    assert len(matches) == 1
    assert matches[0]["message"] == message
    assert matches[0]["message_uid"] == message["message_uid"]
    assert matches[0]["session_id"] == "session-a"
    assert matches[0]["origin"] == "hook"
    assert len(matches[0]["event_id"]) == 64
