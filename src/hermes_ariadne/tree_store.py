"""Profile tree store: one append-only message log and its summary tree (issue #54).

Every primary conversation of a Hermes profile is appended, in arrival order, to a
single global log under ``<hermes_home>/ariadne/tree.sqlite3``. Log positions are
assigned once, strictly increasing from zero, and never change; they are the leaf
coordinates of ``tree.Part``. Summary nodes, durable summary-job reservations with
logical accounting, and the persisted ``chat`` and ``seen`` views live beside it.

Storage follows ``notes.py``: reads never create state; the schema is built under
a temporary name and published with ``os.link``; Ariadne's directory and files are
0700/0600 (drift is tightened with a warning; symlinks, hardlinks and foreign
owners are refused); the Hermes home only has to be absolute and symlink-free.
Writes are ``BEGIN IMMEDIATE`` transactions, multi-query reads use one snapshot.

The log references the exact archive (``events.sqlite3``) by ``event_id``; it holds
no message text. Summaries are derived navigation data, never canonical.

Job states: ``reserved`` (a call may be in flight), ``done`` (node stored),
``uncertain`` (ambiguous outcome or lost owner: blocked, never resubmitted
automatically), ``refused`` (a definitive answer that was not accepted, e.g. over
length: may be reserved again as a new attempt) and ``armed`` (an uncertain job the
user explicitly re-armed with ``retry``). ``counters`` hold the logical totals of
every reservation ever made and never decrease.

Deviations from the issue #54 plan (section B), all additive:
- ``views`` also stores ``size`` so ``load_view`` rebuilds the exact ``tree.View``.
- The ``armed`` job state records an explicit ``retry`` while keeping the attempt
  history (deleting the row would lose ``attempts``/``request_bytes``).
- ``jobs()`` and ``counters()`` read snapshots for the builder and status tool.
- A conversation tag is the first 8 hex chars of sha256(session_id); if another
  stored conversation already has that tag, it is lengthened by 4 hex chars at a
  time, so tags stay unique (the legend, not the hash, is authoritative).
- ``backfill`` only accepts ``<home>/ariadne/events.sqlite3`` (the archive reader's
  admission, reimplemented here because ``rolling_engine`` imports Hermes), and
  skips (and counts) archived rows without a message uid or a known role.
"""

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from types import MappingProxyType

from .notes import NoteStore, _fsync_dir
from .private_paths import StorageRefused, tighten_private
from .tree import Part, View, utf8_size

APPLICATION_ID = 0x41525452  # "ARTR"
SCHEMA_VERSION = 1

ORIGINS = ("live", "import")
ROLES = ("user", "assistant", "tool", "system")
VIEW_NAMES = ("chat", "seen")
MAX_SESSION_ID_CHARS = 200
MAX_TITLE_CHARS = 200
MAX_SOURCE_CHARS = 200
MAX_UID_CHARS = 200
MAX_DATE_CHARS = 64
MAX_PAGE = 10000
MAX_OWNER_CHARS = 200
BACKFILL_CHUNK = 1000
MAX_ERROR_CHARS = 100
# Job states that refuse a new reservation. 'refused' (a definitive answer that was
# not accepted) and 'armed' (an uncertain job explicitly re-armed by retry) do not.
BLOCKING_STATES = ("reserved", "done", "uncertain")

SCHEMA = (
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

# Upgrades for stores written by earlier revisions: {from_version: steps}. Each step
# is SQL text or a callable(conn) and must be idempotent. Version 1 is the first.
MIGRATIONS = {}


@dataclass(frozen=True)
class LogEntry:
    """One log row (field-compatible with the planned ``profile_tree.LogEntry``)."""

    position: int
    session_id: str
    event_id: str
    message_uid: str
    role: str
    date: str | None
    origin: str
    restricted: bool


@dataclass(frozen=True)
class Session:
    """Legend row of one conversation: its short tag, title and log span."""

    session_id: str
    tag: str
    title: str | None
    source: str | None
    origin: str
    first_position: int
    last_position: int


_LOG_COLUMNS = "position,session_id,event_id,message_uid,role,date,origin,restricted"
_SESSION_COLUMNS = "session_id,tag,title,source,origin,first_position,last_position"


def _log_entry(row):
    *rest, restricted = row
    return LogEntry(*rest, bool(restricted))


def _session_id(value):
    if not isinstance(value, str) or not value or len(value) > MAX_SESSION_ID_CHARS:
        raise ValueError("invalid session id")
    return value


def _origin(value):
    if value not in ORIGINS:
        raise ValueError("invalid log origin")
    return value


def _short(value, limit, name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = " ".join(value.split())[:limit]
    return value or None


def _event_id(value):
    if not isinstance(value, str) or len(value) != 64 or value.strip("0123456789abcdef"):
        raise ValueError("event_id must be a canonical lowercase SHA-256 identity")
    return value


def _entry(item):
    if not hasattr(item, "get"):
        raise ValueError("log entries must be mappings")
    uid = item.get("message_uid")
    if not isinstance(uid, str) or not uid or len(uid) > MAX_UID_CHARS:
        raise ValueError("invalid message_uid")
    if item.get("role") not in ROLES:
        raise ValueError("invalid role")
    date = item.get("date")
    if date is not None and (not isinstance(date, str) or len(date) > MAX_DATE_CHARS):
        raise ValueError("invalid date")
    restricted = item.get("restricted", False)
    if type(restricted) is not bool:
        raise ValueError("restricted must be a bool")
    return _event_id(item.get("event_id")), uid, item["role"], date, int(restricted)


def _tag(conn, session_id):
    """8 hex chars of sha256(session_id); lengthened only on a stored-tag collision."""
    digest = hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()
    for length in range(8, 65, 4):
        if not conn.execute("SELECT 1 FROM sessions WHERE tag=?", (digest[:length],)).fetchone():
            return digest[:length]
    raise StorageRefused("conversation tag space exhausted")  # pragma: no cover


@dataclass(frozen=True)
class Job:
    """Durable record of one node's summary calls (``attempts`` reservations)."""

    state: str
    attempts: int
    request_bytes: int
    owner: str | None
    reserved_at: float | None
    finished_at: float | None
    error_type: str | None


_JOB_COLUMNS = "level,idx,state,attempts,request_bytes,owner,reserved_at,finished_at,error_type"


def _part(part):
    if not isinstance(part, Part):
        raise TypeError("expected a tree.Part")
    return part


def _node_text(text):
    if not isinstance(text, str):
        raise ValueError("node text must be a string")
    return text


def _error_type(value):
    if value is not None and (not isinstance(value, str) or len(value) > MAX_ERROR_CHARS):
        raise ValueError("invalid error type")
    return value


def _view_name(name):
    if name not in VIEW_NAMES:
        raise ValueError("invalid view name")


def _count(conn):
    return conn.execute("SELECT coalesce(max(position) + 1, 0) FROM log").fetchone()[0]


def _within(conn, part):
    if part.stop > _count(conn):
        raise ValueError("part extends beyond the log")


def _counters(conn):
    return dict(conn.execute("SELECT name, value FROM counters ORDER BY name"))


def _insert_node(conn, part, text):
    return (
        conn.execute(
            "INSERT OR IGNORE INTO nodes VALUES(?,?,?,?,?)",
            (part.level, part.index, text, utf8_size(text), time.time()),
        ).rowcount
        == 1
    )


def _require_reserved(conn, part, owner, attempt):
    """The job must still be in the exact reservation (owner, attempt) the caller holds."""
    row = conn.execute(
        "SELECT state,owner,attempts FROM jobs WHERE level=? AND idx=?", (part.level, part.index)
    ).fetchone()
    if row is None or row[0] != "reserved":
        raise ValueError("job is not reserved")
    if type(attempt) is not int or row[1] != owner or row[2] != attempt:
        raise ValueError("job is reserved by another attempt")


class TreeStore:
    """Short-lived connections per operation; safe to share across threads/processes."""

    def __init__(self, home):
        self.home = Path(home)
        self.directory = self.home / "ariadne"
        self.path = self.directory / "tree.sqlite3"

    # -- admission and initialization (same rules as NoteStore) -------------------------

    _init_lock = NoteStore._init_lock
    _remove_leftovers = NoteStore._remove_leftovers
    _admit = NoteStore._admit
    _LOCK_NAME = ".tree.lock"
    _INIT_PREFIX = ".tree-init-"

    def _initialize(self):
        fd, name = tempfile.mkstemp(prefix=self._INIT_PREFIX, dir=self.directory)
        os.close(fd)
        try:
            conn = sqlite3.connect(name)
            try:
                with conn:
                    for sql in SCHEMA:
                        conn.execute(sql)
                    conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            finally:
                conn.close()
            try:
                os.link(name, self.path)  # never replaces a concurrently created store
            except FileExistsError:
                pass
            _fsync_dir(self.directory)
        finally:
            for leftover in (name, name + "-journal"):
                if os.path.exists(leftover):
                    os.unlink(leftover)

    @contextmanager
    def _connect(self, *, write):
        if not self._admit(create=write):
            yield None
            return
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            if conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                raise StorageRefused("not an ariadne tree store")
            if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                self._migrate(conn)
            if write:
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.execute("BEGIN")  # one snapshot for multi-query reads
                conn.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchall()
            yield conn
            if write:
                conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _migrate(conn):
        """Upgrade an older tree store in one transaction; unknown versions refuse."""
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            while version != SCHEMA_VERSION:
                if version not in MIGRATIONS:
                    raise StorageRefused("not an ariadne tree store")
                for step in MIGRATIONS[version]:
                    step(conn) if callable(step) else conn.execute(step)
                version += 1
                conn.execute(f"PRAGMA user_version={version}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def prepare(self):
        """Create, admit, identify and migrate the store now (no rows written)."""
        with self._connect(write=True):
            pass

    # -- append-only log -------------------------------------------------------------

    def append(self, session_id, entries, *, origin, title=None, source=None):
        """Append ``entries`` of one conversation; returns each entry's log position.

        Idempotent per ``(session_id, event_id)``: an already logged event returns its
        existing position. New entries get strictly increasing positions in one
        transaction. ``title``/``source`` refresh the legend when given.
        """
        session_id, origin = _session_id(session_id), _origin(origin)
        title = _short(title, MAX_TITLE_CHARS, "title")
        source = _short(source, MAX_SOURCE_CHARS, "source")
        rows = [_entry(item) for item in entries]
        with self._connect(write=True) as conn:
            next_position = _count(conn)
            positions, added = [], []
            for event_id, uid, role, date, restricted in rows:
                found = conn.execute(
                    "SELECT position FROM log WHERE session_id=? AND event_id=?",
                    (session_id, event_id),
                ).fetchone()
                if found:
                    positions.append(found[0])
                    continue
                conn.execute(
                    "INSERT INTO log VALUES(?,?,?,?,?,?,?,?)",
                    (next_position, session_id, uid, event_id, role, date, origin, restricted),
                )
                positions.append(next_position)
                added.append(next_position)
                next_position += 1
            known = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if known:
                conn.execute(
                    "UPDATE sessions SET title=coalesce(?,title), source=coalesce(?,source), "
                    "last_position=max(last_position, ?) WHERE session_id=?",
                    (title, source, max(added, default=-1), session_id),
                )
            elif added:
                conn.execute(
                    "INSERT INTO sessions VALUES(?,?,?,?,?,?,?)",
                    (
                        session_id,
                        _tag(conn, session_id),
                        title,
                        source,
                        origin,
                        added[0],
                        added[-1],
                    ),
                )
        return positions

    def count(self):
        """Number of log entries, which is also the next position to be assigned."""
        with self._connect(write=False) as conn:
            if conn is None:
                return 0
            return _count(conn)

    def entries(self, start, stop):
        """Log entries with ``start <= position < stop`` (at most ``MAX_PAGE``)."""
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop:
            raise ValueError("invalid range")
        if stop - start > MAX_PAGE:
            raise ValueError("range too large")
        with self._connect(write=False) as conn:
            if conn is None:
                return []
            return [
                _log_entry(row)
                for row in conn.execute(
                    f"SELECT {_LOG_COLUMNS} FROM log WHERE position>=? AND position<? "
                    "ORDER BY position",
                    (start, stop),
                )
            ]

    def entry(self, position):
        if type(position) is not int or position < 0:
            raise ValueError("invalid position")
        found = self.entries(position, position + 1)
        return found[0] if found else None

    def sessions(self):
        """Read-only legend snapshot ``{session_id: Session}`` by first position."""
        with self._connect(write=False) as conn:
            if conn is None:
                return MappingProxyType({})
            return MappingProxyType(
                {
                    row[0]: Session(*row)
                    for row in conn.execute(
                        f"SELECT {_SESSION_COLUMNS} FROM sessions ORDER BY first_position"
                    )
                }
            )

    # -- summary nodes ---------------------------------------------------------------

    def nodes(self):
        """Read-only ``{Part: text}`` snapshot of every built node, one transaction."""
        with self._connect(write=False) as conn:
            if conn is None:
                return MappingProxyType({})
            return MappingProxyType(
                {
                    Part(level, idx): text
                    for level, idx, text in conn.execute("SELECT level,idx,text FROM nodes")
                }
            )

    def put_node(self, part, text):
        """Store a built node; False (unchanged) when it already exists.

        Built nodes are immutable so a rendered line never changes under a cache.
        """
        _part(part)
        text = _node_text(text)
        with self._connect(write=True) as conn:
            _within(conn, part)
            return _insert_node(conn, part, text)

    # -- summary jobs ----------------------------------------------------------------

    def reserve(self, part, request_bytes, owner, *, max_jobs, max_total_bytes):
        """Durably reserve one summary call before dispatch; False when refused.

        Atomic with the logical counters: refused when the part is already built,
        reserved, done or uncertain, or when one more job or ``request_bytes`` more
        would exceed ``max_jobs`` / ``max_total_bytes``. Each reservation is one
        attempt; counters only ever grow. Returns the new attempt number (always
        positive); ``finish``/``mark`` must present it with the same ``owner`` so a
        late caller from a replaced reservation can never settle the new one.
        """
        _part(part)
        for name, value in (
            ("request_bytes", request_bytes),
            ("max_jobs", max_jobs),
            ("max_total_bytes", max_total_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(owner, str) or not owner or len(owner) > MAX_OWNER_CHARS:
            raise ValueError("invalid owner")
        with self._connect(write=True) as conn:
            _within(conn, part)
            if conn.execute(
                "SELECT 1 FROM nodes WHERE level=? AND idx=?", (part.level, part.index)
            ).fetchone():
                return False
            row = conn.execute(
                "SELECT state FROM jobs WHERE level=? AND idx=?", (part.level, part.index)
            ).fetchone()
            if row and row[0] in BLOCKING_STATES:
                return False
            counters = _counters(conn)
            if (
                counters["logical_jobs"] + 1 > max_jobs
                or counters["request_bytes"] + request_bytes > max_total_bytes
            ):
                return False
            conn.execute(
                "INSERT INTO jobs VALUES(?,?,'reserved',1,?,?,?,NULL,NULL) "
                "ON CONFLICT(level, idx) DO UPDATE SET state='reserved', "
                "attempts=attempts+1, request_bytes=request_bytes+excluded.request_bytes, "
                "owner=excluded.owner, reserved_at=excluded.reserved_at, "
                "finished_at=NULL, error_type=NULL",
                (part.level, part.index, request_bytes, owner, time.time()),
            )
            conn.execute("UPDATE counters SET value=value+1 WHERE name='logical_jobs'")
            conn.execute(
                "UPDATE counters SET value=value+? WHERE name='request_bytes'", (request_bytes,)
            )
            return conn.execute(
                "SELECT attempts FROM jobs WHERE level=? AND idx=?", (part.level, part.index)
            ).fetchone()[0]

    def finish(self, part, text, *, owner, attempt):
        """Record a reserved job's accepted text: store the node and mark it done.

        Fenced to the reservation (``owner``, ``attempt``) ``reserve`` returned.
        """
        _part(part)
        text = _node_text(text)
        with self._connect(write=True) as conn:
            _require_reserved(conn, part, owner, attempt)
            _insert_node(conn, part, text)
            conn.execute(
                "UPDATE jobs SET state='done', finished_at=? WHERE level=? AND idx=?",
                (time.time(), part.level, part.index),
            )

    def mark(self, part, state, error_type, *, owner, attempt):
        """End a reserved job without a node: ``uncertain`` (ambiguous call, blocked
        until ``retry``) or ``refused`` (definitive answer not accepted; may be
        reserved again as a new attempt). Fenced like ``finish``."""
        _part(part)
        if state not in ("uncertain", "refused"):
            raise ValueError("a job can only be marked uncertain or refused")
        error_type = _error_type(error_type)
        with self._connect(write=True) as conn:
            _require_reserved(conn, part, owner, attempt)
            conn.execute(
                "UPDATE jobs SET state=?, error_type=?, finished_at=? WHERE level=? AND idx=?",
                (state, error_type, time.time(), part.level, part.index),
            )

    def stale_reserved(self, alive):
        """Turn reservations whose owner is not alive into ``uncertain``.

        ``alive`` is a predicate ``owner -> bool`` or a collection of live owners.
        A lost reservation may have dispatched, so it never becomes free again here.
        """
        is_alive = alive if callable(alive) else (lambda owner: owner in alive)
        with self._connect(write=True) as conn:
            stale = [
                Part(level, idx)
                for level, idx, owner in conn.execute(
                    "SELECT level,idx,owner FROM jobs WHERE state='reserved' ORDER BY level,idx"
                ).fetchall()
                if not is_alive(owner)
            ]
            now = time.time()
            for part in stale:
                conn.execute(
                    "UPDATE jobs SET state='uncertain', error_type='owner_lost', finished_at=? "
                    "WHERE level=? AND idx=?",
                    (now, part.level, part.index),
                )
            return stale

    def retry(self, parts=None):
        """Explicitly re-arm uncertain jobs (all when ``parts`` is None); returns them."""
        if parts is not None:
            parts = [_part(part) for part in parts]
        with self._connect(write=True) as conn:
            uncertain = [
                Part(level, idx)
                for level, idx in conn.execute(
                    "SELECT level,idx FROM jobs WHERE state='uncertain' ORDER BY level,idx"
                ).fetchall()
            ]
            wanted = None if parts is None else set(parts)
            chosen = [p for p in uncertain if wanted is None or p in wanted]
            for part in chosen:
                conn.execute(
                    "UPDATE jobs SET state='armed' WHERE level=? AND idx=?",
                    (part.level, part.index),
                )
            return chosen

    def jobs(self):
        """Read-only ``{Part: Job}`` snapshot of every job ever reserved."""
        with self._connect(write=False) as conn:
            if conn is None:
                return MappingProxyType({})
            return MappingProxyType(
                {
                    Part(level, idx): Job(*rest)
                    for level, idx, *rest in conn.execute(f"SELECT {_JOB_COLUMNS} FROM jobs")
                }
            )

    def counters(self):
        """Logical accounting totals (never decrease)."""
        with self._connect(write=False) as conn:
            if conn is None:
                return {"logical_jobs": 0, "request_bytes": 0}
            return _counters(conn)

    # -- views -----------------------------------------------------------------------

    def load_view(self, name):
        """``(View, revision)``; ``(None, 0)`` when the view was never saved."""
        _view_name(name)
        with self._connect(write=False) as conn:
            if conn is None:
                return None, 0
            row = conn.execute(
                "SELECT parts_json,size,batch,high,revision FROM views WHERE name=?", (name,)
            ).fetchone()
        if row is None:
            return None, 0
        parts_json, size, batch, high, revision = row
        parts = tuple(Part(level, index) for level, index in json.loads(parts_json))
        return View(parts, size, bool(batch), high), revision

    def save_view(self, name, view, expected_revision):
        """Compare-and-swap: write ``view`` only if the stored revision (0 when never
        saved) equals ``expected_revision``; returns whether it was written."""
        _view_name(name)
        if not isinstance(view, View):
            raise TypeError("view must be a tree.View")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("invalid revision")
        parts_json = json.dumps([[part.level, part.index] for part in view.parts])
        with self._connect(write=True) as conn:
            end = view.parts[-1].stop if view.parts else 0
            if end > _count(conn):
                raise ValueError("view extends beyond the log")
            row = conn.execute("SELECT revision FROM views WHERE name=?", (name,)).fetchone()
            if (row[0] if row else 0) != expected_revision:
                return False
            conn.execute(
                "INSERT OR REPLACE INTO views VALUES(?,?,?,?,?,?)",
                (name, parts_json, view.size, int(view.batch), view.high, expected_revision + 1),
            )
            return True

    # -- backfill from the exact archive ---------------------------------------------

    def backfill(self, archive_path, *, exclude_sessions=frozenset(), titles=None):
        """Append every archived conversation's events not yet in the log.

        ``archive_path`` must be this store's own home archive,
        ``<home>/ariadne/events.sqlite3`` (compared after resolution; any other path,
        e.g. another profile's archive, is refused). It gets the archive reader's
        admission and is opened read-only. Conversations are
        taken in first-observation order (lowest observation rowid), each one's
        events in observation-position order. ``exclude_sessions`` (subagent/cron
        sessions, decided by the caller from state.db) are skipped. Titles come from
        ``titles`` or, when None, the notes conversation index. Rerunning appends only
        events that are new since; existing positions never move. Every payload is
        rehashed and the whole archive is read before the first append.
        """
        excluded = frozenset(exclude_sessions)
        plan = self._read_archive(archive_path, excluded)
        outcome = {"sessions": 0, "appended": 0, "skipped": 0}
        if not plan:
            return outcome
        before = self.count()
        notes = NoteStore(self.home) if titles is None else None
        for session_id, rows, skipped in plan:
            outcome["sessions"] += 1
            outcome["skipped"] += skipped
            if notes is not None:
                indexed = notes.get_conversation(session_id)
                title = indexed["title"] if indexed else None
            else:
                title = titles.get(session_id)
            for origin, group in groupby(rows, key=lambda row: row[0]):
                group = [entry for _, entry in group]
                for start in range(0, len(group), BACKFILL_CHUNK):
                    chunk = group[start : start + BACKFILL_CHUNK]
                    self.append(session_id, chunk, origin=origin, title=title)
        outcome["appended"] = self.count() - before
        return outcome

    def _read_archive(self, archive_path, excluded):
        """``[(session_id, [(origin, entry), ...], skipped)]`` from one read snapshot."""
        from .retrieval import _require_supported_data_use, _UnsupportedDataUse
        from .store import APPLICATION_ID as ARCHIVE_ID

        archive = Path(archive_path)
        own = self.directory / "events.sqlite3"
        if not archive.is_absolute() or archive.resolve() != own.resolve():
            raise StorageRefused("backfill reads only this home's ariadne/events.sqlite3")
        _admit_archive(archive)
        if not archive.is_file():
            return []
        conn = sqlite3.connect(archive.as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            if conn.execute("PRAGMA application_id").fetchone()[0] != ARCHIVE_ID:
                raise StorageRefused("not an ariadne archive")
            conn.execute("BEGIN")
            sessions = conn.execute(
                "SELECT session_id FROM observations GROUP BY session_id ORDER BY min(rowid)"
            ).fetchall()
            plan = []
            for (session_id,) in sessions:
                if session_id in excluded:
                    continue
                rows, skipped = [], 0
                for event_id, payload, origin in conn.execute(
                    "SELECT e.event_id, e.payload_json, (SELECT o2.origin FROM observations o2 "
                    "WHERE o2.event_id=e.event_id AND o2.session_id=o.session_id "
                    "ORDER BY o2.rowid LIMIT 1) FROM events e JOIN observations o "
                    "USING (event_id) WHERE o.session_id=? GROUP BY e.event_id "
                    "ORDER BY min(o.position), min(o.rowid), e.event_id",
                    (session_id,),
                ):
                    if hashlib.sha256(payload).hexdigest() != event_id:
                        raise ValueError("archived event payload does not match its identity")
                    message = json.loads(payload.decode("utf-8", "surrogatepass"))
                    uid, role = message.get("message_uid"), message.get("role")
                    if (
                        not isinstance(uid, str)
                        or not uid
                        or len(uid) > MAX_UID_CHARS
                        or role not in ROLES
                    ):
                        skipped += 1  # not a loggable message row; counted, never guessed
                        continue
                    try:
                        _require_supported_data_use(message)
                        restricted = False
                    except _UnsupportedDataUse:
                        restricted = True
                    entry = {
                        "event_id": event_id,
                        "message_uid": uid,
                        "role": role,
                        "date": _date(message.get("timestamp")),
                        "restricted": restricted,
                    }
                    rows.append(("import" if origin == "import" else "live", entry))
                if rows or skipped:
                    plan.append((_session_id(session_id), rows, skipped))
            return plan
        finally:
            conn.close()


def _admit_archive(archive):
    """``rolling_engine._archive_path(home, private_home=False)`` without its host
    imports: a symlink-free home; private, single-link Ariadne files (drift tightened)."""
    for item in (archive.parent.parent, *archive.parent.parent.parents):
        if item.is_symlink():
            raise StorageRefused("ariadne home must not traverse symlinks")
    tighten_private(archive.parent, regular=False)
    for suffix in ("", "-wal", "-shm", "-journal"):
        tighten_private(Path(str(archive) + suffix), regular=True)


def _date(value):
    """ISO date of an archived timestamp (Unix seconds or text), else None."""
    if isinstance(value, str):
        return value if 0 < len(value) <= MAX_DATE_CHARS else None
    if type(value) in (int, float) and math.isfinite(value):
        try:
            moment = datetime.fromtimestamp(value, UTC)
        except (OverflowError, OSError, ValueError):
            return None
        return moment.isoformat(timespec="seconds").replace("+00:00", "Z")
    return None
