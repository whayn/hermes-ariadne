"""Append-only, versioned JSON snapshots. Original message values are never rewritten."""

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

APPLICATION_ID = 0x41524941  # "ARIA", only for dedicated Ariadne archives.
SCHEMA = {
    "events": """CREATE TABLE events (
        event_id TEXT PRIMARY KEY,
        message_uid TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        search_text BLOB NOT NULL
    )""",
    "observations": """CREATE TABLE observations (
        event_id TEXT NOT NULL REFERENCES events(event_id),
        session_id TEXT NOT NULL,
        origin TEXT NOT NULL,
        position INTEGER NOT NULL,
        PRIMARY KEY (event_id, session_id, origin, position)
    )""",
    "observations_session": "CREATE INDEX observations_session ON observations(session_id)",
}


def validate_schema(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    application_id = conn.execute("PRAGMA application_id").fetchone()[0]
    objects = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_schema").fetchall()
    if version == 0 and application_id == 0 and not objects:
        return True
    if version != 1 or application_id != APPLICATION_ID:
        raise ValueError("Unsupported or foreign Ariadne schema")
    # Conservative fingerprint: verify definitions, constraints, indexes, and the
    # absence of extra objects (especially triggers), not just table names.
    expected = {
        (
            "table" if name in ("events", "observations") else "index",
            name,
            name if name in ("events", "observations") else "observations",
            " ".join(sql.split()),
        )
        for name, sql in SCHEMA.items()
    } | {
        ("index", "sqlite_autoindex_events_1", "events", None),
        ("index", "sqlite_autoindex_observations_1", "observations", None),
    }
    actual = {
        (
            row["type"],
            row["name"],
            row["tbl_name"],
            " ".join(row["sql"].split()) if row["sql"] else None,
        )
        for row in objects
    }
    if actual != expected:
        raise ValueError("Incompatible Ariadne schema definitions")
    return False


def canonical_json(value):
    # UTF-8/surrogatepass BLOBs distinguish literal UTF-16 pairs from scalar emoji.
    # ASCII JSON escapes would coalesce them, losing the exact observed string.
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8", "surrogatepass")


def search_text(text):
    # SQLite instr(BLOB, BLOB) is literal and avoids surrogate/escape aliases.
    return text.encode("utf-8", "surrogatepass")


class EventStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        with self._connection() as conn:
            # Serialize validation and initialization in one snapshot, before
            # any persistent PRAGMA or DDL changes to a foreign database.
            conn.execute("BEGIN IMMEDIATE")
            if validate_schema(conn):
                for statement in SCHEMA.values():
                    conn.execute(statement)
                conn.execute("PRAGMA application_id=0x41524941")
                conn.execute("PRAGMA user_version=1")
            # SQLite cannot switch journal modes inside a transaction. Competing
            # initializers can make this PRAGMA fail busy without invoking SQLite's
            # busy handler, so retry explicitly with a bounded deadline.
            conn.commit()
            conn.execute("PRAGMA busy_timeout=0")
            deadline = time.monotonic() + 10
            while True:
                try:
                    conn.execute("PRAGMA journal_mode=WAL").fetchone()
                    break
                except sqlite3.OperationalError as exc:
                    if exc.sqlite_errorcode != sqlite3.SQLITE_BUSY or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ingest(self, session_id, messages, *, origin):
        inserted = 0
        skipped = 0
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for position, message in enumerate(messages):
                uid = message.get("message_uid")
                if not isinstance(uid, str) or not uid:
                    skipped += 1
                    continue
                payload = canonical_json(message)
                event_id = hashlib.sha256(payload).hexdigest()
                content = message.get("content")
                text = (
                    content
                    if isinstance(content, str)
                    else json.dumps(
                        content,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                inserted += conn.execute(
                    "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)",
                    (event_id, json.dumps(uid, ensure_ascii=True), payload, search_text(text)),
                ).rowcount
                conn.execute(
                    "INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?)",
                    (event_id, session_id, origin, position),
                )
        return {"inserted": inserted, "skipped": skipped}

    def search(self, query, *, session_id, limit=20, offset=0):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT e.*, o.session_id, o.origin, MIN(o.position) AS position
                FROM events e JOIN observations o USING (event_id)
                WHERE o.session_id = ? AND instr(e.search_text, ?) > 0
                GROUP BY e.event_id, o.session_id, o.origin
                ORDER BY position, e.event_id LIMIT ? OFFSET ?
            """,
                (session_id, search_text(query), limit, offset),
            ).fetchall()
        results = []
        for row in rows:
            message = json.loads(row["payload_json"].decode("utf-8", "surrogatepass"))
            results.append(
                {
                    "event_id": row["event_id"],
                    "message_uid": message["message_uid"],
                    "message": message,
                    "session_id": row["session_id"],
                    "origin": row["origin"],
                    "position": row["position"],
                }
            )
        return results
