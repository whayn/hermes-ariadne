"""Profile-local note store: short persistent facts with revisions and provenance.

Notes are the only agent-written memory. They live in a private SQLite file
under ``<hermes_home>/ariadne/notes.sqlite3``, separate from the conversation
archive. Reads never create state; the schema is built under a temporary name
and renamed into place so an interrupted first write never leaves a half-made
database. Mirrored built-in memory entries carry ``origin='mirror:<target>'``
and are only ever changed by mirroring, never by agent note tools.
"""

import fcntl
import json
import logging
import os
import sqlite3
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .private_paths import StorageRefused, tighten_private  # noqa: F401

logger = logging.getLogger(__name__)

APPLICATION_ID = 0x41524E54  # "ARNT"
SCHEMA_VERSION = 3
MAX_NOTE_CHARS = 4000
MAX_TAGS = 16
MAX_TAG_CHARS = 64
MAX_SOURCE_BYTES = 2048
MAX_NOTES = 20000
MAX_PAGE = 100
MIRROR_TARGETS = ("memory", "user")

SCHEMA = (
    """CREATE TABLE notes (
        id TEXT PRIMARY KEY, text TEXT NOT NULL, tags TEXT NOT NULL, source TEXT,
        origin TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL
    )""",
    """CREATE TABLE revisions (
        note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
        rev INTEGER NOT NULL, text TEXT NOT NULL, at REAL NOT NULL,
        PRIMARY KEY (note_id, rev)
    )""",
    "CREATE INDEX notes_origin_text ON notes(origin, text)",
    """CREATE TABLE conversations (
        session_id TEXT PRIMARY KEY, title TEXT, preview TEXT, summary TEXT,
        origin TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, source TEXT
    )""",
)


# Upgrades for stores written by earlier revisions: {from_version: statements}.
def _add_source_column(conn):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
    if "source" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN source TEXT")


# Statements must be idempotent: pre-release revisions created newer tables while
# still labelling the store version 1.
MIGRATIONS = {
    # v1 -> v2: the conversations index as first released (without ``source``).
    1: (
        """CREATE TABLE IF NOT EXISTS conversations (
        session_id TEXT PRIMARY KEY, title TEXT, preview TEXT, summary TEXT,
        origin TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL
    )""",
    ),
    # v2 -> v3: import provenance on index rows (skipped if already present).
    2: (_add_source_column,),
}
MAX_TITLE_CHARS = 200
MAX_PREVIEW_CHARS = 300
MAX_SUMMARY_CHARS = 8000
CONVERSATION_ORIGINS = ("live", "import")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("note text must be a non-empty string")
    if len(value) > MAX_NOTE_CHARS:
        raise ValueError("note text is too long")
    return value


def _tags(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_TAGS:
        raise ValueError("tags must be a short list")
    for tag in value:
        if not isinstance(tag, str) or not tag.strip() or len(tag) > MAX_TAG_CHARS:
            raise ValueError("invalid tag")
    return list(value)


def _source(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("source must be a mapping")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(encoded.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("source is too large")
    return encoded


_check_private = tighten_private


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _row(row):
    note_id, text, tags, source, origin, created, updated = row
    return {
        "id": note_id,
        "text": text,
        "tags": json.loads(tags),
        "source": json.loads(source) if source else None,
        "origin": origin,
        "created": created,
        "updated": updated,
    }


_COLUMNS = "id,text,tags,source,origin,created,updated"
_CONV_COLUMNS = "session_id,title,preview,summary,origin,created,updated,source"


def _conversation(row):
    result = dict(zip(_CONV_COLUMNS.split(","), row))
    result["source"] = json.loads(result["source"]) if result["source"] else None
    return result


def _clip(value, limit):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("conversation fields must be strings")
    value = " ".join(value.split())
    return value[:limit] or None


def conversation_fields(*, title=None, preview=None, summary=None, source=None):
    """Validate and normalize index metadata without touching storage."""
    if summary is not None:
        if not isinstance(summary, str):
            raise ValueError("summary must be a string")
        summary = summary.strip()[:MAX_SUMMARY_CHARS] or None
    return {
        "title": _clip(title, MAX_TITLE_CHARS),
        "preview": _clip(preview, MAX_PREVIEW_CHARS),
        "summary": summary,
        "source": _source(source),
    }


class NoteStore:
    """Short-lived connections per operation; safe to share across threads."""

    def __init__(self, home):
        self.home = Path(home)
        self.directory = self.home / "ariadne"
        self.path = self.directory / "notes.sqlite3"

    # -- admission and initialization -------------------------------------------------

    def _admit(self, *, create):
        if not self.home.is_absolute():
            raise StorageRefused("hermes home must be absolute")
        for parent in (self.home, *self.home.parents):
            if parent.is_symlink():
                raise StorageRefused("hermes home must not traverse symlinks")
        if not (self.directory.exists() or self.directory.is_symlink()):
            if not create:
                return False
            self.directory.mkdir(mode=0o700, exist_ok=True)
        _check_private(self.directory, regular=False)
        with self._init_lock():
            self._remove_leftovers()
            if self.path.exists() or self.path.is_symlink():
                _check_private(self.path, regular=True)
                for suffix in ("-journal", "-wal", "-shm"):
                    companion = Path(str(self.path) + suffix)
                    try:
                        _check_private(companion, regular=True)
                    except FileNotFoundError:
                        pass  # absent, or removed by a concurrent commit: nothing to admit
                return True
            if not create:
                return False
            self._initialize()
            return True

    @contextmanager
    def _init_lock(self):
        """Serialize initialization and leftover cleanup across threads and processes."""
        lock = self.directory / ".notes.lock"
        if lock.is_symlink():
            raise StorageRefused("ariadne paths must not be symlinks")
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise StorageRefused("invalid ariadne lock file")
            if info.st_uid != os.geteuid():
                raise StorageRefused("ariadne lock file belongs to another user")
            if info.st_mode & 0o077:
                os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # releases the lock

    def _remove_leftovers(self):
        """Under the init lock: drop temporaries an interrupted initializer left behind.

        A crash between publishing the store (``os.link``) and removing the temporary
        leaves a second name for the store's own inode; that residual link is ours and
        is removed before the store's single-link admission. Anything else is refused.
        """
        try:
            published = os.lstat(self.path)
        except FileNotFoundError:
            published = None
        for leftover in self.directory.glob(".notes-init-*"):
            info = leftover.lstat()
            residual = (
                published is not None
                and stat.S_ISREG(info.st_mode)
                and (info.st_dev, info.st_ino) == (published.st_dev, published.st_ino)
            )
            if not residual:
                _check_private(leftover, regular=True)
            leftover.unlink()

    def _initialize(self):
        fd, name = tempfile.mkstemp(prefix=".notes-init-", dir=self.directory)
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
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            if conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                raise StorageRefused("not an ariadne note store")
            if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                self._migrate(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            if write:
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.execute("BEGIN")  # one snapshot for multi-query reads (count + page)
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
        """Upgrade an older Ariadne store in one transaction; unknown versions refuse."""
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            while version != SCHEMA_VERSION:
                if version not in MIGRATIONS:
                    raise StorageRefused("not an ariadne note store")
                for step in MIGRATIONS[version]:
                    step(conn) if callable(step) else conn.execute(step)
                version += 1
                conn.execute(f"PRAGMA user_version={version}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def prepare(self):
        """Create, admit, identify and migrate the store now (no rows written), so a
        caller can refuse before side effects elsewhere instead of failing afterwards."""
        with self._connect(write=True):
            pass

    # -- writes ------------------------------------------------------------------------

    def _insert(self, conn, text, tags, source, origin):
        if conn.execute("SELECT count(*) FROM notes").fetchone()[0] >= MAX_NOTES:
            raise ValueError("note capacity reached")
        now = time.time()
        note_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO notes VALUES(?,?,?,?,?,?,?)",
            (note_id, text, json.dumps(tags, ensure_ascii=False), source, origin, now, now),
        )
        conn.execute("INSERT INTO revisions VALUES(?,?,?,?)", (note_id, 1, text, now))
        return note_id

    def _update(self, conn, note_id, text):
        now = time.time()
        rev = conn.execute(
            "SELECT coalesce(max(rev),0)+1 FROM revisions WHERE note_id=?", (note_id,)
        ).fetchone()[0]
        conn.execute("UPDATE notes SET text=?, updated=? WHERE id=?", (text, now, note_id))
        conn.execute("INSERT INTO revisions VALUES(?,?,?,?)", (note_id, rev, text, now))

    def add(self, text, *, tags=None, source=None, origin="agent"):
        text, tags, source = _text(text), _tags(tags), _source(source)
        with self._connect(write=True) as conn:
            existing = conn.execute(
                "SELECT id FROM notes WHERE origin=? AND text=?", (origin, text)
            ).fetchone()
            if existing:
                return {"id": existing[0], "duplicate": True}
            return {"id": self._insert(conn, text, tags, source, origin), "duplicate": False}

    def edit(self, note_id, text, *, tags=None):
        text = _text(text)
        with self._connect(write=True) as conn:
            row = conn.execute("SELECT origin FROM notes WHERE id=?", (note_id,)).fetchone()
            if row is None or row[0] != "agent":
                raise KeyError("no editable note with that id")
            self._update(conn, note_id, text)
            if tags is not None:
                conn.execute(
                    "UPDATE notes SET tags=? WHERE id=?",
                    (json.dumps(_tags(tags), ensure_ascii=False), note_id),
                )
        return {"id": note_id}

    def delete(self, note_id):
        with self._connect(write=True) as conn:
            if (
                conn.execute("DELETE FROM notes WHERE id=? AND origin='agent'", (note_id,)).rowcount
                != 1
            ):
                raise KeyError("no deletable note with that id")
        return {"id": note_id, "deleted": True}

    def mirror(self, action, target, content, *, previous=None):
        """Apply one built-in memory write. Returns 'applied', 'unchanged' or 'skipped'."""
        if target not in MIRROR_TARGETS or action not in ("add", "replace", "remove"):
            return "skipped"
        origin = "mirror:" + target
        if action in ("replace", "remove") and not (isinstance(previous, str) and previous):
            return "skipped"  # old text alone is not authoritative identity
        with self._connect(write=True) as conn:
            if action == "add":
                content = _text(content)
                if conn.execute(
                    "SELECT 1 FROM notes WHERE origin=? AND text=?", (origin, content)
                ).fetchone():
                    return "unchanged"
                self._insert(conn, content, [], None, origin)
                return "applied"
            row = conn.execute(
                "SELECT id FROM notes WHERE origin=? AND text=? ORDER BY created LIMIT 1",
                (origin, previous),
            ).fetchone()
            if action == "remove":
                if row is None:
                    return "unchanged"
                conn.execute("DELETE FROM notes WHERE id=?", (row[0],))
                return "applied"
            content = _text(content)
            if row is None:
                self._insert(conn, content, [], None, origin)
            else:
                self._update(conn, row[0], content)
            return "applied"

    # -- reads -------------------------------------------------------------------------

    def get(self, note_id):
        with self._connect(write=False) as conn:
            if conn is None:
                return None
            row = conn.execute(f"SELECT {_COLUMNS} FROM notes WHERE id=?", (note_id,)).fetchone()
            return _row(row) if row else None

    def revisions(self, note_id):
        with self._connect(write=False) as conn:
            if conn is None:
                return []
            return [
                {"rev": rev, "text": text, "at": at}
                for rev, text, at in conn.execute(
                    "SELECT rev,text,at FROM revisions WHERE note_id=? ORDER BY rev", (note_id,)
                )
            ]

    def list(self, *, offset=0, limit=50, include_mirrored=True):
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= MAX_PAGE
        ):
            raise ValueError("invalid page")
        where = "" if include_mirrored else "WHERE origin NOT LIKE 'mirror:%'"
        with self._connect(write=False) as conn:
            if conn is None:
                return {"notes": [], "offset": offset, "total": 0}
            total = conn.execute(f"SELECT count(*) FROM notes {where}").fetchone()[0]
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM notes {where} ORDER BY created, id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return {"notes": [_row(r) for r in rows], "offset": offset, "total": total}

    def search(self, query, *, limit=20, include_mirrored=True):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("invalid query")
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE:
            raise ValueError("invalid limit")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clause = "" if include_mirrored else "AND origin NOT LIKE 'mirror:%'"
        with self._connect(write=False) as conn:
            if conn is None:
                return []
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM notes WHERE (text LIKE ? ESCAPE '\\' "
                "OR EXISTS (SELECT 1 FROM json_each(notes.tags) "
                "WHERE json_each.value LIKE ? ESCAPE '\\')) "
                f"{clause} ORDER BY updated DESC LIMIT ?",
                (f"%{escaped}%", f"%{escaped}%", limit),
            ).fetchall()
            return [_row(r) for r in rows]

    def all_texts(self, *, include_mirrored=False, cap=MAX_NOTES):
        """(id, text, tags) for recall ranking; bounded and read-only."""
        clause = "" if include_mirrored else "WHERE origin NOT LIKE 'mirror:%'"
        with self._connect(write=False) as conn:
            if conn is None:
                return []
            return [
                (note_id, text, json.loads(tags))
                for note_id, text, tags in conn.execute(
                    f"SELECT id,text,tags FROM notes {clause} ORDER BY updated DESC LIMIT ?", (cap,)
                )
            ]

    # -- conversation index (derived navigation data, never canonical) -----------------

    def upsert_conversation(
        self, session_id, *, title=None, preview=None, summary=None, origin="live", source=None
    ):
        """Create or refresh one index row. ``preview`` is only set once."""
        if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
            raise ValueError("invalid session id")
        if origin not in CONVERSATION_ORIGINS:
            raise ValueError("invalid conversation origin")
        fields = conversation_fields(title=title, preview=preview, summary=summary, source=source)
        title, preview, summary, source = (
            fields["title"],
            fields["preview"],
            fields["summary"],
            fields["source"],
        )
        now = time.time()
        with self._connect(write=True) as conn:
            if conn.execute(
                "SELECT 1 FROM conversations WHERE session_id=?", (session_id,)
            ).fetchone():
                conn.execute(
                    "UPDATE conversations SET title=coalesce(?,title), "
                    "preview=coalesce(preview,?), summary=coalesce(?,summary), "
                    "source=coalesce(?,source), updated=? WHERE session_id=?",
                    (title, preview, summary, source, now, session_id),
                )
            else:
                conn.execute(
                    "INSERT INTO conversations VALUES(?,?,?,?,?,?,?,?)",
                    (session_id, title, preview, summary, origin, now, now, source),
                )

    def get_conversation(self, session_id):
        with self._connect(write=False) as conn:
            if conn is None:
                return None
            row = conn.execute(
                f"SELECT {_CONV_COLUMNS} FROM conversations WHERE session_id=?", (session_id,)
            ).fetchone()
            return _conversation(row) if row else None

    def conversations(self, *, offset=0, limit=20, exclude=None):
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= MAX_PAGE
        ):
            raise ValueError("invalid page")
        with self._connect(write=False) as conn:
            if conn is None:
                return {"conversations": [], "offset": offset, "total": 0}
            where, params = ("WHERE session_id != ?", (exclude,)) if exclude else ("", ())
            total = conn.execute(f"SELECT count(*) FROM conversations {where}", params).fetchone()[
                0
            ]
            rows = conn.execute(
                f"SELECT {_CONV_COLUMNS} FROM conversations {where} "
                "ORDER BY updated DESC, session_id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            return {
                "conversations": [_conversation(r) for r in rows],
                "offset": offset,
                "total": total,
            }

    def search_conversations(self, query, *, limit=20, exclude=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("invalid query")
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE:
            raise ValueError("invalid limit")
        like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect(write=False) as conn:
            if conn is None:
                return []
            rows = conn.execute(
                f"SELECT {_CONV_COLUMNS} FROM conversations WHERE session_id != ? AND "
                "(title LIKE ? ESCAPE '\\' OR preview LIKE ? ESCAPE '\\' "
                "OR summary LIKE ? ESCAPE '\\') ORDER BY updated DESC LIMIT ?",
                (exclude or "", like, like, like, limit),
            ).fetchall()
            return [_conversation(r) for r in rows]
