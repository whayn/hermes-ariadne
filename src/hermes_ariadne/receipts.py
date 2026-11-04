"""Issue #8: owned, single-campaign SQLite compaction receipts.

This is separate from the canonical archive. The caller owns request dispatch;
only a committed reservation authorizes an attempt. There is no budget reset API.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # POSIX flock is required; unsupported OSes fail closed.
    fcntl = None

from .store import canonical_json

APPLICATION_ID = 0x41524350  # ARCP, never the canonical archive's application id.
SCHEMA = (
    """CREATE TABLE campaign (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        namespace BLOB NOT NULL, namespace_sha256 TEXT NOT NULL,
        limits BLOB NOT NULL, limits_sha256 TEXT NOT NULL
    )""",
    """CREATE TABLE jobs (
        slot BLOB PRIMARY KEY, plan BLOB NOT NULL, plan_sha256 TEXT NOT NULL,
        state TEXT NOT NULL, reason BLOB,
        receipt BLOB, receipt_sha256 TEXT
    )""",
    """CREATE TABLE attempts (
        attempt_id INTEGER PRIMARY KEY, slot BLOB NOT NULL REFERENCES jobs(slot),
        ordinal INTEGER NOT NULL, request_sha256 TEXT NOT NULL,
        request_bytes INTEGER NOT NULL, output_tokens INTEGER,
        state TEXT NOT NULL, reason BLOB,
        usage BLOB NOT NULL, usage_sha256 TEXT NOT NULL,
        UNIQUE (slot, ordinal)
    )""",
)
LIMIT_KEYS = {
    "max_attempts",
    "max_request_bytes",
    "max_total_request_bytes",
    "max_attempts_per_job",
    "max_jobs",
}


class LedgerError(ValueError):
    """Fatal identity, ownership, corruption, or state conflict."""


class LedgerBudgetError(LedgerError):
    """A refusal that leaves all ledger state unchanged."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise LedgerError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise LedgerError("value must be finite JSON")

    try:
        check(value)
        return canonical_json(value)
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise LedgerError("value must be finite JSON") from exc


def _decode(data, digest):
    try:
        if not isinstance(data, bytes) or _sha(data) != digest:
            raise LedgerError("ledger payload hash mismatch")
        value = json.loads(data.decode("utf-8", "surrogatepass"))
        if _json(value) != data:
            raise LedgerError("noncanonical ledger JSON")
        return value
    except (UnicodeError, ValueError, TypeError) as exc:
        raise LedgerError("corrupt ledger payload") from exc


def _slot(slot):
    if not isinstance(slot, str) or not slot:
        raise LedgerError("slot must be a nonempty string")
    return slot.encode("utf-8", "surrogatepass")


FAMILY_SUFFIXES = ("", "-wal", "-shm", "-journal", ".lock")


def _family(path):
    path = Path(path)
    # Include suffixes on both the lexical name and the resolved main name: a
    # symlink's WAL spelling and its target's WAL spelling can differ.
    return {
        Path(str(base) + suffix).resolve()
        for base in (path, path.resolve())
        for suffix in FAMILY_SUFFIXES
    }


def _protect(path, protected_paths):
    own = _family(path)
    for protected in protected_paths:
        other = _family(protected)
        if own & other:
            raise LedgerError("ledger filename family overlaps protected path")
        for left in own:
            for right in other:
                if left.exists() and right.exists() and left.samefile(right):
                    raise LedgerError("ledger filename family aliases protected inode")


def _mkdir_private(path):
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)


def _fingerprint(conn):
    return {
        (row[0], row[1], row[2], " ".join(row[3].split()) if row[3] else None)
        for row in conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_schema")
    }


def _validate_schema(conn):
    if (
        conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
        or conn.execute("PRAGMA user_version").fetchone()[0] != 1
    ):
        raise LedgerError("foreign or unsupported compaction ledger")
    reference = sqlite3.connect(":memory:")
    try:
        for statement in SCHEMA:
            reference.execute(statement)
        if _fingerprint(conn) != _fingerprint(reference):
            raise LedgerError("incompatible compaction ledger schema fingerprint")
    finally:
        reference.close()


def _validate_campaign(conn, namespace, limits):
    row = conn.execute("SELECT * FROM campaign WHERE id=1").fetchone()
    if row is None:
        raise LedgerError("missing campaign")
    _decode(row["namespace"], row["namespace_sha256"])
    _decode(row["limits"], row["limits_sha256"])
    if row["namespace"] != namespace or row["limits"] != limits:
        raise LedgerError("immutable campaign namespace or limits mismatch")


class CompactionLedger:
    """Sequential single writer; requires local POSIX flock semantics.

    Concurrent hostile path replacement and non-cooperating SQLite writers are
    outside this file-ownership contract. Lock files are deliberately not unlinked.
    """

    def __init__(self, path, *, namespace: dict, limits: dict, protected_paths=()):
        if type(namespace) is not dict or type(limits) is not dict:
            raise LedgerError("namespace and limits must be JSON objects")
        if set(limits) != LIMIT_KEYS or any(type(v) is not int or v <= 0 for v in limits.values()):
            raise LedgerError("limits require exactly the five positive integer caps")
        self._namespace = _json(namespace)
        self._limits = dict(limits)
        limits_json = _json(limits)
        _protect(path, protected_paths)
        self.path = Path(path).resolve()
        self._raw_conn = None
        self._lock_fds = []
        if fcntl is None:
            raise LedgerError("CompactionLedger requires POSIX fcntl.flock")
        try:
            for suffix in FAMILY_SUFFIXES[1:]:
                companion = Path(str(self.path) + suffix)
                if companion.is_symlink():
                    raise LedgerError("ledger companions must not be symlinks")
                if companion.exists():
                    if not stat.S_ISREG(companion.stat().st_mode):
                        raise LedgerError("ledger companions must be regular files")
                    if not self.path.exists() and (suffix != ".lock" or companion.stat().st_size):
                        raise LedgerError("orphaned ledger companion is not owned")
            if self.path.exists():
                if not stat.S_ISREG(self.path.stat().st_mode):
                    raise LedgerError("ledger must be a regular file")
                # Probe a disposable copy, including WAL/hot journal. Read-only
                # SQLite itself can update SHM; immutable=1 ignores foreign WAL.
                # Neither is sufficient for a no-modification ownership check.
                scratch = Path(os.environ.get("TMPDIR", Path.home() / ".cache" / "ariadne"))
                scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
                with tempfile.TemporaryDirectory(
                    prefix="ariadne-ledger-", dir=scratch
                ) as directory:
                    copy = Path(directory) / "probe.sqlite"
                    for suffix in ("", "-wal", "-journal"):
                        source = Path(str(self.path) + suffix)
                        if source.exists():
                            target = Path(str(copy) + suffix)
                            target.touch(mode=0o600)
                            shutil.copyfile(source, target)
                    probe = sqlite3.connect(copy)
                    probe.row_factory = sqlite3.Row
                    try:
                        _validate_schema(probe)
                        _validate_campaign(probe, self._namespace, limits_json)
                    finally:
                        probe.close()
            _mkdir_private(self.path.parent)
            for file in (self.path, Path(str(self.path) + ".lock")):
                fd = os.open(file, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                self._lock_fds.append(fd)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise LedgerError("ledger and lock must be regular files")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise LedgerError("compaction ledger already has a writer") from exc
                os.fchmod(fd, 0o600)
            self._raw_conn = sqlite3.connect(self.path, timeout=0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=FULL")
            with self._transaction():
                objects = self._conn.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_schema"
                ).fetchall()
                app = self._conn.execute("PRAGMA application_id").fetchone()[0]
                version = self._conn.execute("PRAGMA user_version").fetchone()[0]
                if not objects and app == version == 0:
                    for statement in SCHEMA:
                        self._conn.execute(statement)
                    self._conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    self._conn.execute("PRAGMA user_version=1")
                    self._conn.execute(
                        "INSERT INTO campaign VALUES (1,?,?,?,?)",
                        (self._namespace, _sha(self._namespace), limits_json, _sha(limits_json)),
                    )
                elif app != APPLICATION_ID or version != 1:
                    raise LedgerError("foreign or unsupported compaction ledger")
                _validate_schema(self._conn)
                _validate_campaign(self._conn, self._namespace, limits_json)
                for job in self._conn.execute("SELECT slot FROM jobs").fetchall():
                    if not isinstance(job["slot"], bytes):
                        raise LedgerError("corrupt slot encoding")
                    self.inspect(job["slot"].decode("utf-8", "surrogatepass"))
                reason = b"interrupted dispatch; outcome unknown"
                self._conn.execute(
                    "UPDATE attempts SET state='UNCERTAIN',reason=? WHERE state='DISPATCHING'",
                    (reason,),
                )
                self._conn.execute(
                    "UPDATE jobs SET state='UNCERTAIN',reason=? WHERE state='DISPATCHING'",
                    (reason,),
                )
            if self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise LedgerError("WAL is required")
        except sqlite3.Error as exc:
            self.close()
            raise LedgerError("cannot open compaction ledger") from exc
        except BaseException:
            self.close()
            raise

    @property
    def _conn(self):
        if self._raw_conn is None:
            raise LedgerError("compaction ledger is closed")
        return self._raw_conn

    @contextmanager
    def _transaction(self):
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            yield
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise LedgerError("compaction ledger transaction failed") from exc
        except BaseException:
            self._conn.rollback()
            raise

    def close(self):
        try:
            if self._raw_conn is not None:
                self._raw_conn.close()
        finally:
            self._raw_conn = None
            while self._lock_fds:
                os.close(self._lock_fds.pop())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get_plan(self, slot):
        row = self._conn.execute("SELECT * FROM jobs WHERE slot=?", (_slot(slot),)).fetchone()
        return None if row is None else _decode(row["plan"], row["plan_sha256"])

    def put_plan(self, slot, spec: dict):
        key = _slot(slot)
        if type(spec) is not dict:
            raise LedgerError("plan must be a JSON object")
        payload = _json(spec)
        with self._transaction():
            old = self.get_plan(slot)
            if old is not None:
                if _json(old) != payload:
                    raise LedgerError("immutable plan mismatch")
                return old
            count = self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if count >= self._limits["max_jobs"]:
                raise LedgerBudgetError("max_jobs exhausted")
            self._conn.execute(
                "INSERT INTO jobs VALUES (?,?,?,'PLANNED',NULL,NULL,NULL)",
                (key, payload, _sha(payload)),
            )
        return _decode(payload, _sha(payload))

    def get_receipt(self, slot):
        row = self._conn.execute("SELECT * FROM jobs WHERE slot=?", (_slot(slot),)).fetchone()
        if row is None:
            return None
        if row["receipt"] is None:
            if row["state"] == "SUCCEEDED" or row["receipt_sha256"] is not None:
                raise LedgerError("missing successful receipt")
            return None
        if row["state"] != "SUCCEEDED":
            raise LedgerError("receipt contradicts job state")
        receipt = _decode(row["receipt"], row["receipt_sha256"])
        if (
            type(receipt) is not dict
            or type(receipt.get("text")) is not str
            or set(receipt) != {"text", "text_sha256", "attempt_id", "usage", "metadata"}
            or (
                receipt["attempt_id"] is not None
                and (type(receipt["attempt_id"]) is not int or receipt["attempt_id"] <= 0)
            )
            or (receipt["usage"] is not None and type(receipt["usage"]) is not dict)
            or (receipt["metadata"] is not None and type(receipt["metadata"]) is not dict)
            or _sha(receipt["text"].encode("utf-8", "surrogatepass")) != receipt.get("text_sha256")
        ):
            raise LedgerError("invalid receipt text identity or shape")
        return receipt

    def _job(self, slot):
        row = self._conn.execute("SELECT * FROM jobs WHERE slot=?", (_slot(slot),)).fetchone()
        if row is None:
            raise LedgerError("slot must be preplanned")
        _decode(row["plan"], row["plan_sha256"])
        return row

    def reserve(self, slot, *, request_sha256: str, request_bytes: int, output_tokens: int | None):
        if (
            type(request_sha256) is not str
            or len(request_sha256) != 64
            or any(c not in "0123456789abcdef" for c in request_sha256)
        ):
            raise LedgerError("request_sha256 must be canonical lowercase SHA-256")
        if type(request_bytes) is not int or request_bytes <= 0:
            raise LedgerError("request_bytes must be a positive integer")
        if output_tokens is not None and (type(output_tokens) is not int or output_tokens <= 0):
            raise LedgerError("output_tokens must be a positive integer or None")
        with self._transaction():
            job = self._job(slot)
            if job["state"] not in ("PLANNED", "FAILED"):
                raise LedgerError("job cannot dispatch in its current state")
            report = self.report()
            ordinal = (
                self._conn.execute(
                    "SELECT COUNT(*) FROM attempts WHERE slot=?", (_slot(slot),)
                ).fetchone()[0]
                + 1
            )
            if request_bytes > self._limits["max_request_bytes"]:
                raise LedgerBudgetError("max_request_bytes exceeded")
            if report["attempts"] >= self._limits["max_attempts"]:
                raise LedgerBudgetError("max_attempts exhausted")
            if request_bytes > report["remaining"]["max_total_request_bytes"]:
                raise LedgerBudgetError("max_total_request_bytes exhausted")
            if ordinal > self._limits["max_attempts_per_job"]:
                raise LedgerBudgetError("max_attempts_per_job exhausted")
            cursor = self._conn.execute(
                "INSERT INTO attempts (slot,ordinal,request_sha256,request_bytes,output_tokens,"
                "state,reason,usage,usage_sha256) VALUES (?,?,?,?,?,'DISPATCHING',NULL,?,?)",
                (
                    _slot(slot),
                    ordinal,
                    request_sha256,
                    request_bytes,
                    output_tokens,
                    b"null",
                    _sha(b"null"),
                ),
            )
            attempt_id = cursor.lastrowid
            self._conn.execute(
                "UPDATE jobs SET state='DISPATCHING',reason=NULL WHERE slot=?", (_slot(slot),)
            )
        return attempt_id

    def succeed(
        self, slot, attempt_id, *, text: str, usage: dict | None, metadata: dict | None = None
    ):
        if type(text) is not str:
            raise LedgerError("receipt text must be a string")
        if usage is not None and type(usage) is not dict:
            raise LedgerError("usage must be an object or None")
        if metadata is not None and type(metadata) is not dict:
            raise LedgerError("metadata must be an object or None")
        if type(attempt_id) is not int or attempt_id <= 0:
            raise LedgerError("attempt_id must be a positive integer")
        receipt = {
            "text": text,
            "text_sha256": _sha(text.encode("utf-8", "surrogatepass")),
            "attempt_id": attempt_id,
            "usage": usage,
            "metadata": metadata,
        }
        payload = _json(receipt)
        with self._transaction():
            job = self._job(slot)
            old = self.get_receipt(slot)
            if old is not None:
                if _json(old) != payload:
                    raise LedgerError("immutable receipt mismatch")
                return old
            self._dispatching(slot, attempt_id, job)
            usage_json = _json(usage)
            self._conn.execute(
                "UPDATE attempts SET state='SUCCEEDED',usage=?,usage_sha256=? WHERE attempt_id=?",
                (usage_json, _sha(usage_json), attempt_id),
            )
            self._save_receipt(slot, payload)
        return _decode(payload, _sha(payload))

    def record_local(self, slot, *, text: str, metadata: dict):
        if type(text) is not str or type(metadata) is not dict:
            raise LedgerError("local receipt requires text and JSON object metadata")
        payload = _json(
            {
                "text": text,
                "text_sha256": _sha(text.encode("utf-8", "surrogatepass")),
                "attempt_id": None,
                "usage": None,
                "metadata": metadata,
            }
        )
        with self._transaction():
            job = self._job(slot)
            old = self.get_receipt(slot)
            if old is not None:
                if _json(old) != payload:
                    raise LedgerError("immutable receipt mismatch")
                return old
            count = self._conn.execute(
                "SELECT COUNT(*) FROM attempts WHERE slot=?", (_slot(slot),)
            ).fetchone()[0]
            if job["state"] != "PLANNED" or count:
                raise LedgerError("local receipt cannot replace an attempted job")
            self._save_receipt(slot, payload)
        return _decode(payload, _sha(payload))

    def fail(
        self, slot, attempt_id, *, reason: str, uncertain: bool = False, usage: dict | None = None
    ):
        if type(reason) is not str or not reason or type(uncertain) is not bool:
            raise LedgerError("failure needs a nonempty reason and boolean uncertain flag")
        if type(attempt_id) is not int or attempt_id <= 0:
            raise LedgerError("attempt_id must be a positive integer")
        if usage is not None and type(usage) is not dict:
            raise LedgerError("usage must be an object or None")
        usage_json = _json(usage)
        state = "UNCERTAIN" if uncertain else "FAILED"
        reason_bytes = reason.encode("utf-8", "surrogatepass")
        with self._transaction():
            job = self._job(slot)
            self._dispatching(slot, attempt_id, job)
            self._conn.execute(
                "UPDATE attempts SET state=?,reason=?,usage=?,usage_sha256=? WHERE attempt_id=?",
                (state, reason_bytes, usage_json, _sha(usage_json), attempt_id),
            )
            self._conn.execute(
                "UPDATE jobs SET state=?,reason=? WHERE slot=?", (state, reason_bytes, _slot(slot))
            )

    def _dispatching(self, slot, attempt_id, job):
        attempt = self._conn.execute(
            "SELECT * FROM attempts WHERE slot=? AND attempt_id=?", (_slot(slot), attempt_id)
        ).fetchone()
        if job["state"] != "DISPATCHING" or attempt is None or attempt["state"] != "DISPATCHING":
            raise LedgerError("attempt is not the current dispatching reservation")
        return attempt

    def _save_receipt(self, slot, payload):
        self._conn.execute(
            "UPDATE jobs SET state='SUCCEEDED',reason=NULL,receipt=?,receipt_sha256=? WHERE slot=?",
            (payload, _sha(payload), _slot(slot)),
        )

    def inspect(self, slot):
        key = _slot(slot)
        row = self._conn.execute("SELECT * FROM jobs WHERE slot=?", (key,)).fetchone()
        if row is None:
            return {"state": "MISSING", "reason": None, "attempts": []}
        self.get_plan(slot)
        receipt = self.get_receipt(slot)
        if row["state"] not in ("PLANNED", "DISPATCHING", "FAILED", "UNCERTAIN", "SUCCEEDED"):
            raise LedgerError("invalid job state")
        attempts = []
        for item in self._conn.execute(
            "SELECT * FROM attempts WHERE slot=? ORDER BY ordinal", (key,)
        ):
            data = dict(item)
            del data["slot"]
            data["usage"] = _decode(data["usage"], data.pop("usage_sha256"))
            data["reason"] = (
                None if data["reason"] is None else data["reason"].decode("utf-8", "surrogatepass")
            )
            if (
                data["ordinal"] != len(attempts) + 1
                or type(data["request_bytes"]) is not int
                or data["request_bytes"] <= 0
                or data["request_bytes"] > self._limits["max_request_bytes"]
                or data["state"] not in ("DISPATCHING", "FAILED", "UNCERTAIN", "SUCCEEDED")
                or (data["usage"] is not None and type(data["usage"]) is not dict)
                or (
                    data["output_tokens"] is not None
                    and (type(data["output_tokens"]) is not int or data["output_tokens"] <= 0)
                )
            ):
                raise LedgerError("invalid attempt accounting or state")
            if attempts and attempts[-1]["state"] != "FAILED":
                raise LedgerError("only known failed attempts may precede a retry")
            attempts.append(data)
        if attempts:
            if (
                row["state"] != attempts[-1]["state"]
                or len(attempts) > self._limits["max_attempts_per_job"]
            ):
                raise LedgerError("job contradicts latest attempt")
            if receipt is not None and (
                receipt["attempt_id"] != attempts[-1]["attempt_id"]
                or receipt["usage"] != attempts[-1]["usage"]
            ):
                raise LedgerError("receipt contradicts successful attempt")
        elif row["state"] not in ("PLANNED", "SUCCEEDED") or (
            receipt is not None
            and (receipt["attempt_id"] is not None or receipt["usage"] is not None)
        ):
            raise LedgerError("job has no matching attempt")
        return {
            "state": row["state"],
            "reason": None
            if row["reason"] is None
            else row["reason"].decode("utf-8", "surrogatepass"),
            "attempts": attempts,
        }

    def report(self):
        attempts, reserved = self._conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(request_bytes),0) FROM attempts"
        ).fetchone()
        states = dict(self._conn.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
        jobs = sum(states.values())
        return {
            "attempts": attempts,
            "reserved_request_bytes": reserved,
            "jobs": jobs,
            "limits": dict(self._limits),
            "states": states,
            "remaining": {
                "max_attempts": self._limits["max_attempts"] - attempts,
                "max_total_request_bytes": self._limits["max_total_request_bytes"] - reserved,
                "max_jobs": self._limits["max_jobs"] - jobs,
            },
        }
