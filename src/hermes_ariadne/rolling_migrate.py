"""Explicit offline schema-2 -> schema-3 conversion; never invokes a facade.

All source audits are read-only. The stable root lock coordinates old and new
code. Ordinary runtime opens only resolve a completely published selector.
"""

import os
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from . import rolling as r
from . import rolling_state as s
from .rolling_delta import DeltaTarget

FENCE = 2147483000
RECEIPT_FORMAT = "ariadne-rolling-migration"
SELECTOR_FORMAT = "ariadne-rolling-storage"
INTENT_FORMAT = "ariadne-rolling-migration-intent"


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(path, limit):
    data = s._read(path, limit)
    value = r.decode_json(data, limit)
    if type(value) is dict and "version" in value and type(value["version"]) is not int:
        raise ValueError("invalid metadata version")
    if r.encode_json(value, limit) != data:
        raise ValueError("noncanonical migration metadata")
    return value, data


def _read_db(path, *, recover=False):
    s._path_check(path, regular=True)
    for suffix in ("-wal", "-shm", "-journal"):
        companion = Path(str(path) + suffix)
        s._path_check(companion, regular=True)
        # The journal writes DELETE-mode transactions. An active recovery journal
        # is not safe for this read-only offline snapshot protocol.
        if not recover and companion.exists() and companion.stat().st_size:
            raise ValueError("offline migration requires checkpointed journal companions")
    if not path.is_file():
        raise ValueError("missing migration database")
    # Only the completed-rollback runtime probe opts into normal SQLite recovery.
    # Every offline audit keeps the strict read-only, checkpointed default.
    mode = "rw" if recover else "ro"
    conn = sqlite3.connect(path.as_uri() + "?mode=" + mode, uri=True, timeout=0)
    conn.row_factory = sqlite3.Row
    if not recover:
        conn.execute("PRAGMA query_only=ON")
    return conn


def _receipt(root, session_id, settings):
    value, data = _json(root / "migration.json", settings.max_logical_jobs * 8192 + 8192)
    if (
        type(value) is not dict
        or set(value)
        != {
            "format",
            "version",
            "session_id",
            "settings_digest",
            "source_fingerprint",
            "destination_fingerprint",
            "source_files",
            "targets",
            "jobs",
            "generations",
        }
        or value["format"] != RECEIPT_FORMAT
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["session_id"] != session_id
        or value["settings_digest"] != settings.digest
    ):
        raise ValueError("invalid migration receipt")
    for key in ("source_fingerprint", "destination_fingerprint"):
        r.require_digest(value[key])
    source_files = value["source_files"]
    if (
        type(source_files) is not dict
        or "journal.sqlite3" not in source_files
        or len(source_files) > settings.max_logical_jobs + 2
    ):
        raise ValueError("invalid migration source file map")
    for name, digest in source_files.items():
        r.require_digest(digest)
        if name in {"journal.sqlite3", "manifest.json"}:
            continue
        if not name.startswith("generations/") or not name.endswith(".json"):
            raise ValueError("invalid migration source file name")
        r.require_digest(name[len("generations/") : -len(".json")])
    for key in ("targets", "jobs", "generations"):
        if type(value[key]) is not dict or len(value[key]) > settings.max_logical_jobs:
            raise ValueError("invalid migration identity map")
        if len(set(value[key].values())) != len(value[key]):
            raise ValueError("noninjective migration identity map")
        for old, new in value[key].items():
            r.require_digest(old)
            r.require_digest(new)
    return value, data


def _intent(session_id, settings, receipt_digest, operation):
    return {
        "format": INTENT_FORMAT,
        "version": 1,
        "session_id": session_id,
        "settings_digest": settings.digest,
        "receipt_sha256": receipt_digest,
        "operation": operation,
    }


def _rollback_marker(session_id, settings, receipt_digest):
    return {
        "format": "ariadne-rolling-rollback",
        "version": 1,
        "session_id": session_id,
        "settings_digest": settings.digest,
        "receipt_sha256": receipt_digest,
    }


def _require_metadata(path, expected):
    _, data = _json(path, 8192)
    if data != r.encode_json(expected, 8192):
        raise ValueError("incomplete or conflicting migration metadata")


def _immutable_metadata(path, value):
    data = r.encode_json(value, 8192)
    if path.exists() or path.is_symlink():
        if s._read(path, 8192) != data:
            raise ValueError("conflicting immutable migration metadata")
        _sync(path.parent)
    else:
        s._atomic(path, data)


def resolve_storage(root, session_id, settings, *, recover_rollback=False):
    """Caller owns ROOT/writer.lock. Never initializes or resumes migration."""
    markers = (
        "v3",
        "legacy-snapshot",
        "migration.json",
        "migration-intent.json",
        "storage.json",
        "rollback-intent.json",
        "rollback.json",
    )
    if not any((root / name).exists() or (root / name).is_symlink() for name in markers):
        return root
    for name in markers:
        s._path_check(root / name, regular=name.endswith(".json"))
    _, receipt_bytes = _receipt(root, session_id, settings)
    digest = r.sha256(receipt_bytes)
    _require_metadata(
        root / "migration-intent.json", _intent(session_id, settings, digest, "upgrade")
    )
    if (root / "rollback-intent.json").exists():
        _require_metadata(
            root / "rollback-intent.json", _intent(session_id, settings, digest, "rollback")
        )
        _require_metadata(root / "rollback.json", _rollback_marker(session_id, settings, digest))
        if (root / "storage.json").exists():
            raise ValueError("incomplete rollback")
        with closing(_read_db(root / "journal.sqlite3", recover=recover_rollback)) as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] != 2:
                raise ValueError("incomplete rollback schema fence")
        return root
    if (root / "rollback.json").exists():
        raise ValueError("rollback intent missing")
    expected_selector = {
        "format": SELECTOR_FORMAT,
        "version": 1,
        "schema": 3,
        "session_id": session_id,
        "settings_digest": settings.digest,
        "receipt_sha256": digest,
    }
    _require_metadata(root / "storage.json", expected_selector)
    with closing(_read_db(root / "journal.sqlite3")) as conn:
        if conn.execute("PRAGMA user_version").fetchone()[0] != FENCE:
            raise ValueError("legacy migration fence missing")
    destination = root / "v3"
    s._path_check(destination)
    if not destination.is_dir():
        raise ValueError("migrated journal missing")
    return destination


@contextmanager
def audit(directory, home, session_id, settings, store, policy=None, *, schema=2):
    """Run all source/SQL/artifact checks WITHOUT normal startup recovery."""
    s._path_check(directory)
    s._path_check(directory / "generations")
    obj = s.RollingJournal.__new__(s.RollingJournal)
    obj.directory, obj.path = directory, directory / "journal.sqlite3"
    obj.generations, obj.manifest_path = directory / "generations", directory / "manifest.json"
    obj.settings, obj.session_id, obj.store, obj.policy = settings, session_id, store, policy
    obj.schema_version, obj.guard, obj.cache = schema, None, None
    obj._lock_fd, obj._busy = None, False
    obj._memo = r._ValidationMemo(store, session_id, policy, settings_digest=settings.digest)
    obj._memo.storage_version = 2 if schema == 3 else 1
    obj._conn = _read_db(obj.path)
    try:
        obj._validate_schema()
        obj._validate_records()
        with obj._operation():
            for row in obj._conn.execute("SELECT target_id FROM targets").fetchall():
                obj._validate_target(obj._target(row[0]))
            for index, path in enumerate(obj.generations.iterdir()):
                if index >= settings.max_logical_jobs:
                    raise ValueError("too many migration artifacts")
                view = obj._view(s._read(path, settings.max_checkpoint_bytes))
                if path.name != view.generation_id + ".json":
                    raise ValueError("rolling generation filename mismatch")
            view = obj._load_view()
            if view is not None:
                job, _ = obj._job(view.job_id)
                campaign = obj._campaign()
                if job["state"] == "ready":
                    if (
                        campaign["published_generation"] != view.parent_generation
                        or campaign["next_sequence"] != view.sequence
                    ):
                        raise ValueError("invalid interrupted publication")
                elif campaign["published_generation"] != view.generation_id:
                    raise ValueError("journal/manifest mismatch")
        yield obj
    finally:
        obj.close()


def _sql_snapshot(directory):
    with closing(_read_db(directory / "journal.sqlite3")) as conn:
        result = {"schema": conn.execute("PRAGMA user_version").fetchone()[0], "tables": {}}
        for table, order in (("campaign", "id"), ("targets", "target_id"), ("jobs", "job_id")):
            result["tables"][table] = [
                {
                    key: {"blob_hex": value.hex()} if type(value) is bytes else value
                    for key, value in dict(row).items()
                }
                for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}")
            ]
    return result


def _file_hash(path):
    s._path_check(path, regular=True)
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _files(directory):
    result = {"journal.sqlite3": _file_hash(directory / "journal.sqlite3")}
    if (directory / "manifest.json").exists():
        result["manifest.json"] = _file_hash(directory / "manifest.json")
    for path in (directory / "generations").iterdir():
        result["generations/" + path.name] = _file_hash(path)
    return result


def fingerprint(directory):
    value = _sql_snapshot(directory)
    value["artifacts"] = {
        key: value for key, value in _files(directory).items() if key != "journal.sqlite3"
    }
    return r.sha256(r.encode_json(value, 2**31 - 1))


def _snapshot(source, destination, files, settings):
    import shutil

    destination.mkdir(mode=0o700)
    (destination / "generations").mkdir(mode=0o700)
    for name in files:
        src, dst = source / name, destination / name
        s._path_check(src, regular=True)
        with src.open("rb") as inp, dst.open("xb") as out:
            os.chmod(dst, 0o600)
            shutil.copyfileobj(inp, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
    if _files(destination) != files:
        raise ValueError("legacy snapshot mismatch")
    _sync(destination / "generations")
    _sync(destination)
    _sync(destination.parent)


def _convert(source, destination, settings):
    """Source has passed read-only audit. No lifecycle operation is called."""
    destination.mkdir(mode=0o700)
    (destination / "generations").mkdir(mode=0o700)
    targets, jobs, views = {}, {}, {}
    for row in source._conn.execute("SELECT * FROM jobs"):
        old_job = dict(row)
        jobs[old_job["job_id"]] = old_job
        targets[old_job["target_id"]] = source._target(old_job["target_id"])
    old_views = []
    with source._operation():
        for path in source.generations.iterdir():
            old_views.append(source._view(s._read(path, settings.max_checkpoint_bytes)))
    new_targets, target_map, job_map, generation_map = {}, {}, {}, {}

    def convert_target(old, parent):
        hit = new_targets.get(old.target_id)
        if hit is not None:
            return hit
        new = DeltaTarget.from_legacy(old, parent, settings)
        new_targets[old.target_id] = new
        target_map[old.target_id], job_map[old.job_id] = new.target_id, new.job_id
        return new

    for old in sorted(old_views, key=lambda view: view.sequence):
        parent = None if old.parent_generation is None else views[old.parent_generation]
        old_target = targets[jobs[old.job_id]["target_id"]]
        new = convert_target(old_target, parent)
        data = r.generation_bytes(new, old.summary_text, sequence=old.sequence, settings=settings)
        # Semantic fields come from the independently validated old source.
        # Complete destination validation occurs before a fence or selector exists.
        from .rolling_delta import Segments

        events = Segments(
            None if parent is None else parent.event_ids, new.delta_event_ids, digests=True
        )
        uids = Segments(None if parent is None else parent.message_uids, new.delta_message_uids)
        value = {f.name: getattr(old, f.name) for f in r.fields(r.RollingView)}
        value.update(
            snapshot_bytes=data,
            generation_id=r.sha256(data),
            parent_generation=new.parent_generation,
            event_ids=events,
            message_uids=uids,
            job_id=new.job_id,
        )
        view = r.RollingView(**value)
        views[old.generation_id] = view
        generation_map[old.generation_id] = view.generation_id
        s._atomic(destination / "generations" / (view.generation_id + ".json"), data)
    for old in targets.values():
        convert_target(old, None if old.parent_generation is None else views[old.parent_generation])
    conn = sqlite3.connect(destination / "journal.sqlite3", timeout=0)
    os.chmod(destination / "journal.sqlite3", 0o600)
    try:
        conn.execute("PRAGMA synchronous=FULL")
        with conn:
            for sql in s.SCHEMA:
                conn.execute(sql)
            conn.execute(f"PRAGMA application_id={s.APPLICATION_ID}")
            conn.execute("PRAGMA user_version=3")
            campaign = dict(source._conn.execute("SELECT * FROM campaign").fetchone())
            if campaign["published_generation"] is not None:
                campaign["published_generation"] = generation_map[campaign["published_generation"]]
            columns = list(campaign)
            conn.execute(
                "INSERT INTO campaign VALUES(" + ",".join("?" for _ in columns) + ")",
                list(campaign.values()),
            )
            for row in source._conn.execute("SELECT * FROM targets"):
                new = new_targets[row["target_id"]]
                data = new.to_bytes(settings)
                conn.execute(
                    "INSERT INTO targets VALUES(?,?,?,?)",
                    (new.target_id, data, r.sha256(data), row["status"]),
                )
            for old in jobs.values():
                value = dict(old)
                value.update(
                    job_id=job_map[value["job_id"]], target_id=target_map[value["target_id"]]
                )
                if value["generation_id"] is not None:
                    value["generation_id"] = generation_map[value["generation_id"]]
                conn.execute(
                    "INSERT INTO jobs VALUES(" + ",".join("?" for _ in value) + ")",
                    list(value.values()),
                )
    finally:
        conn.close()
    if source.manifest_path.exists():
        old_manifest = r.decode_json(s._read(source.manifest_path, 8192), 8192)
        old_manifest.update(
            version=2,
            generation_id=generation_map[old_manifest["generation_id"]],
            job_id=job_map[old_manifest["job_id"]],
        )
        s._atomic(destination / "manifest.json", r.encode_json(old_manifest, 8192))
    _sync(destination / "generations")
    _sync(destination)
    _sync(destination.parent)
    return target_map, job_map, generation_map


@contextmanager
def _owned(home, session_id, settings, store):
    root_home = Path(home).absolute()
    s._path_check(root_home)
    marker = root_home / ".ariadne-offline-migration"
    s._path_check(marker, regular=True)
    if not marker.is_file():
        raise ValueError("explicit offline migration marker required")
    archive = root_home / "ariadne" / "events.sqlite3"
    s._path_check(archive, regular=True)
    if store.path.absolute() != archive or not archive.is_file():
        raise ValueError("migration archive binding mismatch")
    for suffix in ("-wal", "-shm", "-journal"):
        s._path_check(Path(str(archive) + suffix), regular=True)
    root = archive.parent / "rolling" / r.sha256(session_id.encode())
    s._path_check(root)
    if not root.is_dir() or s.fcntl is None:
        raise ValueError("existing offline POSIX session required")
    lock = root / "writer.lock"
    s._path_check(lock, regular=True)
    fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    try:
        s.fcntl.flock(fd, s.fcntl.LOCK_EX | s.fcntl.LOCK_NB)
        yield root_home, root
    finally:
        os.close(fd)


def _finish(root, session_id, settings, store, policy):
    receipt, data = _receipt(root, session_id, settings)
    intent, _ = _json(root / "migration-intent.json", 8192)
    if intent != {
        "format": INTENT_FORMAT,
        "version": 1,
        "session_id": session_id,
        "settings_digest": settings.digest,
        "receipt_sha256": r.sha256(data),
        "operation": "upgrade",
    }:
        raise ValueError("conflicting migration intent")
    if _files(root / "legacy-snapshot") != receipt["source_files"]:
        raise ValueError("legacy migration snapshot changed")
    if fingerprint(root / "legacy-snapshot") != receipt["source_fingerprint"]:
        raise ValueError("legacy migration snapshot changed")
    if fingerprint(root / "v3") != receipt["destination_fingerprint"]:
        raise ValueError("staged migration destination changed")
    with audit(
        root / "v3", root.parent.parent.parent, session_id, settings, store, policy, schema=3
    ):
        pass
    with closing(_read_db(root / "journal.sqlite3")) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 2:
        if (
            fingerprint(root) != receipt["source_fingerprint"]
            or _files(root) != receipt["source_files"]
        ):
            raise ValueError("migration source changed")
        conn = sqlite3.connect(root / "journal.sqlite3", timeout=0)
        try:
            conn.execute("PRAGMA synchronous=FULL")
            with conn:
                conn.execute(f"PRAGMA user_version={FENCE}")
        finally:
            conn.close()
        _sync(root)
    elif version == FENCE:
        with audit(
            root, root.parent.parent.parent, session_id, settings, store, policy, schema=FENCE
        ):
            pass
        source_value = _sql_snapshot(root)
        source_value["schema"] = 2
        source_value["artifacts"] = {
            key: value for key, value in _files(root).items() if key != "journal.sqlite3"
        }
        if r.sha256(r.encode_json(source_value, 2**31 - 1)) != receipt["source_fingerprint"]:
            raise ValueError("fenced migration source changed")
    else:
        raise ValueError("unsupported migration source version")
    selector = {
        "format": SELECTOR_FORMAT,
        "version": 1,
        "schema": 3,
        "session_id": session_id,
        "settings_digest": settings.digest,
        "receipt_sha256": r.sha256(data),
    }
    s._atomic(root / "storage.json", r.encode_json(selector, 8192))
    return receipt


def migrate(home, session_id, settings, *, store, policy=None):
    r.bounded_text(session_id, 4096)
    with _owned(home, session_id, settings, store) as (root_home, root):
        if (root / "rollback-intent.json").exists() or (root / "rollback.json").exists():
            raise ValueError("session already entered rollback; use explicit rollback recovery")
        if (root / "storage.json").exists():
            selected = resolve_storage(root, session_id, settings)
            receipt = _receipt(root, session_id, settings)[0]
            # A selected journal may have advanced legitimately since conversion.
            # Audit its current state without recovery; the conversion fingerprint
            # is a rollback boundary, not a permanent freeze of the live ledger.
            with audit(selected, root_home, session_id, settings, store, policy, schema=3):
                pass
            _cleanup_rollback_copies(root, receipt)
            _sync(root)
            return receipt
        if (root / "migration-intent.json").exists():
            return _finish(root, session_id, settings, store, policy)
        if any(
            (root / name).exists() or (root / name).is_symlink()
            for name in ("v3", "legacy-snapshot", "migration.json")
        ):
            raise ValueError("incomplete migration staging requires inspection")
        with audit(root, root_home, session_id, settings, store, policy) as source:
            source_files = _files(root)
            source_fingerprint = fingerprint(root)
            _snapshot(root, root / "legacy-snapshot", source_files, settings)
            target_map, job_map, generation_map = _convert(source, root / "v3", settings)
        with audit(root / "v3", root_home, session_id, settings, store, policy, schema=3):
            pass
        if _files(root) != source_files:
            raise ValueError("migration source changed")
        receipt = {
            "format": RECEIPT_FORMAT,
            "version": 1,
            "session_id": session_id,
            "settings_digest": settings.digest,
            "source_fingerprint": source_fingerprint,
            "destination_fingerprint": fingerprint(root / "v3"),
            "source_files": source_files,
            "targets": target_map,
            "jobs": job_map,
            "generations": generation_map,
        }
        data = r.encode_json(receipt, settings.max_logical_jobs * 8192 + 8192)
        s._atomic(root / "migration.json", data)
        intent = {
            "format": INTENT_FORMAT,
            "version": 1,
            "session_id": session_id,
            "settings_digest": settings.digest,
            "receipt_sha256": r.sha256(data),
            "operation": "upgrade",
        }
        s._atomic(root / "migration-intent.json", r.encode_json(intent, 8192))
        return _finish(root, session_id, settings, store, policy)


@contextmanager
def _prepared_restore(source, directory, expected_digest):
    """Hash and fsync a private streamed copy BEFORE removing the selector."""
    import hashlib
    import stat
    import tempfile

    s._path_check(source, regular=True)
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    name = None
    try:
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077:
            raise ValueError("invalid rollback database")
        fd, name = tempfile.mkstemp(prefix=".rollback-", dir=directory)
        digest = hashlib.sha256()
        with os.fdopen(source_fd, "rb", closefd=False) as inp, os.fdopen(fd, "wb") as out:
            remaining = info.st_size
            while remaining:
                chunk = inp.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("rollback database changed")
                remaining -= len(chunk)
                digest.update(chunk)
                out.write(chunk)
            if inp.read(1) or digest.hexdigest() != expected_digest:
                raise ValueError("rollback database changed")
            out.flush()
            os.fsync(out.fileno())
        yield Path(name)
    finally:
        os.close(source_fd)
        if name is not None and os.path.exists(name):
            os.unlink(name)


def _cleanup_rollback_copies(root, receipt):
    """Caller holds writer.lock; delete only exact, private snapshot duplicates."""
    candidates = sorted(root.glob(".rollback-*"))
    if not candidates:
        return
    expected = receipt["source_files"]["journal.sqlite3"]
    if _file_hash(root / "legacy-snapshot" / "journal.sqlite3") != expected:
        raise ValueError("legacy snapshot changed")
    for path in candidates:
        suffix = path.name.removeprefix(".rollback-")
        if (
            len(suffix) != 8
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in suffix)
            or _file_hash(path) != expected
        ):
            raise ValueError("unrecognized orphan rollback copy; inspect manually")
    # Validate the entire set before unlinking anything. Unknown/partial files
    # and links are never silently deleted or treated as a reusable database.
    for path in candidates:
        path.unlink()
    _sync(root)


def rollback(home, session_id, settings, *, store, policy=None):
    """Remove selector/publish completion metadata BEFORE final legacy admission."""
    with _owned(home, session_id, settings, store) as (root_home, root):
        receipt, data = _receipt(root, session_id, settings)
        digest = r.sha256(data)
        _require_metadata(
            root / "migration-intent.json", _intent(session_id, settings, digest, "upgrade")
        )
        _cleanup_rollback_copies(root, receipt)
        with closing(_read_db(root / "journal.sqlite3")) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        if (root / "rollback-intent.json").exists():
            _require_metadata(
                root / "rollback-intent.json", _intent(session_id, settings, digest, "rollback")
            )
            if (root / "rollback.json").exists() and version == 2:
                resolve_storage(root, session_id, settings)
                with audit(root, root_home, session_id, settings, store, policy):
                    pass
                _sync(root)
                return receipt  # Already restored; never overwrite subsequent legacy activity.
        else:
            resolve_storage(root, session_id, settings)
        if version != FENCE:
            raise ValueError("invalid rollback source fence")
        if fingerprint(root / "v3") != receipt["destination_fingerprint"]:
            raise ValueError("migration destination ledger diverged; rollback refused")
        snapshot = root / "legacy-snapshot"
        if (
            _files(snapshot) != receipt["source_files"]
            or fingerprint(snapshot) != receipt["source_fingerprint"]
        ):
            raise ValueError("legacy snapshot changed")
        with audit(root / "v3", root_home, session_id, settings, store, policy, schema=3):
            pass
        current = _sql_snapshot(root)
        current["schema"] = 2
        current["artifacts"] = {
            key: value for key, value in _files(root).items() if key != "journal.sqlite3"
        }
        if r.sha256(r.encode_json(current, 2**31 - 1)) != receipt["source_fingerprint"]:
            raise ValueError("legacy ledger diverged; rollback refused")
        with _prepared_restore(
            snapshot / "journal.sqlite3", root, receipt["source_files"]["journal.sqlite3"]
        ) as prepared:
            _immutable_metadata(
                root / "rollback-intent.json", _intent(session_id, settings, digest, "rollback")
            )
            selector = root / "storage.json"
            if selector.exists():
                selector.unlink()
                _sync(root)
            _immutable_metadata(
                root / "rollback.json", _rollback_marker(session_id, settings, digest)
            )
            # Old binaries ignore metadata: schema 2 is the LAST publication.
            os.replace(prepared, root / "journal.sqlite3")
            _sync(root)
        if fingerprint(root) != receipt["source_fingerprint"]:
            raise ValueError("restored legacy ledger mismatch")
        _sync(root)
        return receipt


def main(argv=None):
    import argparse
    import json

    from .prepared import ReasoningReplayPolicy
    from .store import EventStore, validate_schema

    parser = argparse.ArgumentParser(description="Explicit offline rolling journal conversion")
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--settings", default="{}", help="Exact rolling settings JSON (no credentials)"
    )
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args(argv)
    settings_value = r.decode_json(args.settings.encode(), 65536)
    if type(settings_value) is not dict:
        parser.error("settings must be a JSON object")
    settings = r.RollingSettings(**settings_value)
    if bool(args.provider) != bool(args.model):
        parser.error("provider and model must be supplied together")
    policy = ReasoningReplayPolicy(args.provider, args.model) if args.provider else None
    home = args.home.absolute()
    s._path_check(home)
    marker = home / ".ariadne-offline-migration"
    s._path_check(marker, regular=True)
    if not marker.is_file():
        raise ValueError("explicit offline migration marker required")
    path = home / "ariadne" / "events.sqlite3"
    s._path_check(path, regular=True)
    for suffix in ("-wal", "-shm", "-journal"):
        s._path_check(Path(str(path) + suffix), regular=True)
    if not path.is_file():
        raise ValueError("existing archive required")

    class ReadOnlyArchive(EventStore):
        def __init__(self, path):
            self.path = path

        @contextmanager
        def _connection(self):
            conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA query_only=ON")
                if validate_schema(conn):
                    raise ValueError("existing initialized archive required")
                yield conn
            finally:
                conn.close()

        def ingest(self, *args, **kwargs):
            raise ValueError("offline migration archive is read-only")

    store = ReadOnlyArchive(path)
    action = rollback if args.rollback else migrate
    receipt = action(home, args.session, settings, store=store, policy=policy)
    print(
        json.dumps(
            {
                "status": "rolled_back" if args.rollback else "migrated",
                "session_id": args.session,
                "targets": len(receipt["targets"]),
                "generations": len(receipt["generations"]),
            }
        )
    )


if __name__ == "__main__":
    os.umask(0o077)
    main()
