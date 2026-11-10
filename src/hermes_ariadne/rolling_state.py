"""Private per-session rolling logical-job journal; no dispatch or host imports.

A committed reservation is permission for ONE caller-owned facade invocation,
not a physical-send or billing guarantee. Failures and uncertainty stay charged.
POSIX cooperative single-writer ownership lasts until close; readers should use
this same journal, not open a second maintenance writer.
"""

import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

from .receipts import _fingerprint
from .rolling import (
    FORMAT,
    VERSION,
    Target,
    bounded_text,
    decode_json,
    encode_json,
    generation_bytes,
    require_digest,
    sha256,
    validate_generation,
)

APPLICATION_ID = 0x4152524C
SCHEMA_VERSION = 2  # v1 has no actual charging-callback identity: reject, never infer it.
SCHEMA = (
    """CREATE TABLE campaign (
        id INTEGER PRIMARY KEY CHECK(id=1), session_id TEXT NOT NULL,
        settings_digest TEXT NOT NULL, next_sequence INTEGER NOT NULL,
        logical_reservations INTEGER NOT NULL, cumulative_request_bytes INTEGER NOT NULL,
        published_generation TEXT, last_completed_anchor TEXT, blocked_reason TEXT
    )""",
    """CREATE TABLE targets (
        target_id TEXT PRIMARY KEY, payload BLOB NOT NULL, payload_sha256 TEXT NOT NULL,
        status TEXT NOT NULL
    )""",
    """CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY, target_id TEXT NOT NULL UNIQUE REFERENCES targets(target_id),
        state TEXT NOT NULL, reserved_request_bytes INTEGER NOT NULL,
        validated_text TEXT, text_sha256 TEXT, attribution BLOB, usage_observed BLOB,
        error_code TEXT, generation_id TEXT, charging_callback_id TEXT UNIQUE
    )""",
)


def _path_check(path, *, regular=False):
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError("rolling paths must not traverse symlinks")
    if path.exists():
        info = path.stat()
        if regular and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ValueError("rolling file must be regular and single-link")
        if info.st_mode & 0o077:
            raise ValueError("rolling path must be private")


def _read(path, limit):
    _path_check(path, regular=True)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_mode & 0o077
            or info.st_size > limit
        ):
            raise ValueError("invalid or oversized rolling artifact")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("rolling artifact exceeds read bound")
        return data
    finally:
        os.close(fd)


def _atomic(path, data):
    _path_check(path, regular=True)
    fd, name = tempfile.mkstemp(prefix=".rolling-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RollingJournal:
    def __init__(self, home, session_id, settings, *, store, policy=None, guard=None):
        bounded_text(session_id, 4096)
        self.settings, self.session_id, self.store, self.policy = (
            settings,
            session_id,
            store,
            policy,
        )
        self.guard = guard
        self._conn = None
        self._lock_fd = None
        self._busy = False
        self._check()
        root = Path(home).absolute()
        for item in (root, *root.parents):
            if item.is_symlink():
                raise ValueError("rolling home must not traverse symlinks")
        archive = root / "ariadne" / "events.sqlite3"
        if not archive.is_file() or store.path.absolute() != archive:
            raise ValueError("rolling requires the initialized explicit-home archive")
        _path_check(archive, regular=True)
        self.directory = archive.parent / "rolling" / sha256(session_id.encode("utf-8"))
        for directory in (self.directory.parent, self.directory, self.directory / "generations"):
            _path_check(directory)
            directory.mkdir(mode=0o700, exist_ok=True)
            _path_check(directory)
        self.path = self.directory / "journal.sqlite3"
        self.manifest_path = self.directory / "manifest.json"
        self.generations = self.directory / "generations"
        try:
            if fcntl is None:
                raise ValueError("rolling requires POSIX flock")
            lock = self.directory / "writer.lock"
            _path_check(lock, regular=True)
            self._lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            info = os.fstat(self._lock_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077:
                raise ValueError("invalid rolling lock")
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("rolling session already has a writer") from exc
            existed = self.path.exists() or self.path.is_symlink()
            _path_check(self.path, regular=True)
            artifacts = (
                self.manifest_path.exists()
                or self.manifest_path.is_symlink()
                or any(self.generations.iterdir())
            )
            if not existed and artifacts:
                raise ValueError("journal missing with existing generation artifacts")
            for suffix in ("-wal", "-shm", "-journal"):
                companion = Path(str(self.path) + suffix)
                _path_check(companion, regular=True)
                if companion.exists() and not existed:
                    raise ValueError("orphaned rolling journal companion")
            if not existed:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                os.close(fd)
            self._conn = sqlite3.connect(self.path, timeout=0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=FULL")
            with self._transaction():
                if not existed:
                    for sql in SCHEMA:
                        self._conn.execute(sql)
                    self._conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    self._conn.execute(
                        "INSERT INTO campaign VALUES(1,?,?,1,0,0,NULL,NULL,NULL)",
                        (session_id, settings.digest),
                    )
                self._validate_schema()
                self._validate_records()
                had_reserved = self._conn.execute(
                    "SELECT 1 FROM jobs WHERE state='reserved'"
                ).fetchone()
                self._conn.execute(
                    "UPDATE jobs SET state='uncertain',error_code='restart_uncertain' WHERE state='reserved'"
                )
                self._conn.execute(
                    "UPDATE targets SET status='uncertain' WHERE target_id IN (SELECT target_id FROM jobs WHERE state='uncertain')"
                )
                if had_reserved:
                    self._conn.execute("UPDATE campaign SET blocked_reason='restart_uncertain'")
            self._reconcile()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _check(self, *guards):
        for guard in (self.guard, *guards):
            if guard is not None:
                if not callable(guard):
                    raise ValueError("capture guard must be callable")
                guard()

    @contextmanager
    def _operation(self):
        if self._busy or self._conn is None:
            raise ValueError("rolling journal closed or reentrant operation")
        self._busy = True
        try:
            self._check()
            yield
            self._check()
        finally:
            self._busy = False

    @contextmanager
    def _transaction(self):
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            yield
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def _validate_schema(self):
        if (
            self._conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
            or self._conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
        ):
            raise ValueError("unsupported rolling journal schema")
        reference = sqlite3.connect(":memory:")
        try:
            for sql in SCHEMA:
                reference.execute(sql)
            if _fingerprint(reference) != _fingerprint(self._conn):
                raise ValueError("rolling journal schema fingerprint mismatch")
        finally:
            reference.close()

    def _campaign(self):
        if self._conn.execute("SELECT count(*) FROM campaign").fetchone()[0] != 1:
            raise ValueError("missing or multiple rolling campaigns")
        lengths = self._conn.execute(
            "SELECT length(CAST(session_id AS BLOB)),length(settings_digest),"
            "length(published_generation),length(last_completed_anchor),length(blocked_reason) FROM campaign"
        ).fetchone()
        for size, cap in zip(lengths, (4096, 64, 64, 4096, 64)):
            if size is not None and size > cap:
                raise ValueError("campaign field exceeds read bound")
        result = self._conn.execute("SELECT * FROM campaign").fetchall()
        if len(result) != 1:
            raise ValueError("missing or multiple rolling campaigns")
        row = dict(result[0])
        if row["session_id"] != self.session_id or row["settings_digest"] != self.settings.digest:
            raise ValueError("settings_mismatch or session mismatch")
        for key in ("next_sequence", "logical_reservations", "cumulative_request_bytes"):
            if type(row[key]) is not int or row[key] < (1 if key == "next_sequence" else 0):
                raise ValueError("corrupt campaign counters")
        return row

    def report(self):
        with self._operation():
            return self._campaign()

    def charged_callback_ids(self):
        """All known actual charging IDs; NULL legacy-style IDs remain unknown.

        Explicit-ID reservations refuse unknown historical charges. The original
        S1 reserve signature and ready-result recovery still work without guessing.
        """
        with self._operation():
            count = self._conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
            if count > self.settings.max_logical_jobs:
                raise ValueError("journal job count exceeds campaign bound")
            result = set()
            for row in self._conn.execute("SELECT job_id FROM jobs").fetchall():
                job, _ = self._job(row[0])
                callback = job["charging_callback_id"]
                if callback is not None:
                    bounded_text(callback, 4096)
                    if job["state"] == "planned" or callback in result:
                        raise ValueError("invalid charging callback identity")
                    result.add(callback)
            return result

    def _target(self, target_id):
        require_digest(target_id)
        row = self._conn.execute(
            "SELECT length(payload),length(CAST(payload_sha256 AS BLOB)),length(CAST(status AS BLOB)) "
            "FROM targets WHERE target_id=?",
            (target_id,),
        ).fetchone()
        if (
            row is None
            or row[0] > self.settings.max_checkpoint_bytes
            or row[1] != 64
            or row[2] > 16
        ):
            raise ValueError("missing or oversized frozen target")
        row = self._conn.execute(
            "SELECT payload,payload_sha256 FROM targets WHERE target_id=?", (target_id,)
        ).fetchone()
        target = Target.from_bytes(row["payload"], self.settings)
        if sha256(row["payload"]) != row["payload_sha256"] or target.target_id != target_id:
            raise ValueError("frozen target hash mismatch")
        return target

    def _job(self, job_id):
        require_digest(job_id)
        lengths = self._conn.execute(
            "SELECT length(target_id),length(CAST(validated_text AS BLOB)),length(text_sha256),"
            "length(attribution),length(usage_observed),length(error_code),length(generation_id),length(state),"
            "length(CAST(charging_callback_id AS BLOB)) "
            "FROM jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if lengths is None:
            raise ValueError("job must be frozen first")
        for size, cap in zip(
            lengths, (64, self.settings.max_summary_bytes, 64, 8192, 8192, 64, 64, 16, 4096)
        ):
            if size is not None and size > cap:
                raise ValueError("job field exceeds read bound")
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("job must be frozen first")
        result = dict(row)
        target = self._target(result["target_id"])
        if target.job_id != job_id:
            raise ValueError("job identity mismatch")
        return result, target

    def inspect(self, job_id):
        with self._operation():
            return self._job(job_id)[0]

    def _parent_bytes(self, digest):
        require_digest(digest)
        self._check()
        data = _read(self.generations / (digest + ".json"), self.settings.max_checkpoint_bytes)
        self._check()
        return data

    def _view(self, data):
        validated = []
        view = validate_generation(
            data,
            self.store,
            self.session_id,
            self.settings,
            policy=self.policy,
            parent_loader=self._parent_bytes,
            guard=self._check,
            _validated=validated,
        )
        # Reuse the exact bytes/objects already validated, including all ancestors.
        for current in validated:
            job, target = self._job(current.job_id)
            if (
                job["state"] not in {"ready", "published"}
                or job["validated_text"] != current.summary_text
                or generation_bytes(
                    target, job["validated_text"], sequence=current.sequence, settings=self.settings
                )
                != current.snapshot_bytes
            ):
                raise ValueError("unjournaled or mismatching generation")
        return view

    def _validate_target(self, target):
        if (
            type(target) is not Target
            or target.session_id != self.session_id
            or target.settings_digest != self.settings.digest
        ):
            raise ValueError("frozen target scope mismatch")
        if len(target.event_ids) > self.settings.max_prefix_events:
            raise ValueError("target source bound")
        parent = (
            None
            if target.parent_generation is None
            else self._view(self._parent_bytes(target.parent_generation))
        )
        source = []
        for event in target.event_ids:
            require_digest(event)
            self._check()
            source.append(self.store.get_event(event, session_id=self.session_id))
            self._check()
        rebuilt = Target.create(
            source,
            session_id=self.session_id,
            settings=self.settings,
            policy=self.policy,
            parent=parent,
            request_bytes=target.request_bytes,
            route_labels=target.route_labels,
        )
        if rebuilt != target:
            raise ValueError("frozen target source or identity mismatch")
        return target

    def pending_target(self):
        with self._operation():
            rows = self._conn.execute(
                "SELECT target_id FROM jobs WHERE state != 'published'"
            ).fetchall()
            if len(rows) > 1:
                raise ValueError("multiple pending rolling targets")
            return None if not rows else self._target(rows[0][0])

    def freeze_target(self, target):
        with self._operation(), self._transaction():
            self._validate_target(target)
            campaign = self._campaign()
            old = self._conn.execute(
                "SELECT job_id FROM jobs WHERE target_id=?", (target.target_id,)
            ).fetchone()
            if old:
                return self._job(old[0])[1]
            if (
                campaign["blocked_reason"]
                or self._conn.execute("SELECT 1 FROM jobs WHERE state != 'published'").fetchone()
            ):
                raise ValueError("rolling target blocked or already pending")
            if target.parent_generation != campaign["published_generation"]:
                raise ValueError("target parent is not current publication")
            if campaign["logical_reservations"] >= self.settings.max_logical_jobs:
                raise ValueError("logical job budget exhausted")
            payload = target.to_bytes(self.settings)
            self._conn.execute(
                "INSERT INTO targets VALUES(?,?,?,'planned')",
                (target.target_id, payload, sha256(payload)),
            )
            self._conn.execute(
                "INSERT INTO jobs VALUES(?,?,'planned',0,NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
                (target.job_id, target.target_id),
            )
            self._check()
            return target

    def reserve(
        self, job_id, request_bytes, attachment_guard, route_guard, *, charging_callback_id=None
    ):
        if not callable(attachment_guard) or not callable(route_guard):
            raise ValueError("live capture guards required")
        if charging_callback_id is not None:
            bounded_text(charging_callback_id, 4096)
        with self._operation(), self._transaction():
            self._check(attachment_guard, route_guard)
            job, target = self._job(job_id)
            if request_bytes != target.request_bytes:
                raise ValueError("frozen request mismatch")
            if job["state"] in {"reserved", "ready", "published"}:
                return False  # duplicate callback never receives invocation permission
            if job["state"] != "planned":
                raise ValueError("failed or uncertain target cannot retry")
            if charging_callback_id is not None:
                if self._conn.execute(
                    "SELECT 1 FROM jobs WHERE state!='planned' AND charging_callback_id IS NULL"
                ).fetchone():
                    raise ValueError("unknown charging callback in historical reservation")
                if self._conn.execute(
                    "SELECT 1 FROM jobs WHERE charging_callback_id=?", (charging_callback_id,)
                ).fetchone():
                    return False
            self._validate_target(target)
            self._check(attachment_guard, route_guard)
            campaign = self._campaign()
            size = len(request_bytes)
            if (
                campaign["blocked_reason"]
                or size > self.settings.max_request_bytes
                or campaign["logical_reservations"] >= self.settings.max_logical_jobs
                or campaign["cumulative_request_bytes"] + size
                > self.settings.max_total_request_bytes
            ):
                raise ValueError("rolling logical budget exhausted")
            self._conn.execute(
                "UPDATE jobs SET state='reserved',reserved_request_bytes=?,charging_callback_id=? WHERE job_id=?",
                (size, charging_callback_id, job_id),
            )
            self._conn.execute(
                "UPDATE targets SET status='reserved' WHERE target_id=?", (target.target_id,)
            )
            self._conn.execute(
                "UPDATE campaign SET logical_reservations=logical_reservations+1,cumulative_request_bytes=cumulative_request_bytes+?",
                (size,),
            )
            self._check(attachment_guard, route_guard)
        return True

    def record_ready(self, job_id, validated_summary, attribution, observed_usage):
        bounded_text(validated_summary, self.settings.max_summary_bytes)
        if type(attribution) is not dict or set(attribution) != {"provider", "model"}:
            raise ValueError("bounded result attribution required")
        for label in attribution.values():
            bounded_text(label, 1024)
        attribution_bytes = encode_json(attribution, 8192)
        usage = (
            None
            if observed_usage is None
            else encode_json({"kind": "reported_or_defaulted", "value": observed_usage}, 8192)
        )
        with self._operation(), self._transaction():
            job, target = self._job(job_id)
            if job["state"] in {"ready", "published"}:
                if (
                    job["validated_text"] != validated_summary
                    or job["attribution"] != attribution_bytes
                    or job["usage_observed"] != usage
                ):
                    raise ValueError("immutable ready result mismatch")
                return
            if job["state"] != "reserved":
                raise ValueError("ready requires a charged reservation")
            self._conn.execute(
                "UPDATE jobs SET state='ready',validated_text=?,text_sha256=?,attribution=?,usage_observed=? WHERE job_id=?",
                (
                    validated_summary,
                    sha256(validated_summary.encode("utf-8")),
                    attribution_bytes,
                    usage,
                    job_id,
                ),
            )
            self._conn.execute(
                "UPDATE targets SET status='ready' WHERE target_id=?", (target.target_id,)
            )
            self._check()

    def record_failed_or_uncertain(self, job_id, reason, *, uncertain=False):
        if (
            reason
            not in {
                "invalid_output",
                "facade_error",
                "capture_changed",
                "source_changed",
                "interrupted",
                "restart_uncertain",
            }
            or type(uncertain) is not bool
        ):
            raise ValueError("unsupported safe failure code")
        state = "uncertain" if uncertain else "failed"
        with self._operation(), self._transaction():
            job, target = self._job(job_id)
            if job["state"] == state and job["error_code"] == reason:
                return
            if job["state"] != "reserved":
                raise ValueError("failure requires a reserved job")
            self._conn.execute(
                "UPDATE jobs SET state=?,error_code=? WHERE job_id=?", (state, reason, job_id)
            )
            self._conn.execute(
                "UPDATE targets SET status=? WHERE target_id=?", (state, target.target_id)
            )
            self._conn.execute("UPDATE campaign SET blocked_reason=?", (reason,))

    def _manifest(self, view):
        return encode_json(
            {
                "format": FORMAT,
                "version": VERSION,
                "session_id": self.session_id,
                "settings_digest": self.settings.digest,
                "generation_id": view.generation_id,
                "job_id": view.job_id,
            },
            8192,
        )

    def _mark_published(self, view):
        with self._transaction():
            campaign = self._campaign()
            if campaign["published_generation"] == view.generation_id:
                return
            if (
                campaign["published_generation"] != view.parent_generation
                or campaign["next_sequence"] != view.sequence
            ):
                raise ValueError("publication lineage conflict")
            self._conn.execute(
                "UPDATE jobs SET state='published',generation_id=? WHERE job_id=?",
                (view.generation_id, view.job_id),
            )
            self._conn.execute(
                "UPDATE targets SET status='published' WHERE target_id=(SELECT target_id FROM jobs WHERE job_id=?)",
                (view.job_id,),
            )
            self._conn.execute(
                "UPDATE campaign SET published_generation=?,next_sequence=?",
                (view.generation_id, view.sequence + 1),
            )
            self._check()

    def publish_ready(self, job_id, immutable_generation_bytes):
        with self._operation():
            job, target = self._job(job_id)
            if job["state"] not in {"ready", "published"}:
                raise ValueError("publication requires durable ready result")
            self._validate_target(target)
            view = self._view(immutable_generation_bytes)
            if view.job_id != job_id:
                raise ValueError("generation job mismatch")
            campaign = self._campaign()
            if job["state"] == "published":
                if job["generation_id"] != view.generation_id:
                    raise ValueError("published generation mismatch")
                return view
            if (
                campaign["published_generation"] != target.parent_generation
                or campaign["next_sequence"] != view.sequence
                or campaign["blocked_reason"]
            ):
                raise ValueError("publication parent or campaign mismatch")
            path = self.generations / (view.generation_id + ".json")
            self._check()
            if path.exists() or path.is_symlink():
                if _read(path, self.settings.max_checkpoint_bytes) != view.snapshot_bytes:
                    raise ValueError("immutable generation collision")
            else:
                _atomic(path, view.snapshot_bytes)
            self._check()
            _atomic(self.manifest_path, self._manifest(view))
            self._check()
            self._mark_published(view)
            return view

    def _load_view(self):
        campaign = self._campaign()
        if not self.manifest_path.exists() and not self.manifest_path.is_symlink():
            if campaign["published_generation"] is not None:
                raise ValueError("published rolling manifest missing")
            return None
        self._check()
        data = _read(self.manifest_path, 8192)
        self._check()
        value = decode_json(data, 8192)
        if type(value) is not dict or "generation_id" not in value:
            raise ValueError("invalid rolling manifest")
        view = self._view(self._parent_bytes(require_digest(value["generation_id"])))
        if data != self._manifest(view):
            raise ValueError("rolling manifest identity mismatch")
        return view

    def load_view(self):
        with self._operation():
            view = self._load_view()
            if view is not None and self._campaign()["published_generation"] != view.generation_id:
                raise ValueError("rolling manifest not reconciled")
            return view

    def _reconcile(self):
        with self._operation():
            # Charge interrupted reservations before callback-capable source reads.
            for row in self._conn.execute("SELECT job_id FROM jobs").fetchall():
                self._validate_target(self._job(row[0])[1])
            view = self._load_view()
            for index, artifact in enumerate(self.generations.iterdir()):
                if index >= self.settings.max_logical_jobs:
                    raise ValueError("too many rolling generation artifacts")
                checked = self._view(_read(artifact, self.settings.max_checkpoint_bytes))
                if artifact.name != checked.generation_id + ".json":
                    raise ValueError("rolling generation filename mismatch")
            if view is not None:
                job, _ = self._job(view.job_id)
                if job["state"] == "ready":
                    self._mark_published(view)
                elif self._campaign()["published_generation"] != view.generation_id:
                    raise ValueError("journal/manifest mismatch")

    def _validate_records(self):
        campaign = self._campaign()
        count = self._conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
        if count > self.settings.max_logical_jobs:
            raise ValueError("journal job count exceeds campaign bound")
        if self._conn.execute("SELECT count(*) FROM targets").fetchone()[0] != count:
            raise ValueError("orphaned rolling target")
        published = self._conn.execute(
            "SELECT count(*) FROM jobs WHERE state='published'"
        ).fetchone()[0]
        if campaign["next_sequence"] != published + 1:
            raise ValueError("corrupt rolling sequence")
        if (campaign["published_generation"] is None) != (published == 0):
            raise ValueError("corrupt publication counter")
        reservations = total = pending = 0
        for row in self._conn.execute("SELECT job_id FROM jobs").fetchall():
            job, target = self._job(row[0])
            state = job["state"]
            if state not in {"planned", "reserved", "ready", "published", "failed", "uncertain"}:
                raise ValueError("invalid rolling job state")
            status = self._conn.execute(
                "SELECT status FROM targets WHERE target_id=?", (target.target_id,)
            ).fetchone()[0]
            if status != state:
                raise ValueError("target/job state mismatch")
            pending += state != "published"
            charged = state != "planned"
            callback = job["charging_callback_id"]
            if callback is not None:
                bounded_text(callback, 4096)
                if not charged:
                    raise ValueError("uncharged callback identity")
            expected = len(target.request_bytes) if charged else 0
            if (
                type(job["reserved_request_bytes"]) is not int
                or job["reserved_request_bytes"] != expected
            ):
                raise ValueError("invalid rolling reservation")
            reservations += charged
            total += expected
            if state in {"ready", "published"}:
                bounded_text(job["validated_text"], self.settings.max_summary_bytes)
                if sha256(job["validated_text"].encode("utf-8")) != job["text_sha256"]:
                    raise ValueError("ready summary hash mismatch")
                attribution = decode_json(job["attribution"], 8192)
                if type(attribution) is not dict or set(attribution) != {"provider", "model"}:
                    raise ValueError("invalid persisted result attribution")
                for label in attribution.values():
                    bounded_text(label, 1024)
                if encode_json(attribution, 8192) != job["attribution"]:
                    raise ValueError("noncanonical persisted attribution")
                if job["usage_observed"] is not None:
                    usage = decode_json(job["usage_observed"], 8192)
                    if (
                        type(usage) is not dict
                        or set(usage) != {"kind", "value"}
                        or usage["kind"] != "reported_or_defaulted"
                        or encode_json(usage, 8192) != job["usage_observed"]
                    ):
                        raise ValueError("invalid persisted usage observation")
            elif any(
                job[key] is not None
                for key in (
                    "validated_text",
                    "text_sha256",
                    "attribution",
                    "usage_observed",
                    "generation_id",
                )
            ):
                raise ValueError("unexpected rolling result fields")
            if state in {"failed", "uncertain"}:
                if (
                    job["error_code"]
                    not in {
                        "invalid_output",
                        "facade_error",
                        "capture_changed",
                        "source_changed",
                        "interrupted",
                        "restart_uncertain",
                    }
                    or campaign["blocked_reason"] != job["error_code"]
                ):
                    raise ValueError("invalid persisted failure classification")
            elif job["error_code"] is not None:
                raise ValueError("unexpected job failure classification")
            if state == "published":
                require_digest(job["generation_id"])
            elif job["generation_id"] is not None:
                raise ValueError("unpublished generation marker")
        if (
            pending > 1
            or reservations != campaign["logical_reservations"]
            or total != campaign["cumulative_request_bytes"]
            or total > self.settings.max_total_request_bytes
        ):
            raise ValueError("rolling accounting mismatch")
