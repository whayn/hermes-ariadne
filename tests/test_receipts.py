"""Issue #8: synthetic durable compaction receipts; no model or network calls."""

import hashlib
import os
import sqlite3
import stat

import pytest

from hermes_ariadne.receipts import CompactionLedger, LedgerBudgetError, LedgerError


def sha(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


LIMITS = {
    "max_attempts": 4,
    "max_request_bytes": 100,
    "max_total_request_bytes": 250,
    "max_attempts_per_job": 2,
    "max_jobs": 3,
}
NAMESPACE = {"projection": "synthetic", "model": "fixture", "prompt": 1}


def ledger(path, **kwargs):
    return CompactionLedger(path, namespace=NAMESPACE, limits=LIMITS, **kwargs)


def test_plans_are_immutable_persistent_exact_json_bindings(tmp_path):
    path = tmp_path / "receipts.sqlite"
    spec = {"sources": ["synthetic"], "text": "\ud800", "nested": {"x": None}}
    with ledger(path) as store:
        assert store.get_plan("job") is None
        assert store.get_receipt("job") is None
        assert store.put_plan("job", spec) == spec
        assert store.put_plan("job", spec) == spec
        with pytest.raises(LedgerError):
            store.put_plan("job", {"different": True})
    with ledger(path) as store:
        assert store.get_plan("job") == spec
    with pytest.raises(LedgerError):
        CompactionLedger(path, namespace={"different": True}, limits=LIMITS)
    with ledger(path) as store:
        assert store.get_plan("job") == spec


def test_committed_reservation_success_is_reusable_and_immutable(tmp_path):
    path = tmp_path / "receipts.sqlite"
    with ledger(path) as store:
        store.put_plan("job", {"kind": "model"})
        attempt = store.reserve(
            "job", request_sha256=sha("wire"), request_bytes=40, output_tokens=None
        )
        assert store.inspect("job")["state"] == "DISPATCHING"
        receipt = store.succeed("job", attempt, text="summary\ud800", usage=None)
        assert receipt == {
            "text": "summary\ud800",
            "text_sha256": sha("summary\ud800"),
            "attempt_id": attempt,
            "usage": None,
            "metadata": None,
        }
        assert store.succeed("job", attempt, text="summary\ud800", usage=None) == receipt
        with pytest.raises(LedgerError):
            store.succeed("job", attempt, text="different", usage=None)
        with pytest.raises(LedgerError):
            store.reserve("job", request_sha256=sha("wire"), request_bytes=40, output_tokens=10)
    with ledger(path) as store:
        assert store.get_receipt("job") == receipt
        assert store.inspect("job")["state"] == "SUCCEEDED"
        report = store.report()
        assert report["attempts"] == 1
        assert report["reserved_request_bytes"] == 40
        assert report["remaining"]["max_attempts"] == 3
        assert report["remaining"]["max_total_request_bytes"] == 210


@pytest.mark.parametrize("cap", ["max_attempts", "max_total_request_bytes", "max_attempts_per_job"])
def test_failures_remain_charged_and_caps_survive_restart(tmp_path, cap):
    path = tmp_path / "receipts.sqlite"
    limits = {**LIMITS, cap: 1 if cap != "max_total_request_bytes" else 40}
    with CompactionLedger(path, namespace=NAMESPACE, limits=limits) as store:
        store.put_plan("job", {})
        attempt = store.reserve(
            "job", request_sha256=sha("wire"), request_bytes=40, output_tokens=10
        )
        store.fail("job", attempt, reason="known bad output", usage={"input_tokens": 3})
    with CompactionLedger(path, namespace=NAMESPACE, limits=limits) as store:
        before = store.report()
        with pytest.raises(LedgerBudgetError):
            store.reserve("job", request_sha256=sha("retry"), request_bytes=1, output_tokens=10)
        assert store.report() == before
        assert store.inspect("job")["attempts"][0]["usage"] == {"input_tokens": 3}


def test_known_failure_can_retry_with_monotonic_ordinals(tmp_path):
    with ledger(tmp_path / "receipts.sqlite") as store:
        store.put_plan("job", {})
        first = store.reserve("job", request_sha256=sha("wire"), request_bytes=40, output_tokens=10)
        store.fail("job", first, reason="empty", usage=None)
        second = store.reserve(
            "job", request_sha256=sha("retry"), request_bytes=40, output_tokens=10
        )
        assert second > first
        assert [a["ordinal"] for a in store.inspect("job")["attempts"]] == [1, 2]
        with pytest.raises(LedgerError):
            store.succeed("job", first, text="stale", usage=None)
        store.succeed("job", second, text="good", usage={"output_tokens": 2})
        assert store.report()["reserved_request_bytes"] == 80


@pytest.mark.parametrize("explicit", [False, True])
def test_uncertain_dispatch_never_automatically_retries(tmp_path, explicit):
    path = tmp_path / "receipts.sqlite"
    with ledger(path) as store:
        store.put_plan("job", {})
        attempt = store.reserve(
            "job", request_sha256=sha("wire"), request_bytes=40, output_tokens=None
        )
        if explicit:
            store.fail("job", attempt, reason="timeout", uncertain=True)
    with ledger(path) as store:
        detail = store.inspect("job")
        assert detail["state"] == "UNCERTAIN"
        assert detail["attempts"][0]["state"] == "UNCERTAIN"
        assert detail["attempts"][0]["usage"] is None
        before = store.report()
        with pytest.raises(LedgerError):
            store.reserve("job", request_sha256=sha("retry"), request_bytes=40, output_tokens=None)
        assert store.report() == before
        assert store.get_receipt("job") is None


def test_local_receipts_require_unused_preplanned_slot_and_are_immutable(tmp_path):
    path = tmp_path / "receipts.sqlite"
    with ledger(path) as store:
        with pytest.raises(LedgerError):
            store.record_local("missing", text="literal", metadata={})
        store.put_plan("local", {"kind": "literal"})
        receipt = store.record_local("local", text="literal", metadata={"kind": "literal"})
        assert receipt["attempt_id"] is None and receipt["usage"] is None
        assert store.report()["attempts"] == 0
        assert store.record_local("local", text="literal", metadata={"kind": "literal"}) == receipt
        with pytest.raises(LedgerError):
            store.record_local("local", text="changed", metadata={})
        store.put_plan("remote", {})
        attempt = store.reserve(
            "remote", request_sha256=sha("wire"), request_bytes=40, output_tokens=None
        )
        store.fail("remote", attempt, reason="known failure")
        with pytest.raises(LedgerError):
            store.record_local("remote", text="bypass", metadata={})
    with ledger(path) as store:
        assert store.get_receipt("local") == receipt


def test_owned_schema_wal_full_sync_permissions_and_single_writer(tmp_path):
    path = tmp_path / "private" / "receipts.sqlite"
    with ledger(path) as store:
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        with pytest.raises(LedgerError):
            ledger(path)
        alias = tmp_path / "alias.sqlite"
        os.link(path, alias)
        with pytest.raises(LedgerError):
            ledger(alias)
        for suffix in ("", "-wal", "-shm", ".lock"):
            file = path.with_name(path.name + suffix)
            if file.exists():
                assert stat.S_IMODE(file.stat().st_mode) == 0o600
    with ledger(path):
        pass


@pytest.mark.parametrize("kind", ["plain", "empty", "foreign", "extra_schema", "version"])
def test_foreign_or_incompatible_file_preserved_and_init_releases_lock(tmp_path, kind):
    path = tmp_path / "receipts.sqlite"
    if kind == "plain":
        path.write_bytes(b"not sqlite")
    elif kind == "empty":
        path.touch()
    elif kind == "foreign":
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE foreign_data (secret BLOB)")
    else:
        with ledger(path):
            pass
        with sqlite3.connect(path) as conn:
            if kind == "extra_schema":
                conn.execute("CREATE TRIGGER surprise AFTER INSERT ON jobs BEGIN SELECT 1; END")
            else:
                conn.execute("PRAGMA user_version=99")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(LedgerError):
        ledger(path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    if kind not in ("plain", "empty"):
        conn.close()
    path.unlink()
    with ledger(path):
        pass


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal", ".lock"])
@pytest.mark.parametrize("inverse", [False, True])
def test_protected_filename_families_collide_in_both_directions(tmp_path, suffix, inverse):
    base = tmp_path / "archive.sqlite"
    protected, destination = (base, tmp_path / (base.name + suffix))
    if inverse:
        destination, protected = protected, destination
    protected.write_bytes(b"protected synthetic archive")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(LedgerError):
        ledger(destination, protected_paths=[protected])
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("alias", ["directory", "symlink", "hardlink", "sidecar_hardlink"])
def test_protected_aliases_are_rejected_before_any_writes(tmp_path, alias):
    directory = tmp_path / "real"
    directory.mkdir()
    archive = directory / "archive.sqlite"
    archive.write_bytes(b"synthetic protected content")
    if alias == "directory":
        link = tmp_path / "alias"
        link.symlink_to(directory, target_is_directory=True)
        destination = link / "archive.sqlite"
    elif alias == "symlink":
        destination = tmp_path / "alias.sqlite"
        destination.symlink_to(archive)
    elif alias == "hardlink":
        destination = tmp_path / "alias.sqlite"
        os.link(archive, destination)
    else:
        destination = tmp_path / "alias.sqlite"
        os.link(archive, tmp_path / "alias.sqlite-wal")
    with pytest.raises(LedgerError):
        ledger(destination, protected_paths=[archive])
    assert archive.read_bytes() == b"synthetic protected content"
    assert not (tmp_path / "alias.sqlite.lock").exists()


@pytest.mark.parametrize("column", ["plan", "receipt", "receipt_sha256", "usage"])
def test_corrupt_payloads_fail_closed_on_read_and_restart(tmp_path, column):
    path = tmp_path / "receipts.sqlite"
    with ledger(path) as store:
        store.put_plan("job", {})
        attempt = store.reserve(
            "job", request_sha256=sha("wire"), request_bytes=40, output_tokens=10
        )
        store.succeed("job", attempt, text="synthetic", usage=None)
        table = "attempts" if column == "usage" else "jobs"
        bad = "wrong" if column.endswith("sha256") else b'{"corrupt":true}'
        store._conn.execute(f"UPDATE {table} SET {column}=?", (bad,))
        store._conn.commit()
        with pytest.raises(LedgerError):
            store.inspect("job")
    with pytest.raises(LedgerError):
        ledger(path)


def test_caps_and_argument_rejections_do_not_mutate_ledger(tmp_path):
    path = tmp_path / "receipts.sqlite"
    with ledger(path) as store:
        for slot in ("a", "b", "c"):
            store.put_plan(slot, {})
        before = store.report()
        with pytest.raises(LedgerBudgetError):
            store.put_plan("overflow", {})
        with pytest.raises(LedgerBudgetError):
            store.reserve("a", request_sha256=sha("wire"), request_bytes=101, output_tokens=1)
        for invalid in (True, 0, -1, 1.5):
            with pytest.raises(LedgerError):
                store.reserve(
                    "a", request_sha256=sha("wire"), request_bytes=invalid, output_tokens=None
                )
            with pytest.raises(LedgerError):
                store.reserve(
                    "a", request_sha256=sha("wire"), request_bytes=40, output_tokens=invalid
                )
        with pytest.raises(LedgerError):
            store.reserve(
                "unplanned", request_sha256=sha("wire"), request_bytes=40, output_tokens=1
            )
        assert store.report() == before
    with pytest.raises(LedgerError):
        CompactionLedger(path, namespace=NAMESPACE, limits={**LIMITS, "max_jobs": 4})
    with ledger(path) as store:
        assert store.report() == before


@pytest.mark.parametrize(
    "limits",
    [
        {**LIMITS, "unknown": 1},
        {k: v for k, v in LIMITS.items() if k != "max_jobs"},
        *[{**LIMITS, "max_attempts": bad} for bad in (True, False, 0, -1, 1.5, None)],
    ],
)
def test_invalid_limits_do_not_create_files(tmp_path, limits):
    path = tmp_path / "unused" / "receipts.sqlite"
    with pytest.raises(LedgerError):
        CompactionLedger(path, namespace=NAMESPACE, limits=limits)
    assert not path.parent.exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "not a JSON key"}, (1, 2)])
def test_namespace_and_plan_require_finite_exact_json(tmp_path, bad):
    path = tmp_path / "receipts.sqlite"
    with pytest.raises(LedgerError):
        CompactionLedger(path, namespace={"bad": bad}, limits=LIMITS)
    assert not path.exists()
    with ledger(path) as store:
        with pytest.raises(LedgerError):
            store.put_plan("bad", {"bad": bad})
        assert store.report()["jobs"] == 0


def test_abrupt_exit_retains_charged_uncertain_reservation(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    path = tmp_path / "receipts.sqlite"
    script = """
import json, os, sys
from hermes_ariadne.receipts import CompactionLedger
store = CompactionLedger(sys.argv[1], namespace=json.loads(sys.argv[2]), limits=json.loads(sys.argv[3]))
store.put_plan('job', {'kind': 'synthetic'})
store.reserve('job', request_sha256='a' * 64, request_bytes=40, output_tokens=None)
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), json.dumps(NAMESPACE), json.dumps(LIMITS)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    with ledger(path) as store:
        assert store.inspect("job")["state"] == "UNCERTAIN"
        assert store.report()["attempts"] == 1
        assert store.report()["reserved_request_bytes"] == 40
        assert store.get_receipt("job") is None


def test_namespace_mismatch_preserves_owned_wal_family(tmp_path):
    path = tmp_path / "receipts.sqlite"
    with ledger(path):
        pass
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA user_version=1")
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        with pytest.raises(LedgerError, match="namespace"):
            CompactionLedger(path, namespace={"different": True}, limits=LIMITS)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    finally:
        conn.close()


def test_all_new_ledger_directories_are_private(tmp_path):
    path = tmp_path / "new" / "nested" / "receipts.sqlite"
    with ledger(path):
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal", ".lock"])
def test_orphaned_companions_are_not_adopted(tmp_path, suffix):
    path = tmp_path / "receipts.sqlite"
    companion = tmp_path / (path.name + suffix)
    companion.write_bytes(b"foreign companion")
    with pytest.raises(LedgerError):
        ledger(path)
    assert not path.exists()
    assert companion.read_bytes() == b"foreign companion"


def test_commit_failure_cannot_authorize_dispatch_or_publish_result(tmp_path, monkeypatch):
    import hermes_ariadne.receipts as receipts

    class FaultConnection(sqlite3.Connection):
        fail_commit = False

        def commit(self):
            if self.fail_commit:
                raise sqlite3.OperationalError("synthetic commit failure")
            return super().commit()

    real_connect = sqlite3.connect
    monkeypatch.setattr(
        receipts.sqlite3,
        "connect",
        lambda *a, **kw: real_connect(*a, **kw, factory=FaultConnection),
    )
    path = tmp_path / "receipts.sqlite"
    with ledger(path) as store:
        store.put_plan("job", {})
        before = store.report()
        store._conn.fail_commit = True
        with pytest.raises(LedgerError, match="transaction"):
            store.reserve("job", request_sha256=sha("wire"), request_bytes=40, output_tokens=None)
        assert store.report() == before
        assert store.inspect("job")["state"] == "PLANNED"
        store._conn.fail_commit = False
        attempt = store.reserve(
            "job", request_sha256=sha("wire"), request_bytes=40, output_tokens=None
        )
        store._conn.fail_commit = True
        with pytest.raises(LedgerError, match="transaction"):
            store.succeed("job", attempt, text="not durable", usage=None)
        assert store.get_receipt("job") is None
        assert store.report()["attempts"] == 1
    with ledger(path) as store:
        assert store.inspect("job")["state"] == "UNCERTAIN"


@pytest.mark.parametrize(
    "table,column,bad",
    [
        ("jobs", "receipt", None),
        ("jobs", "state", "BROKEN"),
        ("attempts", "request_bytes", -1),
        ("attempts", "ordinal", 2),
        ("attempts", "state", "FAILED"),
    ],
)
def test_inconsistent_persisted_states_are_corruption(tmp_path, table, column, bad):
    path = tmp_path / "receipts.sqlite"
    with ledger(path) as store:
        store.put_plan("job", {})
        attempt = store.reserve(
            "job", request_sha256=sha("wire"), request_bytes=40, output_tokens=10
        )
        store.succeed("job", attempt, text="synthetic", usage=None)
        store._conn.execute(f"UPDATE {table} SET {column}=?", (bad,))
        store._conn.commit()
        with pytest.raises(LedgerError):
            store.inspect("job")
    with pytest.raises(LedgerError):
        ledger(path)
