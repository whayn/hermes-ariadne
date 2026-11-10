"""Pure oracle mutation checks: never import/start Hermes or use sockets."""

import ast
import copy
import hashlib
import ipaddress
import json
import os
import shutil
from pathlib import Path

import pytest
import test_rolling_transport as tracer


def fixture():
    host = [{"role": "system", "content": "SYNTHETIC HOST POLICY"}]
    old = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}]
    tail = [
        {"role": "user", "content": "recent"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "ariadne_find", "arguments": '{"query":"recent"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok":true}'},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current"},
    ]
    nav = "SYNTHETIC navigation rolling:" + "a" * 64
    return (
        host + old + tail,
        copy.deepcopy(host + [{"role": "assistant", "content": nav}] + tail),
        nav,
    )


def test_wire_oracle_exists_and_accepts_exact_positive():
    assert callable(getattr(tracer, "assert_wire", None)), "wire oracle not implemented"
    baseline, selected, nav = fixture()
    tracer.assert_wire(selected, baseline, covered=2, navigation=nav)


@pytest.mark.parametrize("encoding", ["json-text", "decoded-list"])
def test_baseline_oracle_accepts_tool_call_encodings_and_rejects_pair_omission(encoding):
    # SessionDB.get_messages returns tool_calls already decoded to a list (observed
    # in the native tracer); the oracle must accept that shape, not only JSON TEXT.
    assert callable(getattr(tracer, "assert_baseline", None)), (
        "canonical baseline oracle not implemented"
    )
    wire, _, _ = fixture()
    rows = copy.deepcopy(wire[1:-1])
    if encoding == "json-text":
        for row in rows:
            if "tool_calls" in row:
                row["tool_calls"] = json.dumps(row["tool_calls"])
        assert any(isinstance(row.get("tool_calls"), str) for row in rows)
    else:
        assert any(isinstance(row.get("tool_calls"), list) for row in rows)
    tracer.assert_baseline(wire, rows, "current")
    omitted = copy.deepcopy(wire)
    del omitted[4:6]
    with pytest.raises(AssertionError, match="baseline"):
        tracer.assert_baseline(omitted, rows, "current")


@pytest.mark.parametrize(
    "mutation", ["omit-nav", "corrupt-nav", "host", "tail", "tool-order", "drop-pair"]
)
def test_wire_oracle_rejects_each_corruption(mutation):
    baseline, selected, nav = fixture()
    if mutation == "omit-nav":
        del selected[1]
    elif mutation == "corrupt-nav":
        selected[1]["content"] += "!"
    elif mutation == "host":
        selected[0]["content"] += "!"
    elif mutation == "tail":
        selected[-1]["content"] += "!"
    elif mutation == "tool-order":
        selected[3], selected[4] = selected[4], selected[3]
    else:
        del selected[3:5]
    with pytest.raises(AssertionError, match="wire"):
        tracer.assert_wire(selected, baseline, covered=2, navigation=nav)


def test_checkpoint_and_restart_oracles_accept_exact_and_reject_uid_hash_charge_changes():
    rows = [{"role": "user", "content": "synthetic café  ", "message_uid": "uid-1"}]
    receipt = tracer.checkpoint(rows)
    assert receipt["event_ids"] == [hashlib.sha256(tracer.encode(rows[0])).hexdigest()]
    tracer.assert_checkpoint(rows, receipt)
    for key in ("content", "message_uid"):
        changed = copy.deepcopy(rows)
        changed[0][key] += "!"
        with pytest.raises(AssertionError, match="canonical"):
            tracer.assert_checkpoint(changed, receipt)
    campaign = {
        "published_generation": "a" * 64,
        "logical_reservations": 2,
        "cumulative_request_bytes": 2048,
        "session_id": "synthetic",
        "settings_digest": "b" * 64,
        "next_sequence": 3,
    }
    tracer.assert_restart(campaign, copy.deepcopy(campaign))
    for key, value in (
        ("published_generation", "c" * 64),
        ("logical_reservations", 1),
        ("cumulative_request_bytes", 1024),
        ("session_id", "other-synthetic"),
        ("settings_digest", "c" * 64),
        ("next_sequence", 4),
    ):
        changed = {**campaign, key: value}
        with pytest.raises(AssertionError, match="restart"):
            tracer.assert_restart(campaign, changed)


def test_expired_ref_oracle_rejects_retargeting():
    tracer.assert_expired({"ok": False, "error": "generation_expired"})
    with pytest.raises(AssertionError, match="expired"):
        tracer.assert_expired({"ok": True, "ref": "rolling:" + "b" * 64})


@pytest.mark.parametrize("key", sorted(tracer.CARRIERS))
def test_aux_oracle_rejects_opaque_and_readable_carrier_keys(key):
    clean = {"covered_prefix_sha256": "a" * 64, "delta": [{"role": "user", "content": "synthetic"}]}
    tracer.assert_aux(clean)
    leaked = copy.deepcopy(clean)
    leaked["delta"][0][key] = "SYNTHETIC forbidden carrier"
    with pytest.raises(AssertionError, match="carrier"):
        tracer.assert_aux(leaked)
    # JSON-encoded nested envelopes cannot bypass the recursive check.
    leaked = {**clean, "extra": json.dumps({key: "SYNTHETIC forbidden carrier"})}
    with pytest.raises(AssertionError, match="carrier"):
        tracer.assert_aux(leaked)


def test_canonical_codec_uses_utf8_original_not_ascii_escaped_unicode():
    assert tracer.encode({"x": "café 😀"}) == '{"x":"café 😀"}'.encode("utf-8")
    assert tracer.encode({"x": "\ud800"}) == b'{"x":"\xed\xa0\x80"}'


@pytest.mark.parametrize("mutation", ["call-uid", "result-uid", "order", "drop", "event-hash"])
def test_canonical_occurrence_and_order_mutants(mutation):
    rows = [
        {"role": "assistant", "message_uid": "owner", "tool_call_uids": '{"call":"occurrence"}'},
        {"role": "tool", "message_uid": "result", "tool_call_uid": "occurrence"},
    ]
    expected = tracer.checkpoint(rows)
    if mutation == "call-uid":
        rows[0]["tool_call_uids"] = '{"call":"regenerated"}'
    elif mutation == "result-uid":
        rows[1]["tool_call_uid"] = "regenerated"
    elif mutation == "order":
        rows.reverse()
    elif mutation == "drop":
        rows.pop()
    else:
        expected["event_ids"][0] = "0" * 64
    with pytest.raises(AssertionError, match="canonical"):
        tracer.assert_checkpoint(rows, expected)


@pytest.mark.parametrize("escape", ["outside", "symlink", "traversal"])
def test_actual_worker_guard_is_purely_extracted_and_rejects_shared_reads(tmp_path, escape):
    worker = Path(tracer.__file__).parent / "fixtures" / "rolling_worker.py"
    tree = ast.parse(worker.read_text())
    guard = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "credential_read_guard"
    ]
    assert len(guard) == 1
    # The violation record is a module-level list so host code that swallows the
    # guard's AssertionError still cannot hide the attempted read from the worker.
    record = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["credential_violations"]
    ]
    assert len(record) == 1, "worker must record guard violations at module level"
    home = tmp_path / "synthetic" / "isolated-home"
    managed = tmp_path / "synthetic-managed"
    home.mkdir(parents=True)
    managed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "auth.json"
    target.write_text("SYNTHETIC OUTSIDE SENTINEL")
    namespace = {"os": os, "Path": Path, "home": home, "managed": managed}
    exec(compile(ast.Module(body=record + guard, type_ignores=[]), str(worker), "exec"), namespace)
    read_guard = namespace["credential_read_guard"]
    violations = namespace["credential_violations"]
    assert violations == []
    if escape == "outside":
        path = target
    elif escape == "symlink":
        path = home / "harmless.txt"
        path.symlink_to(target)
    else:
        path = home.parent / ".." / "outside" / "auth.json"
    with pytest.raises(AssertionError, match="shared credential/config read forbidden"):
        read_guard("open", (path, "r", 0))
    assert violations == ["auth.json"], "blocked read was not recorded (name only)"
    # Swallowing the exception, as host code may do, must leave the record intact.
    try:
        read_guard("open", (path, "r", 0))
    except AssertionError:
        pass
    assert len(violations) == 2
    safe = home / "auth.json"
    safe.write_text("SYNTHETIC ALLOWED SENTINEL")
    read_guard("open", (safe, "r", 0))
    read_guard("open", (home / "unrelated.txt", "r", 0))
    read_guard("read", (path,))
    assert len(violations) == 2, "allowed/irrelevant events must not be recorded"


def test_loopback_guard_is_extracted_without_socket_or_native_startup():
    worker = Path(tracer.__file__).parent / "fixtures" / "rolling_worker.py"
    tree = ast.parse(worker.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_loopback_address"
    ]
    assert len(functions) == 1
    namespace = {"ipaddress": ipaddress}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(worker), "exec"), namespace)
    check = namespace["_loopback_address"]
    check(("127.0.0.1", 1234))
    check(("::1", 1234, 0, 0))
    for address in (("192.0.2.1", 443), ("provider.invalid", 443), "/synthetic/socket"):
        with pytest.raises((AssertionError, ValueError)):
            check(address)


def _profile(tmp_path):
    root = tmp_path / "synthetic-profile"
    home = root / "isolated-home"
    home.mkdir(parents=True)
    (home / "live-marker.txt").write_text("LIVE")
    return root, home, tmp_path / "parked-baseline-C"


def test_parked_profile_swaps_and_restores_live_profile_on_success(tmp_path):
    root, home, parked = _profile(tmp_path)
    with tracer.parked_profile(root, parked):
        assert not root.exists() and (parked / home.name / "live-marker.txt").is_file()
        home.mkdir(parents=True)
        (home / "baseline-only.txt").write_text("BASELINE")
    assert (home / "live-marker.txt").read_text() == "LIVE"
    assert not (home / "baseline-only.txt").exists() and not parked.exists()


def test_parked_profile_setup_failure_before_root_exists_does_not_mask_original(tmp_path):
    # Regression: the old inline finally called rmtree(root) on a missing root, so
    # FileNotFoundError replaced the real setup error and the live profile stayed parked.
    root, home, parked = _profile(tmp_path)
    with pytest.raises(OSError, match="synthetic setup failure") as caught:
        with tracer.parked_profile(root, parked):
            raise OSError("synthetic setup failure")  # root never recreated
    assert not isinstance(caught.value, FileNotFoundError)
    assert (home / "live-marker.txt").read_text() == "LIVE" and not parked.exists()


def test_parked_profile_partial_baseline_is_discarded_after_failure(tmp_path):
    root, home, parked = _profile(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic copy failure"):
        with tracer.parked_profile(root, parked):
            home.mkdir(parents=True)
            (home / "partial.db").write_text("PARTIAL")
            raise RuntimeError("synthetic copy failure")
    assert (home / "live-marker.txt").read_text() == "LIVE"
    assert not (home / "partial.db").exists() and not parked.exists()


def test_parked_profile_restore_failure_never_masks_original_exception(tmp_path, monkeypatch):
    root, home, parked = _profile(tmp_path)

    def failing_restore(*_args):
        raise OSError("synthetic restore failure detail")

    monkeypatch.setattr(tracer, "_restore_profile", failing_restore)
    with pytest.raises(RuntimeError, match="synthetic body failure") as caught:
        with tracer.parked_profile(root, parked):
            raise RuntimeError("synthetic body failure")
    # The restore error is recorded by exception type only, never raised over the original.
    assert caught.value.__notes__ == ["synthetic profile restore failed: OSError"]


def test_parked_profile_restore_failure_after_success_is_raised(tmp_path, monkeypatch):
    root, home, parked = _profile(tmp_path)

    def failing_restore(*_args):
        raise OSError("synthetic restore failure detail")

    monkeypatch.setattr(tracer, "_restore_profile", failing_restore)
    with pytest.raises(OSError, match="synthetic restore failure detail"):
        with tracer.parked_profile(root, parked):
            pass


def test_parked_profile_missing_parked_copy_is_reported_and_never_deletes_root(tmp_path):
    root, home, parked = _profile(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic body failure") as caught:
        with tracer.parked_profile(root, parked):
            shutil.rmtree(parked)  # the live profile copy is gone: nothing to restore
            home.mkdir(parents=True)
            (home / "baseline-only.txt").write_text("BASELINE")
            raise RuntimeError("synthetic body failure")
    assert caught.value.__notes__ == ["synthetic profile restore failed: FileNotFoundError"]
    assert (home / "baseline-only.txt").read_text() == "BASELINE", "root must not be deleted"

    root, home, parked = _profile(tmp_path / "success")
    with pytest.raises(FileNotFoundError):
        with tracer.parked_profile(root, parked):
            shutil.rmtree(parked)
            home.mkdir(parents=True)
    assert home.is_dir(), "root must not be deleted"
