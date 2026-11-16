"""Schema-2 source evidence, not tests of an unimplemented migration.

Pins are captured from ddc433f. Inspection deliberately avoids RollingJournal:
its constructor changes reserved jobs and reconciles an interrupted manifest.
No provider, facade, migration, or job-submission API is invoked by inspection.
"""

import json
import sqlite3

import pytest
from test_rolling_state import setup, target

from hermes_ariadne import rolling, rolling_state

REQUEST = b'[{"content":"summarize","role":"user"}]'
SUMMARY = "  synthetic résumé\n"
ATTRIBUTION = {"provider": "fixture", "model": "fixture-model"}
USAGE = {"input_tokens": 17, "output_tokens": 3}
STATES = (
    "planned",
    "reserved",
    "ready",
    "published",
    "failed",
    "uncertain",
    "generation_only",
    "manifest_only",
)
# Hash of the complete logical SQL snapshot plus exact artifact bytes (hex),
# not a SQLite page-layout hash. Populated from real base execution, never guessed.
PINS = {
    "failed": "1e7284de3c04b011b49a4674a0b556f8c2b3bca62ec97e3476060186cb0a9677",
    "generation_only": "ddcf2cfb5c60ba2211cc295a40ca6bb41377f20f6188044304f700f4790031f9",
    "manifest_only": "e1656c57d6fab5bf201b4281d9b6de2cd18d44248996606821c042228ea4dd25",
    "planned": "99f45c1f75548d383990eac15f01801561de3fafc1e759320818b82ab5cc473b",
    "published": "8a0a4af3a17cc0d81781ea467de21de9c92535262f7f6a286b07746a2ea5455a",
    "ready": "d017005216ff405215fc81b51d7ada47801e22811a96ac4d98088a53c12ac6e1",
    "reserved": "a943945c714fb870fcf27af48ac8b1b2db8aecdd272772b9b0b02479e5ee168c",
    "uncertain": "42e7f4fa26db9b450501ac9ea8c14ef885b3fd697f0573a86641c9e10e366ef8",
}


def encode(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def bytes_value(value):
    return {"blob_hex": value.hex()} if isinstance(value, bytes) else value


def inspect_readonly(directory):
    """Freeze every SQL column and artifact without recovery or a writable DB."""
    with sqlite3.connect((directory / "journal.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        snapshot = {
            "application_id": conn.execute("PRAGMA application_id").fetchone()[0],
            "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
            "schema": conn.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
            ).fetchall(),
        }
        for table, order in (("campaign", "id"), ("targets", "target_id"), ("jobs", "job_id")):
            cursor = conn.execute(f"SELECT * FROM {table} ORDER BY {order}")
            columns = [item[0] for item in cursor.description]
            snapshot[table] = [
                {key: bytes_value(value) for key, value in zip(columns, row)} for row in cursor
            ]
    snapshot["artifacts"] = {
        str(path.relative_to(directory)): path.read_bytes().hex()
        for path in sorted(directory.rglob("*.json"))
    }
    return snapshot


def inventory(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def build_source(tmp_path, state, monkeypatch):
    store, settings = setup(tmp_path)
    frozen = target(store, settings, 1)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        journal.freeze_target(frozen)
        # Exercise exact non-null callback and anchor retention too.
        with journal._conn:
            journal._conn.execute("UPDATE campaign SET last_completed_anchor=?", ("anchor-α",))
        if state != "planned":
            assert journal.reserve(
                frozen.job_id,
                REQUEST,
                lambda: None,
                lambda: None,
                charging_callback_id="callback-α",
            )
        if state in {"ready", "published", "generation_only", "manifest_only"}:
            journal.record_ready(frozen.job_id, SUMMARY, ATTRIBUTION, USAGE)
        elif state in {"failed", "uncertain"}:
            journal.record_failed_or_uncertain(
                frozen.job_id, "facade_error", uncertain=state == "uncertain"
            )
        data = rolling.generation_bytes(frozen, SUMMARY, sequence=1, settings=settings)
        if state == "published":
            journal.publish_ready(frozen.job_id, data)
        elif state in {"generation_only", "manifest_only"}:
            # Interrupt the real publisher at its durable boundaries; do not
            # fabricate a manifest or claim the job has already been recovered.
            atomic = rolling_state._atomic

            def stop_before_manifest(path, value):
                if path == journal.manifest_path:
                    raise RuntimeError("synthetic interruption")
                atomic(path, value)

            def stop_before_marker(view):
                raise RuntimeError("synthetic interruption")

            with monkeypatch.context() as patch:
                if state == "generation_only":
                    patch.setattr(rolling_state, "_atomic", stop_before_manifest)
                else:
                    patch.setattr(journal, "_mark_published", stop_before_marker)
                with pytest.raises(RuntimeError, match="synthetic interruption"):
                    journal.publish_ready(frozen.job_id, data)
        directory = journal.directory
    return settings, frozen, data, directory


@pytest.mark.parametrize("state", STATES)
def test_literal_legacy_snapshot_survives_readonly_inspection(tmp_path, monkeypatch, state):
    settings, frozen, data, directory = build_source(tmp_path, state, monkeypatch)
    before = inventory(directory)

    def forbidden(*args, **kwargs):
        pytest.fail("read-only inspection must not open/recover/submit a rolling job")

    monkeypatch.setattr(rolling_state.RollingJournal, "__init__", forbidden)
    monkeypatch.setattr(rolling_state.RollingJournal, "reserve", forbidden)
    snapshot = inspect_readonly(directory)
    assert inspect_readonly(directory) == snapshot
    assert inventory(directory) == before
    assert snapshot["user_version"] == 2
    assert snapshot["application_id"] == 0x4152524C
    assert rolling.sha256(encode(snapshot)) == PINS[state]
    campaign = snapshot["campaign"][0]
    job = snapshot["jobs"][0]
    stored_target = snapshot["targets"][0]
    expected_state = "ready" if state.endswith("_only") else state
    assert job["state"] == stored_target["status"] == expected_state
    charged = state != "planned"
    assert campaign["logical_reservations"] == int(charged)
    assert (
        campaign["cumulative_request_bytes"]
        == job["reserved_request_bytes"]
        == (len(REQUEST) if charged else 0)
    )
    assert job["charging_callback_id"] == ("callback-α" if charged else None)
    assert campaign["last_completed_anchor"] == "anchor-α"
    assert (
        campaign["blocked_reason"]
        == job["error_code"]
        == ("facade_error" if state in {"failed", "uncertain"} else None)
    )
    payload = bytes.fromhex(stored_target["payload"]["blob_hex"])
    assert payload == frozen.to_bytes(settings)
    assert rolling.sha256(payload) == stored_target["payload_sha256"] == frozen.target_id
    decoded = json.loads(payload)
    assert decoded["request_bytes"].encode() == REQUEST
    assert decoded["readable_input"].encode() == frozen.readable_input
    assert decoded["event_ids"] == list(frozen.event_ids)
    assert decoded["message_uids"] == list(frozen.message_uids)
    assert frozen.prefix_sha256 == rolling.sha256(encode(list(frozen.event_ids)))
    assert job["job_id"] == frozen.job_id
    has_result = expected_state in {"ready", "published"}
    assert job["validated_text"] == (SUMMARY if has_result else None)
    assert job["text_sha256"] == (rolling.sha256(SUMMARY.encode()) if has_result else None)
    assert job["attribution"] == ({"blob_hex": encode(ATTRIBUTION).hex()} if has_result else None)
    assert job["usage_observed"] == (
        {"blob_hex": encode({"kind": "reported_or_defaulted", "value": USAGE}).hex()}
        if has_result
        else None
    )
    generation_id = rolling.sha256(data)
    assert (
        job["generation_id"]
        == campaign["published_generation"]
        == (generation_id if state == "published" else None)
    )
    assert campaign["next_sequence"] == (2 if state == "published" else 1)
    artifacts = snapshot["artifacts"]
    assert (f"generations/{generation_id}.json" in artifacts) == (
        state in {"published", "generation_only", "manifest_only"}
    )
    assert ("manifest.json" in artifacts) == (state in {"published", "manifest_only"})
