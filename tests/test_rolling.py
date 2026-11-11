"""Pure rolling policy tests: synthetic canonical rows, no host imports."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from hermes_ariadne import rolling
from hermes_ariadne.prepared import ReasoningReplayPolicy
from hermes_ariadne.store import EventStore, canonical_json


def rows(count=1):
    return [
        {"message_uid": f"{i}-{role}", "role": role, "content": f"{role} {i}"}
        for i in range(count)
        for role in ("user", "assistant")
    ]


def encrypted_rows():
    result = rows()
    result[1].update(
        reasoning="readable companion",
        reasoning_content=None,
        codex_reasoning_items=json.dumps(
            [
                {
                    "type": "reasoning",
                    "id": "rs_one",
                    "encrypted_content": "OPAQUE_SENTINEL",
                    "summary": [{"type": "summary_text", "text": "private companion"}],
                    "_issuer_kind": "codex_backend",
                    "_issuer_model": "gpt-test",
                }
            ]
        ),
        codex_message_items=json.dumps(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "id": "msg_one",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "assistant 0"}],
                }
            ]
        ),
    )
    return result


def test_readable_carrier_omission_preserves_exact_archive(tmp_path):
    source = encrypted_rows()
    before = copy.deepcopy(source)
    archive = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    archive.ingest("s", source, origin="canonical")
    result = rolling.readable_delta(
        source, policy=ReasoningReplayPolicy("codex_backend", "gpt-test")
    )
    decoded = json.loads(result)
    assert decoded == [
        {
            "role": row["role"],
            "content": row["content"],
            "event_ref": "event:" + hashlib.sha256(canonical_json(row)).hexdigest(),
        }
        for row in source
    ]
    assert b"OPAQUE_SENTINEL" not in result
    assert b"companion" not in result
    assert source == before
    for row in source:
        event = hashlib.sha256(canonical_json(row)).hexdigest()
        assert canonical_json(archive.get_event(event, session_id="s")) == canonical_json(row)


@pytest.mark.parametrize(
    "patch",
    [
        {"data_use": ["no_training"]},
        {"data_use": None},
        {"display_metadata": '{"data_use":[],"data_use":[]}'},
        {"display_metadata": {"withheld_prompt": ""}},
        {"display_metadata": "[]"},
        {"api_content": "divergent"},
        {"reasoning_details": ["unknown"]},
        {"unknown": "carrier"},
        {"content": [{"type": "image"}]},
    ],
)
def test_restrictions_and_unknown_carriers_refused(patch):
    source = rows()
    source[1].update(patch)
    with pytest.raises(ValueError):
        rolling.readable_delta(source, policy=None)


def test_settings_immutable_digest_and_strict_types():
    settings = rolling.RollingSettings()
    assert settings.max_logical_jobs == 64
    assert settings.max_request_bytes == 65536
    assert settings.digest == rolling.RollingSettings().digest
    with pytest.raises(FrozenInstanceError):
        settings.max_logical_jobs = 1
    with pytest.raises(ValueError):
        rolling.RollingSettings(max_logical_jobs=True)


@pytest.mark.parametrize("metadata", ['{"x":1e999}', '{"x":NaN}', '{"x":"\\ud800"}'])
def test_malformed_inner_metadata_refused(metadata):
    source = rows()
    source[0]["display_metadata"] = metadata
    with pytest.raises(ValueError):
        rolling.readable_delta(source, policy=None)


def test_foreign_canonical_session_refused_by_target():
    source = rows()
    source[0]["session_id"] = "foreign"
    with pytest.raises(ValueError):
        rolling.Target.create(
            source,
            session_id="s",
            settings=rolling.RollingSettings(),
            policy=None,
            parent=None,
            request_bytes=b'[{"content":"x","role":"user"}]',
            route_labels=("codex_backend", "gpt-test"),
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"reasoning_content": []},
        {"reasoning": 1},
        {"codex_reasoning_items": "[]"},
        {"codex_reasoning_items": '[{"type":"reasoning"}]'},
    ],
)
def test_reasoning_codec_and_companion_refusals(patch):
    source = encrypted_rows()
    source[1].update(patch)
    with pytest.raises(ValueError):
        rolling.readable_delta(source, policy=ReasoningReplayPolicy("codex_backend", "gpt-test"))


def test_reasoning_route_and_duplicate_ids_refused():
    source = encrypted_rows()
    with pytest.raises(ValueError):
        rolling.readable_delta(source, policy=ReasoningReplayPolicy("codex_backend", "other"))
    with pytest.raises(ValueError):
        rolling.readable_delta(source, policy=None)
    doubled = source + copy.deepcopy(source)
    for row in doubled[2:]:
        row["message_uid"] += "-later"
    with pytest.raises(ValueError, match="duplicate retained reasoning"):
        rolling.readable_delta(doubled, policy=ReasoningReplayPolicy("codex_backend", "gpt-test"))


def test_tool_owner_readable_projection_uses_argument_strings_not_replay():
    source = encrypted_rows()
    owner = source.pop()
    owner.pop("codex_message_items")
    owner.update(
        content=None,
        tool_calls=[
            {
                "id": "call_one",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"exact":"text"}'},
            }
        ],
        tool_call_uids='{"call_one":"occurrence-one"}',
    )
    source += [
        owner,
        {
            "message_uid": "tool",
            "role": "tool",
            "content": "result",
            "tool_name": "lookup",
            "tool_call_id": "call_one",
            "tool_call_uid": "occurrence-one",
        },
        {"message_uid": "final", "role": "assistant", "content": "done"},
    ]
    before = copy.deepcopy(source)
    result = json.loads(
        rolling.readable_delta(source, policy=ReasoningReplayPolicy("codex_backend", "gpt-test"))
    )
    assert result[1]["tool_calls"] == [{"name": "lookup", "arguments": '{"exact":"text"}'}]
    assert result[2]["content"] == "result"
    assert source == before


@pytest.mark.parametrize(
    "raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'"\\ud800"', b'"\xff"']
)
def test_strict_json_rejects_duplicate_nonfinite_and_invalid_utf8(raw):
    with pytest.raises(ValueError):
        rolling.decode_json(raw, 1024)


def test_bounds_before_scalar_json_serialization(monkeypatch):
    original = json.dumps

    def guarded(value, *args, **kwargs):
        assert not isinstance(value, str) or len(value) <= 4096
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", guarded)
    with pytest.raises(ValueError):
        rolling.encode_json({"x": "x" * 100000}, 1024)
    with pytest.raises(ValueError):
        rolling.readable_delta(rows(5), policy=None)
    with pytest.raises(ValueError):
        rolling.encode_json({"x": "\ud800"}, 1024)


def test_reasoning_preflight_before_generic_serialization(monkeypatch):
    source = encrypted_rows()
    items = json.loads(source[1]["codex_reasoning_items"])
    items[0]["encrypted_content"] = "x" * (1024 * 1024 + 1)
    source[1]["codex_reasoning_items"] = items
    original = json.dumps

    def guarded(value, *args, **kwargs):
        if isinstance(value, str) and value and set(value) == {"x"}:
            raise AssertionError("oversized carrier reached serializer")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", guarded)
    with pytest.raises(ValueError, match="admission bound"):
        rolling.readable_delta(source, policy=ReasoningReplayPolicy("codex_backend", "gpt-test"))


def test_canonical_answer_text_cannot_use_restored_strip_exception():
    source = encrypted_rows()
    items = json.loads(source[1]["codex_message_items"])
    items[0]["content"][0]["text"] = " assistant 0 "
    source[1]["codex_message_items"] = json.dumps(items)
    source[1]["_db_persisted"] = True
    with pytest.raises(ValueError):
        rolling.readable_delta(source, policy=ReasoningReplayPolicy("codex_backend", "gpt-test"))


def test_validated_summary_strict_echo_and_output_bounds():
    source = rows()
    settings = rolling.RollingSettings()
    frozen = rolling.Target.create(
        source,
        session_id="s",
        settings=settings,
        policy=None,
        parent=None,
        request_bytes=b'[{"content":"x","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    valid = {"summary": "bounded summary", "covered_prefix_sha256": frozen.prefix_sha256}
    assert (
        rolling.validate_summary(rolling.encode_json(valid, 8192), frozen, settings)
        == "bounded summary"
    )
    for patch in [
        {"summary": ""},
        {"summary": "x" * 4097},
        {"summary": False},
        {"covered_prefix_sha256": "f" * 64},
        {"extra": "bad"},
    ]:
        bad = {**valid, **patch}
        with pytest.raises(ValueError):
            rolling.validate_summary(rolling.encode_json(bad, 65536), frozen, settings)
    for data in [
        b'{"summary":"one","summary":"two","covered_prefix_sha256":"x"}',
        b'{"summary":NaN}',
        b"x" * 65537,
    ]:
        with pytest.raises(ValueError):
            rolling.validate_summary(data, frozen, settings)


def test_frozen_target_rejects_mutable_fields_and_avoids_deepcopy(monkeypatch):
    from dataclasses import replace

    settings = rolling.RollingSettings()
    frozen = rolling.Target.create(
        rows(),
        session_id="s",
        settings=settings,
        policy=None,
        parent=None,
        request_bytes=b'[{"content":"x","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    with pytest.raises(ValueError):
        replace(frozen, event_ids=list(frozen.event_ids))
    with pytest.raises(ValueError):
        replace(frozen, request_bytes=bytearray(frozen.request_bytes))
    original = rolling.asdict

    def guard(value):
        assert not isinstance(value, rolling.Target), "deepcopy before bound"
        return original(value)

    monkeypatch.setattr(rolling, "asdict", guard)
    assert rolling.Target.from_bytes(frozen.to_bytes(settings), settings) == frozen


def test_active_prefix_seam_accepts_append_but_never_rebases():
    settings = rolling.RollingSettings()
    frozen = rolling.Target.create(
        rows(),
        session_id="s",
        settings=settings,
        policy=None,
        parent=None,
        request_bytes=b'[{"content":"x","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    assert rolling.validate_target_prefix(frozen, rows(3), settings=settings, policy=None) is None
    for active in [rows()[:1], list(reversed(rows())), rows()[1:]]:
        with pytest.raises(ValueError):
            rolling.validate_target_prefix(frozen, active, settings=settings, policy=None)
    edited = rows(2)
    edited[0]["content"] = "revision"
    with pytest.raises(ValueError):
        rolling.validate_target_prefix(frozen, edited, settings=settings, policy=None)


def test_settings_identity_binds_projection_version(monkeypatch):
    before = rolling.RollingSettings().digest
    monkeypatch.setattr(rolling, "PROJECTION_ID", "next-unsupported-version")
    assert rolling.RollingSettings().digest != before


@pytest.mark.parametrize("companion", [None, "", "readable companion"])
def test_native_null_columns_text_replay_and_empty_companions(companion):
    source = encrypted_rows()
    source[1].update(
        tool_calls=None,
        tool_call_id=None,
        tool_name=None,
        effect_disposition=None,
        reasoning=companion,
        absorbed_message_uids="[]",
        tool_call_uids=None,
        tool_call_uid=None,
        display_metadata=None,
        active=1,
    )
    before = canonical_json(source)
    result = json.loads(
        rolling.readable_delta(source, policy=ReasoningReplayPolicy("codex_backend", "gpt-test"))
    )
    assert result[1]["content"] == source[1]["content"]
    assert "codex_reasoning_items" not in result[1]
    assert canonical_json(source) == before


def test_frozen_route_labels_match_the_trusted_reasoning_binding():
    with pytest.raises(ValueError, match="route"):
        rolling.Target.create(
            encrypted_rows(),
            session_id="s",
            settings=rolling.RollingSettings(),
            policy=ReasoningReplayPolicy("codex_backend", "gpt-test"),
            parent=None,
            request_bytes=b'[{"content":"x","role":"user"}]',
            route_labels=("codex_backend", "foreign-model"),
        )
