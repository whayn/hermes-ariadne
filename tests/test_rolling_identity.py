"""Target identity must not reapply default bounds to a target admitted by its campaign."""

import pytest
from test_rolling_state import setup

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.rolling import sha256

REQUEST = b'[{"content":"summarize","role":"user"}]'


def large_target(store, settings, text_chars):
    source = [
        {"message_uid": "0-user", "role": "user", "content": "x" * text_chars},
        {"message_uid": "0-assistant", "role": "assistant", "content": "ok"},
    ]
    store.ingest("s", source, origin="canonical")
    return rolling.Target.create(
        source,
        session_id="s",
        settings=settings,
        policy=None,
        parent=None,
        request_bytes=REQUEST,
        route_labels=("codex_backend", "gpt-test"),
    )


def test_configured_larger_target_has_identity_and_can_be_frozen(tmp_path):
    store, settings = setup(tmp_path, max_request_bytes=131072, max_readable_delta_bytes=100000)
    frozen = large_target(store, settings, 70000)
    assert len(frozen.readable_input) > rolling.RollingSettings().max_request_bytes
    assert frozen.target_id == sha256(frozen.to_bytes(settings))
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        journal.freeze_target(frozen)
        assert journal.pending_target() == frozen
        assert journal.reserve(frozen.job_id, frozen.request_bytes, lambda: None, lambda: None)
        journal.record_ready(frozen.job_id, "summary", {"provider": "t", "model": "t"}, None)
        view = journal.publish_ready(
            frozen.job_id, rolling.generation_bytes(frozen, "summary", sequence=1)
        )
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view() == view
        assert journal.inspect(frozen.job_id)["state"] == "published"


def test_identity_is_unchanged_for_targets_within_default_bounds(tmp_path):
    store, settings = setup(tmp_path)
    frozen = large_target(store, settings, 10)
    # Existing journals key rows by this exact digest; it must not move.
    assert frozen.target_id == sha256(frozen.to_bytes(rolling.RollingSettings()))
    assert frozen.target_id == sha256(frozen.to_bytes(settings))


def test_campaign_bounds_still_refuse_an_oversized_target(tmp_path):
    store, settings = setup(tmp_path, max_request_bytes=131072, max_readable_delta_bytes=100000)
    frozen = large_target(store, settings, 70000)
    small = rolling.RollingSettings(trigger_bytes=1)
    with pytest.raises(ValueError, match="input bound"):
        frozen.to_bytes(small)


def test_identity_digest_is_pinned_to_pre_fix_value():
    # Digests computed with the pre-fix implementation (48f9a2c): durable job
    # and target rows in existing journals must keep these exact keys.
    source = [
        {"message_uid": "0-user", "role": "user", "content": "golden question"},
        {"message_uid": "0-assistant", "role": "assistant", "content": "ok"},
    ]
    frozen = rolling.Target.create(
        source,
        session_id="s",
        settings=rolling.RollingSettings(trigger_bytes=1),
        policy=None,
        parent=None,
        request_bytes=REQUEST,
        route_labels=("codex_backend", "gpt-test"),
    )
    assert frozen.target_id == "94d0aa19f5ee1dc950cae7f62c5f591210ce5de7a056f7129283ed725160499c"
    assert frozen.job_id == "30dd362683583f6b261c0d7e0ec0ff1f46e6123b78d5fb2d638b2503856b0faa"
