"""Delta storage uses new IDs, unchanged coverage and exact frozen inputs."""

import importlib
import importlib.util
import json

import pytest
from test_rolling_state import setup, target

from hermes_ariadne import rolling


def codec():
    assert importlib.util.find_spec("hermes_ariadne.rolling_delta") is not None, (
        "delta-only storage codec is not implemented"
    )
    return importlib.import_module("hermes_ariadne.rolling_delta")


def test_delta_root_roundtrip_preserves_legacy_coverage_and_inputs(tmp_path):
    store, settings = setup(tmp_path)
    old = target(store, settings, 1)
    delta = codec().DeltaTarget.from_legacy(old, None, settings)
    value = json.loads(delta.to_bytes(settings))
    assert value["version"] == 2
    assert "event_ids" not in value and "message_uids" not in value
    assert value["delta_event_ids"] == list(old.event_ids)
    assert value["delta_message_uids"] == list(old.message_uids)
    assert delta.prefix_sha256 == old.prefix_sha256
    assert delta.request_bytes == old.request_bytes
    assert delta.readable_input == old.readable_input
    assert delta.target_id != old.target_id and delta.job_id != old.job_id
    restored = codec().DeltaTarget.from_bytes(delta.to_bytes(settings), settings)
    restored.bind(None, settings)
    assert restored == delta
    assert tuple(restored.event_ids) == old.event_ids
    assert tuple(restored.message_uids) == old.message_uids


def test_delta_journal_two_advances_restart_and_mixed_refusal(tmp_path):
    import pytest

    from hermes_ariadne.rolling_state import RollingJournal

    store, settings = setup(tmp_path)
    module = codec()
    old = target(store, settings, 1)
    with RollingJournal(tmp_path, "s", settings, store=store, storage_version=3) as journal:
        first = journal.encode_target(old, None)
        journal.freeze_target(first)
        assert journal.reserve(first.job_id, first.request_bytes, lambda: None, lambda: None)
        journal.record_ready(first.job_id, "first", {"provider": "t", "model": "t"}, None)
        parent = journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "first", sequence=1, settings=settings)
        )
        second_old = target(store, settings, 2, parent)
        second = journal.encode_target(second_old, parent)
        with pytest.raises(ValueError):
            journal.freeze_target(second_old)
        journal.freeze_target(second)
        assert journal.reserve(second.job_id, second.request_bytes, lambda: None, lambda: None)
        journal.record_ready(second.job_id, "second", {"provider": "t", "model": "t"}, None)
        view = journal.publish_ready(
            second.job_id, rolling.generation_bytes(second, "second", sequence=2, settings=settings)
        )
        assert isinstance(view.event_ids, module.Segments)
    with RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.schema_version == 3
        assert journal.load_view() == view
        assert journal.report()["logical_reservations"] == 2
        assert isinstance(journal.load_view().event_ids, module.Segments)


def test_delta_warm_open_reuses_only_hash_bound_work(tmp_path, monkeypatch):
    from hermes_ariadne.rolling_state import RollingJournal

    store, settings = setup(tmp_path)
    cache = rolling.RollingVerificationCache()
    old = target(store, settings, 1)
    with RollingJournal(
        tmp_path, "s", settings, store=store, storage_version=3, cache=cache
    ) as journal:
        first = journal.encode_target(old, None)
        journal.freeze_target(first)
        journal.reserve(first.job_id, first.request_bytes, lambda: None, lambda: None)
        journal.record_ready(first.job_id, "first", {"provider": "t", "model": "t"}, None)
        journal.publish_ready(
            first.job_id, rolling.generation_bytes(first, "first", sequence=1, settings=settings)
        )
    calls = []
    original = rolling._canonical_rows

    def counted(rows, **kwargs):
        calls.append(len(rows))
        return original(rows, **kwargs)

    verified = []
    verify = store.verify_events

    def counted_verify(event_ids, *, session_id):
        found = verify(event_ids, session_id=session_id)
        verified.extend(found)
        return found

    monkeypatch.setattr(rolling, "_canonical_rows", counted)
    monkeypatch.setattr(store, "verify_events", counted_verify)
    with RollingJournal(tmp_path, "s", settings, store=store, cache=cache) as journal:
        assert journal.load_view().frontier_count == len(old.event_ids)
    assert calls == [], "warm open must not recanonicalize sources"
    assert sorted(verified) == sorted(old.event_ids), "every source must be reverified once"


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 1),
        ("delta_start", 1),
        ("delta_stop", 99),
        ("frontier_count", 99),
        ("prefix_sha256", "0" * 64),
        ("parent_generation", "0" * 64),
        ("parent_summary_sha256", "0" * 64),
        ("delta_event_ids", []),
        ("delta_message_uids", ["wrong"]),
        ("summary_text", "tampered"),
    ],
)
def test_delta_generation_negative_schema_and_binding(tmp_path, field, value):
    store, settings = setup(tmp_path)
    old = target(store, settings, 1)
    module = codec()
    root = module.DeltaTarget.from_legacy(old, None, settings)
    payload = json.loads(module.generation_bytes(root, "first", sequence=1, settings=settings))
    payload[field] = value
    with pytest.raises(ValueError):
        module.validate_generation(
            rolling.encode_json(payload, settings.max_checkpoint_bytes), store, "s", settings
        )


def test_delta_target_detects_distant_ancestor_identity_collision(tmp_path):
    from hermes_ariadne.rolling_state import RollingJournal

    store, settings = setup(tmp_path)
    module, parent, first = codec(), None, None
    with RollingJournal(tmp_path, "s", settings, store=store, storage_version=3) as journal:
        for count in range(1, 7):
            old = target(store, settings, count, parent)
            current = journal.encode_target(old, parent)
            journal.freeze_target(current)
            journal.reserve(current.job_id, current.request_bytes, lambda: None, lambda: None)
            journal.record_ready(current.job_id, "summary", {"provider": "t", "model": "t"}, None)
            parent = journal.publish_ready(
                current.job_id,
                rolling.generation_bytes(current, "summary", sequence=count, settings=settings),
            )
            if first is None:
                first = current
        source = [store.get_event(event, session_id="s") for event in first.delta_event_ids]
        delta = rolling.readable_delta(source, settings=settings, policy=None)
        events = module.Segments(parent.event_ids, first.delta_event_ids, digests=True)
        bad = module.DeltaTarget(
            "s",
            settings.digest,
            parent.generation_id,
            parent.summary_sha256,
            first.delta_event_ids,
            first.delta_message_uids,
            parent.frontier_count,
            len(events),
            events.digest,
            rolling._summary_input(parent.summary_text, delta, settings),
            first.request_bytes,
            first.route_labels,
        )
        with pytest.raises(ValueError):
            journal.freeze_target(bad)
        assert journal.report()["logical_reservations"] == 6


def test_delta_target_rejects_forged_full_prefix_digest(tmp_path):
    from dataclasses import replace

    from hermes_ariadne.rolling_state import RollingJournal

    store, settings = setup(tmp_path)
    old = target(store, settings, 1)
    with RollingJournal(tmp_path, "s", settings, store=store, storage_version=3) as journal:
        encoded = journal.encode_target(old, None)
        with pytest.raises(ValueError, match="prefix hash"):
            journal.freeze_target(replace(encoded, prefix_sha256="0" * 64))
        assert journal.report()["logical_reservations"] == 0
        assert journal.pending_target() is None


def test_delta_generations_store_only_delta_and_reject_changed_sources(tmp_path):
    import pytest

    store, settings = setup(tmp_path)
    old = target(store, settings, 1)
    module = codec()
    root = module.DeltaTarget.from_legacy(old, None, settings)
    data = module.generation_bytes(root, "first", sequence=1, settings=settings)
    view = module.validate_generation(data, store, "s", settings)
    legacy = rolling.validate_generation(
        rolling.generation_bytes(old, "first", sequence=1, settings=settings),
        store,
        "s",
        settings,
    )
    old_child = target(store, settings, 2, parent=legacy)
    child = module.DeltaTarget.from_legacy(old_child, view, settings)
    child_data = module.generation_bytes(child, "second", sequence=2, settings=settings)
    value = json.loads(child_data)
    assert len(value["delta_event_ids"]) == len(old_child.event_ids) - legacy.frontier_count
    assert "event_ids" not in value
    checked = module.validate_generation(
        child_data,
        store,
        "s",
        settings,
        parent_loader=lambda digest: data,
    )
    assert tuple(checked.event_ids) == old_child.event_ids
    assert checked.prefix_sha256 == old_child.prefix_sha256
    assert checked.event_ids.parent.delta == view.event_ids.delta
    value["delta_start"] = 0
    with pytest.raises(ValueError):
        module.validate_generation(
            rolling.encode_json(value, settings.max_checkpoint_bytes),
            store,
            "s",
            settings,
            parent_loader=lambda digest: data,
        )
