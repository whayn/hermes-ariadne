"""Tests for rolling context trigger correction and starvation fix."""

import json
import os

import pytest
from test_rolling_engine import complete, journal_snapshot, select, setup_engine
from test_rolling_selection import POLICY, exchange

from hermes_ariadne import rolling_engine
from hermes_ariadne.rolling import RollingSettings, encode_json, readable_delta

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("HERMES_SOURCE required for ContextEngine ABC", allow_module_level=True)


def test_default_settings_short_exchange_accumulation(tmp_path, monkeypatch):
    """Many short exchanges trigger preparation under default settings (PR18 starvation fix).

    Default settings: max_delta_exchanges=4, trigger_bytes=4096.
    Each 4-exchange candidate is ~1021 bytes (below 4096), but aggregate aged exchanges
    exceed 4096 bytes. Under the old code, preparation never triggered.
    Under the fix, facade is called, summarizing earliest max 4 exchanges, leaving newest unretired.
    """
    settings = RollingSettings()
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)

    # Accumulate 30 short exchanges (each has 4 rows; ~255 bytes descriptor length each)
    # Total aged descriptor bytes will well exceed 4096.
    for i in range(30):
        tag = f"ex{i:02d}"
        db.rows += exchange(tag)
        complete(engine, db, tag)

    # Facade should have been called at least once
    assert len(facade.calls) >= 1

    # Published view exists and covers the earliest prefix
    published = engine._published
    assert published is not None
    assert published.frontier_count <= 4 * settings.max_delta_exchanges
    assert published.frontier_count == 16
    assert published.delta_start == 0
    assert published.delta_stop == 16

    # Inspect the first call request: delta must contain at most 4 exchanges (earliest prefix)
    first_req = json.loads(facade.calls[0]["messages"][-1]["content"])
    delta = first_req["delta"]
    assert len(delta) <= 4 * settings.max_delta_exchanges
    # Earliest exchange was ex00
    assert any("ex00" in d.get("content", "") for d in delta if d.get("content"))
    # Newest exchange (ex29) must not be in the delta
    assert not any("ex29" in d.get("content", "") for d in delta if d.get("content"))


def test_no_target_with_single_exchange_and_newest_excluded_from_threshold(tmp_path, monkeypatch):
    """No target with only one exchange; below-threshold aged bytes do not count newest exchange."""
    # Even with trigger_bytes=1, 1 exchange never triggers
    engine, db, facade = setup_engine(
        tmp_path / "h1", monkeypatch, settings=RollingSettings(trigger_bytes=1)
    )
    db.rows = exchange("A")
    complete(engine, db, "A")
    assert not facade.calls

    # Derive threshold dynamically from actual small canonical readable projection
    ex_a = exchange("A")
    ex_b = exchange("B")
    delta_a = readable_delta(ex_a, policy=POLICY)
    # Set trigger to exactly len(delta_a) + 1
    trigger = len(delta_a) + 1

    engine2, db2, facade2 = setup_engine(
        tmp_path / "h2", monkeypatch, settings=RollingSettings(trigger_bytes=trigger)
    )
    db2.rows = list(ex_a)
    complete(engine2, db2, "A")
    assert not facade2.calls

    db2.rows += ex_b
    complete(engine2, db2, "B")
    # A alone is len(delta_a) < trigger, and B is newest completed so excluded. No trigger!
    assert not facade2.calls

    # Now add Exchange C: Exchange A + B are both eligible.
    # Virtual merged length > trigger. Exchange C is newest. Trigger fires!
    db2.rows += exchange("C")
    complete(engine2, db2, "C")
    assert len(facade2.calls) == 1


def test_exact_virtual_array_byte_accounting_multibyte_and_tool_groups(tmp_path, monkeypatch):
    """Exact virtual-array length matches independently assembled JSON array including Unicode."""
    ex1 = exchange("u1")
    ex1[0]["content"] = "質問: こんにちは世界 🚀"
    ex2 = exchange("u2")
    ex2[0]["content"] = "ツール結果: テスト 🤖"

    desc1 = json.loads(readable_delta(ex1, policy=POLICY).decode("utf-8"))
    desc2 = json.loads(readable_delta(ex2, policy=POLICY).decode("utf-8"))
    expected_descriptors = desc1 + desc2
    expected_bytes = encode_json(expected_descriptors, 65536)
    expected_len = len(expected_bytes)

    active = ex1 + ex2 + exchange("newest")
    stops = [len(ex1), len(ex1) + len(ex2)]

    # Test threshold-1 does NOT trigger
    engine_below, _, _ = setup_engine(
        tmp_path / "h_below",
        monkeypatch,
        settings=RollingSettings(trigger_bytes=expected_len + 1),
    )
    scope_below = engine_below._capture()
    counted = engine_below._count_trigger_bytes(scope_below, active, 0, stops)
    assert counted == expected_len
    assert engine_below._new_target(scope_below, active, None) is None

    # Test exact threshold triggers
    engine_exact, _, _ = setup_engine(
        tmp_path / "h_exact",
        monkeypatch,
        settings=RollingSettings(trigger_bytes=expected_len),
    )
    scope_exact = engine_exact._capture()
    target = engine_exact._new_target(scope_exact, active, None)
    assert target is not None


def test_threshold_stop_with_later_invalid_exchange_unexamined(tmp_path, monkeypatch):
    """Counting stops when threshold reached; later invalid exchange is not examined."""
    ex_a = exchange("A")
    ex_b = exchange("B")
    delta_a = readable_delta(ex_a, policy=POLICY)

    # Trigger threshold reached on exchange A alone
    trigger = len(delta_a)
    engine, _, _ = setup_engine(
        tmp_path, monkeypatch, settings=RollingSettings(trigger_bytes=trigger)
    )

    # Exchange C is deliberately malformed/unsupported (e.g. invalid role)
    ex_c = [
        {
            "role": "invalid_role",
            "content": "broken",
            "message_uid": "c1",
            "session_id": "s",
            "active": 1,
        }
    ]
    ex_newest = exchange("newest")
    active = ex_a + ex_b + ex_c + ex_newest
    # Stops between exchanges A, B, C
    stops = [len(ex_a), len(ex_a) + len(ex_b), len(ex_a) + len(ex_b) + len(ex_c)]

    scope = engine._capture()
    # Counting stops after exchange A reaches threshold; invalid exchange C is not read.
    counted = engine._count_trigger_bytes(scope, active, 0, stops)
    assert counted >= trigger


def test_invalid_or_oversized_examined_exchange_refuses_no_skip(tmp_path, monkeypatch):
    """Examined exchange before threshold that is invalid/oversized refuses without skipping."""
    ex_a = exchange("A")
    # Corrupted second exchange
    ex_b_bad = [
        {
            "role": "bad_role",
            "content": "corrupt",
            "message_uid": "b1",
            "session_id": "s",
            "active": 1,
        }
    ]
    ex_c = exchange("C")
    active = ex_a + ex_b_bad + ex_c + exchange("newest")
    stops = [len(ex_a), len(ex_a) + len(ex_b_bad), len(ex_a) + len(ex_b_bad) + len(ex_c)]

    engine, _, _ = setup_engine(tmp_path, monkeypatch, settings=RollingSettings(trigger_bytes=4096))
    scope = engine._capture()

    # Examination of ex_b_bad must raise ValueError without skipping to ex_c
    with pytest.raises(ValueError):
        engine._count_trigger_bytes(scope, active, 0, stops)


def test_max_prefix_events_scan_bound_and_guard(tmp_path, monkeypatch):
    """Stop index exceeding max_prefix_events or invalid scope guard raises ValueError."""
    ex_a = exchange("A")
    ex_b = exchange("B")
    active = ex_a + ex_b + exchange("newest")
    stops = [len(ex_a), len(ex_a) + len(ex_b)]

    # Setting max_prefix_events smaller than stop index of exchange B
    settings = RollingSettings(max_prefix_events=len(ex_a) + 1, trigger_bytes=1000)
    engine, _, _ = setup_engine(tmp_path / "h_guard", monkeypatch, settings=settings)
    scope = engine._capture()

    with pytest.raises(ValueError, match="exceeded max_prefix_events"):
        engine._count_trigger_bytes(scope, active, 0, stops)

    # Test scope guard check
    valid_stops = [len(ex_a)]
    # Invalidate scope by updating route revision
    engine.update_model("gpt-changed", 100000, "https://invalid.test", "", "fixture", "fixture")
    with pytest.raises(ValueError, match="rolling capture changed"):
        engine._count_trigger_bytes(scope, active, 0, valid_stops)


def test_post_publication_counts_only_newly_aged_delta_and_advances(tmp_path, monkeypatch):
    """After publication, count starts after frontier; old prefix/summary excluded; advances later."""
    settings = RollingSettings(trigger_bytes=1)
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)

    # 1. Complete exchange A and B -> triggers first target
    db.rows = exchange("A")
    complete(engine, db, "A")
    db.rows += exchange("B")
    complete(engine, db, "B")

    assert len(facade.calls) == 1
    first = engine._published
    assert first is not None
    assert first.frontier_count == len(exchange("A"))

    # 2. Incoming user message for exchange C adopts first view
    db.rows += exchange("C", complete=False)[:1]
    assert select(engine, db, "C") is not None
    assert engine._pinned == first

    # 3. Exchange C completes -> triggers second advance starting at first.frontier_count
    db.rows = exchange("A") + exchange("B") + exchange("C")
    complete(engine, db, "C")
    second = engine._published
    assert second is not None
    assert second.sequence == 2
    assert second.frontier_count == len(exchange("A")) + len(exchange("B"))
    assert second.parent_generation == first.generation_id
    assert second.delta_start == first.frontier_count
    assert len(facade.calls) == 2


def _spy_projection(monkeypatch):
    """Record every row slice the engine projects, delegating to the real helper."""
    seen = []
    real = rolling_engine.readable_delta

    def spy(rows, **kwargs):
        seen.append([r["message_uid"] for r in rows])
        return real(rows, **kwargs)

    monkeypatch.setattr(rolling_engine, "readable_delta", spy)
    return seen


def test_oversized_examined_exchange_refuses_without_skipping(tmp_path, monkeypatch):
    """An examined exchange beyond the readable byte bound refuses; later ones are never read."""
    settings = RollingSettings(trigger_bytes=4096)
    ex_a, ex_big, ex_c = exchange("A"), exchange("BIG"), exchange("C")
    ex_big[0]["content"] = "x" * (settings.max_readable_delta_bytes + 1)
    with pytest.raises(ValueError):
        readable_delta(ex_big, policy=POLICY, settings=settings)
    active = ex_a + ex_big + ex_c + exchange("newest")
    stops = [len(ex_a), len(ex_a) + len(ex_big), len(ex_a) + len(ex_big) + len(ex_c)]
    engine, _, _ = setup_engine(tmp_path, monkeypatch, settings=settings)
    scope = engine._capture()
    seen = _spy_projection(monkeypatch)
    with pytest.raises(ValueError):
        engine._count_trigger_bytes(scope, active, 0, stops)
    # Exactly one exchange per projection, in order, stopping at the oversized one.
    assert seen == [[r["message_uid"] for r in ex_a], [r["message_uid"] for r in ex_big]]


def test_eligible_but_unfittable_earliest_job_does_no_work(tmp_path, monkeypatch):
    """Threshold reached, but the earliest job exceeds max_request_bytes: no target, no charge."""
    settings = RollingSettings(trigger_bytes=1, max_request_bytes=512)
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)
    db.rows = exchange("A") + exchange("B") + exchange("C")
    scope = engine._capture()
    starts = rolling_engine.completed_exchange_starts(db.rows)
    assert engine._count_trigger_bytes(scope, db.rows, 0, starts[1:]) >= settings.trigger_bytes
    assert engine._new_target(scope, db.rows, None) is None
    db.rows = exchange("A")
    complete(engine, db, "A")
    db.rows += exchange("B")
    complete(engine, db, "B")
    db.rows += exchange("C")
    complete(engine, db, "C")
    assert not facade.calls
    assert engine._published is None
    report, pending, _ = journal_snapshot(tmp_path, settings, engine._store)
    assert report["logical_reservations"] == 0
    assert pending is None


def test_advance_threshold_excludes_published_prefix_and_parent_summary(tmp_path, monkeypatch):
    """Discriminating threshold: only exchanges after the published frontier count.

    A is large enough to trigger alone; B alone is one byte short. Counting A again
    (old prefix) or the parent summary would wrongly fire the advance after C.
    """
    ex_a, ex_b, ex_c, ex_d = exchange("A"), exchange("B"), exchange("C"), exchange("D")
    ex_a[0]["content"] = "long question " + "a" * 400
    len_b = len(readable_delta(ex_b, policy=POLICY))
    settings = RollingSettings(trigger_bytes=len_b + 1)
    assert len(readable_delta(ex_a, policy=POLICY)) >= settings.trigger_bytes
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)

    db.rows = list(ex_a)
    complete(engine, db, "A")
    assert not facade.calls
    db.rows += ex_b
    complete(engine, db, "B")
    first = engine._published
    assert len(facade.calls) == 1 and first.frontier_count == len(ex_a)

    db.rows += exchange("C", complete=False)[:1]
    assert select(engine, db, "C") is not None and engine._pinned == first
    db.rows = ex_a + ex_b + ex_c
    complete(engine, db, "C")
    # Only B is newly aged (C is newest) and B < trigger: no advance.
    assert len(facade.calls) == 1 and engine._published == first

    db.rows += exchange("D", complete=False)[:1]
    assert select(engine, db, "D") is not None and engine._pinned == first
    db.rows = ex_a + ex_b + ex_c + ex_d
    complete(engine, db, "D")
    second = engine._published
    assert len(facade.calls) == 2
    assert second.sequence == 2 and second.parent_generation == first.generation_id
    assert second.delta_start == first.frontier_count
    assert second.frontier_count == len(ex_a) + len(ex_b) + len(ex_c)
    request = json.loads(facade.calls[1]["messages"][-1]["content"])
    assert request["previous_summary"] == "summary 1"
    uids = json.dumps(request["delta"])
    assert "question A" not in uids and "long question" not in uids
    assert " question B " in uids or "question B" in uids
    assert "question D" not in uids

    # Next turn adopts G2; select() checks host prefix and exact original tail.
    db.rows += exchange("E", complete=False)[:1]
    assert select(engine, db, "E") is not None
    assert engine._pinned == second
