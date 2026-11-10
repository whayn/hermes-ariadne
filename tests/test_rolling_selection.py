"""Pure native-shaped correspondence/selection fixtures; no Hermes imports."""

import copy
import json

import pytest

from hermes_ariadne import rolling
from hermes_ariadne.kernel import MemoryKernel
from hermes_ariadne.prepared import PreparedView, ReasoningReplayPolicy
from hermes_ariadne.sources import Projection
from hermes_ariadne.store import EventStore, canonical_json

POLICY = ReasoningReplayPolicy("codex_backend", "gpt-test")


def exchange(tag, *, complete=True):
    def row(role, content, suffix, **extra):
        return {
            "session_id": "s",
            "message_uid": tag + suffix,
            "role": role,
            "content": content,
            "tool_calls": None,
            "tool_call_id": None,
            "tool_name": None,
            "effect_disposition": None,
            "reasoning": None,
            "reasoning_content": None,
            "reasoning_details": None,
            "codex_reasoning_items": None,
            "codex_message_items": None,
            "tool_call_uids": None,
            "tool_call_uid": None,
            "absorbed_message_uids": "[]",
            "display_metadata": None,
            "active": 1,
            **extra,
        }

    def reasoning(suffix, count):
        return json.dumps(
            [
                {
                    "type": "reasoning",
                    "id": "rs_" + tag + suffix + str(i),
                    "encrypted_content": "OPAQUE_" + tag + suffix + str(i),
                    "summary": []
                    if i % 2 == 0
                    else [{"type": "summary_text", "text": "companion"}],
                    "_issuer_kind": "codex_backend",
                    "_issuer_model": "gpt-test",
                }
                for i in range(count)
            ]
        )

    result = [
        row("user", " question " + tag + " ", "u"),
        row(
            "assistant",
            None,
            "o",
            tool_calls=[
                {
                    "id": "call_" + tag,
                    "call_id": "call_" + tag,
                    "response_item_id": "fc_native_" + tag,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"exact"}'},
                }
            ],
            tool_call_uids=json.dumps({"call_" + tag: "occ_" + tag}),
            codex_reasoning_items=reasoning("owner", 1),
            reasoning="",
        ),
        row(
            "tool",
            "result " + tag,
            "t",
            tool_call_id="call_" + tag,
            tool_name="lookup",
            effect_disposition="none",
            tool_call_uid="occ_" + tag,
        ),
    ]
    if complete:
        text = " answer " + tag + " "
        result.append(
            row(
                "assistant",
                text,
                "a",
                reasoning="readable companion",
                reasoning_content="",
                codex_reasoning_items=reasoning("answer", 4),
                codex_message_items=json.dumps(
                    [
                        {
                            "type": "message",
                            "id": "msg_" + tag,
                            "role": "assistant",
                            "status": "completed",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    ]
                ),
            )
        )
    return result


def restored(source):
    result = []
    for row in source:
        item = {k: copy.deepcopy(row[k]) for k in ("role", "content", "message_uid")}
        item["_db_persisted"] = True
        if item["role"] in {"user", "assistant"} and isinstance(item["content"], str):
            item["content"] = item["content"].strip()
        for key in ("codex_reasoning_items", "codex_message_items", "tool_call_uids"):
            if row[key] is not None:
                item[key] = json.loads(row[key])
        for key in (
            "tool_calls",
            "tool_call_id",
            "tool_call_uid",
            "tool_name",
            "effect_disposition",
            "reasoning",
            "reasoning_content",
        ):
            if row[key] is not None and not (key == "reasoning" and row[key] == ""):
                item[key] = copy.deepcopy(row[key])
        result.append(item)
    return result


def assembled(history):
    return [
        {"role": "system", "content": "host"},
        {"role": "developer", "content": "host rules"},
    ] + [
        {
            k: copy.deepcopy(v)
            for k, v in row.items()
            if k
            in {
                "role",
                "content",
                "tool_calls",
                "tool_call_id",
                "tool_name",
                "effect_disposition",
                "codex_reasoning_items",
                "codex_message_items",
                "_db_persisted",
            }
        }
        for row in history
    ]


def fixture(tmp_path, *, settings=None):
    settings = settings or rolling.RollingSettings()
    source = exchange("old") + exchange("prev") + exchange("now", complete=False)
    history = restored(source)
    request = assembled(history)
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    store.ingest("s", source, origin="canonical")
    target = rolling.Target.create(
        source[:4],
        session_id="s",
        settings=settings,
        policy=POLICY,
        parent=None,
        request_bytes=b'[{"content":"bounded input","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    view = rolling.validate_generation(
        rolling.generation_bytes(target, "historical summary", sequence=1, settings=settings),
        store,
        "s",
        settings,
        policy=POLICY,
    )
    return store, settings, view, source, history, request


def choose(view, store, settings, source, history, request, **kwargs):
    from hermes_ariadne.rolling_selection import select_rolling

    return select_rolling(
        view,
        store=store,
        session_id="s",
        settings=settings,
        request_messages=request,
        conversation_messages=history,
        incoming_message=kwargs.pop("incoming_message", history[8]),
        active_messages=source,
        budget_tokens=kwargs.pop("budget_tokens", 10000),
        measure_tokens=kwargs.pop("measure_tokens", lambda _: 100),
        reasoning_policy=kwargs.pop("reasoning_policy", POLICY),
        retirement_policy=kwargs.pop("retirement_policy", rolling.POLICY_ID),
        **kwargs,
    )


def test_aged_native_four_item_envelope_old_declines_explicit_rolling_keeps_exact_suffix(tmp_path):
    store, settings, view, source, history, request = fixture(tmp_path)
    prefix = Projection(store, "s", view.event_ids)
    kernel = MemoryKernel(prefix, tmp_path / "old-kernel.json", node_bytes=128)
    kernel.build(lambda _: "old navigation")
    old = PreparedView(kernel, "f" * 64, 8192, 65536, 4096, "old navigation")
    before = copy.deepcopy((source, history, request))
    old_decision = old.select(
        request,
        conversation_messages=history,
        incoming_message=history[8],
        active_messages=source,
        budget_tokens=10000,
        measure_tokens=lambda _: 100,
        reasoning_policy=POLICY,
    )
    assert old_decision.status == "declined" and old_decision.messages is None
    decision = choose(view, store, settings, source, history, request)
    assert decision.status == "selected"
    assert decision.messages[:2] == request[:2]
    assert decision.messages[3:] == request[6:]
    navigation = decision.messages[2]["content"]
    for text in (
        "untrusted",
        "historical",
        "retired",
        "rolling:" + view.generation_id,
        "4",
        "ariadne_rolling_page",
        "ariadne_find",
        "ariadne_zoom",
    ):
        assert text in navigation
    assert "OPAQUE" not in navigation and "event:" not in navigation
    assert (
        decision.messages[3:][-2]["codex_reasoning_items"] == request[-2]["codex_reasoning_items"]
    )
    decision.messages[-2]["tool_calls"][0]["function"]["arguments"] = "isolated"
    assert (source, history, request) == before
    assert [store.get_event(event, session_id="s") for event in view.event_ids] == source[:4]
    assert all(canonical_json(row) for row in source)


def test_completed_turn_covers_full_real_observation_without_fabricated_assembled(
    tmp_path, monkeypatch
):
    from hermes_ariadne import prepared

    source = exchange("old") + exchange("prev")
    observed = restored(source)
    before = copy.deepcopy((source, observed))
    comparison = prepared._comparison

    def no_fabricated_wire(message, **kwargs):
        assert kwargs.get("wire") is not True, "completion is raw observation, not assembled"
        return comparison(message, **kwargs)

    monkeypatch.setattr(prepared, "_comparison", no_fabricated_wire)
    coverage = rolling.validate_completed_turn(source, observed, reasoning_policy=POLICY)
    assert coverage.event_ids == tuple(rolling.sha256(canonical_json(row)) for row in source)
    assert coverage.message_uids == tuple(row["message_uid"] for row in source)
    assert coverage.final_assistant_uid == source[-1]["message_uid"]
    assert coverage.completed_exchanges == 2
    assert (source, observed) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_active_tail",
        "missing_observed_tail",
        "old_prefix_only",
        "missing_user",
        "missing_tool",
        "tool_owner_tail",
        "tool_result_tail",
        "late_payload",
        "late_uid",
        "reordered",
        "late_reasoning_item",
        "late_occurrence",
        "missing_companion",
        "null_companion",
        "restored_text_carrier",
        "missing_response",
        "stamp_only",
    ],
)
def test_completed_turn_refuses_incomplete_or_changed_full_coverage(mutation):
    source = exchange("old") + exchange("prev")
    observed = restored(source)
    if mutation == "missing_active_tail":
        source.pop()
    elif mutation == "missing_observed_tail":
        observed.pop()
    elif mutation == "old_prefix_only":
        source = source[:4]
    elif mutation == "missing_user":
        source.pop(4)
    elif mutation == "missing_tool":
        source.pop(6)
        observed.pop(6)
    elif mutation == "tool_owner_tail":
        source = source[:6]
        observed = observed[:6]
    elif mutation == "tool_result_tail":
        source.pop()
        observed.pop()
    elif mutation == "late_payload":
        observed[-2]["content"] = "changed"
    elif mutation == "late_uid":
        observed[-1]["message_uid"] = "changed"
    elif mutation == "reordered":
        observed[4], observed[5] = observed[5], observed[4]
    elif mutation == "late_reasoning_item":
        observed[-1]["codex_reasoning_items"][-1]["encrypted_content"] = "CHANGED"
    elif mutation == "late_occurrence":
        observed[-2]["tool_call_uid"] = "other"
    elif mutation == "missing_companion":
        observed[-1].pop("reasoning_content")
    elif mutation == "null_companion":
        observed[5]["reasoning_content"] = "not SQL NULL"
    elif mutation == "restored_text_carrier":
        observed[-1]["codex_reasoning_items"] = source[-1]["codex_reasoning_items"]
    elif mutation == "missing_response":
        source.pop()
    elif mutation == "stamp_only":
        source = source[:4]
        for row in observed:
            row["_db_persisted"] = True
    with pytest.raises((ValueError, KeyError)):
        rolling.validate_completed_turn(source, observed, reasoning_policy=POLICY)


def test_completed_turn_cannot_skip_an_earlier_canonical_response_in_both_views():
    source = exchange("old") + exchange("prev")
    source.pop(3)  # A tool result alone is not a completed exchange.
    observed = restored(source)
    with pytest.raises(ValueError):
        rolling.validate_completed_turn(source, observed, reasoning_policy=POLICY)


def test_selection_requires_a_completed_previous_exchange(tmp_path):
    store, settings, view, source, history, request = fixture(tmp_path)
    source.pop(7)
    history.pop(7)
    request.pop(9)
    decision = choose(view, store, settings, source, history, request, incoming_message=history[7])
    assert decision.status == "declined" and decision.messages is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_request",
        "reordered_request",
        "late_item",
        "late_item_order",
        "native_item",
        "source_changed",
        "uid_changed",
        "occurrence",
        "companion",
        "canonical_null_missing",
        "restricted",
        "withheld",
        "unknown",
        "incomplete_tool",
        "wrong_anchor_payload",
        "wrong_anchor_uid",
        "not_latest_user",
        "foreign_session",
        "rewound",
    ],
)
def test_selection_refuses_full_history_source_and_replay_mismatches_before_measurement(
    tmp_path, mutation
):
    store, settings, view, source, history, request = fixture(tmp_path)
    incoming = copy.deepcopy(history[8])
    if mutation == "missing_request":
        request.pop(4)
    elif mutation == "reordered_request":
        request[3], request[4] = request[4], request[3]
    elif mutation == "late_item":
        request[9]["codex_reasoning_items"][-1]["encrypted_content"] = "CHANGED"
    elif mutation == "late_item_order":
        request[9]["codex_reasoning_items"].reverse()
    elif mutation == "native_item":
        request[-2]["tool_calls"][0]["response_item_id"] = "fc_different"
    elif mutation == "source_changed":
        source[0]["content"] = "changed prefix"
    elif mutation == "uid_changed":
        source[-1]["message_uid"] = "changed"
    elif mutation == "occurrence":
        history[-1]["tool_call_uid"] = "different"
    elif mutation == "companion":
        history[3]["reasoning"] = "changed companion"
    elif mutation == "canonical_null_missing":
        source[5].pop("reasoning_content")
    elif mutation == "restricted":
        history[7]["data_use"] = ["no_training"]
    elif mutation == "withheld":
        source[-1]["display_metadata"] = '{"withheld_prompt":""}'
    elif mutation == "unknown":
        request[-1]["unknown_carrier"] = "opaque"
    elif mutation == "incomplete_tool":
        source.pop()
        history.pop()
        request.pop()
    elif mutation == "wrong_anchor_payload":
        incoming["content"] = "different"
    elif mutation == "wrong_anchor_uid":
        incoming["message_uid"] = "different"
    elif mutation == "not_latest_user":
        incoming = history[4]
    elif mutation == "foreign_session":
        source[-1]["session_id"] = "foreign"
    elif mutation == "rewound":
        source = source[:3]
    before = copy.deepcopy((source, history, request, incoming))
    measurements = []
    decision = choose(
        view,
        store,
        settings,
        source,
        history,
        request,
        incoming_message=incoming,
        measure_tokens=lambda value: measurements.append(value) or 100,
    )
    assert decision.status == "declined" and decision.messages is None
    assert measurements == []
    assert (source, history, request, incoming) == before


@pytest.mark.parametrize(
    "retirement_policy,reasoning_policy",
    [
        (None, POLICY),
        ("future-policy", POLICY),
        (rolling.POLICY_ID, None),
        (rolling.POLICY_ID, ReasoningReplayPolicy("codex_backend", "wrong-model")),
    ],
)
def test_aged_state_needs_both_explicit_versioned_permission_and_bound_route(
    tmp_path, retirement_policy, reasoning_policy
):
    store, settings, view, source, history, request = fixture(tmp_path)
    decision = choose(
        view,
        store,
        settings,
        source,
        history,
        request,
        retirement_policy=retirement_policy,
        reasoning_policy=reasoning_policy,
    )
    assert decision.status == "declined" and decision.messages is None


@pytest.mark.parametrize(
    "budget,measurement,reason",
    [
        (0, 100, "unknown_budget"),
        (None, 100, "unknown_budget"),
        (True, 100, "unknown_budget"),
        (4096, 1, "over_budget"),
        (4196, 101, "over_budget"),
        (10000, None, "measurement_error"),
        (10000, True, "measurement_error"),
        (10000, -1, "measurement_error"),
    ],
)
def test_declines_are_explicit_original_request_fallback_not_dispatch_veto(
    tmp_path, budget, measurement, reason
):
    store, settings, view, source, history, request = fixture(tmp_path)
    decision = choose(
        view,
        store,
        settings,
        source,
        history,
        request,
        budget_tokens=budget,
        measure_tokens=lambda _: measurement,
    )
    assert decision.status == "declined" and decision.messages is None
    assert decision.diagnostics["reason"] == reason
    assert decision.diagnostics["provider_fit_verified"] is False
    assert decision.diagnostics["host_fallback"] == "original_request_on_decline"


def test_measurement_gets_whole_isolated_candidate_and_exact_reserve_boundary(tmp_path):
    store, settings, view, source, history, request = fixture(tmp_path)
    seen = []

    def measure(candidate):
        seen.append(copy.deepcopy(candidate))
        candidate[-2]["tool_calls"].clear()
        return 100

    decision = choose(
        view, store, settings, source, history, request, budget_tokens=4196, measure_tokens=measure
    )
    assert decision.status == "selected"
    assert seen == [decision.messages]
    assert decision.messages[-2]["tool_calls"] == request[-2]["tool_calls"]
    assert decision.diagnostics["available_tokens"] == 100
    assert decision.diagnostics["reserve_tokens"] == 4096

    def broken(_):
        raise RuntimeError("measurement failed")

    assert (
        choose(view, store, settings, source, history, request, measure_tokens=broken).diagnostics[
            "reason"
        ]
        == "measurement_error"
    )


@pytest.mark.parametrize("bound", ["history", "candidate"])
def test_navigation_and_candidate_byte_bounds_precede_measurement(tmp_path, bound):
    settings = rolling.RollingSettings(max_navigation_bytes=32) if bound == "history" else None
    store, settings, view, source, history, request = fixture(tmp_path, settings=settings)
    if bound == "candidate":
        for row in (source[-1], history[-1], request[-1]):
            row["content"] = "x" * 65536
    seen = []
    decision = choose(
        view,
        store,
        settings,
        source,
        history,
        request,
        measure_tokens=lambda candidate: seen.append(candidate) or 100,
    )
    assert decision.messages is None and seen == []
    assert decision.diagnostics["reason"] == (
        "history_bytes_exceeded" if bound == "history" else "request_bytes_exceeded"
    )


def test_anchor_key_survives_tool_round_and_no_view_is_an_explicit_decline(tmp_path):
    from hermes_ariadne.rolling_selection import incoming_anchor

    store, settings, view, source, history, request = fixture(tmp_path)
    before_tools = incoming_anchor(history[:9], history[8])
    after_tools = incoming_anchor(history, history[8])
    assert before_tools == after_tools
    before_tool_decision = choose(view, store, settings, source[:9], history[:9], request[:11])
    after_tool_decision = choose(view, store, settings, source, history, request)
    assert before_tool_decision.status == after_tool_decision.status == "selected"
    assert before_tool_decision.messages[2] == after_tool_decision.messages[2]
    assert before_tool_decision.messages[3:] == request[6:11]
    assert after_tool_decision.messages[3:] == request[6:]
    decision = choose(None, store, settings, source, history, request)
    assert decision.messages is None and decision.status == "declined"
    assert choose(view, store, settings, source, history, request).status == "selected"


def test_reasoning_free_text_and_tools_use_same_pure_coverage_and_selection(tmp_path):
    source = exchange("old") + exchange("prev") + exchange("now", complete=False)
    for row in source:
        for key in (
            "reasoning",
            "reasoning_content",
            "codex_reasoning_items",
            "codex_message_items",
        ):
            row[key] = None
    history = restored(source)
    request = assembled(history)
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    store.ingest("s", source, origin="canonical")
    settings = rolling.RollingSettings()
    target = rolling.Target.create(
        source[:4],
        session_id="s",
        settings=settings,
        policy=None,
        parent=None,
        request_bytes=b'[{"content":"input","role":"user"}]',
        route_labels=("ordinary", "test"),
    )
    view = rolling.validate_generation(
        rolling.generation_bytes(target, "ordinary summary", sequence=1), store, "s", settings
    )
    decision = choose(view, store, settings, source, history, request, reasoning_policy=None)
    assert decision.status == "selected" and decision.messages[3:] == request[6:]
    coverage = rolling.validate_completed_turn(source[:8], history[:8], reasoning_policy=None)
    assert coverage.completed_exchanges == 2


@pytest.mark.parametrize(
    "mutation", ["settings", "session", "forged", "missing_source", "changed_source"]
)
def test_selection_revalidates_immutable_view_and_all_archived_dependencies(
    tmp_path, monkeypatch, mutation
):
    from dataclasses import replace

    store, settings, view, source, history, request = fixture(tmp_path)
    if mutation == "settings":
        settings = replace(settings, max_summary_bytes=2048)
    elif mutation == "session":
        view = replace(view, session_id="foreign")
    elif mutation == "forged":
        view = replace(view, frontier_count=8)
    else:
        original = store.get_event

        def changed(event, *, session_id):
            row = original(event, session_id=session_id)
            if event == view.event_ids[-1]:
                if mutation == "missing_source":
                    return None
                row["content"] = "changed source"
            return row

        monkeypatch.setattr(store, "get_event", changed)
    decision = choose(view, store, settings, source, history, request)
    assert decision.status == "declined" and decision.messages is None


def test_successor_selection_requires_exact_parent_artifact_and_complete_lineage(tmp_path):
    store, settings, first, _, _, _ = fixture(tmp_path)
    source = (
        exchange("old") + exchange("prev") + exchange("third") + exchange("now", complete=False)
    )
    history = restored(source)
    request = assembled(history)
    store.ingest("s", source, origin="canonical")
    target = rolling.Target.create(
        source[:8],
        session_id="s",
        settings=settings,
        policy=POLICY,
        parent=first,
        request_bytes=b'[{"content":"next input","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    second = rolling.validate_generation(
        rolling.generation_bytes(target, "successor summary", sequence=2),
        store,
        "s",
        settings,
        policy=POLICY,
        parent_loader=lambda _: first.snapshot_bytes,
    )
    selected = choose(
        second,
        store,
        settings,
        source,
        history,
        request,
        incoming_message=history[12],
        parent_loader=lambda _: first.snapshot_bytes,
    )
    assert selected.status == "selected" and selected.messages[3:] == request[10:]
    for loader in (None, lambda _: second.snapshot_bytes, lambda _: first.snapshot_bytes + b" "):
        decision = choose(
            second,
            store,
            settings,
            source,
            history,
            request,
            incoming_message=history[12],
            parent_loader=loader,
        )
        assert decision.status == "declined" and decision.messages is None
