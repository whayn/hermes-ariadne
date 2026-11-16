"""Rolling-only admission of non-Codex reasoning carriers (#29, design points A-E).

Carriers are opaque: excluded from the canonical/restored/assembled comparison,
admitted structurally (assistant rows only, JSON-shaped, bounded, present on the
assembled side only where the canonical row has one or as the must-echo pad),
never summarized, logged or returned by a tool. Retired exchanges drop theirs.
Passive/prepared validation keeps refusing them. Shapes are synthetic fixtures
modelled on the host's persisted/assembled forms, not provider evidence.
"""

import copy
import json

import pytest
from test_rolling_engine import DB, Facade, engine_class, engine_records

from hermes_ariadne import prepared, rolling

SENT = "CARRIER-SENTINEL"
SIGNED = [{"type": "thinking", "thinking": SENT + " thought", "signature": "c2lnbmF0dXJl"}]


def base_row(tag, suffix, role, content, **extra):
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


# Per route: (canonical carriers on the tool-call turn, on the final answer,
# extra tool-call keys, assembled shaping). Canonical reasoning_details is JSON
# TEXT (SessionDB leaves it encoded); the restored view decodes it.
ROUTES = {
    "anthropic": dict(
        call={"reasoning": SENT + " r", "reasoning_details": json.dumps(SIGNED)},
        answer={"reasoning": SENT + " r", "reasoning_details": json.dumps(SIGNED)},
        assembled=("reasoning_details",),
        live_blocks=True,
    ),
    "reasoning_content": dict(
        call={"reasoning_content": SENT + " rc"},
        answer={"reasoning_content": SENT + " rc"},
        assembled=("reasoning_content",),
    ),
    "reasoning": dict(
        call={"reasoning": SENT + " r"},
        answer={"reasoning": SENT + " r"},
        assembled=("reasoning",),
    ),
    "openrouter_text": dict(
        call={
            "reasoning": SENT,
            "reasoning_details": json.dumps([{"type": "reasoning.text", "text": SENT}]),
        },
        answer={"reasoning": SENT},
        assembled=("reasoning", "reasoning_details"),
    ),
    "openrouter_encrypted": dict(
        call={"reasoning_details": json.dumps([{"type": "reasoning.encrypted", "data": SENT}])},
        answer={},
        assembled=("reasoning_details",),
    ),
    "gemini": dict(
        call={
            "reasoning_details": json.dumps(
                [{"type": "hermes.native_assistant", "thought_signature": SENT}]
            )
        },
        answer={},
        assembled=("reasoning_details",),
        call_extra={"extra_content": {"google": {"thought_signature": SENT}}},
    ),
    "must_echo": dict(
        call={"reasoning_content": " "},
        answer={},
        assembled=(),
        pad_all=True,
    ),
}


def exchange(tag, route, *, complete=True):
    spec = ROUTES[route]
    call = {
        "id": "toolu_" + tag,
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"key":"exact"}'},
        **copy.deepcopy(spec.get("call_extra", {})),
    }
    rows = [
        base_row(tag, "u", "user", " question " + tag + " "),
        base_row(
            tag,
            "o",
            "assistant",
            None,
            tool_calls=[call],
            tool_call_uids=json.dumps({"toolu_" + tag: "occ_" + tag}),
            **copy.deepcopy(spec["call"]),
        ),
        base_row(
            tag,
            "t",
            "tool",
            "result " + tag,
            tool_call_id="toolu_" + tag,
            tool_name="lookup",
            effect_disposition="none",
            tool_call_uid="occ_" + tag,
        ),
    ]
    if complete:
        rows.append(
            base_row(tag, "a", "assistant", " answer " + tag + " ", **copy.deepcopy(spec["answer"]))
        )
    return rows


def restored(rows, route):
    """The host's conversation view: decoded JSON columns, live-only Anthropic blocks."""
    result = []
    for row in rows:
        item = {k: copy.deepcopy(row[k]) for k in ("role", "content", "message_uid")}
        item["_db_persisted"] = True
        if item["role"] in {"user", "assistant"} and isinstance(item["content"], str):
            item["content"] = item["content"].strip()
        for key in ("tool_call_uids", "reasoning_details"):
            if row[key] is not None:
                item[key] = json.loads(row[key]) if type(row[key]) is str else row[key]
        for key in (
            "tool_calls",
            "tool_call_id",
            "tool_call_uid",
            "tool_name",
            "effect_disposition",
            "reasoning",
            "reasoning_content",
        ):
            if row[key] is not None:
                item[key] = copy.deepcopy(row[key])
        if ROUTES[route].get("live_blocks") and row.get("tool_calls"):
            item["anthropic_content_blocks"] = [
                *copy.deepcopy(SIGNED),
                {"type": "tool_use", "id": row["tool_calls"][0]["id"], "name": "lookup"},
            ]
        result.append(item)
    return result


def assembled(history, route):
    spec = ROUTES[route]
    request = [
        {"role": "system", "content": "host"},
        {"role": "developer", "content": "host rules"},
    ]
    for row in history:
        item = {
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
                "_db_persisted",
                *spec["assembled"],
                "anthropic_content_blocks",
            }
        }
        if spec.get("pad_all") and item["role"] == "assistant":
            item["reasoning_content"] = row.get("reasoning_content") or " "
        request.append(item)
    return request


def setup(tmp_path, monkeypatch, *, db=None, facade=None):
    cls = engine_class()
    monkeypatch.setattr(cls, "_bound_reasoning_policy", staticmethod(lambda *a: None))
    facade, db = facade or Facade(), db or DB()
    engine = cls(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("claude-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    return engine, db, facade


def complete_turn(engine, db, tag, route):
    engine.on_turn_complete(
        restored(db.rows, route),
        usage={},
        turn_id=tag,
        interrupted=False,
        failed=False,
        turn_exit_reason="text_response(finish_reason=stop)",
    )


def request_for(db, tag, route):
    history = restored(db.rows, route)
    incoming = next(m for m in history if m["message_uid"] == tag + "u")
    return assembled(history, route), history, incoming


def choose(engine, db, tag, route, request=None):
    built, history, incoming = request_for(db, tag, route)
    request = built if request is None else request
    before = copy.deepcopy(request)
    result = engine.select_context(
        request, conversation_messages=history, incoming_message=incoming, budget_tokens=100000
    )
    assert request == before, "selection must not mutate the host request"
    return request, result


def assert_retires_prefix_keeps_tail(engine, request, result):
    count = engine._pinned.frontier_count
    assert result[:2] == request[:2]
    assert result[2]["role"] == "assistant" and set(result[2]) == {"role", "content"}
    # B: the retained tail is the host's own shaping, carriers included, unchanged.
    assert result[3:] == request[2 + count :]
    assert SENT not in json.dumps(result[2])


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_route_shaped_carriers_select_from_the_first_eligible_turn(
    tmp_path, monkeypatch, caplog, route
):
    caplog.set_level("DEBUG", logger="hermes_ariadne.rolling_engine")
    engine, db, facade = setup(tmp_path, monkeypatch)
    db.rows = exchange("A", route)
    complete_turn(engine, db, "A", route)
    db.rows += exchange("B", route)
    complete_turn(engine, db, "B", route)
    assert engine._published is not None and len(facade.calls) == 1
    # Turn C: first request, then two requests inside its tool loop -- S, S, S.
    db.rows += exchange("C", route, complete=False)[:1]
    request, result = choose(engine, db, "C", route)
    assert result is not None, engine.get_status()
    assert_retires_prefix_keeps_tail(engine, request, result)
    db.rows = db.rows[:-1] + exchange("C", route, complete=False)
    for _ in range(2):
        request, result = choose(engine, db, "C", route)
        assert result is not None, engine.get_status()
        assert_retires_prefix_keeps_tail(engine, request, result)
        if ROUTES[route]["assembled"] or ROUTES[route].get("pad_all"):
            tail_keys = set().union(*(m.keys() for m in result[3:]))
            assert tail_keys & prepared._REASONING_CARRIERS, "kept tail lost its carriers"
    db.rows = db.rows[:-3] + exchange("C", route)
    complete_turn(engine, db, "C", route)
    assert len(facade.calls) == 2 and engine._published.frontier_count == 8
    db.rows += exchange("D", route, complete=False)[:1]
    request, result = choose(engine, db, "D", route)
    assert result is not None
    assert engine._pinned.frontier_count == 8
    assert_retires_prefix_keeps_tail(engine, request, result)
    status = engine.get_status()
    assert status["rolling"]["status"] != "unsupported_carrier", status
    assert "carrier" not in status["rolling"] and "carrier" not in status["selection"]
    # C: no carrier text reaches ctx.llm payloads or logs.
    assert SENT not in json.dumps(facade.calls)
    assert "signature" not in json.dumps(facade.calls)
    engine_records(caplog)
    assert SENT not in caplog.text
    # D: retrieval never shows them either.
    page = json.loads(
        engine.handle_tool_call(
            "ariadne_rolling_page", {"ref": "rolling:" + engine._pinned.generation_id}
        )
    )
    assert page["ok"]
    for item in page["results"]:
        zoom = engine.handle_tool_call("ariadne_zoom", {"ref": item["event_ref"]})
        assert json.loads(zoom)["ok"] and SENT not in zoom
    # Restart in a new engine: the view, charges and selection survive.
    charges, published = status["rolling"]["logical_reservations"], engine._published
    engine.on_session_end("s", [])
    restarted, _, _ = setup(tmp_path, monkeypatch, db=db, facade=facade)
    assert restarted._published == published
    assert restarted.get_status()["rolling"]["logical_reservations"] == charges
    request, result = choose(restarted, db, "D", route)
    assert result is not None
    assert len(facade.calls) == 2


def test_summary_input_is_reasoning_free_for_every_carrier():
    rows = []
    for index, route in enumerate(sorted(ROUTES)):
        rows += exchange("T" + str(index), route)
    delta = rolling.readable_delta(
        rows, policy=None, settings=rolling.RollingSettings(max_delta_exchanges=len(ROUTES))
    )
    assert SENT.encode() not in delta and b"signature" not in delta
    decoded = json.loads(delta)
    assert {key for item in decoded for key in item} <= {
        "role",
        "content",
        "event_ref",
        "tool_name",
        "tool_calls",
        "name",
    }


def _engine_after_publication(tmp_path, monkeypatch, route="anthropic"):
    engine, db, facade = setup(tmp_path, monkeypatch)
    db.rows = exchange("A", route) + exchange("B", route)
    complete_turn(engine, db, "B", route)
    assert engine._published is not None
    return engine, db, facade


@pytest.mark.parametrize(
    "row, key, value",
    [
        (0, "reasoning", SENT),  # user row
        (2, "reasoning_details", json.dumps(SIGNED)),  # tool row
        (3, "reasoning", 7),  # non-JSON-text value
        (3, "reasoning_details", True),
        (3, "bedrock_content_blocks", [{"reasoningContent": {"text": SENT}}]),  # not admitted
    ],
)
def test_canonical_carrier_failing_admission_is_reported_by_name(
    tmp_path, monkeypatch, caplog, row, key, value
):
    caplog.set_level("DEBUG", logger="hermes_ariadne.rolling_engine")
    engine, db, facade = _engine_after_publication(tmp_path, monkeypatch)
    rows = exchange("C", "anthropic")
    rows[row][key] = value
    db.rows += rows
    complete_turn(engine, db, "C", "anthropic")
    status = engine.get_status()["rolling"]
    assert status["status"] == "unsupported_carrier" and status["carrier"] == key, status
    assert len(facade.calls) == 1
    assert SENT not in caplog.text


def test_oversized_carrier_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(prepared, "_MAX_CARRIER_MESSAGE_BYTES", 256)
    engine, db, facade = _engine_after_publication(tmp_path, monkeypatch)
    rows = exchange("C", "anthropic")
    rows[3]["reasoning"] = "x" * 300
    db.rows += rows
    complete_turn(engine, db, "C", "anthropic")
    status = engine.get_status()["rolling"]
    assert status["status"] == "unsupported_carrier" and status["carrier"] == "reasoning"


def test_per_message_bound_counts_every_carrier_of_the_message():
    message = {"role": "assistant", "content": "x", "reasoning": "a" * 600}
    message["reasoning_details"] = [{"type": "thinking", "thinking": "b" * 600}]
    prepared._admit_opaque_carriers(message)
    original = prepared._MAX_CARRIER_MESSAGE_BYTES
    try:
        prepared._MAX_CARRIER_MESSAGE_BYTES = 1000
        with pytest.raises(ValueError) as caught:
            prepared._admit_opaque_carriers(message)
    finally:
        prepared._MAX_CARRIER_MESSAGE_BYTES = original
    assert caught.value.carrier == "reasoning_details"


@pytest.mark.parametrize(
    "target, key, value",
    [
        # An assembled carrier the canonical row does not have (not the pad).
        (-2, "reasoning_details", SIGNED),
        (-2, "reasoning_content", "injected"),
        (-2, "reasoning", "  "),  # whitespace other than the exact pad
        # On a user message (the incoming one).
        (-1, "reasoning", "x"),
        # A carrier key the rolling admission does not cover.
        (-2, "bedrock_content_blocks", [{"x": 1}]),
    ],
)
def test_assembled_carrier_without_canonical_origin_declines_by_name(
    tmp_path, monkeypatch, target, key, value
):
    route = "reasoning"
    engine, db, _ = setup(tmp_path, monkeypatch)
    rows = exchange("B", route)
    rows[3]["reasoning"] = None  # B's final answer has no canonical carrier.
    db.rows = exchange("A", route) + rows
    complete_turn(engine, db, "B", route)
    assert engine._published is not None
    db.rows += exchange("C", route, complete=False)[:1]
    request, _, _ = request_for(db, "C", route)
    assert request[-2]["content"] == "answer B" and "reasoning" not in request[-2]
    assert request[-1]["role"] == "user"
    _, control = choose(engine, db, "C", route, request=copy.deepcopy(request))
    assert control is not None, engine.get_status()
    request[target][key] = value
    _, result = choose(engine, db, "C", route, request=request)
    assert result is None
    selection = engine.get_status()["selection"]
    assert selection["reason"] == "unsupported_carrier" and selection["carrier"] == key


def test_must_echo_pad_without_canonical_reasoning_is_admitted(tmp_path, monkeypatch):
    route = "reasoning"
    engine, db, _ = setup(tmp_path, monkeypatch)
    rows = exchange("B", route)
    rows[3]["reasoning"] = None
    db.rows = exchange("A", route) + rows
    complete_turn(engine, db, "B", route)
    db.rows += exchange("C", route, complete=False)[:1]
    request, _, _ = request_for(db, "C", route)
    for message in request:
        if message["role"] == "assistant":
            message.setdefault("reasoning_content", " ")
    _, result = choose(engine, db, "C", route, request=request)
    assert result is not None, engine.get_status()


def test_request_bound_counts_carriers_across_the_assembled_request(tmp_path, monkeypatch):
    route = "anthropic"
    engine, db, _ = _engine_after_publication(tmp_path, monkeypatch, route)
    db.rows += exchange("C", route, complete=False)[:1]
    request, _, _ = request_for(db, "C", route)
    one = prepared._carrier_bytes(request[-2])
    assert one and sum(prepared._carrier_bytes(m) for m in request) > one
    monkeypatch.setattr(prepared, "_MAX_CARRIER_REQUEST_BYTES", one)
    _, result = choose(engine, db, "C", route, request=request)
    assert result is None
    assert engine.get_status()["selection"]["reason"] == "unsupported_carrier"


ADMITTED = [
    "_anthropic_content_blocks",
    "anthropic_content_blocks",
    "extra_content",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "thinking",
    "thought_signature",
]


@pytest.mark.parametrize("key", ADMITTED)
@pytest.mark.parametrize("wire", [False, True])
def test_shared_validation_still_refuses_every_admitted_carrier_by_default(key, wire):
    """Passive/prepared engines never pass the rolling-only flag: refusal is unchanged."""
    value = [{"type": "thinking", "thinking": "x"}]
    message = {"role": "assistant", "content": "x", key: value}
    with pytest.raises(ValueError) as caught:
        prepared._comparison_payloads([message], [None], wire=wire)
    assert type(caught.value) is ValueError
    assert caught.value.carrier == key
    # The rolling-only flag admits it and excludes it from the compared payload.
    payload = prepared._comparison_payloads([message], [None], wire=wire, carriers=True)
    assert payload == [{"role": "assistant", "content": "x"}]


@pytest.mark.parametrize(
    "call",
    [
        {"extra_content": {"google": {"thought_signature": "x"}}},
        {"function": {"name": "f", "arguments": "{}", "thought_signature": "x"}},
    ],
)
def test_tool_call_signatures_are_refused_by_default_and_excluded_when_admitted(call):
    base = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{**base, **call}]},
        {"role": "tool", "content": "r", "tool_call_id": "c1"},
    ]
    with pytest.raises(ValueError):
        prepared._occurrences(messages)
    prepared._occurrences(messages, carriers=True)
    payload = prepared._comparison_payloads(messages, [None] * 3, carriers=True)
    assert payload[1]["tool_calls"] == [base]


def test_restored_carrier_needs_a_canonical_one_at_completion():
    rows = exchange("A", "reasoning")
    rows[3]["reasoning"] = None
    observed = restored(rows, "reasoning")
    rolling.validate_completed_turn(rows, observed, reasoning_policy=None)
    observed[3]["reasoning_details"] = SIGNED  # Not in the canonical row.
    with pytest.raises(ValueError) as caught:
        rolling.validate_completed_turn(rows, observed, reasoning_policy=None)
    assert caught.value.carrier == "reasoning_details"


def test_codex_reasoning_rows_keep_their_audited_codec_under_the_rolling_flag():
    """A Codex reasoning row admits no other carrier: the opaque path never applies to it."""
    from test_rolling_selection import POLICY
    from test_rolling_selection import exchange as codex_exchange

    rows = codex_exchange("A")
    answer = rows[3]
    assert answer["codex_reasoning_items"]
    payloads = prepared._comparison_payloads(rows, [POLICY] * 4, durable=True, carriers=True)
    assert payloads[3]["codex_reasoning_items"]  # Still compared, unchanged.
    answer["reasoning_details"] = json.dumps(SIGNED)
    with pytest.raises(ValueError) as caught:
        prepared._comparison_payloads(rows, [POLICY] * 4, durable=True, carriers=True)
    assert str(caught.value) == "unsupported replay carrier"
    assert caught.value.carrier == "reasoning_details"


@pytest.mark.parametrize("value", [False, 0, 0.0, True, 7])
def test_scalar_carrier_values_are_refused_not_treated_as_absent(value):
    message = {"role": "assistant", "content": "x", "reasoning": value}
    with pytest.raises(ValueError) as caught:
        prepared._comparison_payloads([message], [None], carriers=True)
    assert caught.value.carrier == "reasoning"


@pytest.mark.parametrize("key", ["reasoning", "reasoning_details"])
def test_restored_scalar_carrier_needs_canonical_origin(key):
    rows = exchange("A", "reasoning")
    rows[3]["reasoning"] = None
    observed = restored(rows, "reasoning")
    observed[3][key] = 0
    with pytest.raises(ValueError) as caught:
        rolling.validate_completed_turn(rows, observed, reasoning_policy=None)
    assert caught.value.carrier == key


def test_carrier_admission_runs_before_generic_serialization(monkeypatch):
    """Oversized or non-JSON carriers are named, not lost to a generic size/JSON error."""
    monkeypatch.setattr(prepared, "_MAX_CARRIER_MESSAGE_BYTES", 256)
    settings = rolling.RollingSettings(max_checkpoint_bytes=4096)
    for value in ("x" * 5000, [float("nan")]):
        rows = exchange("A", "reasoning")
        rows[3]["reasoning_details"] = value
        with pytest.raises(ValueError) as caught:
            rolling.readable_delta(rows, policy=None, settings=settings)
        assert caught.value.carrier == "reasoning_details", value
        with pytest.raises(ValueError) as caught:
            rolling.validate_completed_turn(
                exchange("A", "reasoning"),
                [
                    {**m, "reasoning_details": value} if m["role"] == "assistant" else m
                    for m in restored(exchange("A", "reasoning"), "reasoning")
                ],
                reasoning_policy=None,
                settings=settings,
            )
        assert caught.value.carrier == "reasoning_details", value


def test_oversized_string_carrier_is_refused_before_it_is_serialized(monkeypatch):
    monkeypatch.setattr(prepared, "_MAX_CARRIER_MESSAGE_BYTES", 256)

    def no_serialization(value):
        raise AssertionError("oversized carrier was serialized")

    monkeypatch.setattr(prepared, "ascii_json_units", no_serialization)
    monkeypatch.setattr(prepared, "_safe_json", no_serialization)
    with pytest.raises(ValueError) as caught:
        prepared._admit_opaque_carriers({"role": "assistant", "reasoning": "x" * 300})
    assert caught.value.carrier == "reasoning"


def test_non_json_assembled_carrier_is_named_at_selection(tmp_path, monkeypatch):
    engine, db, _ = _engine_after_publication(tmp_path, monkeypatch)
    db.rows += exchange("C", "anthropic", complete=False)[:1]
    request, _, _ = request_for(db, "C", "anthropic")
    request[-2]["reasoning_details"] = [float("nan")]  # B's answer has a canonical carrier.
    _, result = choose(engine, db, "C", "anthropic", request=request)
    assert result is None
    selection = engine.get_status()["selection"]
    assert selection["reason"] == "unsupported_carrier"
    assert selection["carrier"] == "reasoning_details"
