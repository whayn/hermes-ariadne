"""Synthetic, framework-free prepared publication and request selection tests."""

import copy
import hashlib
import json
import stat

import pytest

from hermes_ariadne.kernel import MemoryKernel
from hermes_ariadne.sources import Projection
from hermes_ariadne.store import EventStore, canonical_json


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def fixture(tmp_path, *, count=2, source_whitespace="", **limits):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    rows = [
        {
            "message_uid": f"m{i}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": source_whitespace + f"source {i} " + "λ" * 200 + source_whitespace,
            "timestamp": i,
        }
        for i in range(count)
    ]
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    store.ingest("chosen", rows, origin="hermes_db")
    projection = Projection(store, "chosen", [digest(row) for row in rows])
    kernel = MemoryKernel(projection, tmp_path / "offline.json", node_bytes=128)
    kernel.build(lambda request: f"summary {request.part.start}+{request.part.count}")
    path = publish_prepared(kernel, home=tmp_path, **limits)
    prepared = PreparedView.load(store, "chosen", tmp_path)
    assert prepared is not None
    return store, rows, kernel, path, prepared


def test_publication_loads_exact_frozen_checkpoint(tmp_path):
    from hermes_ariadne.prepared import FORMAT, VERSION, PreparedView

    store, rows, kernel, path, prepared = fixture(tmp_path)
    assert (
        path
        == tmp_path
        / "ariadne"
        / "prepared"
        / hashlib.sha256(b"chosen").hexdigest()
        / "manifest.json"
    )
    manifest = json.loads(path.read_bytes())
    assert manifest["format"] == FORMAT == "hermes-ariadne-prepared"
    assert manifest["version"] == VERSION == 1
    assert prepared is not None
    assert prepared.projection.event_ids == tuple(digest(row) for row in rows)
    assert prepared.kernel.view() == kernel.view()
    assert prepared.kernel.path != kernel.path
    for target in (path, prepared.kernel.path):
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    store.ingest(
        "chosen", [{"message_uid": "tail", "role": "user", "content": "new"}], origin="hermes_db"
    )
    reloaded = PreparedView.load(store, "chosen", tmp_path)
    assert reloaded is not None
    assert reloaded.kernel.view() == prepared.kernel.view()
    assert PreparedView.load(store, "absent", tmp_path) is None


def selection_fixture(tmp_path, **limits):
    store, rows, kernel, path, prepared = fixture(tmp_path, **limits)
    conversation = copy.deepcopy(rows) + [
        {"message_uid": "previous", "role": "user", "content": "previous"},
        {"message_uid": "answer", "role": "assistant", "content": "answer"},
        {"message_uid": "current", "role": "user", "content": "current"},
    ]
    wire = [{"role": "system", "content": "host"}, {"role": "developer", "content": "policy"}] + [
        {key: value for key, value in row.items() if key not in {"message_uid", "timestamp"}}
        for row in conversation
    ]
    return prepared, rows, conversation, wire


def choose(prepared, active, conversation, wire, **kwargs):
    return prepared.select(
        wire,
        conversation_messages=conversation,
        incoming_message=conversation[4],
        active_messages=active,
        budget_tokens=kwargs.pop("budget_tokens", 10000),
        measure_tokens=kwargs.pop("measure_tokens", lambda _: 100),
        **kwargs,
    )


def test_select_exact_prefix_preserves_actual_request_and_sources(tmp_path):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    original = copy.deepcopy((active, conversation, wire))
    measured = []

    def measure(messages):
        measured.append(copy.deepcopy(messages))
        messages[0]["content"] = "counter mutation is isolated"
        return 100

    decision = choose(prepared, active, conversation, wire, measure_tokens=measure)
    assert decision.status == "selected"
    assert decision.messages is not None
    assert decision.messages[:2] == wire[:2]
    assert decision.messages[3:] == wire[4:]
    memory = decision.messages[2]
    assert memory["role"] == "assistant"
    assert "untrusted" in memory["content"]
    assert prepared.projection.projection_id in memory["content"]
    assert all(event not in memory["content"] for event in prepared.projection.event_ids)
    assert decision.messages == measured[0]
    assert decision.diagnostics["provider_fit_verified"] is False
    assert decision.diagnostics["measured_tokens"] == 100
    assert decision.diagnostics["request_units"] == "ascii-json-bytes"
    assert decision.diagnostics["history_units"] == "compiled-history-utf8-bytes"
    assert (active, conversation, wire) == original
    decision.messages[0]["content"] = "changed return"
    assert (active, conversation, wire) == original


def test_restored_db_persisted_marker_allows_exact_host_mapping(tmp_path):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    for row in conversation:
        row["_db_persisted"] = True
    original = copy.deepcopy((active, conversation, wire))

    result = choose(prepared, active, conversation, wire)

    assert result.status == "selected"
    assert result.messages is not None
    assert result.messages[:2] == wire[:2]
    assert result.messages[3:] == wire[4:]
    assert all("_db_persisted" not in row for row in result.messages)
    assert (active, conversation, wire) == original


@pytest.mark.parametrize("marker", [True, False, 1, "true", None])
@pytest.mark.parametrize("location", ["conversation", "request"])
def test_assembled_persistence_marker_is_exact_true(tmp_path, marker, location):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    target = conversation if location == "conversation" else wire[2:]
    for row in target:
        row["_db_persisted"] = marker
    before = copy.deepcopy((active, conversation, wire))
    archive = prepared.projection._archive.path.read_bytes()
    result = choose(prepared, active, conversation, wire)
    assert result.status == ("selected" if marker is True else "declined")
    if marker is True:
        assert result.messages[:2] == wire[:2]
        assert result.messages[3:] == wire[4:]
    else:
        assert result.messages is None
    assert (active, conversation, wire) == before
    assert prepared.projection._archive.path.read_bytes() == archive


@pytest.mark.parametrize("mutation", [None, "unstamped", "interior", "tool"])
def test_only_supported_restored_whitespace_projection_can_align(tmp_path, mutation):
    prepared, active, conversation, _ = selection_fixture(tmp_path, source_whitespace=" \n\t")
    conversation = tool_tail(conversation)
    active = copy.deepcopy(active + conversation[2:])
    if mutation == "tool":
        active[-1]["content"] = " third "
        conversation[-1]["content"] = "third"
    for row in conversation:
        row["_db_persisted"] = True
        if row["role"] in {"user", "assistant"} and isinstance(row["content"], str):
            row["content"] = row["content"].strip()
    if mutation == "unstamped":
        conversation[0].pop("_db_persisted")
    elif mutation == "interior":
        conversation[0]["content"] = conversation[0]["content"].replace("source 0", "source  0")
    wire = as_wire(conversation)
    before = copy.deepcopy((active, conversation, wire))
    archive = prepared.projection._archive.path.read_bytes()

    result = choose(prepared, active, conversation, wire)

    if mutation is None:
        assert result.status == "selected"
        assert result.messages is not None
        assert result.messages[:2] == wire[:2]
        assert result.messages[3:] == wire[4:]
    else:
        assert result.status == "declined" and result.messages is None
        assert result.diagnostics["reason"] == "unsupported_alignment"
    assert (active, conversation, wire) == before
    assert prepared.projection._archive.path.read_bytes() == archive


@pytest.mark.parametrize(
    "budget,measurement,limits,reason",
    [
        (0, 100, {}, "unknown_budget"),
        (True, 100, {}, "unknown_budget"),
        (-1, 100, {}, "unknown_budget"),
        (None, 100, {}, "unknown_budget"),
        (4195, 100, {}, "over_budget"),
        (4000, 0, {}, "over_budget"),
        (10000, True, {}, "measurement_error"),
        (10000, -1, {}, "measurement_error"),
        (10000, 1.5, {}, "measurement_error"),
        (10000, None, {}, "measurement_error"),
        (10000, 100, {"history_bytes": 1}, "history_bytes_exceeded"),
        (10000, 100, {"request_bytes": 1}, "request_bytes_exceeded"),
    ],
)
def test_budget_uncertainty_declines_without_a_request(
    tmp_path, budget, measurement, limits, reason
):
    prepared, active, conversation, wire = selection_fixture(tmp_path, **limits)
    result = choose(
        prepared,
        active,
        conversation,
        wire,
        budget_tokens=budget,
        measure_tokens=lambda _: measurement,
    )
    assert result.status == "declined"
    assert result.messages is None
    assert result.diagnostics["reason"] == reason
    assert result.diagnostics["provider_fit_verified"] is False


def test_counter_error_is_a_decline_and_exact_allowance_can_select(tmp_path):
    prepared, active, conversation, wire = selection_fixture(tmp_path)

    def error(_):
        raise RuntimeError("secret caller path")

    result = choose(prepared, active, conversation, wire, measure_tokens=error)
    assert result.status == "declined"
    assert result.diagnostics["reason"] == "measurement_error"
    assert "secret" not in str(result.diagnostics)
    assert choose(prepared, active, conversation, wire, budget_tokens=4196).status == "selected"


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "metadata_revision",
        "active_missing",
        "active_order",
        "duplicate_active",
        "missing_uid",
        "duplicate_conversation",
        "conversation_uid",
        "conversation_payload",
        "wire_payload",
        "wire_extra",
        "wire_uid",
        "wire_db_persisted",
        "unknown_internal_carrier",
        "wire_reorder",
        "api_content",
        "thinking",
        "absorbed",
        "db_absorbed",
        "incoming_uid",
        "incoming_payload",
        "inactive",
        "unknown_raw",
        "prefill",
        "history_system",
        "no_host_prefix",
    ],
)
def test_unsupported_or_unproved_mapping_declines(tmp_path, mutation):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    active, conversation = copy.deepcopy(active), copy.deepcopy(conversation)
    incoming = copy.deepcopy(conversation[4])
    if mutation == "revision":
        active[0]["content"] = "edited"
    elif mutation == "metadata_revision":
        active[0]["timestamp"] = 99
    elif mutation == "active_missing":
        active.pop(0)
    elif mutation == "active_order":
        active.reverse()
    elif mutation == "duplicate_active":
        active.append(copy.deepcopy(active[0]))
    elif mutation == "missing_uid":
        conversation[0].pop("message_uid")
    elif mutation == "duplicate_conversation":
        conversation[2]["message_uid"] = "m0"
    elif mutation == "conversation_uid":
        conversation[0]["message_uid"] = "different"
    elif mutation == "conversation_payload":
        conversation[0]["content"] = "edited"
    elif mutation == "wire_payload":
        wire[2]["content"] = "injected"
    elif mutation == "wire_extra":
        wire[2]["reasoning"] = "opaque"
    elif mutation == "wire_uid":
        wire[2]["message_uid"] = "m0"
    elif mutation == "wire_db_persisted":
        wire[2]["_db_persisted"] = False
    elif mutation == "unknown_internal_carrier":
        conversation[0]["_unknown_replay"] = "opaque"
    elif mutation == "wire_reorder":
        wire[2], wire[3] = wire[3], wire[2]
    elif mutation == "api_content":
        conversation[0]["api_content"] = "different wire replay"
    elif mutation == "thinking":
        conversation[0]["_anthropic_content_blocks"] = [{"thinking": "x"}]
    elif mutation == "absorbed":
        conversation[0]["_absorbed_message_uids"] = ["gone"]
    elif mutation == "db_absorbed":
        active[0]["absorbed_message_uids"] = '["gone"]'
    elif mutation == "incoming_uid":
        incoming["message_uid"] = "elsewhere"
    elif mutation == "incoming_payload":
        incoming["content"] = "different"
    elif mutation == "inactive":
        active[0]["active"] = 0
    elif mutation == "unknown_raw":
        conversation[0]["unknown"] = None
    elif mutation == "prefill":
        wire.insert(2, {"role": "assistant", "content": "prefill"})
    elif mutation == "history_system":
        conversation[2]["role"] = wire[4]["role"] = "system"
    elif mutation == "no_host_prefix":
        wire = wire[2:]
    result = prepared.select(
        wire,
        conversation_messages=conversation,
        incoming_message=incoming,
        active_messages=active,
        budget_tokens=10000,
        measure_tokens=lambda _: 100,
    )
    assert result.status == "declined"
    assert result.messages is None
    assert result.diagnostics["reason"] == "unsupported_alignment"


def test_supported_raw_host_prefix_and_identical_sidecar_are_not_rebuilt(tmp_path):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    conversation[0]["api_content"] = conversation[0]["content"]
    conversation = copy.deepcopy(wire[:2]) + conversation
    result = prepared.select(
        wire,
        conversation_messages=conversation,
        incoming_message=conversation[-1],
        active_messages=active,
        budget_tokens=10000,
        measure_tokens=lambda _: 100,
    )
    assert result.status == "selected"
    assert result.messages is not None
    assert result.messages[:2] == wire[:2]


def call(call_id):
    return {"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


def tool_tail(conversation):
    return conversation + [
        {
            "message_uid": "a1",
            "role": "assistant",
            "content": None,
            "tool_calls": [call("repeat"), call("parallel")],
            "_tool_call_uids": {"repeat": "occ1", "parallel": "occ2"},
        },
        {
            "message_uid": "t1",
            "role": "tool",
            "content": "first",
            "tool_call_id": "repeat",
            "_tool_call_uid": "occ1",
        },
        {
            "message_uid": "t2",
            "role": "tool",
            "content": "second",
            "tool_call_id": "parallel",
            "_tool_call_uid": "occ2",
        },
        {
            "message_uid": "a2",
            "role": "assistant",
            "content": None,
            "tool_calls": [call("repeat")],
            "_tool_call_uids": {"repeat": "occ3"},
        },
        {
            "message_uid": "t3",
            "role": "tool",
            "content": "third",
            "tool_call_id": "repeat",
            "_tool_call_uid": "occ3",
        },
    ]


def as_wire(conversation):
    return [{"role": "system", "content": "host"}, {"role": "developer", "content": "policy"}] + [
        {
            key: value
            for key, value in row.items()
            if key in {"role", "content", "tool_calls", "tool_call_id", "name"}
        }
        for row in conversation
    ]


@pytest.mark.parametrize(
    "call_id,item_id",
    [
        ("repeat", "fc_repeat"),
        ("call_abc", "fc_abc"),
        ("fc_abc", "fc_abc"),
        ("call_a:b", "fc_a:b"),
        ("a:b", "fc_ab"),
        ("f:c_abc", "fc_abc"),
        ("c:all_abc", "fc_abc"),
        ("!!!", "fc_" + hashlib.sha1(b"!!!").hexdigest()[:24]),
        ("x" * 60, "fc_" + "x" * 48),
    ],
)
@pytest.mark.parametrize("request_sidecars", [False, True])
def test_canonical_function_sidecars_compare_without_changing_hook_suffix(
    tmp_path, call_id, item_id, request_sidecars
):
    prepared, active, conversation, _ = selection_fixture(tmp_path)
    conversation = tool_tail(conversation)
    for row in conversation:
        if "tool_calls" in row:
            for tc in row["tool_calls"]:
                old = tc["id"]
                if old == "repeat":
                    tc["id"] = call_id
                    row["_tool_call_uids"][call_id] = row["_tool_call_uids"].pop(old)
                tc.update(
                    call_id=tc["id"], response_item_id=item_id if old == "repeat" else "fc_parallel"
                )
        if row.get("tool_call_id") == "repeat":
            row["tool_call_id"] = call_id
    active = copy.deepcopy(conversation)
    wire = copy.deepcopy(as_wire(conversation))
    if not request_sidecars:
        for row in wire:
            for tc in row.get("tool_calls", []):
                tc.pop("call_id")
                tc.pop("response_item_id")
    before = copy.deepcopy((active, conversation, wire))
    archive = prepared.projection._archive.path.read_bytes()
    result = choose(prepared, active, conversation, wire)
    assert result.status == "selected" and result.messages is not None
    assert result.messages[3:] == wire[4:]
    result.messages[-2]["tool_calls"][0]["function"]["arguments"] = "isolated"
    assert (active, conversation, wire) == before
    assert prepared.projection._archive.path.read_bytes() == archive


def test_archived_function_prefix_keeps_exact_sidecars_hashes_and_records(tmp_path):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    store, rows, _, _, _ = fixture(tmp_path, count=4)
    rows[1].update(
        tool_calls=[{**call("repeat"), "call_id": "repeat", "response_item_id": "fc_repeat"}],
        _tool_call_uids={"repeat": "archived_occurrence"},
    )
    rows[2].update(
        role="tool", tool_call_id="repeat", tool_name="lookup", _tool_call_uid="archived_occurrence"
    )
    store.ingest("chosen", rows, origin="hermes_db")
    events = tuple(digest(row) for row in rows)
    projection = Projection(store, "chosen", events)
    kernel = MemoryKernel(projection, tmp_path / "function-prefix.json", node_bytes=128)
    kernel.build(lambda _: "archived function navigation")
    publish_prepared(kernel, home=tmp_path)
    prepared = PreparedView.load(store, "chosen", tmp_path)
    assert prepared is not None
    conversation = copy.deepcopy(rows) + [
        {"message_uid": "previous", "role": "user", "content": "previous"},
        {"message_uid": "answer", "role": "assistant", "content": "answer"},
        {"message_uid": "current", "role": "user", "content": "current"},
    ]
    wire = copy.deepcopy(as_wire(conversation))
    wire[3]["tool_calls"][0].pop("call_id")
    wire[3]["tool_calls"][0].pop("response_item_id")
    wire[4]["tool_name"] = "lookup"
    before = copy.deepcopy((rows, conversation, wire))
    archive = store.path.read_bytes()
    decision = prepared.select(
        wire,
        conversation_messages=conversation,
        incoming_message=conversation[-1],
        active_messages=rows,
        budget_tokens=10000,
        measure_tokens=lambda _: 100,
    )
    assert decision.status == "selected" and decision.messages is not None
    assert decision.messages[3:] == wire[6:]
    assert prepared.projection.event_ids == events
    assert [store.get_event(event, session_id="chosen") for event in events] == rows
    assert (rows, conversation, wire) == before
    assert store.path.read_bytes() == archive


@pytest.mark.parametrize("assembled_name", ["name", "tool_name", "both"])
def test_tool_result_names_and_effect_metadata_have_role_scoped_comparison(
    tmp_path, assembled_name
):
    prepared, active, conversation, _ = selection_fixture(tmp_path)
    conversation = tool_tail(conversation)
    for row in conversation:
        if row["role"] == "tool":
            row.update(tool_name="lookup", name="lookup", effect_disposition="none")
        elif row["role"] in {"user", "assistant"}:
            row["name"] = "speaker_" + row["role"]
    # Durable/restored tool rows do not have the live name alias.
    active = copy.deepcopy(conversation)
    for row in active:
        if row["role"] == "tool":
            row.pop("name")
    # Prepared source rows remain the original exact hash-bound prefix.
    for row in active[:2]:
        row.pop("name")
    for row in conversation[:2]:
        row.pop("name")
    wire = copy.deepcopy(as_wire(conversation))
    for row in wire:
        if row["role"] == "tool":
            row.update(tool_name="lookup", effect_disposition="none")
            if assembled_name == "name":
                row.pop("tool_name")
            elif assembled_name == "tool_name":
                row.pop("name")
    before = copy.deepcopy((active, conversation, wire))
    archive = prepared.projection._archive.path.read_bytes()
    result = choose(prepared, active, conversation, wire)
    assert result.status == "selected" and result.messages is not None
    assert result.messages[3:] == wire[4:]
    assert (active, conversation, wire) == before
    assert prepared.projection._archive.path.read_bytes() == archive


@pytest.mark.parametrize("location", ["all", "request", "conversation", "active"])
@pytest.mark.parametrize(
    "mutation",
    [
        "call_id",
        "call_id_type",
        "item_id",
        "item_id_type",
        "item_id_whitespace",
        "signature",
        "extra_content",
        "function_signature",
        "tool_name_conflict",
        "tool_name_type",
        "tool_name_empty",
        "name_type",
        "effect_type",
        "cross_role_name",
        "cross_role_effect",
        "unknown_marker",
    ],
)
def test_comparison_rejects_malformed_or_unaudited_fields(tmp_path, location, mutation):
    prepared, active, conversation, _ = selection_fixture(tmp_path)
    conversation = tool_tail(conversation)
    for row in conversation:
        for tc in row.get("tool_calls", []):
            tc.update(call_id=tc["id"], response_item_id="fc_" + tc["id"])
        if row["role"] == "tool":
            row.update(name="lookup", tool_name="lookup", effect_disposition="none")
    active = copy.deepcopy(conversation)
    wire = copy.deepcopy(as_wire(conversation))
    for row in wire:
        if row["role"] == "tool":
            row.update(tool_name="lookup", effect_disposition="none")
    views = {"request": wire[2:], "conversation": conversation, "active": active}
    targets = views.values() if location == "all" else [views[location]]
    for rows in targets:
        tc = rows[5]["tool_calls"][0]
        if mutation == "call_id":
            tc["call_id"] = "contradiction"
        elif mutation == "call_id_type":
            tc["call_id"] = None
        elif mutation == "item_id":
            tc["response_item_id"] = "fc_native_unrelated"
        elif mutation == "item_id_type":
            tc["response_item_id"] = False
        elif mutation == "item_id_whitespace":
            tc["response_item_id"] = " fc_repeat "
        elif mutation in {"signature", "extra_content"}:
            tc[mutation] = "opaque"
        elif mutation == "function_signature":
            tc["function"]["thought_signature"] = "opaque"
        elif mutation == "tool_name_conflict":
            rows[6]["tool_name"] = "different"
        elif mutation == "tool_name_type":
            rows[6]["tool_name"] = 1
        elif mutation == "tool_name_empty":
            rows[6]["tool_name"] = ""
        elif mutation == "name_type":
            rows[6]["name"] = None
        elif mutation == "effect_type":
            rows[6]["effect_disposition"] = []
        elif mutation == "cross_role_name":
            rows[4]["tool_name"] = "lookup"
        elif mutation == "cross_role_effect":
            rows[5]["effect_disposition"] = "none"
        elif mutation == "unknown_marker":
            rows[4]["_anything_else"] = True
    before = copy.deepcopy((active, conversation, wire))
    archive = prepared.projection._archive.path.read_bytes()
    measured = []
    result = choose(
        prepared, active, conversation, wire, measure_tokens=lambda m: measured.append(m) or 100
    )
    assert result.status == "declined" and result.messages is None
    assert result.diagnostics["reason"] == "unsupported_alignment"
    assert measured == []
    assert (active, conversation, wire) == before
    assert prepared.projection._archive.path.read_bytes() == archive


@pytest.mark.parametrize(
    "mutation",
    ["function_name", "function_arguments", "user_name", "assistant_name", "effect", "occurrence"],
)
def test_hook_payload_remains_exact_after_comparison(tmp_path, mutation):
    prepared, active, conversation, _ = selection_fixture(tmp_path)
    conversation = tool_tail(conversation)
    conversation[2]["name"] = "speaker_user"
    conversation[3]["name"] = "speaker_assistant"
    for row in conversation:
        for tc in row.get("tool_calls", []):
            tc.update(call_id=tc["id"], response_item_id="fc_" + tc["id"])
        if row["role"] == "tool":
            row.update(tool_name="lookup", effect_disposition="none")
    active = copy.deepcopy(conversation)
    wire = copy.deepcopy(as_wire(conversation))
    for row in wire:
        if row["role"] == "tool":
            row.update(tool_name="lookup", effect_disposition="none")
    if mutation == "function_name":
        wire[7]["tool_calls"][0]["function"]["name"] = " lookup "
    elif mutation == "function_arguments":
        wire[7]["tool_calls"][0]["function"]["arguments"] = "{ }"
    elif mutation == "user_name":
        wire[4]["name"] = "other_user"
    elif mutation == "assistant_name":
        wire[5]["name"] = "other_assistant"
    elif mutation == "effect":
        wire[8]["effect_disposition"] = "unknown"
    elif mutation == "occurrence":
        # Even agreed active/live metadata cannot detach a result from its call.
        active[6]["_tool_call_uid"] = conversation[6]["_tool_call_uid"] = "wrong"
    before = copy.deepcopy((active, conversation, wire))
    result = choose(prepared, active, conversation, wire)
    assert result.status == "declined" and result.messages is None
    assert (active, conversation, wire) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_result",
        "duplicate_result",
        "pending_user",
        "orphan",
        "cross_role",
        "duplicate_parallel",
        "wrong_occurrence",
        "reuse_occurrence",
        "missing_occurrence",
        "unknown_call_field",
        "bad_arguments",
        "assistant_prefill",
        "previous_exchange",
        "cut_mid_turn",
    ],
)
def test_tool_or_retention_uncertainty_declines(tmp_path, mutation):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    conversation = tool_tail(conversation)
    if mutation == "missing_result":
        conversation.pop(6)
    elif mutation == "duplicate_result":
        conversation[7]["tool_call_id"] = "repeat"
    elif mutation == "pending_user":
        conversation[6]["role"] = "user"
    elif mutation == "orphan":
        conversation[6]["tool_call_id"] = "unissued"
    elif mutation == "cross_role":
        conversation[5]["role"] = "user"
    elif mutation == "duplicate_parallel":
        conversation[5]["tool_calls"][1]["id"] = "repeat"
    elif mutation == "wrong_occurrence":
        conversation[6]["_tool_call_uid"] = "wrong"
    elif mutation == "reuse_occurrence":
        conversation[8]["_tool_call_uids"]["repeat"] = "occ1"
        conversation[9]["_tool_call_uid"] = "occ1"
    elif mutation == "missing_occurrence":
        conversation[6].pop("_tool_call_uid")
    elif mutation == "unknown_call_field":
        conversation[5]["tool_calls"][0]["extra_content"] = {"thinking": "opaque"}
    elif mutation == "bad_arguments":
        conversation[5]["tool_calls"][0]["function"]["arguments"] = {}
    elif mutation == "assistant_prefill":
        conversation.append({"message_uid": "prefill", "role": "assistant", "content": "partial"})
    elif mutation == "previous_exchange":
        conversation = conversation[:2] + [conversation[4]]
    elif mutation == "cut_mid_turn":
        store, active, kernel, path, prepared = fixture(tmp_path / "other", count=1)
        conversation = (
            active
            + [{"message_uid": "old_a", "role": "assistant", "content": "a"}]
            + conversation[2:5]
        )
    wire = as_wire(conversation)
    incoming = next(row for row in reversed(conversation) if row["role"] == "user")
    result = prepared.select(
        wire,
        conversation_messages=conversation,
        incoming_message=incoming,
        active_messages=active,
        budget_tokens=10000,
        measure_tokens=lambda _: 100,
    )
    assert result.status == "declined"
    assert result.messages is None


def test_complete_parallel_and_reused_calls_preserve_frozen_prefix(tmp_path):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    before = choose(prepared, active, conversation, wire)
    assert before.messages is not None
    conversation = tool_tail(conversation)
    wire = as_wire(conversation)
    original = copy.deepcopy((active, conversation, wire))
    result = choose(prepared, active, conversation, wire)
    assert result.status == "selected"
    assert result.messages is not None
    assert result.messages[:3] == before.messages[:3]
    assert result.messages[3:] == wire[4:]
    assert (active, conversation, wire) == original


@pytest.mark.parametrize("field", ["history_bytes", "request_bytes", "reserve_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "12"])
def test_publication_limits_are_strict_positive_integers(tmp_path, field, value):
    from hermes_ariadne.prepared import publish_prepared

    store, rows, kernel, path, prepared = fixture(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        publish_prepared(kernel, home=tmp_path, **{field: value})
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "format",
        "session",
        "projection",
        "settings",
        "limit",
        "digest",
        "path",
        "duplicate_json",
        "oversize",
        "checkpoint_digest",
        "checkpoint_invalid",
        "unknown_field",
        "public_manifest",
        "public_checkpoint",
        "public_directory",
    ],
)
def test_invalid_artifacts_never_load_or_create_a_replacement(tmp_path, mutation):
    from hermes_ariadne.prepared import MAX_MANIFEST_BYTES, PreparedView

    store, rows, kernel, path, prepared = fixture(tmp_path)
    manifest = json.loads(path.read_bytes())
    checkpoint = prepared.kernel.path
    if mutation == "version":
        manifest["version"] = True
    elif mutation == "format":
        manifest["format"] = "foreign"
    elif mutation == "session":
        manifest["session_id"] = "elsewhere"
    elif mutation == "projection":
        manifest["projection_id"] = "0" * 64
    elif mutation == "settings":
        manifest["settings"]["node_bytes"] += 1
    elif mutation == "limit":
        manifest["limits"]["reserve_tokens"] = False
    elif mutation == "digest":
        manifest["checkpoint_sha256"] = "0" * 64
    elif mutation == "path":
        manifest["checkpoint_sha256"] = "../../offline"
    elif mutation == "duplicate_json":
        path.write_bytes(b'{"format":"x","format":"y"}')
    elif mutation == "oversize":
        path.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    elif mutation == "checkpoint_digest":
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
    elif mutation == "checkpoint_invalid":
        snapshot = json.loads(checkpoint.read_bytes())
        snapshot["chat"]["size"] += 1
        data = canonical_json(snapshot)
        manifest["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
        target = path.parent / (manifest["checkpoint_sha256"] + ".json")
        target.write_bytes(data)
        target.chmod(0o600)
    elif mutation == "unknown_field":
        manifest["checkpoint_path"] = "/secret/caller/path"
    elif mutation == "public_manifest":
        path.chmod(0o644)
    elif mutation == "public_checkpoint":
        checkpoint.chmod(0o644)
    elif mutation == "public_directory":
        path.parent.chmod(0o755)
    if mutation not in {"duplicate_json", "oversize"}:
        path.write_bytes(canonical_json(manifest))
    before = {p: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    assert PreparedView.load(store, "chosen", tmp_path) is None
    assert {p: p.read_bytes() for p in path.parent.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("replacement", ["text", "version-and-text", "same-summary-settings"])
def test_generation_handles_bind_exact_loaded_checkpoint_across_republication(
    tmp_path, replacement
):
    import base64

    from hermes_ariadne.prepared import PreparedView, publish_prepared
    from hermes_ariadne.retrieval import handle_evidence_tool, tool_schemas
    from hermes_ariadne.tree import Part

    store, rows, kernel, manifest_path, old = fixture(tmp_path, count=4)
    manifest = json.loads(manifest_path.read_bytes())
    assert old.checkpoint_id == manifest["checkpoint_sha256"]
    assert old.checkpoint_id == hashlib.sha256(old.kernel.path.read_bytes()).hexdigest()
    old_memory = old._memory
    root = f"node:{old.checkpoint_id}:" + kernel.reference(Part(2, 0))
    pattern = tool_schemas()[1]["parameters"]["properties"]["ref"]["pattern"]

    def zoom(view, ref):
        import re

        assert re.fullmatch(pattern, ref)
        result = handle_evidence_tool(
            "ariadne_zoom",
            {"ref": ref},
            store=store,
            session_id="chosen",
            kernel=view.kernel,
            checkpoint_id=view.checkpoint_id,
        )
        assert (
            len(json.dumps(result, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
            <= 8192
        )
        return result

    # Collect every emitted node recursively, including root, internal children and leaves.
    refs, responses, pending = [], [], [root]
    while pending:
        ref = pending.pop()
        response = zoom(old, ref)
        assert response["ok"]
        assert response["node"]["ref"] == ref
        assert ref.startswith(f"node:{old.checkpoint_id}:")
        refs.append(ref)
        responses.append(response)
        if response["kind"] == "node":
            pending.extend(child["ref"] for child in response["children"])
    assert len(refs) == 7
    # Unrelated archive appends and a fresh load preserve this exact generation.
    store.ingest("chosen", [{"message_uid": "tail", "role": "user", "content": "new"}], origin="x")
    same = PreparedView.load(store, "chosen", tmp_path)
    assert same is not None and same.checkpoint_id == old.checkpoint_id
    assert [zoom(same, ref) for ref in refs] == responses
    settings = kernel._settings()
    if replacement == "version-and-text":
        settings["summary_version"] = "replacement-v2"
    elif replacement == "same-summary-settings":
        settings["compactor_high"] += 1
    newer = MemoryKernel(kernel.projection, tmp_path / "replacement.json", **settings)
    prefix = "summary" if replacement == "same-summary-settings" else "replacement"
    newer.build(lambda req: f"{prefix} {req.part.start}+{req.part.count}")
    publish_prepared(newer, home=tmp_path)
    assert old._memory == old_memory
    assert [zoom(old, ref) for ref in refs] == responses
    new = PreparedView.load(store, "chosen", tmp_path)
    assert new is not None and new.checkpoint_id != old.checkpoint_id
    assert new.projection.projection_id == old.projection.projection_id
    assert new.projection.event_ids == old.projection.event_ids
    for ref in refs:
        result = zoom(new, ref)
        assert result["ok"] is False and result["error"] == "checkpoint_mismatch"
        assert set(result) == {"trust", "ok", "error"}
        fresh_ref = ref.replace(old.checkpoint_id, new.checkpoint_id, 1)
        assert zoom(new, fresh_ref)["ok"]
    # Source handles are independent of prepared generations and still recover exact bytes.
    for row in rows:
        result = zoom(new, "event:" + digest(row))
        assert result["ok"]
        assert base64.b64decode(result["page"]["data_base64"]) == canonical_json(row)


def test_republication_keeps_old_generation_and_loaded_view_frozen(tmp_path):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    store, rows, kernel, path, old = fixture(tmp_path)
    old_path, old_view = old.kernel.path, old.kernel.view()
    newer = MemoryKernel(
        kernel.projection, tmp_path / "new-offline.json", node_bytes=128, summary_version="new"
    )
    newer.build(lambda request: f"replacement {request.part.start}+{request.part.count}")
    publish_prepared(newer, home=tmp_path)
    new = PreparedView.load(store, "chosen", tmp_path)
    assert new is not None
    assert new.kernel.path != old_path
    assert old.kernel.view() == old_view
    assert old_path.exists()
    assert new.kernel.view() != old_view


@pytest.mark.parametrize(
    "target", ["prepared", "session", "manifest", "checkpoint", "manifest_hardlink"]
)
def test_linked_publication_targets_are_refused_without_archive_changes(tmp_path, target):
    import os

    from hermes_ariadne.prepared import PreparedView, publish_prepared

    store, rows, kernel, path, prepared = fixture(tmp_path)
    archive_before = [store.get_event(digest(row), session_id="chosen") for row in rows]
    checkpoint = prepared.kernel.path
    if target == "manifest_hardlink":
        path.unlink()
        os.link(store.path, path)
    elif target in {"manifest", "checkpoint"}:
        selected = path if target == "manifest" else checkpoint
        selected.unlink()
        selected.symlink_to(store.path)
    else:
        selected = path.parent if target == "session" else path.parent.parent
        destination = tmp_path / "moved"
        selected.rename(destination)
        selected.symlink_to(destination, target_is_directory=True)
    assert PreparedView.load(store, "chosen", tmp_path) is None
    with pytest.raises(ValueError):
        publish_prepared(kernel, home=tmp_path)
    assert [store.get_event(digest(row), session_id="chosen") for row in rows] == archive_before


def test_absent_manifest_does_not_create_a_blank_ready_kernel(tmp_path):
    from hermes_ariadne.prepared import PreparedView

    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    assert PreparedView.load(store, "chosen", tmp_path) is None
    assert not (tmp_path / "ariadne" / "prepared").exists()


def test_wrong_home_cannot_publish_or_load(tmp_path):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    store, rows, kernel, path, prepared = fixture(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    with pytest.raises(ValueError):
        publish_prepared(kernel, home=elsewhere)
    assert PreparedView.load(store, "chosen", elsewhere) is None
    assert not elsewhere.exists()


def test_failed_manifest_replace_leaves_previous_generation_loadable(tmp_path, monkeypatch):
    import os

    from hermes_ariadne.prepared import PreparedView, publish_prepared

    store, rows, kernel, path, prepared = fixture(tmp_path)
    previous = path.read_bytes()
    real_replace = os.replace

    def fail_manifest(source, target):
        if target == path:
            raise OSError("synthetic manifest publication failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_manifest)
    with pytest.raises(OSError):
        publish_prepared(kernel, home=tmp_path, history_bytes=999)
    assert path.read_bytes() == previous
    reloaded = PreparedView.load(store, "chosen", tmp_path)
    assert reloaded is not None
    assert reloaded.kernel.view() == prepared.kernel.view()
    assert not list(path.parent.glob(".publish-*"))


@pytest.mark.parametrize(
    "setting", ["node_bytes", "chat_high", "compactor_high", "summary_version"]
)
def test_publication_rejects_checkpoint_settings_drift(tmp_path, setting):
    from hermes_ariadne.prepared import publish_prepared

    store, rows, kernel, path, prepared = fixture(tmp_path)
    before = path.read_bytes()
    data = json.loads(kernel.path.read_bytes())
    data["settings"][setting] = (
        "other" if setting == "summary_version" else data["settings"][setting] + 1
    )
    # Keep the serialized view settings consistent so MemoryKernel itself can
    # load it: the operator's supplied object still describes different settings.
    if setting == "chat_high":
        data["chat"]["high"] += 1
    if setting == "compactor_high":
        data["seen"]["high"] += 1
    kernel.path.write_bytes(canonical_json(data))
    with pytest.raises(ValueError):
        publish_prepared(kernel, home=tmp_path)
    assert path.read_bytes() == before


def test_pending_kernel_is_not_generated_or_published(tmp_path):
    from hermes_ariadne.prepared import publish_prepared

    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    row = {"message_uid": "m", "role": "user", "content": "x" * 200}
    store.ingest("chosen", [row], origin="hermes_db")
    kernel = MemoryKernel(Projection(store, "chosen", [digest(row)]), tmp_path / "pending.json")
    before = kernel.path.read_bytes()
    with pytest.raises(ValueError):
        publish_prepared(kernel, home=tmp_path)
    assert kernel.path.read_bytes() == before
    assert not (tmp_path / "ariadne" / "prepared").exists()


def test_nonfinite_checkpoint_json_is_rejected_even_with_matching_digest(tmp_path):
    from hermes_ariadne.prepared import PreparedView

    store, rows, kernel, path, prepared = fixture(tmp_path)
    data = prepared.kernel.path.read_bytes()
    data = b'{"unused":1e999,' + data[1:]
    sha = hashlib.sha256(data).hexdigest()
    target = path.parent / f"{sha}.json"
    target.write_bytes(data)
    target.chmod(0o600)
    manifest = json.loads(path.read_bytes())
    manifest["checkpoint_sha256"] = sha
    path.write_bytes(canonical_json(manifest))
    assert PreparedView.load(store, "chosen", tmp_path) is None


def test_full_active_tail_and_nested_counter_are_isolated(tmp_path):
    prepared, active, conversation, wire = selection_fixture(tmp_path)
    conversation = tool_tail(conversation)
    active = copy.deepcopy(active + conversation[2:])
    for row in active[2:]:
        row.update(
            {
                "session_id": "chosen",
                "active": 1,
                "api_content": None,
                "reasoning": None,
                "finish_reason": None,
            }
        )
    wire = as_wire(conversation)
    before = copy.deepcopy((active, conversation, wire))
    checkpoint = prepared.kernel.path.read_bytes()
    tree_view = prepared.kernel.view()

    def mutate_counter(candidate):
        candidate[-2]["tool_calls"][0]["function"]["arguments"] = "counter mutation"
        return 100

    result = choose(prepared, active, conversation, wire, measure_tokens=mutate_counter)
    assert result.status == "selected"
    assert result.messages is not None
    assert result.messages[-2]["tool_calls"][0]["function"]["arguments"] == "{}"
    result.messages[-2]["tool_calls"][0]["function"]["arguments"] = "caller mutation"
    assert (active, conversation, wire) == before
    assert prepared.kernel.path.read_bytes() == checkpoint
    assert prepared.kernel.view() == tree_view


def test_exact_message_byte_bounds_can_select(tmp_path):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    prepared, active, conversation, wire = selection_fixture(tmp_path)
    wire[0]["content"] = "host λ"
    result = choose(prepared, active, conversation, wire)
    assert result.status == "selected"
    assert result.messages is not None
    publish_prepared(
        prepared.kernel,
        home=tmp_path,
        history_bytes=result.diagnostics["history_bytes"],
        request_bytes=result.diagnostics["request_bytes"],
    )
    bounded = PreparedView.load(prepared.projection._archive, "chosen", tmp_path)
    assert bounded is not None
    assert choose(bounded, active, conversation, wire).status == "selected"


@pytest.mark.parametrize("stage", ["publish", "load"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"data_use": ["no_training"]},
        {"data_use": None},
        {"data_use": False},
        {"data_use": {}},
        {"data_use": "no_training"},
        {"display_metadata": {"data_use": ["no_training"]}},
        {"display_metadata": {"data_use": None}},
        {"display_metadata": '{"data_use":["no_training"]}'},
        {"display_metadata": json.dumps('{"data_use":["no_training"]}')},
        {"display_metadata": '{"data_use":["no_training"],"data_use":[]}'},
        {"display_metadata": "broken JSON"},
        {"display_metadata": []},
        {"data_use": [], "display_metadata": {"data_use": ["unknown_requirement"]}},
    ],
)
def test_restricted_prepared_prefix_cannot_be_published_or_loaded(tmp_path, metadata, stage):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    store, rows, _, path, old = fixture(tmp_path)
    rows = copy.deepcopy(rows)
    rows[0].update(metadata)
    store.ingest("chosen", rows, origin="fixture")
    kernel = MemoryKernel(
        Projection(store, "chosen", [digest(row) for row in rows]),
        tmp_path / "restricted.json",
        node_bytes=128,
    )
    kernel.build(lambda _: "PRIVATE derived summary")
    if stage == "load":
        # A structurally valid, already-built legacy artifact bypasses today's
        # publisher; admission must re-check canonical evidence on load too.
        manifest = json.loads(path.read_bytes())
        data = kernel.path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        target = path.parent / f"{sha}.json"
        target.write_bytes(data)
        target.chmod(0o600)
        manifest.update(
            projection_id=kernel.projection.projection_id,
            event_ids=list(kernel.projection.event_ids),
            checkpoint_sha256=sha,
        )
        path.write_bytes(canonical_json(manifest))
    before = {p: p.read_bytes() for p in path.parent.iterdir()}
    archive = store.path.read_bytes()
    checkpoint = kernel.path.read_bytes()
    if stage == "publish":
        with pytest.raises(ValueError, match="^unsupported_data_use$"):
            publish_prepared(kernel, home=tmp_path)
        assert PreparedView.load(store, "chosen", tmp_path) is not None
    else:
        assert PreparedView.load(store, "chosen", tmp_path) is None
    assert {p: p.read_bytes() for p in path.parent.iterdir()} == before
    assert store.path.read_bytes() == archive and kernel.path.read_bytes() == checkpoint
    assert old.kernel.view()["usable"]


@pytest.mark.parametrize("location", ["active", "prefix", "retained_tool", "incoming", "wire"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"data_use": ["no_training"]},
        {"data_use": None},
        {"display_metadata": {"data_use": ["no_training"]}},
        {"display_metadata": {"data_use": False}},
        {"display_metadata": '{"data_use":["unknown_requirement"]}'},
        {"display_metadata": json.dumps('{"data_use":["unknown_requirement"]}')},
        {"display_metadata": "broken JSON"},
    ],
)
def test_selection_refuses_machine_restrictions_in_every_supplied_view(
    tmp_path, location, metadata
):
    prepared, active, conversation, _ = selection_fixture(tmp_path)
    conversation = tool_tail(conversation)
    wire = as_wire(conversation)
    incoming = copy.deepcopy(conversation[4])
    target = {
        "active": active[0],
        "prefix": conversation[0],
        "retained_tool": conversation[-1],
        "incoming": incoming,
        "wire": wire[2],
    }[location]
    target.update(copy.deepcopy(metadata))
    before = copy.deepcopy((active, conversation, wire, incoming))
    measured = []
    decision = prepared.select(
        wire,
        conversation_messages=conversation,
        incoming_message=incoming,
        active_messages=active,
        budget_tokens=10000,
        measure_tokens=lambda candidate: measured.append(candidate) or 100,
    )
    assert decision.status == "declined" and decision.messages is None
    assert decision.diagnostics["reason"] == "unsupported_data_use"
    assert decision.diagnostics["host_fallback"] == "original_request_on_decline"
    assert measured == []
    assert (active, conversation, wire, incoming) == before


def test_selection_rechecks_canonical_projection_even_if_constructed_without_load(tmp_path):
    from dataclasses import replace

    prepared, active, conversation, wire = selection_fixture(tmp_path)
    store = prepared.projection._archive
    restricted = {**active[0], "display_metadata": {"data_use": ["no_training"]}}
    store.ingest("chosen", [restricted], origin="fixture")
    kernel = MemoryKernel(
        Projection(store, "chosen", [digest(restricted), digest(active[1])]),
        tmp_path / "legacy.json",
        node_bytes=128,
    )
    kernel.build(lambda _: "PRIVATE")
    direct = replace(prepared, kernel=kernel)
    result = choose(direct, active, conversation, wire)
    assert result.status == "declined" and result.messages is None
    assert result.diagnostics["reason"] == "unsupported_data_use"


def test_empty_requirement_and_common_metadata_allow_exact_selection(tmp_path):
    from hermes_ariadne.prepared import PreparedView, publish_prepared

    prepared, active, conversation, _ = selection_fixture(tmp_path)
    for row in active:
        row.update(
            data_use=[], display_metadata=json.dumps(json.dumps({"data_use": [], "color": "blue"}))
        )
    store = prepared.projection._archive
    store.ingest("chosen", active, origin="fixture")
    kernel = MemoryKernel(
        Projection(store, "chosen", [digest(row) for row in active]),
        tmp_path / "empty-requirements.json",
        node_bytes=128,
    )
    kernel.build(lambda _: "unrestricted summary")
    publish_prepared(kernel, home=tmp_path)
    prepared = PreparedView.load(store, "chosen", tmp_path)
    assert prepared is not None
    conversation[:2] = copy.deepcopy(active)
    conversation[2].update(display_metadata={"task_count": 2})
    conversation[4]["content"] = (
        "Please do not train on this; a preference, not a machine guarantee"
    )
    wire = as_wire(conversation)
    before = copy.deepcopy((active, conversation, wire))
    result = choose(prepared, active, conversation, wire)
    assert result.status == "selected"
    assert (active, conversation, wire) == before
    assert [store.get_event(digest(row), session_id="chosen") for row in active] == active


def test_selected_navigation_handles_are_accepted_by_ariadne_zoom(tmp_path):
    import re

    from hermes_ariadne.retrieval import handle_evidence_tool, tool_schemas

    prepared, active, conversation, wire = selection_fixture(tmp_path)
    result = choose(prepared, active, conversation, wire)
    assert result.status == "selected" and result.messages is not None
    handles = [json.loads(line)["ref"] for line in result.messages[2]["content"].splitlines()[1:]]
    pattern = tool_schemas()[1]["parameters"]["properties"]["ref"]["pattern"]
    store = prepared.projection._archive
    before = (store.path.read_bytes(), prepared.kernel.path.read_bytes())
    recovered = []
    for ref in handles:
        assert re.fullmatch(pattern, ref), "compiled handles must satisfy the tool's public schema"
        response = handle_evidence_tool(
            "ariadne_zoom",
            {"ref": ref},
            store=store,
            session_id="chosen",
            kernel=prepared.kernel,
            checkpoint_id=prepared.checkpoint_id,
        )
        assert response["ok"]
        while response["kind"] == "node":
            response = handle_evidence_tool(
                "ariadne_zoom",
                {"ref": response["children"][0]["ref"]},
                store=store,
                session_id="chosen",
                kernel=prepared.kernel,
                checkpoint_id=prepared.checkpoint_id,
            )
            assert response["ok"]
        recovered.append(response["page"]["event_ref"])
    assert handles and all(ref in {"event:" + digest(row) for row in active} for ref in recovered)
    assert (store.path.read_bytes(), prepared.kernel.path.read_bytes()) == before
