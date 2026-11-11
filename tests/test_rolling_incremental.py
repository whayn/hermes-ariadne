"""Incremental generation validation (#19 S2): O(delta) cost, proven equivalent.

``validate_generation`` validates only each generation's delta rows plus the
boundary state of its already-validated parent. The oracle below compares that
verdict with the unchanged full-prefix ``_canonical_rows`` on realistic
native-shaped rows (tool calls, occurrence uids, Codex reasoning/message items),
including violations that span the parent/delta boundary.

``incremental_verdict`` MODELS the composition (``_canonical_rows`` per chunk +
``_extend_prefix``); it is not the shipped code path. The ``E2E`` cases exercise
the real ``validate_generation`` on forged generation bytes over a real store.
"""

import copy
import json

import pytest
from test_rolling_journal_validation import chain
from test_rolling_selection import POLICY, exchange
from test_rolling_validation_memo import loader

from hermes_ariadne import prepared, rolling
from hermes_ariadne.store import EventStore

SETTINGS = rolling.RollingSettings()
REQUEST = b'[{"content":"summarize","role":"user"}]'
ROUTE = ("codex_backend", "gpt-test")


# --- cost: every row is canonicalized once, not once per descendant generation ---


@pytest.mark.parametrize("count", [1, 3, 6])
def test_each_row_is_canonicalized_once_across_a_chain(tmp_path, monkeypatch, count):
    store, settings, views, *_ = chain(tmp_path, count)
    head = views[-1]
    canonicalized, comparisons = [], []
    original, original_comparison = rolling._canonical_rows, prepared._comparison

    def spy(source, **kwargs):
        canonicalized.append(len(source))
        return original(source, **kwargs)

    def spy_comparison(*args, **kwargs):
        comparisons.append(1)
        return original_comparison(*args, **kwargs)

    monkeypatch.setattr(rolling, "_canonical_rows", spy)
    monkeypatch.setattr(prepared, "_comparison", spy_comparison)
    checked = rolling.validate_generation(
        head.snapshot_bytes, store, "s", settings, parent_loader=loader(views)
    )
    assert checked == head
    assert sum(canonicalized) == len(head.event_ids)  # was quadratic in generations
    assert len(comparisons) <= 2 * len(head.event_ids)


def test_revalidate_view_still_verifies_every_chain_row_once(tmp_path, monkeypatch):
    store, settings, views, *_ = chain(tmp_path, 6)
    head = views[-1]
    canonicalized = []
    original = rolling._canonical_rows
    monkeypatch.setattr(
        rolling,
        "_canonical_rows",
        lambda source, **kw: canonicalized.append(len(source)) or original(source, **kw),
    )
    assert rolling.revalidate_view(head, store, "s", settings, parent_loader=loader(views)) == head
    assert sum(canonicalized) == len(head.event_ids)


# --- differential oracle: incremental verdict == full-prefix verdict ---


def full_verdict(chunks, settings):
    rows = copy.deepcopy([row for chunk in chunks for row in chunk])
    try:
        rolling._canonical_rows(rows, policy=POLICY, settings=settings)
    except ValueError as exc:
        return str(exc)
    return "ok"


def incremental_verdict(chunks, settings):
    """The generation-by-generation path: root validated whole, then delta + boundary."""
    state = rolling._EMPTY_PREFIX
    try:
        for chunk in copy.deepcopy(chunks):
            _, uids, payloads = rolling._canonical_rows(chunk, policy=POLICY, settings=settings)
            state = rolling._extend_prefix(state, chunk, uids, payloads, settings)
    except ValueError as exc:
        return str(exc)
    return "ok"


def reasoning_with_id(row, ident):
    items = json.loads(row["codex_reasoning_items"])
    items[0]["id"] = ident
    row["codex_reasoning_items"] = json.dumps(items)


def set_occurrence(call_row, result_row, call_id, occurrence):
    call_row["tool_calls"][0].update(id=call_id, call_id=call_id)
    call_row["tool_call_uids"] = json.dumps({call_id: occurrence})
    result_row["tool_call_id"] = call_id
    result_row["tool_call_uid"] = occurrence


def case(name, expect, build, **limits):
    return pytest.param(expect, build, limits, id=name)


def dup_message_uid_first(a, b, c):
    c[0]["message_uid"] = a[0]["message_uid"]


def dup_message_uid_middle(a, b, c):
    c[3]["message_uid"] = b[1]["message_uid"]


def dup_reasoning_owner(a, b, c):
    reasoning_with_id(c[1], "rs_aowner0")


def dup_reasoning_answer(a, b, c):
    reasoning_with_id(c[3], "rs_banswer2")


def dup_occurrence(a, b, c):
    set_occurrence(c[1], c[2], "call_c", "occ_a")


def reused_call_id(a, b, c):  # a fresh occurrence under a previously used call id
    set_occurrence(c[1], c[2], "call_a", "occ_c")


def reused_native_ids(a, b, c):  # response item / message item ids are not cross-row identity
    c[1]["tool_calls"][0]["response_item_id"] = "fc_native_a"
    items = json.loads(c[3]["codex_message_items"])
    items[0]["id"] = "msg_a"
    c[3]["codex_message_items"] = json.dumps(items)


def split_tool_result_into_parent_call(a, b, c):
    c[2]["tool_call_id"] = "call_a"  # result for a call from the parent, call dropped here
    del c[1]


def tool_first(a, b, c):
    c[:] = [c[2], c[3]]


def assistant_first(a, b, c):
    c[:] = [c[3]]


def no_trailing_assistant(a, b, c):
    c[:] = c[:3]


def bad_role(chunk):
    chunk[0]["role"] = "system"


def bad_companion(chunk):
    chunk[3]["reasoning"] = 5


def message_text_mismatch(chunk):
    chunk[3]["content"] = chunk[3]["content"] + "!"


def restricted_data_use(chunk):
    chunk[0]["display_metadata"] = json.dumps({"data_use": ["restricted"]})


def orphan_tool(chunk):
    del chunk[1]


def in_delta(mutate):
    return lambda a, b, c: mutate(c)


def in_parent(mutate):
    return lambda a, b, c: mutate(a)


def in_middle(mutate):
    return lambda a, b, c: mutate(b)


CASES = [
    # accepted
    case("valid", "ok", lambda a, b, c: None),
    case("reused-call-id-across-exchanges", "ok", reused_call_id),
    case("reused-native-item-ids-across-exchanges", "ok", reused_native_ids),
    case("bound-exact-length", "ok", lambda a, b, c: None, max_prefix_events=12),
    # violations spanning the boundary
    case("dup-message-uid-first", "reject", dup_message_uid_first),
    case("dup-message-uid-middle", "reject", dup_message_uid_middle),
    case("dup-reasoning-id-owner", "reject", dup_reasoning_owner),
    case("dup-reasoning-id-answer", "reject", dup_reasoning_answer),
    case("dup-occurrence-uid", "reject", dup_occurrence),
    case("tool-result-for-parent-call", "reject", split_tool_result_into_parent_call),
    case("delta-starts-with-tool", "reject", tool_first),
    case("delta-starts-with-assistant", "reject", assistant_first),
    case("beyond-max-prefix-events", "reject", lambda a, b, c: None, max_prefix_events=11),
    # entirely inside the delta
    case("delta-bad-role", "reject", in_delta(bad_role)),
    case("delta-bad-companion", "reject", in_delta(bad_companion)),
    case("delta-text-mismatch", "reject", in_delta(message_text_mismatch)),
    case("delta-restricted-data-use", "reject", in_delta(restricted_data_use)),
    case("delta-orphan-tool", "reject", in_delta(orphan_tool)),
    case("delta-no-trailing-assistant", "reject", no_trailing_assistant),
    # entirely inside an earlier part (the parent chain)
    case("root-bad-role", "reject", in_parent(bad_role)),
    case("root-bad-companion", "reject", in_parent(bad_companion)),
    case("root-text-mismatch", "reject", in_parent(message_text_mismatch)),
    case("root-restricted-data-use", "reject", in_parent(restricted_data_use)),
    case("root-orphan-tool", "reject", in_parent(orphan_tool)),
    case("middle-bad-companion", "reject", in_middle(bad_companion)),
    case("middle-restricted-data-use", "reject", in_middle(restricted_data_use)),
]

SPLITS = {
    "a|b|c": lambda a, b, c: [a, b, c],
    "ab|c": lambda a, b, c: [a + b, c],
    "a|bc": lambda a, b, c: [a, b + c],
}


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("expect,build,limits", CASES)
def test_incremental_verdict_equals_full_prefix_verdict(expect, build, limits, split):
    settings = rolling.RollingSettings(**limits)
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    build(a, b, c)
    chunks = SPLITS[split](a, b, c)
    full = full_verdict(chunks, settings)
    incremental = incremental_verdict(chunks, settings)
    # The corpus must really exercise the violation (not vacuously pass).
    assert (full == "ok") == (expect == "ok"), full
    assert (incremental == "ok") == (full == "ok"), (full, incremental)


@pytest.mark.parametrize(
    "build,message",
    [
        (dup_message_uid_first, "missing or duplicate logical identity"),
        (dup_reasoning_owner, "duplicate retained reasoning identity"),
        (dup_occurrence, "occurrence identities must be unique"),
        (split_tool_result_into_parent_call, "orphan, duplicate or mismatched tool result"),
    ],
)
def test_boundary_rejections_keep_the_audited_messages(build, message):
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    build(a, b, c)
    assert full_verdict([a + b, c], SETTINGS) == message
    assert incremental_verdict([a + b, c], SETTINGS) == message


def test_state_is_built_from_audited_outputs_only():
    a = exchange("a")
    _, uids, payloads = rolling._canonical_rows(copy.deepcopy(a), policy=POLICY, settings=SETTINGS)
    state = rolling._extend_prefix(rolling._EMPTY_PREFIX, a, uids, payloads, SETTINGS)
    assert state.length == 4
    assert state.message_uids == frozenset(row["message_uid"] for row in a)
    assert state.reasoning_ids == {"rs_aowner0"} | {f"rs_aanswer{i}" for i in range(4)}
    assert state.occurrence_uids == {"occ_a"}


# --- end to end: the code that actually runs (forged generation bytes + EventStore) ---


def child_generation(parent, parent_rows, delta, settings):
    """Self-consistent bytes naming ``parent_rows + delta``; the delta may be invalid."""
    rows = parent_rows + delta
    limit = settings.max_checkpoint_bytes
    try:
        readable = rolling._summary_input(
            "parent", rolling.readable_delta(copy.deepcopy(delta), policy=POLICY), settings
        )
    except ValueError:
        readable = b'{"unusable":"delta"}'
    child = rolling.Target(
        session_id="s",
        settings_digest=settings.digest,
        parent_generation=parent.generation_id,
        parent_summary_sha256=parent.summary_sha256,
        event_ids=tuple(rolling.sha256(rolling.encode_json(row, limit)) for row in rows),
        message_uids=tuple(row["message_uid"] for row in rows),
        delta_start=len(parent_rows),
        delta_stop=len(rows),
        readable_input=readable,
        request_bytes=REQUEST,
        route_labels=ROUTE,
    )
    return rolling.generation_bytes(child, "child", sequence=2, settings=settings)


def forge(tmp_path, parent_chunks, delta, settings):
    """Valid parent view, then bytes for a child whose delta is ``delta`` (maybe invalid)."""
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    parent_rows = [row for chunk in parent_chunks for row in chunk]
    store.ingest("s", parent_rows + delta, origin="canonical")
    parent_target = rolling.Target.create(
        copy.deepcopy(parent_rows),
        session_id="s",
        settings=settings,
        policy=POLICY,
        parent=None,
        request_bytes=REQUEST,
        route_labels=ROUTE,
    )
    parent_bytes = rolling.generation_bytes(parent_target, "parent", sequence=1, settings=settings)
    parent = rolling.validate_generation(
        parent_bytes, store, "s", settings, policy=POLICY, parent_loader=None
    )
    child_bytes = child_generation(parent, parent_rows, delta, settings)
    return store, parent, parent_bytes, child_bytes


def validate_child(store, settings, parent, parent_bytes, child_bytes):
    return rolling.validate_generation(
        child_bytes,
        store,
        "s",
        settings,
        policy=POLICY,
        parent_loader=lambda digest: {parent.generation_id: parent_bytes}[digest],
    )


E2E = [
    ("valid", "ok", lambda a, b, c: None, {}, None),
    ("reused-call-id", "ok", reused_call_id, {}, None),
    ("reused-native-ids", "ok", reused_native_ids, {}, None),
    # The generation header already rejects duplicate uids before any row is read.
    ("dup-message-uid", "reject", dup_message_uid_first, {}, "duplicate generation source"),
    ("dup-reasoning-id", "reject", dup_reasoning_owner, {}, None),
    ("dup-reasoning-id-answer", "reject", dup_reasoning_answer, {}, None),
    ("dup-occurrence-uid", "reject", dup_occurrence, {}, None),
    ("tool-result-for-parent-call", "reject", split_tool_result_into_parent_call, {}, None),
    # Alternation is reported as a non-user first row of the standalone delta.
    ("delta-starts-with-assistant", "reject", assistant_first, {}, "whole completed exchanges"),
    ("delta-bad-companion", "reject", in_delta(bad_companion), {}, None),
    ("delta-text-mismatch", "reject", in_delta(message_text_mismatch), {}, None),
    ("delta-restricted-data-use", "reject", in_delta(restricted_data_use), {}, None),
    # The header bound also rejects this one first; the state-level bound is tested above.
    (
        "beyond-max-prefix-events",
        "reject",
        lambda a, b, c: None,
        {"max_prefix_events": 11},
        "source index",
    ),
]


@pytest.mark.parametrize("name,expect,build,limits,header", E2E, ids=[e[0] for e in E2E])
def test_validate_generation_matches_full_prefix_verdict(
    tmp_path, name, expect, build, limits, header
):
    settings = rolling.RollingSettings(**limits)
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    build(a, b, c)
    parent_chunks = [a, b]
    full = full_verdict([*parent_chunks, c], settings)
    assert (full == "ok") == (expect == "ok"), full
    store, parent, parent_bytes, child_bytes = forge(tmp_path, parent_chunks, c, settings)
    if expect == "ok":
        view = validate_child(store, settings, parent, parent_bytes, child_bytes)
        assert view.sequence == 2 and view.delta_start == len(a) + len(b)
    else:
        with pytest.raises(ValueError) as caught:
            validate_child(store, settings, parent, parent_bytes, child_bytes)
        expected = header or full
        assert expected in str(caught.value), (str(caught.value), full)


def test_delta_rows_still_get_the_per_row_hash_and_uid_check(tmp_path):
    settings = SETTINGS
    a, b = exchange("a"), exchange("b")
    store, parent, parent_bytes, child_bytes = forge(tmp_path, [a], b, settings)
    assert validate_child(store, settings, parent, parent_bytes, child_bytes).sequence == 2
    # A forged generation naming the right events with a different uid for a delta row.
    value = json.loads(child_bytes)
    value["message_uids"][len(a)] = "forged-uid"
    forged = rolling.encode_json(value, settings.max_checkpoint_bytes)
    with pytest.raises(ValueError, match="generation source unavailable or changed"):
        validate_child(store, settings, parent, parent_bytes, forged)
    # A poisoned cached copy of a delta row fails the hash re-verification.
    memo = rolling._ValidationMemo(store, "s", POLICY)
    delta_event = json.loads(child_bytes)["event_ids"][len(a)]
    memo.rows[delta_event] = {**store.get_event(delta_event, session_id="s"), "content": "x"}
    with pytest.raises(ValueError, match="generation source unavailable or changed"):
        rolling.validate_generation(
            child_bytes,
            store,
            "s",
            settings,
            policy=POLICY,
            parent_loader=lambda digest: parent_bytes,
            _memo=memo,
        )


def test_parent_state_comes_from_the_validated_parent_not_an_accepted_lookalike(tmp_path):
    """An accepted entry the memo cannot tie to a validated state is revalidated, not trusted."""
    settings = SETTINGS
    a, b = exchange("a"), exchange("b")
    dup = copy.deepcopy(b)
    reasoning_with_id(dup[1], "rs_aowner0")  # collides with the parent's reasoning id
    store, parent, parent_bytes, child_bytes = forge(tmp_path, [a], dup, settings)
    memo = rolling._ValidationMemo(store, "s", POLICY)
    memo.accepted[parent.generation_id] = parent  # no validated state recorded for it
    with pytest.raises(ValueError, match="duplicate retained reasoning identity"):
        rolling.validate_generation(
            child_bytes,
            store,
            "s",
            settings,
            policy=POLICY,
            parent_loader=lambda digest: parent_bytes,
            _memo=memo,
        )


def test_poisoned_delta_row_with_a_foreign_session_is_rejected(tmp_path):
    """A memo row that hashes to its event id still must belong to this session."""
    settings = SETTINGS
    a, b = exchange("a"), exchange("b")
    store, parent, parent_bytes, _ = forge(tmp_path, [a], b, settings)
    foreign = copy.deepcopy(b)
    foreign[0]["session_id"] = "other"
    child_bytes = child_generation(parent, a, foreign, settings)
    memo = rolling._ValidationMemo(store, "s", POLICY)
    for event, row in zip(json.loads(child_bytes)["event_ids"][len(a) :], foreign):
        memo.rows[event] = row  # consistent hash, uid and content: only the session differs
    with pytest.raises(ValueError, match="foreign generation source"):
        rolling.validate_generation(
            child_bytes,
            store,
            "s",
            settings,
            policy=POLICY,
            parent_loader=lambda digest: parent_bytes,
            _memo=memo,
        )


def forge_chain(tmp_path, chunks, settings):
    """Self-consistent generation bytes for every prefix of ``chunks``; rows may be invalid."""
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    store.ingest("s", [row for chunk in chunks for row in chunk], origin="canonical")
    limit, rows, generations, parent, parent_text = (
        settings.max_checkpoint_bytes,
        [],
        [],
        None,
        None,
    )
    for sequence, chunk in enumerate(chunks, 1):
        start, rows = len(rows), rows + chunk
        try:
            delta = rolling.readable_delta(copy.deepcopy(chunk), policy=POLICY)
        except ValueError:
            delta = b'{"unusable":"delta"}'
        text = f"summary {sequence}"
        target = rolling.Target(
            session_id="s",
            settings_digest=settings.digest,
            parent_generation=None if parent is None else rolling.sha256(parent),
            parent_summary_sha256=None
            if parent_text is None
            else rolling.sha256(parent_text.encode()),
            event_ids=tuple(rolling.sha256(rolling.encode_json(row, limit)) for row in rows),
            message_uids=tuple(row["message_uid"] for row in rows),
            delta_start=start,
            delta_stop=len(rows),
            readable_input=rolling._summary_input(parent_text, delta, settings),
            request_bytes=REQUEST,
            route_labels=ROUTE,
        )
        parent = rolling.generation_bytes(target, text, sequence=sequence, settings=settings)
        parent_text = text
        generations.append(parent)
    return store, generations


@pytest.mark.parametrize("mutate", [bad_companion, restricted_data_use, orphan_tool])
def test_invalid_middle_generation_of_three_rejects_the_head(tmp_path, mutate):
    settings = SETTINGS
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    mutate(b)
    assert full_verdict([a, b, c], settings) != "ok"
    store, generations = forge_chain(tmp_path, [a, b, c], settings)
    by_id = {rolling.sha256(data): data for data in generations}
    with pytest.raises(ValueError) as caught:
        rolling.validate_generation(
            generations[2], store, "s", settings, policy=POLICY, parent_loader=by_id.__getitem__
        )
    assert full_verdict([a, b], settings) in str(caught.value)


def test_valid_three_generation_chain_validates(tmp_path):
    settings = SETTINGS
    store, generations = forge_chain(
        tmp_path, [exchange("a"), exchange("b"), exchange("c")], settings
    )
    by_id = {rolling.sha256(data): data for data in generations}
    view = rolling.validate_generation(
        generations[2], store, "s", settings, policy=POLICY, parent_loader=by_id.__getitem__
    )
    assert view.sequence == 3 and len(view.event_ids) == 12
