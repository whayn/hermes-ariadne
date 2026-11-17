"""Frozen b7b432a full validators; helpers/codecs remain unchanged.

Do not route recursion or target validation through the optimized functions.
"""

from dataclasses import fields

from hermes_ariadne import rolling as r
from hermes_ariadne.rolling_delta import VERSION, _segments


def validate_generation(
    snapshot_bytes,
    store,
    session_id,
    settings,
    *,
    policy=None,
    parent_loader=None,
    guard=None,
    _depth=0,
    _validated=None,
    _memo=None,
):
    if _depth >= settings.max_logical_jobs:
        raise ValueError("generation lineage exceeds campaign bound")
    r._admit_json_bytes(snapshot_bytes, settings.max_checkpoint_bytes)
    digest = r.sha256(snapshot_bytes)
    memo = _memo if _memo is not None else r._ValidationMemo(store, session_id, policy)
    if (
        memo.store is not store
        or memo.session_id != session_id
        or memo.policy is not policy
        or memo.settings_digest not in (None, settings.digest)
    ):
        raise ValueError("validation memo scope mismatch")
    memo.settings_digest = settings.digest

    def load_parent(parent_id):
        r.require_digest(parent_id)
        if memo.bound is not None and parent_id in memo.bound:
            hit = memo.accepted.get(parent_id)
            recorded = memo.prefix_states.get(parent_id)
            if hit is not None and recorded is not None and recorded[0] is hit:
                if _depth + 1 + hit.sequence > settings.max_logical_jobs:
                    raise ValueError("generation lineage exceeds campaign bound")
                return hit
        if parent_loader is None:
            raise ValueError("parent generation unavailable")
        if guard:
            guard()
        data = parent_loader(parent_id)
        if guard:
            guard()
        if r.sha256(data) != parent_id:
            raise ValueError("generation parent lineage mismatch")
        parent = validate_generation(
            data,
            store,
            session_id,
            settings,
            policy=policy,
            parent_loader=parent_loader,
            guard=guard,
            _depth=_depth + 1,
            _validated=_validated,
            _memo=memo,
        )
        if memo.bound is not None:
            memo.bound.add(parent_id)
        return parent

    accepted = memo.accepted.get(digest)
    if accepted is not None:
        if (
            accepted.snapshot_bytes != snapshot_bytes
            or accepted.settings_digest != settings.digest
            or _depth + accepted.sequence > settings.max_logical_jobs
        ):
            raise ValueError("rolling generation identity mismatch")
        if accepted.parent_generation is not None:
            load_parent(accepted.parent_generation)
        return accepted

    known = memo.known_generation(digest)
    if known is not None:
        value = dict(known.fields)
        parent, state = None, memo.empty_prefix()
        if value["parent_generation"] is not None:
            parent = load_parent(value["parent_generation"])
            if (
                parent.summary_sha256 != value["parent_summary_sha256"]
                or parent.frontier_count != value["delta_start"]
                or parent.sequence + 1 != value["sequence"]
            ):
                raise ValueError("generation parent lineage mismatch")
            state = memo.prefix_states[parent.generation_id][1]
        events, uids = value["event_ids"].delta, value["message_uids"].delta
        memo.require_sources(events, guard)
        r._check_known_extension(state, len(events), known.delta, settings)
        coverage = _segments(parent, events, uids)
        if (
            len(coverage[0]) != value["frontier_count"]
            or coverage[0].digest != value["prefix_sha256"]
        ):
            raise ValueError("prefix hash mismatch")
        value.update(event_ids=coverage[0], message_uids=coverage[1])
        view = r.RollingView(snapshot_bytes, digest, **value)
        r._record_identities(memo.identities, digest, known.delta)
        memo.prefix_states[digest] = (
            view,
            r._PrefixState(len(coverage[0]), state.lineage | {digest}, memo.identities),
        )
        memo.deltas[digest] = known.delta
        if _validated is not None:
            _validated.append(view)
        return view

    value = r.decode_json(snapshot_bytes, settings.max_checkpoint_bytes)
    keys = {f.name for f in fields(r.RollingView)} - {
        "snapshot_bytes",
        "generation_id",
        "event_ids",
        "message_uids",
    }
    headers = {"format", "version", "policy_id", "readable_projection_id", "prompt_version"}
    if type(value) is not dict or set(value) != keys | headers | {
        "delta_event_ids",
        "delta_message_uids",
    }:
        raise ValueError("invalid delta generation schema")
    if (
        value["format"] != r.FORMAT
        or type(value["version"]) is not int
        or value["version"] != VERSION
        or value["policy_id"] != r.POLICY_ID
        or value["readable_projection_id"] != r.PROJECTION_ID
        or value["prompt_version"] != r.PROMPT_VERSION
        or value["session_id"] != session_id
        or value["settings_digest"] != settings.digest
    ):
        raise ValueError("rolling generation identity mismatch")
    for key in ("sequence", "frontier_count", "delta_stop"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("positive generation counts required")
    start, stop = value["delta_start"], value["delta_stop"]
    events, uids = value["delta_event_ids"], value["delta_message_uids"]
    if (
        type(start) is not int
        or not 0 <= start < stop == value["frontier_count"]
        or stop > settings.max_prefix_events
        or type(events) is not list
        or type(uids) is not list
        or len(events) != stop - start
        or len(uids) != len(events)
    ):
        raise ValueError("invalid delta generation source index")
    for event in events:
        r.require_digest(event)
    for uid in uids:
        r.bounded_text(uid, settings.max_checkpoint_bytes)
    for key in ("summary_sha256", "summary_input_sha256", "prefix_sha256", "job_id"):
        r.require_digest(value[key])
    r.bounded_text(value["summary_text"], settings.max_summary_bytes)
    if r.sha256(value["summary_text"].encode()) != value["summary_sha256"]:
        raise ValueError("summary hash mismatch")
    parent, state = None, memo.empty_prefix()
    if value["parent_generation"] is not None:
        parent = load_parent(value["parent_generation"])
        if (
            value["parent_summary_sha256"] != parent.summary_sha256
            or start != parent.frontier_count
            or value["sequence"] != parent.sequence + 1
        ):
            raise ValueError("generation parent lineage mismatch")
        recorded = memo.prefix_states.get(parent.generation_id)
        if recorded is None or recorded[0] is not parent or not recorded[1].extends(parent):
            raise ValueError("generation parent state unavailable")
        state = recorded[1]
    elif start != 0 or value["sequence"] != 1 or value["parent_summary_sha256"] is not None:
        raise ValueError("invalid root generation")
    coverage = _segments(parent, events, uids)
    if coverage[0].digest != value["prefix_sha256"]:
        raise ValueError("prefix hash mismatch")
    source = []
    for event, uid in zip(events, uids):
        if guard:
            guard()
        row = memo.get_event(event)
        if guard:
            guard()
        if (
            row is None
            or row.get("message_uid") != uid
            or row.get("session_id", session_id) != session_id
            or r.sha256(r.encode_json(row, settings.max_checkpoint_bytes)) != event
        ):
            raise ValueError("generation source unavailable or changed")
        source.append(row)
    delta_events, delta_uids, payloads = r._canonical_rows(source, policy=policy, settings=settings)
    if delta_events != tuple(events) or delta_uids != tuple(uids):
        raise ValueError("generation source unavailable or changed")
    identities = r._check_prefix_extension(state, source, delta_uids, payloads, settings)
    readable = r._render_delta(source, delta_events, payloads, settings)
    if (
        r.sha256(
            r._summary_input(None if parent is None else parent.summary_text, readable, settings)
        )
        != value["summary_input_sha256"]
    ):
        raise ValueError("summary input hash mismatch")
    if r.encode_json(value, settings.max_checkpoint_bytes) != snapshot_bytes:
        raise ValueError("noncanonical generation bytes")
    for key in headers | {"delta_event_ids", "delta_message_uids"}:
        del value[key]
    view = r.RollingView(
        snapshot_bytes, digest, event_ids=coverage[0], message_uids=coverage[1], **value
    )
    r._record_identities(memo.identities, digest, identities)
    memo.prefix_states[digest] = (
        view,
        r._PrefixState(stop, state.lineage | {digest}, memo.identities),
    )
    if memo.cache is not None:
        memo.deltas[digest] = identities
    if _validated is not None:
        _validated.append(view)
    return view


def validate_target(target, parent, store, session_id, settings, *, policy, memo, guard):
    if target.session_id != session_id or target.settings_digest != settings.digest:
        raise ValueError("frozen target scope mismatch")
    target.bind(parent, settings)
    r._require_request_and_route(
        target.request_bytes, target.route_labels, settings=settings, policy=policy
    )
    known = memo.known_target(target.target_id)
    if known is not None and known.delta is not None and known.target == target:
        memo.require_sources(target.delta_event_ids, guard, "foreign canonical session")
        state = (
            memo.empty_prefix() if parent is None else memo.prefix_states[parent.generation_id][1]
        )
        r._check_known_extension(state, len(target.delta_event_ids), known.delta, settings)
        memo.record_target(target, known.delta)
        return target
    source = []
    for event in target.delta_event_ids:
        guard()
        row = memo.get_event(event)
        guard()
        if row is None or row.get("session_id", session_id) != session_id:
            raise ValueError("foreign canonical session")
        source.append(row)
    events, uids, payloads = r._canonical_rows(source, policy=policy, settings=settings)
    if events != target.delta_event_ids or uids != target.delta_message_uids:
        raise ValueError("frozen target source or identity mismatch")
    state = memo.empty_prefix() if parent is None else memo.prefix_states[parent.generation_id][1]
    identities = r._check_prefix_extension(state, source, uids, payloads, settings)
    readable = r._render_delta(source, events, payloads, settings)
    if target.readable_input != r._summary_input(
        None if parent is None else parent.summary_text, readable, settings
    ):
        raise ValueError("frozen target source or identity mismatch")
    target.to_bytes(settings)
    memo.record_target(target, identities)
    return target
