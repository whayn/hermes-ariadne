"""Opt-in rolling representation, validated views and completed-turn coverage.

Only derived compactor text omits supported replay state. Canonical source rows
are never rewritten. Completion validates the real canonical/restored transcript,
not assembled API messages; selection separately checks all three real views in
rolling_selection. Neither module schedules work or changes the host runtime.
"""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, fields

from .prepared import (
    ReasoningReplayPolicy,
    _comparison_payloads,
    _occurrences,
    _reasoning_resource_preflight,
    _require_reasoning_policy,
    _uids,
)
from .retrieval import _require_supported_data_use

FORMAT = "hermes-ariadne-rolling"
VERSION = 1
POLICY_ID = "aged-codex-retirement-v1"
PROJECTION_ID = "conversation-text-tools-v1"
PROMPT_VERSION = "rolling-summary-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_digest(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid SHA-256 identity")
    return value


def bounded_text(value, limit, *, nonempty=True):
    if type(value) is not str or (nonempty and not value) or len(value) > limit:
        raise ValueError("invalid or oversized text")
    size = 0
    for start in range(0, len(value), 4096):
        size += len(value[start : start + 4096].encode("utf-8"))
        if size > limit:
            raise ValueError("text exceeds UTF-8 bound")
    return value


def encode_json(value, limit):
    """Canonical strict UTF-8 JSON, bounded BEFORE any full serialization.

    Scalar strings are escaped in small chunks; containers are traversed with a
    depth/cycle gate and a running byte budget. No coerced keys or surrogate text.
    """
    output = bytearray()
    active = set()

    def emit(data):
        if len(output) + len(data) > limit:
            raise ValueError("JSON exceeds byte bound")
        output.extend(data)

    def string(text):
        if len(text) > limit - len(output):
            raise ValueError("JSON exceeds byte bound")
        emit(b'"')
        for start in range(0, len(text), 4096):
            emit(json.dumps(text[start : start + 4096], ensure_ascii=False)[1:-1].encode("utf-8"))
        emit(b'"')

    def visit(item, depth=0):
        if depth > 64:
            raise ValueError("JSON nesting exceeds bound")
        kind = type(item)
        if kind is str:
            string(item)
        elif kind in (dict, list):
            if id(item) in active or len(item) > limit - len(output):
                raise ValueError("cyclic or oversized JSON")
            active.add(id(item))
            if kind is dict:
                if any(type(k) is not str for k in item):
                    raise ValueError("non-string JSON key")
                # Bound key lengths before sorting; no large secondary allocation.
                if sum(len(k) + 3 for k in item) > limit - len(output):
                    raise ValueError("JSON exceeds byte bound")
                emit(b"{")
                for index, key in enumerate(sorted(item)):
                    if index:
                        emit(b",")
                    string(key)
                    emit(b":")
                    visit(item[key], depth + 1)
                emit(b"}")
            else:
                emit(b"[")
                for index, child in enumerate(item):
                    if index:
                        emit(b",")
                    visit(child, depth + 1)
                emit(b"]")
            active.remove(id(item))
        elif item is None:
            emit(b"null")
        elif kind is bool:
            emit(b"true" if item else b"false")
        elif kind is int:
            if item.bit_length() > min(limit * 4, 4096):
                raise ValueError("oversized JSON integer")
            emit(str(item).encode("ascii"))
        elif kind is float and math.isfinite(item):
            emit(json.dumps(item, allow_nan=False).encode("ascii"))
        else:
            raise ValueError("unsupported or nonfinite JSON value")

    visit(value)
    return bytes(output)


def decode_json(data, limit):
    if type(data) is not bytes or len(data) > limit:
        raise ValueError("invalid or oversized JSON bytes")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
        encode_json(value, limit)  # strict Unicode, finite overflowing floats, depth
        return value
    except (RecursionError, UnicodeError, OverflowError) as exc:
        raise ValueError("invalid strict JSON") from exc


@dataclass(frozen=True)
class RollingSettings:
    max_logical_jobs: int = 64
    max_request_bytes: int = 65536
    max_total_request_bytes: int = 2 * 1024 * 1024
    max_readable_delta_bytes: int = 32768
    max_delta_exchanges: int = 4
    trigger_bytes: int = 4096
    max_summary_bytes: int = 4096
    max_navigation_bytes: int = 8192
    max_candidate_bytes: int = 65536
    reserve_tokens: int = 4096
    max_prefix_events: int = 100000
    max_checkpoint_bytes: int = 16 * 1024 * 1024
    max_tokens: int = 2048
    timeout: int = 60

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 0 < value <= 2**31 - 1:
                raise ValueError("rolling limits must be positive bounded integers")

    @property
    def digest(self):
        return sha256(
            encode_json(
                {
                    "format": FORMAT,
                    "version": VERSION,
                    "policy_id": POLICY_ID,
                    "readable_projection_id": PROJECTION_ID,
                    "prompt_version": PROMPT_VERSION,
                    "settings": asdict(self),
                },
                8192,
            )
        )


def _canonical_rows(rows, *, policy, settings):
    if type(rows) is not list or not rows or len(rows) > settings.max_prefix_events:
        raise ValueError("nonempty bounded canonical rows required")
    if policy is not None:
        _require_reasoning_policy(policy)
    policies = []
    event_ids = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("canonical row must be a mapping")
        # Carrier-specific resource admission precedes ANY generic serializer.
        _reasoning_resource_preflight(row)
        data = encode_json(row, settings.max_checkpoint_bytes)
        metadata = row.get("display_metadata")
        for _ in range(2):
            if type(metadata) is str:
                bounded_text(metadata, settings.max_checkpoint_bytes, nonempty=False)
                metadata = decode_json(metadata.encode("utf-8"), settings.max_checkpoint_bytes)
        _require_supported_data_use(row)
        has_reasoning = row.get("codex_reasoning_items") not in (None, "", False, [], {})
        if has_reasoning:
            _require_reasoning_policy(policy)
            for key in ("reasoning", "reasoning_content"):
                if key not in row or (row[key] is not None and type(row[key]) is not str):
                    raise ValueError("invalid canonical reasoning companion")
        policies.append(policy if has_reasoning else None)
        if row.get("role") not in {"user", "assistant", "tool"}:
            raise ValueError("unsupported historical role")
        if row.get("name") is not None:
            bounded_text(row["name"], settings.max_checkpoint_bytes)
        event_ids.append(sha256(data))
    _uids(rows)
    payloads = _comparison_payloads(rows, policies, durable=True)
    for row, payload in zip(rows, payloads):
        items = payload.get("codex_message_items")
        if items and items[0]["content"][0]["text"] != row["content"]:
            raise ValueError("Codex replay requires exact canonical source text")
    _occurrences(rows, durable=True, reasoning_policies=policies)
    if rows[0]["role"] != "user" or rows[-1]["role"] != "assistant" or rows[-1].get("tool_calls"):
        raise ValueError("whole completed exchanges required")
    return tuple(event_ids), tuple(row["message_uid"] for row in rows), payloads


def readable_delta(
    canonical_rows: list[dict],
    *,
    policy: ReasoningReplayPolicy | None,
    settings: RollingSettings | None = None,
) -> bytes:
    """Whitelist exact canonical conversational values, never replay companions.

    ``policy`` is a trusted bound ReasoningReplayPolicy or None. It authorizes
    only the audited Codex codec. It does not bypass the later whole-request
    correspondence gate or establish provider provenance by itself.
    """
    settings = settings or RollingSettings()
    events, _, payloads = _canonical_rows(canonical_rows, policy=policy, settings=settings)
    if sum(row["role"] == "user" for row in canonical_rows) > settings.max_delta_exchanges:
        raise ValueError("too many delta exchanges")
    result = []
    for row, payload, event in zip(canonical_rows, payloads, events):
        item = {"role": row["role"], "content": row["content"], "event_ref": "event:" + event}
        if row.get("name") is not None:
            item["name"] = row["name"]
        if row.get("tool_name") is not None:
            item["tool_name"] = row["tool_name"]
        if payload.get("tool_calls"):
            item["tool_calls"] = [
                {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                for c in payload["tool_calls"]
            ]
        result.append(item)
    return encode_json(result, settings.max_readable_delta_bytes)


def _summary_input(parent_summary, delta, settings):
    return encode_json(
        {
            "previous_summary": parent_summary,
            "delta": decode_json(delta, settings.max_readable_delta_bytes),
        },
        settings.max_request_bytes,
    )


@dataclass(frozen=True)
class Target:
    """Frozen whole-prefix identity; constructing one does not select a frontier."""

    session_id: str
    settings_digest: str
    parent_generation: str | None
    parent_summary_sha256: str | None
    event_ids: tuple[str, ...]
    message_uids: tuple[str, ...]
    delta_start: int
    delta_stop: int
    readable_input: bytes
    request_bytes: bytes
    route_labels: tuple[str, str]

    def __post_init__(self):
        bounded_text(self.session_id, 4096)
        require_digest(self.settings_digest)
        for value in (self.parent_generation, self.parent_summary_sha256):
            if value is not None:
                require_digest(value)
        if (self.parent_generation is None) != (self.parent_summary_sha256 is None):
            raise ValueError("incomplete target parent identity")
        for values in (self.event_ids, self.message_uids, self.route_labels):
            if type(values) is not tuple or any(type(v) is not str for v in values):
                raise ValueError("immutable target tuples required")
        if (
            not self.event_ids
            or len(self.event_ids) != len(self.message_uids)
            or type(self.delta_start) is not int
            or type(self.delta_stop) is not int
            or not 0 <= self.delta_start < self.delta_stop == len(self.event_ids)
            or type(self.readable_input) is not bytes
            or type(self.request_bytes) is not bytes
        ):
            raise ValueError("invalid immutable target bounds or input")
        if len(self.route_labels) != 2:
            raise ValueError("bound route labels required")
        for label in self.route_labels:
            bounded_text(label, 1024)

    @classmethod
    def create(
        cls, canonical_rows, *, session_id, settings, policy, parent, request_bytes, route_labels
    ):
        bounded_text(session_id, 4096)
        if any(
            type(row) is not dict or row.get("session_id", session_id) != session_id
            for row in canonical_rows
        ):
            raise ValueError("foreign canonical session")
        events, uids, _ = _canonical_rows(canonical_rows, policy=policy, settings=settings)
        start = 0 if parent is None else parent.frontier_count
        if parent is not None and (
            parent.session_id != session_id
            or parent.settings_digest != settings.digest
            or events[:start] != parent.event_ids
            or uids[:start] != parent.message_uids
        ):
            raise ValueError("target is not a parent prefix extension")
        if start >= len(events):
            raise ValueError("target must advance")
        delta = readable_delta(canonical_rows[start:], policy=policy, settings=settings)
        request = decode_json(request_bytes, settings.max_request_bytes)
        if (
            type(request) is not list
            or not request
            or encode_json(request, settings.max_request_bytes) != request_bytes
        ):
            raise ValueError("request must be canonical facade-message JSON")
        for message in request:
            if (
                type(message) is not dict
                or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or type(message["content"]) is not str
            ):
                raise ValueError("unsupported compactor request")
        if type(route_labels) is not tuple or len(route_labels) != 2:
            raise ValueError("bound route labels required")
        for label in route_labels:
            bounded_text(label, 1024)
        if policy is not None and route_labels != (policy.issuer_kind, policy.issuer_model):
            raise ValueError("frozen route differs from trusted reasoning binding")
        return cls(
            session_id,
            settings.digest,
            None if parent is None else parent.generation_id,
            None if parent is None else parent.summary_sha256,
            events,
            uids,
            start,
            len(events),
            _summary_input(None if parent is None else parent.summary_text, delta, settings),
            request_bytes,
            route_labels,
        )

    def to_bytes(self, settings):
        if len(self.event_ids) > settings.max_prefix_events:
            raise ValueError("target exceeds prefix bound")
        if (
            len(self.readable_input) > settings.max_request_bytes
            or len(self.request_bytes) > settings.max_request_bytes
        ):
            raise ValueError("target exceeds input bound")
        for key in ("event_ids", "message_uids", "route_labels"):
            if len(getattr(self, key)) > settings.max_prefix_events:
                raise ValueError("target exceeds sequence bound")
        return self._canonical_bytes(settings.max_checkpoint_bytes)

    def _canonical_bytes(self, limit):
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        for key in ("event_ids", "message_uids", "route_labels"):
            data[key] = list(data[key])
        for key in ("readable_input", "request_bytes"):
            data[key] = data[key].decode("utf-8")
        return encode_json(data, limit)

    @classmethod
    def from_bytes(cls, data, settings):
        value = decode_json(data, settings.max_checkpoint_bytes)
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("invalid target schema")
        for key in ("event_ids", "message_uids", "route_labels"):
            if type(value[key]) is not list:
                raise ValueError("invalid target sequence")
            value[key] = tuple(value[key])
        for key in ("readable_input", "request_bytes"):
            if type(value[key]) is not str:
                raise ValueError("invalid target input")
            value[key] = value[key].encode("utf-8")
        result = cls(**value)
        if result.to_bytes(settings) != data:
            raise ValueError("noncanonical target")
        return result

    @property
    def prefix_sha256(self):
        return sha256(encode_json(list(self.event_ids), 16 * 1024 * 1024))

    @property
    def readable_input_digest(self):
        return sha256(self.readable_input)

    @property
    def request_digest(self):
        return sha256(self.request_bytes)

    @property
    def target_id(self):
        # Identity is the canonical serialization only. Campaign bounds were
        # enforced when the target was admitted (create/freeze/from_bytes);
        # reapplying default bounds here rejected legitimately configured targets.
        return sha256(self._canonical_bytes(2**31 - 1))

    @property
    def job_id(self):
        return sha256(
            encode_json([self.target_id, POLICY_ID, PROMPT_VERSION, self.request_digest], 1024)
        )


def validate_target_prefix(
    target: Target,
    active_rows: list[dict],
    *,
    settings: RollingSettings,
    policy: ReasoningReplayPolicy | None,
) -> None:
    """Mandatory future engine seam: an append is allowed, a rebase is not.

    This checks the captured prefix, NOT callback persistence or assembled replay
    correspondence. Slice 2 must establish those independently before freezing.
    """
    if (
        type(active_rows) is not list
        or len(active_rows) < target.delta_stop
        or target.settings_digest != settings.digest
    ):
        raise ValueError("target prefix rewound or settings changed")
    source = active_rows[: target.delta_stop]
    if any(
        type(row) is not dict or row.get("session_id", target.session_id) != target.session_id
        for row in source
    ):
        raise ValueError("foreign active source")
    events, uids, _ = _canonical_rows(source, policy=policy, settings=settings)
    if events != target.event_ids or uids != target.message_uids:
        raise ValueError("target is not the exact active prefix")


@dataclass(frozen=True)
class RollingView:
    snapshot_bytes: bytes
    generation_id: str
    session_id: str
    settings_digest: str
    sequence: int
    parent_generation: str | None
    parent_summary_sha256: str | None
    event_ids: tuple[str, ...]
    message_uids: tuple[str, ...]
    frontier_count: int
    prefix_sha256: str
    delta_start: int
    delta_stop: int
    summary_text: str
    summary_sha256: str
    summary_input_sha256: str
    job_id: str


def validate_summary(response_bytes: bytes, target: Target, settings: RollingSettings) -> str:
    """Application JSON/source binding, NOT a provider terminal/semantic receipt."""
    value = decode_json(response_bytes, settings.max_candidate_bytes)
    if type(value) is not dict or set(value) != {"summary", "covered_prefix_sha256"}:
        raise ValueError("invalid rolling summary schema")
    if value["covered_prefix_sha256"] != target.prefix_sha256:
        raise ValueError("summary coverage echo mismatch")
    return bounded_text(value["summary"], settings.max_summary_bytes)


def generation_bytes(
    target: Target, summary_text: str, *, sequence: int, settings: RollingSettings | None = None
) -> bytes:
    settings = settings or RollingSettings()
    bounded_text(summary_text, settings.max_summary_bytes)
    if type(sequence) is not int or sequence <= 0:
        raise ValueError("positive generation sequence required")
    return encode_json(
        {
            "format": FORMAT,
            "version": VERSION,
            "session_id": target.session_id,
            "policy_id": POLICY_ID,
            "readable_projection_id": PROJECTION_ID,
            "prompt_version": PROMPT_VERSION,
            "settings_digest": target.settings_digest,
            "sequence": sequence,
            "parent_generation": target.parent_generation,
            "parent_summary_sha256": target.parent_summary_sha256,
            "event_ids": list(target.event_ids),
            "message_uids": list(target.message_uids),
            "frontier_count": len(target.event_ids),
            "prefix_sha256": target.prefix_sha256,
            "delta_start": target.delta_start,
            "delta_stop": target.delta_stop,
            "summary_text": summary_text,
            "summary_sha256": sha256(summary_text.encode("utf-8")),
            "summary_input_sha256": target.readable_input_digest,
            "job_id": target.job_id,
        },
        settings.max_checkpoint_bytes,
    )


class _ValidationMemo:
    """Per-operation validation memo; never shared across operations or requests.

    Scoped to one (store, session, reasoning policy); validate_generation rejects
    a memo whose scope differs from the call's. ``rows`` is a snapshot of the
    archive for this operation and only avoids repeated reads: every use still
    re-verifies the row hash, message_uid and session, and callers receive
    independent copies. An archive change during an operation is not seen by it;
    the next operation (a new memo) rereads and detects it.
    ``accepted`` holds views the journal itself already cross-checked (see
    RollingJournal._view); validate_generation only reads it. Keys are the SHA-256
    of the exact bytes just reread for this use, never a parent name.
    """

    def __init__(self, store, session_id, policy):
        self.store, self.session_id, self.policy = store, session_id, policy
        self.rows, self.accepted = {}, {}

    def get_event(self, event_id):
        row = deepcopy(self.rows.get(event_id))
        if row is None:
            row = self.store.get_event(event_id, session_id=self.session_id)
            if row is not None:  # a missing row is never cached
                self.rows[event_id] = deepcopy(row)
        return row


def validate_generation(
    snapshot_bytes: bytes,
    store,
    session_id: str,
    settings: RollingSettings,
    *,
    policy=None,
    parent_loader=None,
    guard=None,
    _depth=0,
    _validated=None,
    _memo=None,
) -> RollingView:
    """Hash the supplied exact bytes once, then validate those same bytes.

    Parent loader returns exact bounded bytes, never an unchecked prebuilt view.
    Optional guard is an actual raising capture callback around archive/parent IO.

    ``_memo`` is journal-internal (RollingJournal._operation); callers outside the
    journal must not pass it. Without one, a call-local memo dedupes archive row
    reads within this call only. An accepted hit returns a view already validated
    and journal-cross-checked in this operation, keyed by the sha256 of these exact
    bytes (and re-checked against bytes, settings digest and the depth bound).
    Archive rows are read once per memo; a memo is bound to this store, session
    and policy object and is rejected otherwise.
    """
    if _depth >= settings.max_logical_jobs:
        raise ValueError("generation lineage exceeds campaign bound")
    value = decode_json(snapshot_bytes, settings.max_checkpoint_bytes)
    generation_id = sha256(snapshot_bytes)
    if _memo is None:  # rows only: dedupe shared prefix reads within this one call
        _memo = _ValidationMemo(store, session_id, policy)
    if _memo.store is not store or _memo.session_id != session_id or _memo.policy is not policy:
        raise ValueError("validation memo scope mismatch")
    accepted = _memo.accepted.get(generation_id)
    if accepted is not None:
        if accepted.snapshot_bytes != snapshot_bytes or accepted.settings_digest != settings.digest:
            raise ValueError("rolling generation identity mismatch")
        # The accepted chain has ``sequence`` generations below this depth.
        if _depth + accepted.sequence > settings.max_logical_jobs:
            raise ValueError("generation lineage exceeds campaign bound")
        return accepted
    keys = {f.name for f in fields(RollingView)} - {"snapshot_bytes", "generation_id"}
    headers = {"format", "version", "policy_id", "readable_projection_id", "prompt_version"}
    if type(value) is not dict or set(value) != keys | headers:
        raise ValueError("invalid rolling generation schema")
    if (
        value["format"] != FORMAT
        or type(value["version"]) is not int
        or value["version"] != VERSION
        or value["policy_id"] != POLICY_ID
        or value["readable_projection_id"] != PROJECTION_ID
        or value["prompt_version"] != PROMPT_VERSION
        or value["session_id"] != session_id
        or value["settings_digest"] != settings.digest
    ):
        raise ValueError("rolling generation identity mismatch")
    for key in ("sequence", "frontier_count", "delta_stop"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("positive generation counts required")
    if type(value["delta_start"]) is not int or value["delta_start"] < 0:
        raise ValueError("invalid delta start")
    events, uids = value["event_ids"], value["message_uids"]
    if (
        type(events) is not list
        or type(uids) is not list
        or not events
        or len(events) > settings.max_prefix_events
        or len(events) != len(uids)
        or len(events) != value["frontier_count"]
        or len(events) != value["delta_stop"]
    ):
        raise ValueError("invalid generation source index")
    for event in events:
        require_digest(event)
    for uid in uids:
        bounded_text(uid, settings.max_checkpoint_bytes)
    if len(set(events)) != len(events) or len(set(uids)) != len(uids):
        raise ValueError("duplicate generation source identity")
    for key in ("summary_sha256", "summary_input_sha256", "prefix_sha256", "job_id"):
        require_digest(value[key])
    bounded_text(value["summary_text"], settings.max_summary_bytes)
    if sha256(value["summary_text"].encode("utf-8")) != value["summary_sha256"]:
        raise ValueError("summary hash mismatch")
    if sha256(encode_json(events, settings.max_checkpoint_bytes)) != value["prefix_sha256"]:
        raise ValueError("prefix hash mismatch")
    parent = None
    if value["parent_generation"] is not None:
        require_digest(value["parent_generation"])
        require_digest(value["parent_summary_sha256"])
        if parent_loader is None:
            raise ValueError("parent generation unavailable")
        if guard:
            guard()
        data = parent_loader(value["parent_generation"])
        if guard:
            guard()
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
            _memo=_memo,
        )
        if (
            parent.generation_id != value["parent_generation"]
            or parent.summary_sha256 != value["parent_summary_sha256"]
            or value["sequence"] != parent.sequence + 1
            or value["delta_start"] != parent.frontier_count
            or tuple(events[: parent.frontier_count]) != parent.event_ids
            or tuple(uids[: parent.frontier_count]) != parent.message_uids
        ):
            raise ValueError("generation parent lineage mismatch")
    elif (
        value["parent_summary_sha256"] is not None
        or value["delta_start"] != 0
        or value["sequence"] != 1
    ):
        raise ValueError("invalid root generation")
    source = []
    for event, uid in zip(events, uids):
        if guard:
            guard()
        row = _memo.get_event(event)
        if guard:
            guard()
        if (
            row is None
            or sha256(encode_json(row, settings.max_checkpoint_bytes)) != event
            or row.get("message_uid") != uid
        ):
            raise ValueError("generation source unavailable or changed")
        if row.get("session_id", session_id) != session_id:
            raise ValueError("foreign generation source")
        source.append(row)
    _canonical_rows(source, policy=policy, settings=settings)
    delta = readable_delta(source[value["delta_start"] :], policy=policy, settings=settings)
    if (
        sha256(_summary_input(None if parent is None else parent.summary_text, delta, settings))
        != value["summary_input_sha256"]
    ):
        raise ValueError("summary input hash mismatch")
    if encode_json(value, settings.max_checkpoint_bytes) != snapshot_bytes:
        raise ValueError("noncanonical generation bytes")
    for key in headers:
        del value[key]
    value["event_ids"], value["message_uids"] = tuple(events), tuple(uids)
    view = RollingView(snapshot_bytes, generation_id, **value)
    if _validated is not None:
        _validated.append(view)
    return view


def revalidate_view(
    view, store, session_id, settings, *, policy=None, parent_loader=None, guard=None
):
    """Recheck exact immutable bytes, all source dependencies and parent lineage.

    Never trust a dataclass's exposed fields instead of its validated snapshot.
    Scope/settings are supplied by the caller, not by model-controlled arguments.
    """
    if type(view) is not RollingView:
        raise ValueError("trusted rolling view required")
    checked = validate_generation(
        view.snapshot_bytes,
        store,
        session_id,
        settings,
        policy=policy,
        parent_loader=parent_loader,
        guard=guard,
    )
    if checked != view:
        raise ValueError("rolling view differs from its exact snapshot")
    return checked


@dataclass(frozen=True)
class Coverage:
    """Full canonical/restored coverage only; no callback-success/dispatch proof."""

    event_ids: tuple[str, ...]
    message_uids: tuple[str, ...]
    final_assistant_uid: str
    completed_exchanges: int


def completed_exchange_starts(payloads, *, allow_current=False):
    """Require a final assistant before EVERY next user boundary.

    Tool-result completeness alone is not completed-response coverage. Inputs
    must already have passed the shared payload and tool occurrence validators.
    """
    starts = [i for i, message in enumerate(payloads) if message["role"] == "user"]
    if not starts or starts[0] != 0:
        raise ValueError("history must start with a user exchange")
    ends = starts[1:] + ([] if allow_current else [len(payloads)])
    for end in ends:
        final = payloads[end - 1]
        if final["role"] != "assistant" or final.get("tool_calls"):
            raise ValueError("completed exchange requires a final assistant")
    return tuple(starts)


def validate_completed_turn(active_rows, observed_messages, *, reasoning_policy, settings=None):
    """Compare the COMPLETE real callback transcript against ALL active rows.

    The native completion caller clones raw conversation messages, not assembled
    API messages (conversation_loop._notify_context_engine_turn_complete). There
    is deliberately no synthetic assembled argument here. The engine must still
    gate success metadata, session scope and archive resolution independently.
    """
    from .prepared import _exact_occurrences, _exact_payloads, _reasoning_permissions

    settings = settings or RollingSettings()
    if (
        type(active_rows) is not list
        or type(observed_messages) is not list
        or not active_rows
        or len(active_rows) != len(observed_messages)
        or any(type(m) is not dict for m in observed_messages)
    ):
        raise ValueError("full completed canonical/observed coverage required")
    events, uids, _ = _canonical_rows(active_rows, policy=reasoning_policy, settings=settings)
    if tuple(_uids(observed_messages)) != uids:
        raise ValueError("completed observation UID coverage mismatch")
    for message in observed_messages:
        _require_supported_data_use(message)
        _reasoning_resource_preflight(message)
        encode_json(message, settings.max_checkpoint_bytes)
    policies = _reasoning_permissions(
        observed_messages,
        active_rows,
        None,
        0,
        reasoning_policy,
        retirement_policy=POLICY_ID,
    )
    payloads = _exact_payloads(observed_messages, active_rows, policies)
    _exact_occurrences(observed_messages, active_rows, policies)
    starts = completed_exchange_starts(payloads)
    return Coverage(events, uids, uids[-1], len(starts))
