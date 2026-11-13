"""Opt-in rolling representation, validated views and completed-turn coverage.

Only derived compactor text omits supported replay state. Canonical source rows
are never rewritten. Completion validates the real canonical/restored transcript,
not assembled API messages; selection separately checks all three real views in
rolling_selection. Neither module schedules work or changes the host runtime.
"""

import hashlib
import json
import math
import threading
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property

from .prepared import (
    ReasoningReplayPolicy,
    _comparison_payloads,
    _identity_value,
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


_HEX = "0123456789abcdef"


def require_digest(value):
    # ``strip`` leaves nothing exactly when every character is a lowercase hex digit.
    if type(value) is not str or len(value) != 64 or value.strip(_HEX):
        raise ValueError("invalid SHA-256 identity")
    return value


def _encode_digest_list(values, limit):
    """``encode_json(list(values), limit)``, without the per-item walk for digests.

    A SHA-256 hex digest needs no JSON escaping, so a nonempty list of them that
    fits the bound encodes as plain concatenation (67 bytes per digest plus the
    closing bracket). Anything else, including an oversized list, goes through
    encode_json itself and so gets exactly its bytes or its error.
    """
    if (
        values
        and 67 * len(values) + 1 <= limit
        and all(type(value) is str and len(value) == 64 for value in values)
        and not "".join(values).strip(_HEX)
    ):
        return ('["' + '","'.join(values) + '"]').encode("ascii")
    return encode_json(list(values), limit)


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


def _admit_json_bytes(data, limit):
    """The admission check for JSON input: exact ``bytes`` within ``limit``."""
    if type(data) is not bytes or len(data) > limit:
        raise ValueError("invalid or oversized JSON bytes")


def decode_json(data, limit):
    return _decode_json(data, limit)[0]


def _decode_json(data, limit):
    """``decode_json`` plus the canonical encoding its strictness check produced.

    The second value is ``encode_json(value, limit)``; comparing it with ``data``
    is exactly the canonical-bytes check a caller would otherwise encode again.
    """
    _admit_json_bytes(data, limit)

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
        # strict Unicode, finite overflowing floats, depth
        return value, encode_json(value, limit)
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
        for spec in fields(self):
            value = getattr(self, spec.name)
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


@dataclass(frozen=True)
class _PrefixState:
    """Cross-row identity state of a validated completed-exchange prefix.

    A valid prefix ends at a completed exchange (last row an assistant without
    tool calls), so no tool call is pending and the next row must be a user row:
    the ONLY cross-row facts a following delta can violate are uniqueness of
    message uids, retained reasoning item ids and tool-call occurrence uids, and
    the total length bound. Everything else ``_canonical_rows`` checks is per row
    or per exchange. The values come from the audited functions' own outputs
    (``_canonical_rows`` uids/payloads, ``_identity_value`` of validated rows).

    Precondition: every link of the chain this state summarizes was canonicalized
    with the SAME reasoning policy object and settings digest as the delta it is
    extended with. ``_ValidationMemo`` is scoped to one policy object and callers
    pass one ``RollingSettings`` (a generation's header must carry its digest), so
    a state is never extended under a different policy or limit.

    Representation: a state does not hold its prefix's identities. It holds the
    ``lineage`` (the introducer ids of every link of its validated chain,
    itself included) and a reference to a shared ``index`` mapping each kind's
    identity to the introducer(s) whose validated delta contained it. The
    prefix contains an identity exactly when one of its introducers is in the
    lineage, so sibling branches extending one parent never see each other's
    identities. Only a completed validation records into the index (see
    ``_record_identities``); checking an extension never writes.

    Memory per index: one entry per distinct (identity, introducing
    generation) pair plus O(G) lineage ids per state, i.e. O(distinct
    identities + G^2) for a linear chain, instead of a whole-prefix union per
    generation.
    """

    length: int
    lineage: frozenset
    # Shared by every state of one memo: excluded from equality and repr.
    index: dict = field(compare=False, repr=False)

    def extends(self, view):
        """Whether this is the validated state of ``view``'s whole prefix."""
        return self.length == len(view.event_ids) and view.generation_id in self.lineage


_IDENTITY_KINDS = (
    ("message", "missing or duplicate logical identity"),
    ("reasoning", "duplicate retained reasoning identity"),
    ("occurrence", "occurrence identities must be unique"),
)


def _new_identity_index():
    return {kind: {} for kind, _ in _IDENTITY_KINDS}


def _introduced_in(introducers, lineage):
    """Whether an index entry (one introducer, or a set of them) meets ``lineage``."""
    if type(introducers) is set:
        return not introducers.isdisjoint(lineage)
    return introducers in lineage


def _delta_identities(rows, uids, payloads):
    """(message uids, reasoning ids, occurrence uids) of already-validated delta rows."""
    reasoning = []
    for payload in payloads:
        items = payload.get("codex_reasoning_items")
        if items:
            reasoning.extend(item["id"] for item in items)
    occurrences = []
    for row in rows:
        mapping = _identity_value(row, "tool_call_uids")
        if mapping is not None:
            occurrences.extend(uid for uid in mapping.values() if uid is not None)
    return frozenset(uids), frozenset(reasoning), frozenset(occurrences)


def _check_prefix_extension(prefix, rows, uids, payloads, settings):
    """Reject ``rows`` (already ``_canonical_rows``-valid) extending ``prefix``.

    Rejects a total length over the bound and any identity the delta shares
    with the prefix, exactly where ``_canonical_rows`` of the whole prefix
    would, and in the same order. Read-only: returns the delta's identity sets
    for a caller that later records them after ITS validation completes. The
    disjointness checks are defence in depth where another layer already
    rejects: a repeated message uid is pre-empted by the generation header's
    duplicate-source check, but a state must not rely on its caller.
    """
    if prefix.length + len(rows) > settings.max_prefix_events:
        raise ValueError("nonempty bounded canonical rows required")
    delta = _delta_identities(rows, uids, payloads)
    _check_delta_identities(prefix, delta)
    return delta


def _check_known_extension(prefix, count, delta, settings):
    """``_check_prefix_extension`` for a delta of ``count`` rows whose identity
    sets ``delta`` a completed validation already derived from the same rows."""
    if prefix.length + count > settings.max_prefix_events:
        raise ValueError("nonempty bounded canonical rows required")
    _check_delta_identities(prefix, delta)


def _check_delta_identities(prefix, delta):
    if not prefix.lineage:
        return  # a root prefix has nothing to collide with
    for (kind, message), identities in zip(_IDENTITY_KINDS, delta):
        table = prefix.index[kind]
        for identity in identities:
            introducers = table.get(identity)
            if introducers is not None and _introduced_in(introducers, prefix.lineage):
                raise ValueError(message)


def _record_identities(index, introducer, delta):
    """Record that ``introducer``'s validated delta contains ``delta``'s identities."""
    for (kind, _), identities in zip(_IDENTITY_KINDS, delta):
        table = index[kind]
        for identity in identities:
            introducers = table.get(identity)
            if introducers is None:
                table[identity] = introducer
            elif type(introducers) is set:
                introducers.add(introducer)
            elif introducers != introducer:
                table[identity] = {introducers, introducer}


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
    return _render_delta(canonical_rows, events, payloads, settings)


def _render_delta(canonical_rows, events, payloads, settings):
    """Whitelist already-canonicalized delta rows (``events``/``payloads`` are theirs)."""
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


def _require_request_and_route(request_bytes, route_labels, *, settings, policy):
    """Request/route admission shared by every Target constructor."""
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
        return cls._assemble(
            session_id=session_id,
            settings=settings,
            policy=policy,
            parent=parent,
            event_ids=events,
            message_uids=uids,
            delta_start=start,
            delta=delta,
            request_bytes=request_bytes,
            route_labels=route_labels,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        session_id,
        settings,
        policy,
        parent,
        event_ids,
        message_uids,
        delta_start,
        delta,
        request_bytes,
        route_labels,
    ):
        """Request/route admission, then the Target; shared by every constructor.

        ``delta`` is the already-rendered readable delta of the rows from
        ``delta_start``. One place builds the Target so ``create`` and
        ``_create_extension`` cannot drift apart field by field.
        """
        _require_request_and_route(request_bytes, route_labels, settings=settings, policy=policy)
        return cls(
            session_id=session_id,
            settings_digest=settings.digest,
            parent_generation=None if parent is None else parent.generation_id,
            parent_summary_sha256=None if parent is None else parent.summary_sha256,
            event_ids=event_ids,
            message_uids=message_uids,
            delta_start=delta_start,
            delta_stop=len(event_ids),
            readable_input=_summary_input(
                None if parent is None else parent.summary_text, delta, settings
            ),
            request_bytes=request_bytes,
            route_labels=route_labels,
        )

    @classmethod
    def _create_extension(
        cls,
        delta_rows,
        *,
        session_id,
        settings,
        policy,
        parent,
        prefix,
        request_bytes,
        route_labels,
        _identities=None,
    ):
        """``Target.create(parent's rows + delta_rows, parent=parent, ...)``, O(delta).

        ``parent`` is a validated RollingView and ``prefix`` the ``_PrefixState`` its
        validation produced (callers must tie both to the same validated object).
        Only ``delta_rows`` are read and canonicalized; the parent's own rows were
        verified when it was validated, and ``prefix`` carries every cross-row fact
        the delta can violate (see ``_PrefixState``). Every other check, and the
        resulting Target, is exactly ``create``'s. An empty ``delta_rows`` is
        rejected as non-advancing.
        """
        bounded_text(session_id, 4096)
        if any(
            type(row) is not dict or row.get("session_id", session_id) != session_id
            for row in delta_rows
        ):
            raise ValueError("foreign canonical session")
        if parent.session_id != session_id or parent.settings_digest != settings.digest:
            raise ValueError("target is not a parent prefix extension")
        if not delta_rows:
            raise ValueError("target must advance")
        delta_events, delta_uids, payloads = _canonical_rows(
            delta_rows, policy=policy, settings=settings
        )
        # Fail closed on a prefix state that is not the parent's (callers tie them).
        if not prefix.extends(parent):
            raise ValueError("target is not a parent prefix extension")
        # Check only: a target never records identity state. ``_identities``
        # (journal-internal) receives the checked delta identity sets.
        identities = _check_prefix_extension(prefix, delta_rows, delta_uids, payloads, settings)
        if _identities is not None:
            _identities.append(identities)
        delta = _render_delta(delta_rows, delta_events, payloads, settings)
        prefix_events = parent.event_ids + delta_events
        prefix_uids = parent.message_uids + delta_uids
        return cls._assemble(
            session_id=session_id,
            settings=settings,
            policy=policy,
            parent=parent,
            event_ids=prefix_events,
            message_uids=prefix_uids,
            delta_start=parent.frontier_count,
            delta=delta,
            request_bytes=request_bytes,
            route_labels=route_labels,
        )

    def to_bytes(self, settings):
        self._check_bounds(settings)
        return self._canonical_bytes(settings.max_checkpoint_bytes)

    def _check_bounds(self, settings):
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

    def _canonical_bytes(self, limit):
        data = {spec.name: getattr(self, spec.name) for spec in fields(self)}
        for key in ("event_ids", "message_uids", "route_labels"):
            data[key] = list(data[key])
        for key in ("readable_input", "request_bytes"):
            data[key] = data[key].decode("utf-8")
        return encode_json(data, limit)

    @classmethod
    def from_bytes(cls, data, settings):
        value, canonical = _decode_json(data, settings.max_checkpoint_bytes)
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
        # ``canonical`` is the encoding of exactly the decoded value. Every field
        # that passed the schema and __post_init__ checks round-trips through
        # _canonical_bytes to that same JSON value (tuples as lists, the two byte
        # fields as their UTF-8 text), so it IS ``to_bytes``'s output and need
        # not be encoded a second time.
        result._check_bounds(settings)
        if canonical != data:
            raise ValueError("noncanonical target")
        # target_id hashes the same canonical serialization: these exact bytes.
        result.__dict__["target_id"] = sha256(data)
        return result

    # Targets are frozen and every field is immutable, so derived identities are
    # computed at most once per object (dataclass equality/hash use fields only).
    @cached_property
    def prefix_sha256(self):
        return sha256(_encode_digest_list(self.event_ids, 16 * 1024 * 1024))

    @cached_property
    def readable_input_digest(self):
        return sha256(self.readable_input)

    @cached_property
    def request_digest(self):
        return sha256(self.request_bytes)

    @cached_property
    def target_id(self):
        # Identity is the canonical serialization only. Campaign bounds were
        # enforced when the target was admitted (create/freeze/from_bytes);
        # reapplying default bounds here rejected legitimately configured targets.
        return sha256(self._canonical_bytes(2**31 - 1))

    @cached_property
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


def _generation_matches(view, target, summary_text):
    """``generation_bytes(target, summary_text, sequence=view.sequence) == snapshot``.

    ``view`` must be a RollingView that validate_generation just produced, so its
    snapshot is exactly the canonical encoding of its fields plus the fixed
    headers, with str/None/exact-int/tuple-of-str values and the same key set
    generation_bytes writes. Canonical encoding is injective on such values, so
    the bytes are equal exactly when every field is; this compares the fields
    instead of re-encoding the whole prefix. ``summary_text`` must already equal
    ``view.summary_text`` (so generation_bytes' own text bound cannot fail).
    """
    return (
        view.session_id == target.session_id
        and view.settings_digest == target.settings_digest
        and view.parent_generation == target.parent_generation
        and view.parent_summary_sha256 == target.parent_summary_sha256
        and view.event_ids == target.event_ids
        and view.message_uids == target.message_uids
        and view.frontier_count == len(target.event_ids)
        and view.prefix_sha256 == target.prefix_sha256
        and view.delta_start == target.delta_start
        and view.delta_stop == target.delta_stop
        and view.summary_text == summary_text
        and view.summary_sha256 == sha256(summary_text.encode("utf-8"))
        and view.summary_input_sha256 == target.readable_input_digest
        and view.job_id == target.job_id
    )


@dataclass(frozen=True)
class _KnownGeneration:
    """Fields of a generation a completed validation accepted, minus its bytes."""

    fields: dict
    delta: tuple  # (message uids, reasoning ids, occurrence uids) of its own delta


@dataclass(frozen=True)
class _KnownTarget:
    target: "Target"
    delta: tuple | None  # identity sets of its validated delta; None: parsed only


class RollingVerificationCache:
    """Content-addressed verification results shared by one owner's journal opens.

    Process-local and never persisted. Every key is the SHA-256 of exact bytes
    (a generation artifact or a frozen target payload), and every one of those
    bytes is still reread and hashed on each use; every archive row is still
    present in the session with bytes hashing to its id (``EventStore``
    rehashes on every read). What is reused is only what is a deterministic
    function of those hashed bytes under one scope: decoding, canonicalization,
    source-row canonicalization, rendering and summary-input hashing. Journal
    state (job states, texts, reservations, recovery), lineage, ancestor
    rereads, identity collisions and bounds are rechecked on every open.

    The scope is (archive path, session, settings digest, reasoning policy);
    binding a different scope discards everything. Entries are recorded only
    after a journal operation completes successfully. ``max_source_ids`` bounds
    the total length of the source-id sequences the entries carry (ids and
    uids counted once per entry; shared tuples are not deduplicated), and
    ``max_entries`` the entry count; the least recently used entries are
    evicted first. Eviction only costs a full revalidation.
    """

    def __init__(self, *, max_entries=512, max_source_ids=2 * 64 * 100000):
        for value in (max_entries, max_source_ids):
            if type(value) is not int or value <= 0:
                raise ValueError("cache bounds must be positive integers")
        self.max_entries, self.max_source_ids = max_entries, max_source_ids
        self._lock = threading.Lock()
        self._scope = None
        self._entries = OrderedDict()  # (kind, digest) -> (_Known*, weight)
        self._weight = 0

    @staticmethod
    def scope_key(store, session_id, settings_digest, policy):
        return (str(store.path.absolute()), session_id, settings_digest, type(policy), policy)

    def _bind(self, scope):
        if scope != self._scope:
            self._scope, self._entries, self._weight = scope, OrderedDict(), 0

    def get(self, scope, kind, digest):
        with self._lock:
            if scope != self._scope:
                return None
            entry = self._entries.get((kind, digest))
            if entry is None:
                return None
            self._entries.move_to_end((kind, digest))
            return entry[0]

    def put_all(self, scope, items):
        """Record ``(kind, digest, known, weight)`` items, then evict to the bounds."""
        with self._lock:
            self._bind(scope)
            for kind, digest, known, weight in items:
                old = self._entries.pop((kind, digest), None)
                if old is not None:
                    self._weight -= old[1]
                    if kind == "target" and known.delta is None and old[0].delta is not None:
                        known = old[0]  # never downgrade a validated target to parsed-only
                if weight > self.max_source_ids:
                    continue
                self._entries[(kind, digest)] = (known, weight)
                self._weight += weight
            while self._entries and (
                len(self._entries) > self.max_entries or self._weight > self.max_source_ids
            ):
                _, (_, weight) = self._entries.popitem(last=False)
                self._weight -= weight

    def clear(self):
        with self._lock:
            self._scope, self._entries, self._weight = None, OrderedDict(), 0

    def stats(self):
        with self._lock:
            return {"entries": len(self._entries), "source_ids": self._weight}


def _share(known, tuples):
    """``known`` whose target reuses equal source tuples of accepted views.

    Equal fields serialize identically, so the target id (the hash of the
    canonical bytes it was parsed from) carries over to the rebuilt object.
    """
    target = known.target
    shared = {}
    for name in ("event_ids", "message_uids"):
        sequence = getattr(target, name)
        candidate = tuples.get((len(sequence), sequence[-1]))
        if candidate is not None and candidate is not sequence and candidate == sequence:
            shared[name] = candidate
    if not shared:
        return known
    rebuilt = replace(target, **shared)
    rebuilt.__dict__["target_id"] = target.target_id
    return _KnownTarget(rebuilt, known.delta)


class _ValidationMemo:
    """Request-local validation memo; never shared across journal opens.

    Scoped to one (store, session, reasoning policy); validate_generation rejects
    a memo whose scope differs from the call's. ``rows`` is a snapshot of the
    insert-only archive for this open and only avoids repeated reads: every row
    use still re-verifies its hash, message_uid and session, and callers receive
    independent copies. Direct archive mutation within an open is unsupported;
    the next open (a new memo) rereads and detects it. Settings bind on first use.
    ``accepted`` holds views the journal itself already cross-checked (see
    RollingJournal._view); validate_generation only reads it. Keys are the SHA-256
    of the exact bytes just reread for this use, never a parent name.
    ``prefix_states`` is written only by validate_generation after it fully
    validated that view; an accepted entry is only short-circuited when it is
    that very validated object.
    ``bound`` is None except inside ``operation()`` (one RollingJournal
    operation: a fresh set on entry, None again on exit; a failed operation
    drops the whole memo). validate_generation adds a parent id only after it
    reread that parent's exact bytes through ``parent_loader`` in this
    operation, they hashed to the id and validated (with every ancestor reached
    the same way) to the view its prefix state is tied to. A bound id skips the
    reread only while that view is also the accepted one; see ``parent_view``.
    """

    def __init__(self, store, session_id, policy, cache=None, settings_digest=None):
        self.store, self.session_id, self.policy = store, session_id, policy
        self.rows, self.accepted = {}, {}
        # Optional cross-open RollingVerificationCache (journal-owned; see there).
        # Read through ``known_*``; written only by ``promote`` after a journal
        # operation succeeded. ``present`` holds archive ids whose exact bytes
        # were rehashed by the store in this open without being decoded (ids
        # in ``rows`` were read in full).
        self.cache, self.present = cache, set()
        # Only with a cache, for ``promote`` (and reuse within this open):
        # generation id -> delta identity sets, target id -> _KnownTarget, as
        # recorded by completed validations in this open. Without a cache the
        # memo retains no identity state beyond the shared index and lineages.
        self.deltas, self.validated_targets = {}, {}
        # sha256(exact target payload bytes) -> Target parsed from those bytes in
        # this open. The payload is still reread and hashed on every use; only the
        # decode/canonical/identity work on bytes already verified is reused.
        self.targets = {}
        # Bound on first validation, or up front by a journal (its settings).
        self.settings_digest: str | None = settings_digest
        # generation_id -> (view, _PrefixState) recorded only by a completed
        # validation of that exact view object; see validate_generation.
        self.prefix_states = {}
        # Shared identity index of every validated delta (see _PrefixState):
        # {kind: {identity: introducer | {introducers}}}. A value is a single
        # generation id, or a set of two or more when sibling branches
        # introduced the same identity; never an empty set. Written only by
        # _record_identities, after a whole validation has passed.
        self.identities = _new_identity_index()
        self.bound: set | None = None

    def record_target(self, target, delta):
        if self.cache is not None:
            self.validated_targets[target.target_id] = _KnownTarget(target, delta)

    @contextmanager
    def operation(self):
        """Scope of one journal operation: ``bound`` is a fresh set inside, None outside.

        Every operation rereads and rebinds each ancestor itself; nothing bound
        here survives it. The journal drops the whole memo when the operation
        fails, so a failed operation never leaves a binding behind either.
        """
        self.bound = set()
        try:
            yield
        finally:
            self.bound = None

    def empty_prefix(self):
        """The state of a root generation's (empty) parent prefix in this memo."""
        return _PrefixState(0, frozenset(), self.identities)

    def get_event(self, event_id):
        row = deepcopy(self.rows.get(event_id))
        if row is None:
            row = self.store.get_event(event_id, session_id=self.session_id)
            if row is not None:  # a missing row is never cached
                self.rows[event_id] = deepcopy(row)
        return row

    def _scope(self):
        if self.cache is None or self.settings_digest is None:
            return None
        return RollingVerificationCache.scope_key(
            self.store, self.session_id, self.settings_digest, self.policy
        )

    def known_generation(self, generation_id):
        scope = self._scope()
        return None if scope is None else self.cache.get(scope, "generation", generation_id)

    def known_target(self, digest):
        """A _KnownTarget for these exact payload bytes: this open's, else the cache's."""
        hit = self.validated_targets.get(digest)
        if hit is not None:
            return hit
        scope = self._scope()
        hit = None if scope is None else self.cache.get(scope, "target", digest)
        if hit is not None:
            return hit
        target = self.targets.get(digest)
        return None if target is None else _KnownTarget(target, None)

    def require_sources(
        self, event_ids, guard=None, message="generation source unavailable or changed"
    ):
        """Every id is archived in this session with bytes that still hash to it.

        The store rehashes each payload it returns, so a present id is exactly
        the row a completed validation of these ids already canonicalized.
        """
        missing = [
            event for event in event_ids if event not in self.rows and event not in self.present
        ]
        if not missing:
            return
        if guard:
            guard()
        found = self.store.verify_events(missing, session_id=self.session_id)
        if guard:
            guard()
        if len(found) != len(set(missing)):
            raise ValueError(message)
        self.present.update(found)

    def promote(self):
        """Record this open's completed, journal-accepted validations in the cache."""
        scope = self._scope()
        if scope is None:
            return
        items = []
        tuples = {}
        for generation_id, view in self.accepted.items():
            recorded = self.prefix_states.get(generation_id)
            delta = self.deltas.get(generation_id)
            if recorded is None or recorded[0] is not view or delta is None:
                continue
            fields_ = {
                spec.name: getattr(view, spec.name)
                for spec in fields(RollingView)
                if spec.name not in ("snapshot_bytes", "generation_id")
            }
            for sequence in (view.event_ids, view.message_uids):
                tuples.setdefault((len(sequence), sequence[-1]), sequence)
            items.append(
                ("generation", generation_id, _KnownGeneration(fields_, delta), len(view.event_ids))
            )
        for digest, known in self.validated_targets.items():
            items.append(("target", digest, _share(known, tuples), len(known.target.event_ids)))
        for digest, target in self.targets.items():
            if digest not in self.validated_targets:
                known = _share(_KnownTarget(target, None), tuples)
                items.append(("target", digest, known, len(target.event_ids)))
        self.cache.put_all(scope, items)


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
    and journal-cross-checked in this open, keyed by the sha256 of these exact
    bytes (and re-checked against bytes, settings digest and the depth bound); it
    short-circuits only when ``prefix_states`` ties it to the very view object a
    completed validation here produced, otherwise it is validated in full.
    Validation is incremental: only this generation's own delta rows
    (``events[delta_start:]``) are read and canonicalized; every earlier row was
    verified by the generation that introduced it, whose validated prefix state
    carries what a delta can collide with. Archive rows are read once per memo; a
    memo is bound to this store, session and policy object and is rejected
    otherwise.
    """
    if _depth >= settings.max_logical_jobs:
        raise ValueError("generation lineage exceeds campaign bound")
    # decode_json's own admission, applied before hashing: accepted bytes are
    # bound by hash and exact comparison and never decoded again.
    _admit_json_bytes(snapshot_bytes, settings.max_checkpoint_bytes)
    generation_id = sha256(snapshot_bytes)
    if _memo is None:  # rows only: dedupe shared prefix reads within this one call
        _memo = _ValidationMemo(store, session_id, policy)
    if _memo.store is not store or _memo.session_id != session_id or _memo.policy is not policy:
        raise ValueError("validation memo scope mismatch")
    if _memo.settings_digest is None:
        _memo.settings_digest = settings.digest
    elif _memo.settings_digest != settings.digest:
        raise ValueError("validation memo scope mismatch")

    def parent_view(digest, *, prehash):
        """The validated parent ``digest``: reread and hash-bound once per operation.

        ``prehash`` hashes the reread bytes against ``digest`` before validating
        them (the accepted-head walk, where nothing else ties them to the name).

        ``_memo.bound`` (journal operations only, else None) holds the parents
        whose exact bytes were reread through ``parent_loader`` in THIS
        operation, hashed to their name and resolved to the view their
        prefix state is tied to, after their own ancestors were reached the
        same way. A bound parent skips the reread only while it is ALSO the
        tied accepted (journal-cross-checked) view; the depth bound is still
        applied to the skipped chain. Anything else is reread.
        """
        bound = _memo.bound
        if bound is not None and digest in bound:
            hit = _memo.accepted.get(digest)
            recorded = _memo.prefix_states.get(digest)
            if hit is not None and recorded is not None and recorded[0] is hit:
                if _depth + 1 + hit.sequence > settings.max_logical_jobs:
                    raise ValueError("generation lineage exceeds campaign bound")
                return hit
        if parent_loader is None:
            raise ValueError("parent generation unavailable")
        if guard:
            guard()
        data = parent_loader(digest)
        if guard:
            guard()
        if prehash and sha256(data) != digest:
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
            _memo=_memo,
        )
        recorded = _memo.prefix_states.get(digest)
        if (
            bound is not None
            and parent.generation_id == digest
            and recorded is not None
            and recorded[0] is parent
        ):
            bound.add(digest)
        return parent

    accepted = _memo.accepted.get(generation_id)
    if accepted is not None:
        if accepted.snapshot_bytes != snapshot_bytes or accepted.settings_digest != settings.digest:
            raise ValueError("rolling generation identity mismatch")
        # The accepted chain has ``sequence`` generations below this depth.
        if _depth + accepted.sequence > settings.max_logical_jobs:
            raise ValueError("generation lineage exceeds campaign bound")
        recorded = _memo.prefix_states.get(generation_id)
        if recorded is not None and recorded[0] is accepted:
            # Archive rows are insert-only, artifacts are not: reread and bind
            # every ancestor's exact bytes (once per journal operation) even
            # when this head is accepted.
            if accepted.parent_generation is not None:
                parent_view(accepted.parent_generation, prehash=True)
            return accepted
        # An accepted view this memo cannot tie to a validated prefix state (it
        # was not produced here) is not trusted as a parent: validate it in full.

    def parent_lineage(value, events, uids):
        """The validated parent of ``value`` and its tied prefix state (shared checks)."""
        parent = parent_view(value["parent_generation"], prehash=False)
        if (
            parent.generation_id != value["parent_generation"]
            or parent.summary_sha256 != value["parent_summary_sha256"]
            or value["sequence"] != parent.sequence + 1
            or value["delta_start"] != parent.frontier_count
            or tuple(events[: parent.frontier_count]) != parent.event_ids
            or tuple(uids[: parent.frontier_count]) != parent.message_uids
        ):
            raise ValueError("generation parent lineage mismatch")
        # Unreachable while the recursive call above is the only producer of
        # ``parent`` (it records the state of exactly the view it returns); kept
        # as a fail-closed guard so a future refactor cannot silently extend an
        # empty or foreign prefix state.
        recorded = _memo.prefix_states.get(parent.generation_id)
        if (
            recorded is None
            or recorded[0] is not parent
            or recorded[1].index is not _memo.identities
            or not recorded[1].extends(parent)
        ):
            raise ValueError("generation parent state unavailable")
        return parent, recorded[1]

    def accept(value, delta_identities, parent_state, length):
        view = RollingView(snapshot_bytes, generation_id, **value)
        # Validation is complete: record this delta's identities under this
        # generation, and its state as its parent's lineage plus itself.
        _record_identities(_memo.identities, generation_id, delta_identities)
        state = _PrefixState(length, parent_state.lineage | {generation_id}, _memo.identities)
        _memo.prefix_states[generation_id] = (view, state)
        if _memo.cache is not None:
            _memo.deltas[generation_id] = delta_identities
        if _validated is not None:
            _validated.append(view)
        return view

    known = _memo.known_generation(generation_id)
    if known is not None:
        # A completed validation in this scope already accepted exactly these
        # bytes (they hash to ``generation_id``): every check that is a function
        # of the bytes, of rows bound by their ids and of ancestors bound by
        # theirs gave the verdict recorded here. Recheck what is not: the
        # parent chain (reread and validated as always), that every delta row
        # is still archived in this session with its exact bytes, identity
        # collisions against this lineage, and the length bound.
        value = dict(known.fields)
        events, uids = value["event_ids"], value["message_uids"]
        parent_state = _memo.empty_prefix()
        if value["parent_generation"] is not None:
            _, parent_state = parent_lineage(value, events, uids)
        start = value["delta_start"]
        _memo.require_sources(events[start:], guard)
        _check_known_extension(parent_state, len(events) - start, known.delta, settings)
        length = parent_state.length + len(events) - start
        if len(events) != length:
            raise ValueError("generation source unavailable or changed")
        return accept(value, known.delta, parent_state, length)
    value, canonical = _decode_json(snapshot_bytes, settings.max_checkpoint_bytes)
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
    if sha256(_encode_digest_list(events, settings.max_checkpoint_bytes)) != value["prefix_sha256"]:
        raise ValueError("prefix hash mismatch")
    parent, parent_state = None, _memo.empty_prefix()
    if value["parent_generation"] is not None:
        require_digest(value["parent_generation"])
        require_digest(value["parent_summary_sha256"])
        parent, parent_state = parent_lineage(value, events, uids)
    elif (
        value["parent_summary_sha256"] is not None
        or value["delta_start"] != 0
        or value["sequence"] != 1
    ):
        raise ValueError("invalid root generation")
    # Only this generation's own delta is read and canonicalized: every earlier
    # row was verified by the generation that introduced it (parent chain above),
    # and the parent's prefix state carries everything a delta can collide with.
    start = value["delta_start"]
    source = []
    for event, uid in zip(events[start:], uids[start:]):
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
    delta_events, delta_uids, payloads = _canonical_rows(source, policy=policy, settings=settings)
    # Check only: this generation's identities are recorded after every check below.
    delta_identities = _check_prefix_extension(parent_state, source, delta_uids, payloads, settings)
    length = parent_state.length + len(source)
    # Redundant by construction (each row's hash was re-verified above and the
    # parent lineage fixed ``start``/length), re-asserted as defence in depth.
    if len(events) != length or tuple(events[start:]) != delta_events:
        raise ValueError("generation source unavailable or changed")
    delta = _render_delta(source, delta_events, payloads, settings)
    if (
        sha256(_summary_input(None if parent is None else parent.summary_text, delta, settings))
        != value["summary_input_sha256"]
    ):
        raise ValueError("summary input hash mismatch")
    # ``canonical`` is encode_json of ``value`` exactly as decoded; nothing above
    # mutates ``value``, so this is the same check as encoding it again here.
    if canonical != snapshot_bytes:
        raise ValueError("noncanonical generation bytes")
    for key in headers:
        del value[key]
    if parent is None:
        value["event_ids"], value["message_uids"] = tuple(events), tuple(uids)
    else:
        # Equal to tuple(events)/tuple(uids) (the lineage check above compared the
        # prefix with the parent's), but shares the parent's validated strings, so
        # a chain retains each source identity once rather than once per generation.
        value["event_ids"] = parent.event_ids + tuple(events[start:])
        value["message_uids"] = parent.message_uids + tuple(uids[start:])
    return accept(value, delta_identities, parent_state, length)


def revalidate_view(
    view,
    store,
    session_id,
    settings,
    *,
    policy=None,
    parent_loader=None,
    guard=None,
    _journal=None,
):
    """Recheck exact immutable bytes, all source dependencies and parent lineage.

    Never trust a dataclass's exposed fields instead of its validated snapshot.
    Scope/settings are supplied by the caller, not by model-controlled arguments.
    The engine may supply its explicit open journal to reuse that request's
    validation; its bounded artifact reader is authoritative and rereads the
    head and every ancestor. Independent calls retain fresh call-local validation.
    """
    if _journal is not None:
        with _journal._operation():
            if type(view) is not RollingView:
                raise ValueError("trusted rolling view required")
            if (
                _journal.store is not store
                or _journal.session_id != session_id
                or _journal.settings.digest != settings.digest
                or _journal.policy is not policy
                or (parent_loader is not None and parent_loader != _journal._parent_bytes)
            ):
                raise ValueError("validation journal scope mismatch")
            _journal._check(guard)
            checked = _journal._view(_journal._parent_bytes(view.generation_id))
            _journal._check(guard)
            if checked != view:
                raise ValueError("rolling view differs from its exact snapshot")
            return checked
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
