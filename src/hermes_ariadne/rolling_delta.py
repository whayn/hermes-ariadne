"""Version-2 delta encodings; legacy settings, prompt and coverage stay unchanged.

Storage identities are new. No provider or host imports and no persisted hash state.
"""

import hashlib
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property

from . import rolling as r

VERSION = 2

# Optimization limits, not settings or durable identity inputs. They cover
# staged AND accepted input bytes/records; pressure only selects full validation.
_PROOF_BYTES = 16 * 1024 * 1024
_PROOF_RECORDS = 256


@dataclass(frozen=True, slots=True)
class _SourceProof:
    view: object
    parent_state: object
    readable_input: bytes


class _ProofBatch:
    """One generation walk's candidates, not yet journal-accepted evidence."""

    def __init__(self, owner):
        self.owner, self.pending = owner, {}

    def stage(self, view, parent_state, readable_input):
        owner = self.owner
        if (
            owner.records >= _PROOF_RECORDS
            or owner.input_bytes + len(readable_input) > _PROOF_BYTES
        ):
            owner.pressure += 1
            return
        self.pending[view.generation_id] = _SourceProof(view, parent_state, readable_input)
        owner.records += 1
        owner.input_bytes += len(readable_input)

    def accept(self, view, target):
        proof = self.pending.pop(view.generation_id, None)
        if proof is None:
            return
        owner, memo = self.owner, self.owner.memo()
        if (
            memo is not None
            and owner.in_scope(memo, memo.store, memo.session_id, memo.settings_digest, memo.policy)
            and memo.accepted.get(view.generation_id) is view
            and proof.view is view
            and proof.readable_input == target.readable_input
        ):
            previous = owner.accepted.get(target.target_id)
            if previous is not None:
                owner.release(previous)
            # Borrow the already-owned immutable target bytes after exact
            # comparison, but conservatively keep counting their full length.
            owner.accepted[target.target_id] = _SourceProof(
                view, proof.parent_state, target.readable_input
            )
        else:
            owner.release(proof)

    def clear(self):
        for proof in self.pending.values():
            self.owner.release(proof)
        self.pending.clear()


class _SourceProofs:
    """Bounded generation -> target evidence, owned by exactly one open's memo.

    No source payloads, identity sets or full-prefix copies. Parent states and
    delta segments are borrowed from already-owned views. Not a cross-open cache
    and never an authority for job state, dispatch, recovery or publication.
    """

    def __init__(self, memo):
        self.memo = weakref.ref(memo)
        self.store, self.session, self.settings, self.policy = (
            memo.store,
            memo.session_id,
            memo.settings_digest,
            memo.policy,
        )
        self.index = memo.identities
        self.accepted = {}
        self.records = self.input_bytes = self.hits = self.misses = self.pressure = 0

    def in_scope(self, memo, store, session, settings, policy):
        return (
            self.memo() is memo
            and memo.source_proofs is self
            and self.store is store is memo.store
            and self.session == session == memo.session_id
            and self.settings == settings == memo.settings_digest
            and self.policy is policy is memo.policy
            and self.index is memo.identities
        )

    def release(self, proof):
        self.records -= 1
        self.input_bytes -= len(proof.readable_input)

    def find(self, target, parent, memo, store, session, settings, policy):
        self.misses += 1
        if not self.in_scope(memo, store, session, settings.digest, policy) or memo.bound is None:
            return None
        proof = self.accepted.get(target.target_id)
        if proof is None:
            return None
        view = proof.view
        recorded = memo.prefix_states.get(view.generation_id)
        if (
            memo.accepted.get(view.generation_id) is not view
            or recorded is None
            or recorded[0] is not view
            or not recorded[1].extends(view)
            or not r._generation_matches(view, target, view.summary_text)
            or target.readable_input != proof.readable_input
            or (memo.cache is not None and view.generation_id not in memo.deltas)
        ):
            return None
        if parent is None:
            if proof.parent_state is not None or view.parent_generation is not None:
                return None
        else:
            state = memo.prefix_states.get(parent.generation_id)
            if (
                parent.generation_id not in memo.bound
                or memo.accepted.get(parent.generation_id) is not parent
                or state is None
                or state[0] is not parent
                or state[1] is not proof.parent_state
                or not state[1].extends(parent)
            ):
                return None
        self.misses -= 1
        self.hits += 1
        return proof

    def stats(self):
        return {
            "records": self.records,
            "input_bytes": self.input_bytes,
            "accepted": len(self.accepted),
            "hits": self.hits,
            "misses": self.misses,
            "pressure": self.pressure,
        }


class Segments(Sequence):
    """Immutable ordered parent/delta sequence, retaining each segment once."""

    __slots__ = ("parent", "delta", "_count", "_hash")

    def __init__(self, parent, delta, *, digests=False):
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "delta", tuple(delta))
        object.__setattr__(self, "_count", (0 if parent is None else len(parent)) + len(delta))
        state = None
        if digests:
            state = hashlib.sha256(b"[") if parent is None else parent._hash.copy()
            offset = 0 if parent is None else len(parent)
            for value in delta:
                r.require_digest(value)
                state.update((b"," if offset else b"") + b'"' + value.encode("ascii") + b'"')
                offset += 1
        object.__setattr__(self, "_hash", state)

    def __setattr__(self, name, value):
        raise AttributeError("immutable coverage")

    def __len__(self):
        return self._count

    def __iter__(self):
        stack, current = [], self
        while current is not None:
            stack.append(current.delta)
            current = current.parent
        for segment in reversed(stack):
            yield from segment

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._count)
            if start == self._count - len(self.delta) and stop == self._count and step == 1:
                return self.delta
            return tuple(self)[key]
        if type(key) is not int:
            raise TypeError("coverage index must be integer")
        if key < 0:
            key += self._count
        if not 0 <= key < self._count:
            raise IndexError(key)
        current = self
        while current.parent is not None and key < len(current.parent):
            current = current.parent
        return current.delta[key - (0 if current.parent is None else len(current.parent))]

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, (Segments, tuple, list)):
            return NotImplemented
        if len(self) != len(other):
            return False
        if isinstance(other, Segments) and self.parent is other.parent:
            return self.delta == other.delta
        return all(a == b for a, b in zip(self, other))

    @property
    def digest(self):
        if self._hash is None:
            raise ValueError("not an event digest sequence")
        state = self._hash.copy()
        state.update(b"]")
        return state.hexdigest()


def _segments(parent, events, uids):
    return (
        Segments(None if parent is None else parent.event_ids, events, digests=True),
        Segments(None if parent is None else parent.message_uids, uids),
    )


@dataclass(frozen=True)
class DeltaTarget:
    session_id: str
    settings_digest: str
    parent_generation: str | None
    parent_summary_sha256: str | None
    delta_event_ids: tuple
    delta_message_uids: tuple
    delta_start: int
    delta_stop: int
    prefix_sha256: str
    readable_input: bytes
    request_bytes: bytes
    route_labels: tuple
    _coverage: tuple | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        r.bounded_text(self.session_id, 4096)
        for digest in (self.settings_digest, self.prefix_sha256):
            r.require_digest(digest)
        if (self.parent_generation is None) != (self.parent_summary_sha256 is None):
            raise ValueError("incomplete target parent identity")
        for value in (self.parent_generation, self.parent_summary_sha256):
            if value is not None:
                r.require_digest(value)
        if (
            type(self.delta_start) is not int
            or type(self.delta_stop) is not int
            or not 0 <= self.delta_start < self.delta_stop
            or type(self.delta_event_ids) is not tuple
            or type(self.delta_message_uids) is not tuple
            or len(self.delta_event_ids) != self.delta_stop - self.delta_start
            or len(self.delta_message_uids) != len(self.delta_event_ids)
            or type(self.readable_input) is not bytes
            or type(self.request_bytes) is not bytes
            or type(self.route_labels) is not tuple
            or len(self.route_labels) != 2
        ):
            raise ValueError("invalid delta target bounds or input")
        for event in self.delta_event_ids:
            r.require_digest(event)
        for uid in self.delta_message_uids:
            r.bounded_text(uid, 2**31 - 1)
        for label in self.route_labels:
            r.bounded_text(label, 1024)

    @property
    def event_ids(self):
        if self._coverage is None:
            raise ValueError("target coverage not bound to a validated parent")
        return self._coverage[0]

    @property
    def message_uids(self):
        if self._coverage is None:
            raise ValueError("target coverage not bound to a validated parent")
        return self._coverage[1]

    def bind(self, parent, settings):
        if self.delta_stop > settings.max_prefix_events:
            raise ValueError("target exceeds prefix bound")
        if parent is None:
            if self.parent_generation is not None or self.delta_start != 0:
                raise ValueError("invalid root target")
        elif (
            self.parent_generation != parent.generation_id
            or self.parent_summary_sha256 != parent.summary_sha256
            or self.delta_start != parent.frontier_count
            or self.session_id != parent.session_id
            or self.settings_digest != parent.settings_digest
        ):
            raise ValueError("target is not a parent prefix extension")
        coverage = _segments(parent, self.delta_event_ids, self.delta_message_uids)
        if coverage[0].digest != self.prefix_sha256:
            raise ValueError("prefix hash mismatch")
        object.__setattr__(self, "_coverage", coverage)
        return self

    @classmethod
    def from_legacy(cls, old, parent, settings):
        start = old.delta_start
        if parent is not None and (
            tuple(old.event_ids[:start]) != tuple(parent.event_ids)
            or tuple(old.message_uids[:start]) != tuple(parent.message_uids)
        ):
            raise ValueError("target is not a parent prefix extension")
        result = cls(
            old.session_id,
            old.settings_digest,
            None if parent is None else parent.generation_id,
            None if parent is None else parent.summary_sha256,
            tuple(old.event_ids[start:]),
            tuple(old.message_uids[start:]),
            start,
            old.delta_stop,
            old.prefix_sha256,
            old.readable_input,
            old.request_bytes,
            old.route_labels,
        )
        return result.bind(parent, settings)

    @classmethod
    def create(cls, canonical_rows, *, parent, settings, **kwargs):
        old = r.Target.create(canonical_rows, parent=parent, settings=settings, **kwargs)
        return cls.from_legacy(old, parent, settings)

    def to_bytes(self, settings):
        if self.delta_stop > settings.max_prefix_events:
            raise ValueError("target exceeds prefix bound")
        if max(len(self.request_bytes), len(self.readable_input)) > settings.max_request_bytes:
            raise ValueError("target exceeds input bound")
        value = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "_coverage"}
        value.update(format=r.FORMAT, version=VERSION, frontier_count=self.delta_stop)
        for key in ("delta_event_ids", "delta_message_uids", "route_labels"):
            value[key] = list(value[key])
        for key in ("readable_input", "request_bytes"):
            value[key] = value[key].decode("utf-8")
        return r.encode_json(value, settings.max_checkpoint_bytes)

    @classmethod
    def from_bytes(cls, data, settings):
        value = r.decode_json(data, settings.max_checkpoint_bytes)
        expected = {f.name for f in fields(cls) if f.name != "_coverage"}
        if type(value) is not dict or set(value) != expected | {
            "format",
            "version",
            "frontier_count",
        }:
            raise ValueError("invalid delta target schema")
        if (
            value.pop("format") != r.FORMAT
            or type(value["version"]) is not int
            or value.pop("version") != VERSION
            or type(value["frontier_count"]) is not int
            or value.pop("frontier_count") != value["delta_stop"]
        ):
            raise ValueError("invalid delta target identity")
        for key in ("delta_event_ids", "delta_message_uids", "route_labels"):
            if type(value[key]) is not list:
                raise ValueError("invalid delta target sequence")
            value[key] = tuple(value[key])
        for key in ("readable_input", "request_bytes"):
            if type(value[key]) is not str:
                raise ValueError("invalid delta target input")
            value[key] = value[key].encode("utf-8")
        result = cls(**value)
        if result.to_bytes(settings) != data:
            raise ValueError("noncanonical delta target")
        return result

    @cached_property
    def target_id(self):
        return r.sha256(
            self.to_bytes(
                r.RollingSettings(
                    max_request_bytes=2**31 - 1,
                    max_checkpoint_bytes=2**31 - 1,
                    max_prefix_events=2**31 - 1,
                )
            )
        )

    @cached_property
    def readable_input_digest(self):
        return r.sha256(self.readable_input)

    @cached_property
    def request_digest(self):
        return r.sha256(self.request_bytes)

    @cached_property
    def job_id(self):
        return r.sha256(
            r.encode_json(
                [self.target_id, r.POLICY_ID, r.PROMPT_VERSION, self.request_digest], 1024
            )
        )


def generation_bytes(target, summary_text, *, sequence, settings):
    if type(target) is not DeltaTarget or type(sequence) is not int or sequence <= 0:
        raise ValueError("delta target and positive sequence required")
    r.bounded_text(summary_text, settings.max_summary_bytes)
    return r.encode_json(
        {
            "format": r.FORMAT,
            "version": VERSION,
            "policy_id": r.POLICY_ID,
            "readable_projection_id": r.PROJECTION_ID,
            "prompt_version": r.PROMPT_VERSION,
            "session_id": target.session_id,
            "settings_digest": target.settings_digest,
            "sequence": sequence,
            "parent_generation": target.parent_generation,
            "parent_summary_sha256": target.parent_summary_sha256,
            "delta_event_ids": list(target.delta_event_ids),
            "delta_message_uids": list(target.delta_message_uids),
            "frontier_count": target.delta_stop,
            "prefix_sha256": target.prefix_sha256,
            "delta_start": target.delta_start,
            "delta_stop": target.delta_stop,
            "summary_text": summary_text,
            "summary_sha256": r.sha256(summary_text.encode()),
            "summary_input_sha256": target.readable_input_digest,
            "job_id": target.job_id,
        },
        settings.max_checkpoint_bytes,
    )


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
    _proofs=None,
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
            _proofs=_proofs,
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
    expected_input = r._summary_input(
        None if parent is None else parent.summary_text, readable, settings
    )
    if r.sha256(expected_input) != value["summary_input_sha256"]:
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
    if _proofs is not None:
        _proofs.stage(view, None if parent is None else state, expected_input)
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
    proof = (
        None
        if memo.source_proofs is None
        else memo.source_proofs.find(target, parent, memo, store, session_id, settings, policy)
    )
    if proof is not None:
        # Same capture cadence as source traversal; no new source copy or
        # canonicalization. Parent binding was independently refreshed above.
        for _ in target.delta_event_ids:
            guard()
            guard()
        target.to_bytes(settings)
        if memo.cache is not None:
            memo.record_target(target, memo.deltas[proof.view.generation_id])
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
