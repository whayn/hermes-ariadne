"""Background summary builder for the profile-wide memory tree (issue #54, phase C).

``profile_tree`` plans (pure), ``tree_store`` keeps the log, nodes, durable job
reservations and views (SQLite). This module connects them to the host model facade:
it decides what to build next, reserves each call durably BEFORE sending it, validates
the answer and stores the node. It follows UniiChat's ``Memory.ts`` build semantics:

* A leaf whose source already fits ``node_bytes`` is its own node: stored with no call
  and no reservation. A restricted message is the fixed placeholder, which fits.
* Every other node (a long leaf, every merge) is one summary call made with the
  ``<chat>`` view (``seen``) up to the node's cutoff. A line over the limit is asked
  again with the ``over`` note, each ask its own reservation, ``tries`` asks at most.
  The shortest line is kept (cut to ``node_bytes`` if every try was over: UniiChat keeps
  it whole, but a node past ``node_bytes`` would break the view's size accounting). An
  empty answer is a refusal: the node is its source clipped (``cut(flat(source))``).
* Patient, like UniiChat's ``AskAI``: an outage only delays a node, never stops it
  (#70). After a failed call no NEW call is sent until that call's retry is due, so
  an outage is probed once per backoff instead of once per pending node. A facade exception, a timeout or a non-text result makes the job ``uncertain``
  and it is asked again after a backoff: ``retry_base`` seconds, doubling with each
  consecutive failure up to ``retry_max``, persisted in the store (``retry_at``) so a
  restart keeps the schedule. A job whose owner process died (``owner_lost``) is due at
  once. Each re-ask is a new durable reservation, charged in the logical totals; a
  failed call may still have been billed, so a long outage costs at most one call per
  node per ``retry_max``. ``TreeBuilder.retry`` asks now and resets the backoff. A
  reservation is a LOGICAL job, not a physical send or billing ceiling. Missing or
  zero usage is UNKNOWN, never measured zero.
* Two sides, called lanes: ``live`` and ``import`` (by the origin of a node's LAST
  message, like UniiChat). Each lane has its own ``jobs`` concurrency and its own first
  unbuilt leaf (the context cutoff), and a pass always plans the live lane first, so an
  import backlog never takes a live node's slot or call. A job also ignores unbuilt
  parts of the OTHER lane in its context (they are skipped, not waited for), and when
  the other lane's built lines push the context over ``seen_high`` its oldest such lines
  are left out of that job's context (#59): only the job's own lane waits for merges.
  Caps (``max_jobs``, request bytes) are shared: live work reserves first.

Deliberate choices beyond the plan:

* One pass plans from one snapshot (nodes, jobs, ``seen``); jobs in it are independent,
  like UniiChat's concurrent builds. ``step()`` runs a pass sequentially (deterministic);
  the worker runs the same plan with up to ``jobs`` calls per lane in flight, replanning
  as each finishes. Worker calls use plain daemon threads (not a pool), so an in-flight
  call never blocks interpreter exit by itself. ``stop(timeout)`` stops scheduling at once
  (no launch, no reservation, no over-limit re-ask after it) and then waits up to
  ``timeout`` seconds for the calls in flight to commit; a caller that may block (process
  exit, ``exit_wait``) passes a timeout, a long-lived process passes 0. A call still
  running after the timeout is left ``reserved`` and the next start turns it
  ``uncertain`` (``owner_lost``), due again at once.
* The worker stays alive while a retry or an unreadable source is due later, sleeping
  until then (``stop()`` wakes it), and an error while planning is recorded and retried
  after ``retry_base`` seconds instead of ending the worker (#67).
* Liveness of a reservation owner (``recover``): owners are ``"<pid>:<token>"``. An owner
  is alive when it is a builder of this process, or when another process with that pid
  exists (``os.kill(pid, 0)``; on Windows other processes are assumed alive because that
  call is not a probe there). Everything else, including this pid with an unknown token,
  is dead and its reservations become ``uncertain``. This assumes one host per home; a
  pid reused by an unrelated process keeps a dead reservation ``reserved`` (never
  resubmitted, so safe) until that pid is gone.
* A leaf source larger than ``max_request_bytes`` allows is clipped inside the request
  (never in the archive or the fallback), so one huge tool result cannot stall the
  tree behind it. If even that cannot fit the cap, the node is held (no reservation)
  and ``status()["held"]`` counts it.
* ``estimate`` is informational and never a cap.
"""

import contextvars
import json
import math
import os
import sqlite3
import threading
import time
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path

from . import profile_tree as pt
from .store import EventStore
from .tree import Part, View, advance_view, flat, render_lines, utf8_size
from .tree_store import BLOCKING_STATES, MAX_OWNER_CHARS, MAX_PAGE, TreeStore

TASK = "ariadne_summary"
PURPOSE = "ariadne-tree-summary"
LANES = ("live", "import")
_MAX_INT = (1 << 31) - 1
SWEEP_READS = 512  # unclassified leaves read per pass to find the ones that fit
UNAVAILABLE_RETRY = 30.0  # seconds before an unreadable message is read again
LINE_RATIO = 15 / 16  # UniiChat measured 460-510 byte lines for a 512 byte node
_CLIP_STEPS = 40  # bounds the search for a prefix whose serialized request fits


# --------------------------------------------------------------------------- #
# Settings and host registration
# --------------------------------------------------------------------------- #

_LIMITS = {
    "high": (1, _MAX_INT),
    "seen_high": (1, _MAX_INT),
    "node_bytes": (32, 8192),
    "jobs": (1, 64),
    "max_jobs": (0, _MAX_INT),
    "max_request_bytes": (4096, 1 << 26),
    "max_total_request_bytes": (0, 1 << 40),
    "tail_bytes": (0, _MAX_INT),
    "tries": (1, 20),
    "max_tokens": (1, 1 << 20),
    "timeout": (1, 3600),
    "exit_wait": (0, 3600),
    "retry_base": (1, 86400),
    "retry_max": (1, 86400),
}
# Accepted for configs written for v0.3.0 and ignored: lost jobs now retry like any other.
_RETIRED = frozenset({"auto_reask_lost"})


@dataclass(frozen=True)
class TreeSettings:
    """Tree limits. ``jobs`` is the concurrency per lane; ``max_jobs`` and the request
    byte caps count LOGICAL reservations (the host may retry or fall back internally).
    A zero lifetime job or total byte cap means unlimited; accounting continues.

    ``max_tokens`` leaves room above the ~130 tokens a 512-byte line needs because a
    reasoning model spends its budget on thinking first: an empty answer is read as a
    refusal and would clip the node.

    ``exit_wait`` is the seconds (int or float, 0 = never wait) a process may wait at its
    exit for in-flight summary calls to commit. ``retry_base``/``retry_max`` are the
    first and the longest wait in seconds before a failed call is asked again; see the
    module docs.
    """

    high: int = 128000
    seen_high: int = 32000
    node_bytes: int = 512
    jobs: int = 4
    max_jobs: int = 0
    max_request_bytes: int = 65536
    max_total_request_bytes: int = 0
    tail_bytes: int = 32768
    tries: int = 5
    max_tokens: int = 1024
    timeout: int = 300
    exit_wait: int | float = 120
    retry_base: int = 30
    retry_max: int = 3600

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            low, high = _LIMITS[spec.name]
            kinds = (int, float) if spec.name == "exit_wait" else (int,)
            if type(value) not in kinds or not low <= value <= high:
                raise ValueError(f"tree setting {spec.name} must be an integer in {low}..{high}")
        if self.retry_base > self.retry_max:
            raise ValueError("tree setting retry_base must not exceed retry_max")

    def retry_delay(self, failures: int) -> int:
        """Seconds before the next ask after ``failures`` consecutive failed calls."""
        return min(self.retry_max, self.retry_base << min(max(failures - 1, 0), 32))

    @classmethod
    def from_config(cls, config: Mapping | None) -> "TreeSettings":
        """Settings from a config mapping; unknown keys and invalid values are refused."""
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise ValueError("tree settings must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = sorted(str(key) for key in config.keys() - known - _RETIRED)
        if unknown:
            raise ValueError("unknown tree setting: " + ", ".join(unknown))
        return cls(**{key: value for key, value in config.items() if key in known})


def register_summary_task(ctx) -> None:
    """Register the plugin-owned auxiliary task every summary call is made under.

    With no ``inherit_from`` Hermes resolves it to the user's chat provider and model;
    users can pin another under ``auxiliary.ariadne_summary``.
    """
    ctx.register_auxiliary_task(
        TASK,
        display_name="Ariadne summaries",
        description=(
            "Writes the one-line summaries of Ariadne's memory tree of every conversation. "
            "Defaults to the chat model; pin a cheaper model under auxiliary.ariadne_summary."
        ),
    )


# --------------------------------------------------------------------------- #
# Estimate
# --------------------------------------------------------------------------- #


def _request_json(messages) -> bytes:
    """The bytes a request is measured by: UTF-8 of its canonical JSON."""
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8", "replace"
    )


def _internal_nodes(leaves: int) -> int:
    """Parents in the full tree over ``leaves`` leaves: every complete pair, level by level."""
    return leaves - leaves.bit_count()


def estimate(sizes, node_bytes: int = pt.NODE_BYTES, *, existing: int = 0) -> dict:
    """The expected number of summary calls for appending messages of ``sizes`` bytes.

    ``leaf_calls`` counts leaves whose source exceeds ``node_bytes``; ``internal_calls``
    the parents that become complete when ``len(sizes)`` leaves are appended after
    ``existing`` ones (every parent takes a call). ``request_bytes`` is approximate:
    prompt text plus inputs, without the ``seen`` context a call also carries (up to
    ``seen_high`` bytes each). Informational only, never a cap (UniiChat ``import_guess``).
    """
    if type(node_bytes) is not int or node_bytes < 1:
        raise ValueError("node_bytes must be a positive integer")
    if type(existing) is not int or existing < 0:
        raise ValueError("existing must be a nonnegative integer")
    sizes = list(sizes)
    for size in sizes:
        if type(size) is not int:
            raise TypeError("sizes must be integers")
        if size < 0:
            raise ValueError("sizes must be nonnegative")
    big = [size for size in sizes if size > node_bytes]
    internal = _internal_nodes(existing + len(sizes)) - _internal_nodes(existing)
    if not sizes:
        request = 0
    else:
        leaf_head = len(_request_json(pt.summary_messages(Part(0, 0), ("",), (), node_bytes)))
        merge_head = len(_request_json(pt.summary_messages(Part(1, 0), ("", ""), (), node_bytes)))
        line = max(1, int(node_bytes * LINE_RATIO))
        request = len(big) * leaf_head + sum(big) + internal * (merge_head + 2 * line + 1)
    return {
        "leaves": len(sizes),
        "leaf_calls": len(big),
        "internal_calls": internal,
        "calls": len(big) + internal,
        "request_bytes": request,
    }


# --------------------------------------------------------------------------- #
# Owner liveness
# --------------------------------------------------------------------------- #

_ACTIVE: set[str] = set()  # owners of the builders of this process
_ACTIVE_LOCK = threading.Lock()


def owner_alive(owner: str) -> bool:
    """Whether the reservations of ``owner`` may still be completed (see module docs)."""
    with _ACTIVE_LOCK:
        if owner in _ACTIVE:
            return True
    head = owner.partition(":")[0]
    if not head.isdigit():
        return False
    pid = int(head)
    if pid == os.getpid():
        return False  # this process, but no builder of it owns that token
    if os.name == "nt":
        return True  # os.kill is not a probe on Windows: never presume another dead
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


class _Unavailable(Exception):
    """A leaf's message cannot be read now; ``args[0]`` is the reason (a short code)."""


@dataclass(frozen=True)
class _Job:
    part: Part
    lane: str
    inputs: tuple[str, ...]
    context: tuple[str, ...]
    messages: list
    request_bytes: int
    tries: int


@dataclass
class _Plan:
    jobs: list
    fits: int = 0
    held: int = 0


def _clean(text: str) -> str:
    """A model answer as one stored line: stripped, flat, a well-formed string."""
    return pt.flat(text.strip().encode("utf-8", "replace").decode("utf-8"))


class TreeBuilder:
    """Builds the profile tree of ``home`` with ``llm`` (``ctx.llm``); see the module docs."""

    def __init__(self, home, llm, settings=None, *, owner=None, clock=time.time):
        self.home = Path(home)
        self.settings = settings if settings is not None else TreeSettings()
        if not isinstance(self.settings, TreeSettings):
            raise TypeError("settings must be TreeSettings")
        self._store = TreeStore(self.home)
        self._llm = llm
        self._clock = clock
        self.owner = owner if owner is not None else f"{os.getpid()}:{uuid.uuid4().hex}"
        if not isinstance(self.owner, str) or not self.owner or len(self.owner) > MAX_OWNER_CHARS:
            raise ValueError("invalid owner")
        with _ACTIVE_LOCK:
            _ACTIVE.add(self.owner)
        weakref.finalize(self, _discard_owner, self.owner)
        self._lock = threading.RLock()
        self._plan_lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Condition(self._lock)  # notified when a call leaves flight
        self._thread: threading.Thread | None = None
        self._context: contextvars.Context | None = None
        self._stopped = False
        self._stalled = False
        self._recovered = False
        self._inflight: dict[Part, str] = {}
        self._origins = bytearray()  # per log position: 1 for an import, else 0
        self._cursor = {"live": 0, "import": 0, "any": 0}
        self._nonfit: set[int] = set()
        self._unavailable: dict[int, float] = {}
        self._archive: EventStore | None = None
        self._tags: dict[str, str] = {}
        self._caps_exhausted = False
        self._last_error: str | None = None
        self._held = 0
        self._pause_until = 0.0  # after a failed call: no new call before this clock time
        self._calls = 0
        self._usage_known = 0
        self._usage_unknown = 0
        self._tokens = 0

    # -- reading the log -------------------------------------------------------------

    def _lane(self, part: Part) -> str:
        return "import" if self._origins[part.stop - 1] else "live"

    def _sync_origins(self, count: int) -> None:
        while len(self._origins) < count:
            start = len(self._origins)
            stop = min(count, start + MAX_PAGE)
            found = self._store.entries(start, stop)
            if len(found) != stop - start:
                raise RuntimeError("profile log has a gap")
            self._origins.extend(1 if item.origin == "import" else 0 for item in found)

    def _first(self, lane: str, count: int, nodes) -> int:
        """The first leaf of ``lane`` that is not built (``count`` when none)."""
        flag = 1 if lane == "import" else 0
        position = self._cursor[lane]
        while position < count and (self._origins[position] != flag or Part(0, position) in nodes):
            position += 1
        self._cursor[lane] = position
        return position

    def _archive_store(self) -> EventStore | None:
        """The exact archive, only when it already exists: a read never creates it."""
        if self._archive is None:
            path = self.home / "ariadne" / "events.sqlite3"
            try:
                if not path.is_file() or path.is_symlink():
                    return None
                self._archive = EventStore(path)
            except OSError:
                return None
        return self._archive

    def _tag(self, session_id: str) -> str:
        """The conversation's stored legend tag (lengthened on a collision), cached."""
        tag = self._tags.get(session_id)
        if tag is None:
            found = self._store.sessions().get(session_id)
            if found is None:
                raise _Unavailable("session_missing")
            tag = self._tags[session_id] = found.tag
        return tag

    def _read_source(self, position: int) -> str:
        entry = self._store.entry(position)
        if entry is None:
            raise _Unavailable("entry_missing")
        fields_ = dict(
            position=entry.position,
            session_id=entry.session_id,
            event_id=entry.event_id,
            message_uid=entry.message_uid,
            role=entry.role,
            date=entry.date,
            origin=entry.origin,
            restricted=entry.restricted,
        )
        try:
            item = pt.LogEntry(**fields_)
        except ValueError:
            try:  # a stored date that is not ISO is shown as unknown, not fatal
                item = pt.LogEntry(**{**fields_, "date": None})
            except ValueError:
                raise _Unavailable("entry_invalid") from None
        tag = self._tag(entry.session_id)
        if item.restricted:
            return pt.leaf_source(item, None, tag)  # the archive is not even opened
        archive = self._archive_store()
        if archive is None:
            raise _Unavailable("source_unavailable")
        try:
            message = archive.get_event(entry.event_id, session_id=entry.session_id)
        except (ValueError, sqlite3.Error, OSError):
            raise _Unavailable("source_unavailable") from None
        if message is None:
            raise _Unavailable("source_unavailable")
        try:
            return pt.leaf_source(item, message, tag)
        except (TypeError, ValueError):
            raise _Unavailable("source_invalid") from None

    def _cooling(self, position: int) -> bool:
        """Whether the leaf's source was unreadable and is not due to be read again."""
        until = self._unavailable.get(position)
        return until is not None and self._clock() < until

    def _source_or_skip(self, position: int) -> str | None:
        """A leaf's source, or None (recorded) while it cannot be read."""
        if self._cooling(position):
            return None
        try:
            source = self._read_source(position)
        except _Unavailable as error:
            self._last_error = str(error)
            self._unavailable[position] = self._clock() + UNAVAILABLE_RETRY
            return None
        self._unavailable.pop(position, None)
        return source

    # -- planning --------------------------------------------------------------------

    def _messages(self, part, inputs, context, note):
        """``(messages, request_bytes)`` within ``max_request_bytes``, else None.

        An oversized leaf source is clipped inside the request (see module docs).
        """
        cap = self.settings.max_request_bytes
        nb = self.settings.node_bytes
        messages = pt.summary_messages(part, inputs, context, nb, note)
        size = len(_request_json(messages))
        if size <= cap:
            return messages, size
        if part.level != 0:
            return None
        text = inputs[0]
        total = utf8_size(text)

        def attempt(budget: int):
            clipped = pt.cut(text, budget) + f" …[clipped to {budget} of {total} bytes]"
            clipped_messages = pt.summary_messages(part, (clipped,), context, nb, note)
            return clipped_messages, len(_request_json(clipped_messages))

        # JSON escaping makes the request longer than the text by an unknown amount, so
        # the prefix length is searched against the real serialized size (bounded).
        low, high = nb, total  # `low` is the smallest prefix worth sending
        best = None
        for _ in range(_CLIP_STEPS):
            if low > high:
                break
            budget = (low + high) // 2
            found = attempt(budget)
            if found[1] <= cap:
                best, low = found, budget + 1
            else:
                high = budget - 1
        return best

    def _sweep(self, count: int, nodes: dict, blocked) -> int:
        """Store the unbuilt leaves that fit (no call, no reservation); returns how many."""
        nb = self.settings.node_bytes
        budget, fits = SWEEP_READS, 0
        start = self._cursor["any"]
        while start < count and Part(0, start) in nodes:
            start += 1
        self._cursor["any"] = start
        for position in range(start, count):
            if position in self._nonfit:
                continue
            part = Part(0, position)
            if part in nodes or part in blocked or self._cooling(position):
                continue  # a cooling (unreadable) leaf must not use up the read budget
            if budget == 0:
                break
            budget -= 1
            source = self._source_or_skip(position)
            if source is None:
                continue
            text = pt.leaf_text_if_fits(source, nb)
            if text is None:
                self._nonfit.add(position)
                continue
            self._store.put_node(part, text)
            nodes[part] = text
            fits += 1
        return fits

    def _plan(self, room: Mapping[str, int]) -> _Plan:
        """What to build next from one snapshot, at most ``room[lane]`` jobs per lane."""
        plan = _Plan([])
        store, s = self._store, self.settings
        count = store.count()
        if count == 0:
            return plan
        store.rearm_due(self._clock())
        self._sync_origins(count)
        nodes = dict(store.nodes())
        jobs = store.jobs()
        with self._lock:
            running = set(self._inflight)
        blocked = {p for p, job in jobs.items() if job.state in BLOCKING_STATES} | running
        plan.fits = self._sweep(count, nodes, blocked)
        self._advance_chat(count, nodes)
        if self._clock() < self._pause_until:
            return plan  # a call just failed: send nothing new until its retry is due
        ready = pt.ready_jobs(count, nodes, blocked)
        seen, revision = store.load_view("seen")
        seen = View(high=s.seen_high) if seen is None else replace(seen, high=s.seen_high)
        loaded = seen
        picked = {lane: [] for lane in LANES}
        scanned = dict.fromkeys(LANES, 0)
        by_lane = {lane: [p for p in ready if self._lane(p) == lane] for lane in LANES}
        for lane in LANES:  # live first: an import never takes a live node's place
            for part in by_lane[lane]:
                if len(picked[lane]) >= room.get(lane, 0):
                    break
                if part.level == 0 and self._cooling(part.start):
                    continue  # unreadable until its retry: free, so it cannot hide later leaves
                if scanned[lane] >= 8 * s.jobs + 64:
                    break
                scanned[lane] += 1
                first = self._first(lane, count, nodes)
                seen = pt.advance_seen(seen, part, nodes, first)  # kept even if skipped
                if any(p not in nodes and self._lane(p) == lane for p in seen.parts):
                    continue  # its own lane has a hole: wait
                context = self._lines_for(seen, nodes, lane)
                if context is None:
                    continue  # its own lane's lines are over the high: wait for merges
                if part.level == 0:
                    source = self._source_or_skip(part.start)
                    if source is None:
                        continue
                    text = pt.leaf_text_if_fits(source, s.node_bytes)
                    if text is not None:  # beyond the sweep: store it, no call
                        store.put_node(part, text)
                        nodes[part] = text
                        plan.fits += 1
                        continue
                    inputs = (source,)
                else:
                    inputs = tuple(nodes[child] for child in part.children)
                request = self._messages(part, inputs, context, None)
                if request is None:
                    plan.held += 1
                    continue
                job = jobs.get(part)
                tries = s.tries
                if job is not None and job.state == "refused":
                    tries = max(1, s.tries - job.attempts)  # an unfinished earlier try
                picked[lane].append(_Job(part, lane, inputs, context, *request, tries))
        if seen != loaded and (revision > 0 or seen.parts):
            store.save_view("seen", seen, revision)  # a lost race is fine: a cache
        self._held = plan.held
        plan.jobs = picked["live"] + picked["import"]
        return plan

    def _lines_for(self, seen: View, nodes, lane: str) -> tuple[str, ...] | None:
        """The ``<chat>`` lines of a ``lane`` job: ``seen``'s built lines within its high.

        Unbuilt parts are skipped (only the other lane's can be left). Over the high, the
        OTHER lane's oldest lines are dropped first (#59); None if the job's own lane
        alone is over it.
        """
        if seen.size <= seen.high:
            return render_lines(seen, nodes)
        built = [p for p in seen.parts if p in nodes]
        size = sum(utf8_size(nodes[p]) for p in built)
        keep = []
        for p in built:  # oldest first
            if size > seen.high and self._lane(p) != lane:
                size -= utf8_size(nodes[p])
                continue
            keep.append(p)
        if size > seen.high:
            return None
        return tuple(f"{p.start}+{p.count}|{flat(nodes[p])}" for p in keep)

    def _next_due(self) -> float | None:
        """The clock time of the earliest retry or unreadable-source read, if any."""
        due = list(self._unavailable.values())
        if self._pause_until > self._clock():
            due.append(self._pause_until)
        for job in self._store.jobs().values():
            if job.state == "uncertain":
                due.append(job.retry_at if job.retry_at is not None else self._clock())
        return min(due) if due else None

    def _advance_chat(self, count: int, nodes) -> None:
        """Advance and persist the ``chat`` view to the end of the log (CAS, re-read)."""
        high = self.settings.high
        for _ in range(5):
            stored, revision = self._store.load_view("chat")
            base = View(high=high) if stored is None else replace(stored, high=high)
            new = advance_view(base, count, nodes, next_message=True)
            if stored is not None and new == stored:
                return
            if self._store.save_view("chat", new, revision):
                return

    # -- one job ---------------------------------------------------------------------

    def _reserve(self, part: Part, size: int) -> tuple[str, int]:
        """``("ok", attempt)`` for the reservation now held, else ``(reason, 0)``."""
        s = self.settings
        attempt = self._store.reserve(
            part, size, self.owner, max_jobs=s.max_jobs, max_total_bytes=s.max_total_request_bytes
        )
        if attempt:
            self._caps_exhausted = False
            return "ok", attempt
        counters = self._store.counters()
        if (s.max_jobs and counters["logical_jobs"] + 1 > s.max_jobs) or (
            s.max_total_request_bytes
            and counters["request_bytes"] + size > s.max_total_request_bytes
        ):
            self._caps_exhausted = True
            return "caps", 0
        return "taken", 0  # built, reserved or blocked by another builder meanwhile

    def _release(self, part: Part, attempt: int) -> None:
        """End a reservation whose call was never sent (it may be reserved again)."""
        try:
            self._store.mark(part, "refused", "not_sent", owner=self.owner, attempt=attempt)
        except Exception:  # noqa: BLE001 - it stays reserved; the next start reconciles it
            pass

    def _uncertain(self, part: Part, error: str, attempt: int) -> None:
        self._last_error = error
        now = self._clock()

        def schedule(failures: int) -> float:
            due = now + self.settings.retry_delay(failures)
            self._pause_until = due  # an outage is probed once per backoff (see _plan)
            return due

        try:
            self._store.mark(
                part, "uncertain", error[:100], owner=self.owner, attempt=attempt, schedule=schedule
            )
        except Exception:  # noqa: BLE001 - it stays reserved; the next start reconciles it
            pass

    def _count_usage(self, result) -> None:
        total = getattr(getattr(result, "usage", None), "total_tokens", 0)
        with self._lock:
            if type(total) is int and total > 0:
                self._usage_known += 1
                self._tokens += total
            else:
                self._usage_unknown += 1  # missing or zero usage is unknown, not zero

    def _execute(self, job: _Job, *, worker: bool = False, held: int = 0) -> str:
        """Run one job to an outcome: built, taken, caps, held, stopped, uncertain or error.

        ``worker`` jobs obey ``stop()``: once it was called they send no further call.
        """
        with self._lock:
            self._inflight[job.part] = job.lane
        try:
            outcome = self._build(job, worker, held)
            if outcome == "built":
                self._pause_until = 0.0  # the model answers again
            return outcome
        except Exception as error:  # noqa: BLE001 - a store failure must not kill the worker
            self._last_error = type(error).__name__
            return "error"
        finally:
            with self._idle:
                self._inflight.pop(job.part, None)
                self._idle.notify_all()

    def _build(self, job: _Job, worker: bool = False, held: int = 0) -> str:
        s, part, store = self.settings, job.part, self._store
        source = job.inputs[0] if part.level == 0 else pt.merge_input(*job.inputs)
        lines: list[str] = []
        messages, size = job.messages, job.request_bytes
        for turn in range(job.tries):
            if turn:
                request = self._messages(
                    part, job.inputs, job.context, pt.over(lines[-1], s.node_bytes)
                )
                if request is None:
                    return "held"
                messages, size = request
            if held and not turn:  # the planner reserved the first attempt, in lane order
                outcome, attempt = "ok", held
                if self._stopped:  # reserved but never sent: a definitive, free answer
                    self._release(part, attempt)
                    return "stopped"
            else:
                if worker and self._stopped:
                    return "stopped"  # nothing is reserved or sent once stop() was called
                outcome, attempt = self._reserve(part, size)
                if outcome != "ok":
                    return outcome
                if worker and self._stopped:  # stop() came while reserving: release, don't send
                    self._release(part, attempt)
                    return "stopped"
            with self._lock:
                self._calls += 1  # counted before the send: an exception is still a call
            try:
                result = self._llm.complete(
                    messages,
                    max_tokens=s.max_tokens,
                    timeout=s.timeout,
                    purpose=PURPOSE,
                    task=TASK,
                )
            except Exception as error:  # noqa: BLE001 - any failure is an ambiguous outcome
                self._uncertain(part, type(error).__name__, attempt)
                return "uncertain"
            raw = getattr(result, "text", None)
            if raw is not None and not isinstance(raw, str):
                self._uncertain(part, "invalid_result", attempt)
                return "uncertain"
            self._count_usage(result)
            line = _clean(raw or "")
            if not line:  # a refusal: the best earlier line, else the source clipped
                best = min(lines, key=utf8_size) if lines else pt.flat(source)
                store.finish(part, pt.cut(best, s.node_bytes), owner=self.owner, attempt=attempt)
                return "built"
            if utf8_size(line) <= s.node_bytes:
                store.finish(part, line, owner=self.owner, attempt=attempt)
                return "built"
            lines.append(line)
            if turn + 1 == job.tries:
                best = pt.cut(min(lines, key=utf8_size), s.node_bytes)
                store.finish(part, best, owner=self.owner, attempt=attempt)
                return "built"
            # a definitive answer: ask again
            store.mark(part, "refused", "over_limit", owner=self.owner, attempt=attempt)
        return "held"  # pragma: no cover - the loop always returns

    # -- synchronous pass ------------------------------------------------------------

    def recover(self) -> list[Part]:
        """Turn reservations of dead owners ``uncertain``; nothing is created if absent."""
        self._recovered = True
        if not self._store.path.exists():
            return []
        return self._store.stale_reserved(owner_alive)

    def step(self) -> dict:
        """One synchronous pass: plan, then run the planned jobs one after another.

        Live jobs run before import jobs. Stops early when a cap refuses a reservation.
        ``progress`` says whether any node was stored or any call made, so ``while
        step()["progress"]`` runs the tree to completion (or to a cap or an uncertain job).
        """
        if not self._recovered:
            self.recover()
        with self._plan_lock:
            plan = self._plan({lane: self.settings.jobs for lane in LANES})
        report = {"fits": plan.fits, "built": 0, "calls": 0, "uncertain": 0, "held": plan.held}
        calls = self._calls
        for job in plan.jobs:
            outcome = self._execute(job)
            if outcome == "built":
                report["built"] += 1
            elif outcome == "uncertain":
                report["uncertain"] += 1
            if outcome in ("caps", "error", "uncertain"):
                break  # a failed call pauses the rest (see _plan)
        report["calls"] = self._calls - calls
        report["caps_exhausted"] = self._caps_exhausted
        report["progress"] = bool(report["fits"] or report["built"] or report["calls"])
        if report["progress"]:
            self._advance_chat(self._store.count(), dict(self._store.nodes()))
        return report

    def retry(self, parts=None) -> list[Part]:
        """Ask uncertain jobs now (all when ``parts`` is None), resetting their backoff.
        Also lifts the pause after a failure. Does not wake the worker."""
        self._pause_until = 0.0
        return self._store.retry(parts)

    # -- background worker -----------------------------------------------------------

    def start(self) -> None:
        """Reconcile dead reservations and start the worker. Never blocks on a model.

        Reservations of dead owners become ``uncertain``, due at once.
        """
        with self._lock:
            self._stopped = False
        self.recover()
        self.kick()

    def kick(self) -> bool:
        """Wake the worker (starting it if idle) after new log entries; returns at once.

        The worker runs in a copy of the context of the first ``start``/``kick`` caller,
        so Hermes's profile-bound context survives.
        """
        with self._lock:
            if self._stopped:
                return False
            if self._context is None:
                self._context = contextvars.copy_context()
            self._wake.set()
            if self._thread is None or not self._thread.is_alive():
                self._stalled = False
                thread = threading.Thread(
                    target=self._context.copy().run,
                    args=(self._loop,),
                    name="ariadne-tree-builder",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Stop scheduling now, then wait up to ``timeout`` seconds in total.

        Scheduling stops at once: no job is launched or reserved and no over-limit re-ask
        is sent after this call. It then waits for the worker thread and for the calls
        already in flight to finish and commit as usual. A call still running when
        ``timeout`` ends is not cancelled: it stays ``reserved`` (the next start reconciles
        it) or, if the process lives on, finishes on its own daemon thread. ``timeout=0``
        waits for nothing. Safe to repeat: a later call with a longer timeout waits for
        what an earlier one did not.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            self._stopped = True
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, deadline - time.monotonic()))
        with self._idle:
            while self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._idle.wait(remaining)
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def _launch(self, job: _Job) -> str:
        """Reserve ``job``'s first attempt here, in plan order, then run it on a thread.

        Reserving in the planning thread (not in each job thread) is what lets the live
        lane take a scarce job/byte cap before an import: ``plan.jobs`` is live first.
        """
        with self._lock:
            if self._stopped:
                return "stopped"  # the worker was planning when stop() came
            self._inflight[job.part] = job.lane
        outcome, held = "error", 0
        try:
            outcome, held = self._reserve(job.part, job.request_bytes)
            if outcome == "ok":
                context = contextvars.copy_context()
                threading.Thread(
                    target=context.run,
                    args=(self._job_thread, job, held),
                    name="ariadne-tree-job",
                    daemon=True,
                ).start()
                return "ok"
        except Exception as error:  # noqa: BLE001 - a store or thread failure
            self._last_error = type(error).__name__
            outcome = "error"
            if held:
                self._release(job.part, held)
        with self._idle:
            self._inflight.pop(job.part, None)
            self._idle.notify_all()
        return outcome

    def _job_thread(self, job: _Job, held: int) -> None:
        try:
            outcome = self._execute(job, worker=True, held=held)
            if outcome in ("caps", "error"):
                self._stalled = True
        finally:
            self._wake.set()

    def _loop(self) -> None:
        try:
            while not self._stopped:
                self._wake.clear()
                try:
                    if not self._recovered:
                        self.recover()
                    active = self._iteration()
                except Exception as error:  # noqa: BLE001 - recorded by type, then retried
                    self._last_error = type(error).__name__
                    self._wake.wait(self.settings.retry_base)
                    continue
                if active:
                    continue
                due = None if self._stalled else self._next_due()
                if due is None:
                    with self._lock:
                        if not self._wake.is_set():
                            # Retire under the lock kick() takes: a later kick sees no
                            # worker and starts one; an earlier one is seen just above.
                            if self._thread is threading.current_thread():
                                self._thread = None
                            return
                    continue
                # Nothing runnable now, but a retry or a source read is due later.
                delay = min(max(due - self._clock(), 0.05), self.settings.retry_max)
                self._wake.wait(delay)
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def _iteration(self) -> bool:
        """Plan and launch what has room; whether work is running or was just done."""
        launched, fits = 0, 0
        with self._lock:
            room = {
                lane: self.settings.jobs - sum(1 for v in self._inflight.values() if v == lane)
                for lane in LANES
            }
        if not self._stalled and any(v > 0 for v in room.values()):
            with self._plan_lock:
                plan = self._plan(room)
            fits = plan.fits
            for job in plan.jobs:
                outcome = self._launch(job)
                if outcome == "ok":
                    launched += 1
                elif outcome in ("caps", "error"):
                    self._stalled = True  # as step(): stop at a refusing cap
                    break
                elif outcome == "stopped":
                    break
        with self._lock:
            busy = bool(self._inflight)
        if not (launched or fits or busy):
            return False
        if not fits:
            self._wake.wait(1.0)  # a finished job or a kick wakes it at once
        return True

    # -- status ----------------------------------------------------------------------

    def status(self) -> dict:
        """Counts for diagnostics: no message text, error TYPES only."""
        store = self._store
        count = store.count()
        nodes = store.nodes()
        jobs = store.jobs()
        with self._lock:
            running = set(self._inflight)
            usage = {
                "known_calls": self._usage_known,
                "unknown_calls": self._usage_unknown,
                "total_tokens": self._tokens,
            }
            calls = self._calls
        blocked = {p for p, job in jobs.items() if job.state in BLOCKING_STATES} | running
        states = [job.state for job in jobs.values()]
        now = self._clock()
        waits = [
            max(0, math.ceil((job.retry_at if job.retry_at is not None else now) - now))
            for job in jobs.values()
            if job.state == "uncertain"
        ]
        return {
            "count": count,
            "built": len(nodes),
            "pending_ready": len(pt.ready_jobs(count, nodes, blocked)),
            "running": len(running),
            "reserved": states.count("reserved"),
            "retrying": states.count("uncertain"),
            "next_retry_in": min(waits) if waits else None,
            "paused_for": max(0, math.ceil(self._pause_until - now)) or None,
            "refused": states.count("refused"),
            "armed": states.count("armed"),
            "held": self._held,
            "counters": store.counters(),
            "caps_exhausted": self._caps_exhausted,
            "last_error": self._last_error,
            "calls": calls,
            "usage": usage,
        }


def _discard_owner(owner: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.discard(owner)
