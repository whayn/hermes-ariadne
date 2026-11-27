"""Session binding, archive admission and request correspondence for the tree engine.

Issue #70, phase 1: what ``TreeAriadneEngine`` used from the development engines
(``engine.AriadneEngine`` and ``rolling_engine.RollingAriadneEngine``), copied without
the rolling journal, pins and preparation state. Behaviour is unchanged:

* binding and revocation scopes (``_capture``/``_current``/``_guard``): every host
  callback captures the binding first and re-checks it before writing, so a session
  switch or an archive refusal revokes in-flight work;
* archive admission (``private_paths.archive_path``) before EventStore opens or creates SQLite
  files, and observation of the session's canonical SessionDB rows into the archive;
* the request correspondence proof ``_validated_anchor`` (``host_request``);
* declines (``_decline``) and the sticky ``unsupported_carrier`` refusal;
* diagnostics: ``_note`` only queues (names and exception types, never content);
  ``_flushing`` public entry points emit them after the engine lock is released.

Declines return None: the host sends its original request.
"""

import copy
import functools
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from types import SimpleNamespace
from typing import Any

from agent.context_engine import ContextEngine

from .host_request import (
    _REASONING_CARRIERS,
    POLICY_ID,
    ReasoningReplayPolicy,
    _admit_messages,
    _comparison,
    _exact_occurrences,
    _exact_payloads,
    _leading,
    _reasoning_permissions,
    _require_reasoning_policy,
    _uids,
    completed_exchange_starts,
    incoming_anchor,
)
from .private_paths import archive_path
from .store import canonical_json

logger = logging.getLogger(__name__)
_NOTE_TOKEN = re.compile(r"[^A-Za-z0-9_.=-]")
_MAX_PENDING_NOTES = 64


def _refused_carrier(error):
    """The reasoning carrier key a validation refusal named, if any (cause chain, bounded)."""
    for _ in range(8):
        if error is None:
            return None
        carrier = getattr(error, "carrier", None)
        if carrier in _REASONING_CARRIERS:
            return carrier  # A fixed key name: safe for status and logs.
        error = error.__cause__ or error.__context__
    return None


def _flushing(method):
    """Emit queued diagnostics once the public entry point has left the engine lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self._flush_notes()

    return wrapper


def bound_reasoning_policy(model, base_url, provider, api_mode):
    """The Codex Responses replay policy of the agent's bound route, or None.

    These are the agent's bound labels, NOT effective per-request wire data. Invoke
    native profile lookup in this callback's Hermes home/context; never borrow a probe
    registry, cache a URL, or inspect credentials.
    """
    if any(
        type(value) is not str or not value.strip()
        for value in (model, base_url, provider, api_mode)
    ):
        return None
    if provider != "openai-codex" or api_mode != "codex_responses":
        return None
    try:
        from agent.codex_responses_adapter import (
            _classify_responses_issuer,
            _wire_model_identity,
            classify_responses_route,
        )
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():  # Native lookup deliberately bypasses profiles here.
            return None
        flags = classify_responses_route(SimpleNamespace(provider=provider, base_url=base_url))
        issuer = _classify_responses_issuer(base_url=base_url, **flags._asdict())
        normalized = _wire_model_identity(model)
        if type(normalized) is not str:
            return None
        policy = ReasoningReplayPolicy(issuer, normalized)
        _require_reasoning_policy(policy)
        return policy
    except Exception:
        return None


@dataclass(frozen=True)
class _Scope:
    epoch: int
    revision: int
    home: Any
    store: Any
    db: Any
    session_id: Any
    policy: Any
    route_labels: tuple


class SessionBoundEngine(ContextEngine):
    """Per-agent binding state; host reads never hold the engine lock.

    Subclasses provide ``name``, ``_initialize(scope)`` (attach the archive and their
    own state), ``select_context``, the tools and ``get_status``.
    """

    def __init__(self):
        self._store = None
        self._session_id = None
        self._session_db = None
        self._archive_error = None
        self._skipped = 0
        self._lock = RLock()
        self._generation = self._route_revision = 0
        self._profile_home = self._home = self._reasoning_policy = None
        self._selection = {
            "status": "not_initialized",
            "provider_fit_verified": False,
            "host_fallback": "original_request_on_decline",
        }
        self._route_labels = ("unbound", "unbound")
        # Sticky per session binding: a refused reasoning carrier stays in canonical history
        # (or is re-added to every assembled request), so retrying would only refuse again.
        self._carrier = None
        self._selected_incoming = None  # uid of the user turn whose request was last selected
        self._notes = []
        self._notes_lock = Lock()
        self._last_note = None

    # -- binding scopes ----------------------------------------------------------------

    def _capture(self):
        with self._lock:
            return _Scope(
                self._generation,
                self._route_revision,
                self._home,
                self._store,
                self._session_db,
                self._session_id,
                self._reasoning_policy,
                self._route_labels,
            )

    def _current(self, scope):
        return scope.epoch == self._generation and scope.revision == self._route_revision

    def _guard(self, scope):
        with self._lock:
            if not self._current(scope):
                raise ValueError("rolling capture changed")

    def update_model(
        self, model, context_length, base_url="", api_key="", provider="", api_mode=""
    ):
        with self._lock:
            self._route_revision += 1
            revision = self._route_revision
            self._reasoning_policy = None
            self._route_labels = (str(provider or "unbound"), str(model or "unbound"))
            super().update_model(model, context_length, base_url, api_key, provider, api_mode)
        # Reuse the audited native classifier, never an alternate route registry.
        policy = self._bound_reasoning_policy(model, base_url, provider, api_mode)
        with self._lock:
            if revision == self._route_revision:
                self._reasoning_policy = policy
                if policy:
                    self._route_labels = (policy.issuer_kind, policy.issuer_model)

    _bound_reasoning_policy = staticmethod(bound_reasoning_policy)

    # -- diagnostics -------------------------------------------------------------------

    def _note(self, level, event, **fields):
        """Queue a back-out diagnostic; names and exception types only.

        Never include messages, payloads or paths: histories are private. Values are
        reduced to a fixed-token alphabet so caller-supplied text cannot reach a log
        line or forge one. The record is only queued: handlers may block, so it is
        emitted by ``_flush_notes`` after the public entry point has released the
        engine lock. The engine is otherwise silent, so this is the only trace of why
        a host turn selected nothing.
        """
        exc = sys.exc_info()[0]
        if exc is not None:
            fields["error"] = exc.__name__
        text = " ".join(f"{k}={_NOTE_TOKEN.sub('?', str(v))[:64]}" for k, v in fields.items())
        with self._notes_lock:
            if len(self._notes) >= _MAX_PENDING_NOTES:
                del self._notes[0]
            self._notes.append((level, event, text))

    def _flush_notes(self):
        with self._notes_lock:
            pending, self._notes = self._notes, []
            emit = []
            for level, event, text in pending:
                key = (event, text)
                if key != self._last_note:
                    self._last_note = key
                    emit.append((level, event, text))
        for level, event, text in emit:  # Outside every engine lock.
            logger.log(level, "Ariadne %s%s", event, f" ({text})" if text else "")

    def _mid_turn(self, incoming):
        """Whether this user turn already had a selected request (caller holds the lock)."""
        uid = incoming.get("message_uid") if type(incoming) is dict else None
        return uid is not None and uid == self._selected_incoming

    def _decline(self, reason, scope, *, incoming=None, carrier=None, shape_check=None):
        with self._lock:
            if self._current(scope):
                if self._carrier is not None:
                    # Sticky, whichever path declined (e.g. a concurrent refusal followed
                    # by a host-read failure): report what actually disables selection.
                    reason, carrier = "unsupported_carrier", self._carrier
                # A decline after this turn already selected makes the host switch the
                # in-flight tool loop back to the full original history.
                mid_turn = self._mid_turn(incoming)
                fields = {"reason": reason}
                if carrier:
                    fields["carrier"] = carrier
                if reason == "unsupported_shape" and shape_check:
                    fields["shape_check"] = shape_check
                if mid_turn:
                    fields["mid_turn"] = 1
                self._note(logging.INFO, "declined", **fields)
                self._selection = {
                    "status": "declined",
                    **fields,
                    "mid_turn": mid_turn,
                    "provider_fit_verified": False,
                    "host_fallback": "original_request_on_decline",
                }
        return None

    def _unsupported_carrier(self, scope, carrier):
        with self._lock:
            if self._current(scope):
                if self._carrier is None:
                    self._note(
                        logging.WARNING,
                        "unsupported carrier; selection disabled for this session",
                        carrier=carrier,
                    )
                self._carrier = self._carrier or carrier

    # -- session lifecycle -------------------------------------------------------------

    def _detach(self):
        self._generation += 1
        self._home = self._store = None
        self._carrier = self._selected_incoming = None
        self._selection = {
            "status": "not_initialized",
            "provider_fit_verified": False,
            "host_fallback": "original_request_on_decline",
        }

    def _initialize(self, scope):
        raise NotImplementedError

    @_flushing
    def bind_session_state(self, session_db=None, session_id=None):
        with self._lock:
            if (
                session_db is self._session_db
                and session_id == self._session_id
                and self._home is not None
            ):
                return
            self._detach()
            self._session_db, self._session_id = session_db, session_id
            if self._profile_home is None:
                return
            self._home = self._profile_home
            scope = self._capture()
        self._initialize(scope)

    @_flushing
    def on_session_start(self, session_id, **kwargs):
        # Explicit initial home only: never resolve an ambient/live profile here.
        with self._lock:
            home = kwargs.get("hermes_home") or self._profile_home
            self._detach()
            self._session_id = session_id
            if home is None:
                return
            self._profile_home = self._home = Path(home).absolute()
            scope = self._capture()
        self._initialize(scope)

    def on_session_reset(self):
        with self._lock:
            super().on_session_reset()
            self._archive_error = None
            self._skipped = 0
            self._detach()
            self._session_db = self._session_id = None

    def on_session_end(self, session_id, messages):
        with self._lock:
            if session_id == self._session_id:
                self._detach()
                self._session_db = self._session_id = None

    # -- archive -----------------------------------------------------------------------

    def _admit_archive(self, scope):
        with self._lock:
            self._guard(scope)
            try:
                if scope.store is None or scope.store is not self._store:
                    raise ValueError("rolling archive attachment changed")
                archive_path(scope.home, store=scope.store)
            except Exception:
                self._refuse_archive(scope)
                raise

    def _refuse_archive(self, scope):
        with self._lock:
            if self._current(scope):
                self._note(logging.WARNING, "archive refused; engine detached")
                # Revoke captured writers as well as the current attachment.
                self._detach()

    def _read(self, scope, inactive):
        self._admit_archive(scope)
        rows = scope.db.get_messages(scope.session_id, include_inactive=inactive)
        self._admit_archive(scope)
        if type(rows) is not list or any(
            type(r) is not dict or r.get("session_id", scope.session_id) != scope.session_id
            for r in rows
        ):
            raise ValueError("unsupported canonical binding")
        return copy.deepcopy(rows)

    def _observe_scope(self, scope):
        rows = self._read(scope, True)
        with self._lock:
            self._admit_archive(scope)
            outcome = scope.store.ingest(scope.session_id, rows, origin="hermes_db")
            self._guard(scope)
            self._skipped = outcome["skipped"]
        return rows

    # -- request correspondence --------------------------------------------------------

    def _validated_anchor(self, scope, request, history, incoming, active):
        _admit_messages(request, history, active, [incoming])
        split, raw_split = _leading(request), _leading(history)
        if not split:
            raise ValueError("explicit host prefix required")
        if raw_split and canonical_json(
            [_comparison(m) for m in history[:raw_split]]
        ) != canonical_json([_comparison(m, wire=True) for m in request[:split]]):
            raise ValueError("host prefix mismatch")
        history = history[raw_split:]
        uids = _uids(history)
        if len(active) > len(history) or _uids(active) != uids[: len(active)]:
            raise ValueError("canonical branch mismatch")
        policies = _reasoning_permissions(
            history, active, request[split:], 0, scope.policy, retirement_policy=POLICY_ID
        )
        payloads = _exact_payloads(
            history,
            active,
            policies,
            assembled=request[split:],
            delivery_ids=True,
            carriers=True,
            memory_context=True,
        )
        _exact_occurrences(history, active, policies, carriers=True, memory_context=True)
        completed_exchange_starts(payloads, allow_current=True)
        if payloads[-1]["role"] not in {"user", "tool"}:
            raise ValueError("not an incoming request")
        return incoming_anchor(history, incoming), uids

    @staticmethod
    def _measure(messages):
        from agent.model_metadata import estimate_request_tokens_rough

        return estimate_request_tokens_rough(messages)

    # -- ContextEngine: never compresses -----------------------------------------------

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(
        self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""
    ):
        return messages

    def get_status(self):
        return {
            **super().get_status(),
            "archive_error": self._archive_error,
            "archive_skipped_last_scan": self._skipped,
            "archive_bound_to_durable_store": self._session_db is not None,
        }
