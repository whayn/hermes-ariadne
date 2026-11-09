"""Explicitly opt-in prepared-prefix engine; not automatic live compaction."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

from .engine import AriadneEngine, logger


@dataclass(frozen=True)
class _Scope:
    generation: int
    store: Any
    db: Any
    session_id: str | None
    home: Path | None
    prepared: Any
    # Only selection snapshots include route state; evidence/archive-only hooks
    # remain attachment-scoped and model updates do not change _capture().
    route_revision: int | None = None
    reasoning_policy: Any = None


class PreparedAriadneEngine(AriadneEngine):
    """Freeze one profile/session attachment and discard work crossing its boundary.

    The host clones engines per agent and normally dispatches tools sequentially.
    The lock protects local capture/ingest/publication, never callback-capable host
    reads; generation checks also fence same-thread reentry, which RLock cannot.
    Hooks have no caller lifetime token: a call beginning *after* a rebind belongs
    to that new attachment, even if an external caller still has an old request.
    """

    @property
    def name(self):
        return "ariadne-prepared"

    def __init__(self):
        super().__init__()
        self._lock = RLock()
        self._generation = 0
        self._route_revision = 0
        self._profile_home = None
        self._prepared = None
        self._reasoning_policy = None
        self._home = None
        self._selection = {
            "status": "not_initialized",
            "provider_fit_verified": False,
            "host_fallback": "original_request_on_decline",
        }

    def _capture(self):
        with self._lock:
            return _Scope(
                self._generation,
                self._store,
                self._session_db,
                self._session_id,
                self._home,
                self._prepared,
            )

    def _observe(self, messages, *, scope=None):
        scope = scope or self._capture()
        if scope.home is None or scope.store is None or scope.session_id is None:
            return
        try:
            # Host reads can re-enter lifecycle hooks or wait for another thread.
            # Do not hold our lock here. Never resolve scope again after the read.
            if scope.db is not None:
                messages = scope.db.get_messages(scope.session_id, include_inactive=True)
                origin = "hermes_db"
            else:
                origin = "hook"
            with self._lock:
                if not self._current(scope):
                    return
                outcome = scope.store.ingest(scope.session_id, messages, origin=origin)
                if not self._current(scope):
                    return
                self._skipped = outcome["skipped"]
                self._archive_error = None
        except Exception as exc:
            self._record_scoped_error(scope, exc)

    def _record_scoped_error(self, scope, exc):
        error = type(exc).__name__
        with self._lock:
            if not self._current(scope):
                return
            self._archive_error = error
        # Logging handlers can re-enter lifecycle hooks, including host reads.
        logger.warning("Ariadne archive unavailable (%s); live context unchanged", error)

    def _clear_prepared(self):
        self._generation += 1
        self._prepared = None
        self._home = None
        self._selection = {
            "status": "not_initialized",
            "provider_fit_verified": False,
            "host_fallback": "original_request_on_decline",
        }

    def _begin_attachment(self, session_id, home):
        # Caller holds the lock. Even an absent view is frozen for this lifetime.
        self._clear_prepared()
        self._session_id = session_id
        self._home = home
        self._store = None
        self._archive_error = None
        self._skipped = 0
        return self._capture()

    def bind_session_state(self, session_db=None, session_id=None):
        with self._lock:
            changed = session_id != self._session_id or session_db is not self._session_db
            if not changed and self._home is not None:
                return
            super().bind_session_state(session_db, session_id)
            if self._profile_home is None:
                # agent_init binds before start/schema discovery: no archive IO.
                if changed:
                    self._clear_prepared()
                return
            scope = self._begin_attachment(session_id, self._profile_home)
        self._initialize(scope)

    def on_session_reset(self):
        with self._lock:
            super().on_session_reset()
            self._clear_prepared()

    def on_session_end(self, session_id, messages):
        scope = self._capture()
        if session_id != scope.session_id:
            return
        self._observe(messages, scope=scope)
        with self._lock:
            if scope.generation == self._generation:
                self._clear_prepared()

    def on_session_start(self, session_id, **kwargs):
        from hermes_constants import get_hermes_home

        with self._lock:
            home = Path(kwargs.get("hermes_home") or self._profile_home or get_hermes_home())
            self._profile_home = home
            scope = self._begin_attachment(session_id, home)
        self._initialize(scope)

    def _initialize(self, scope):
        from .prepared import PreparedView
        from .store import EventStore

        try:
            store = EventStore(scope.home / "ariadne" / "events.sqlite3")
        except Exception as exc:
            self._record_scoped_error(scope, exc)
            with self._lock:
                if scope.generation == self._generation:
                    self._selection = {
                        "status": "unavailable",
                        "provider_fit_verified": False,
                        "host_fallback": "original_request_on_decline",
                    }
            return
        with self._lock:
            if scope.generation != self._generation:
                return
            self._store = store
            scope = replace(scope, store=store)
        self._observe([], scope=scope)
        with self._lock:
            if scope.generation != self._generation:
                return
            archive_error = self._archive_error
        prepared = None if archive_error else PreparedView.load(store, scope.session_id, scope.home)
        with self._lock:
            if scope.generation != self._generation:
                return
            self._prepared = prepared
            self._selection = {
                "status": "ready" if prepared is not None else "unavailable",
                "provider_fit_verified": False,
                "host_fallback": "original_request_on_decline",
            }

    def _current(self, scope):
        # Caller holds the lock. Epochs fence away/back AND same-value updates.
        return scope.generation == self._generation and (
            scope.route_revision is None or scope.route_revision == self._route_revision
        )

    def _decline(self, reason, scope):
        with self._lock:
            if self._current(scope):
                self._selection = {
                    "status": "declined",
                    "reason": reason,
                    "provider_fit_verified": False,
                    "host_fallback": "original_request_on_decline",
                }
        return None

    def update_model(
        self, model, context_length, base_url="", api_key="", provider="", api_mode=""
    ):
        with self._lock:
            self._route_revision += 1
            revision = self._route_revision
            self._reasoning_policy = None
            # Source identity is model-independent, but the preceding measurement isn't.
            if self._prepared is not None:
                self._selection = {
                    "status": "unmeasured",
                    "provider_fit_verified": False,
                    "host_fallback": "original_request_on_decline",
                }
            super().update_model(model, context_length, base_url, api_key, provider, api_mode)
        # Native registry/profile lookup may call back or block. A newer update
        # wins even when this older computation fails or finishes last.
        policy = self._bound_reasoning_policy(model, base_url, provider, api_mode)
        with self._lock:
            if revision == self._route_revision:
                self._reasoning_policy = policy

    @staticmethod
    def _bound_reasoning_policy(model, base_url, provider, api_mode):
        # These are the agent's bound labels, NOT effective per-request wire data.
        # Invoke native profile lookup in this callback's Hermes home/context;
        # never borrow a probe registry, cache a URL, or inspect credentials.
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

            from .prepared import ReasoningReplayPolicy, _require_reasoning_policy

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

    def select_context(
        self,
        request_messages,
        *,
        conversation_messages=None,
        incoming_message=None,
        budget_tokens=0,
    ):
        with self._lock:
            scope = replace(
                self._capture(),
                route_revision=self._route_revision,
                reasoning_policy=self._reasoning_policy,
            )
        if scope.home is None:
            return None  # Construction/reset/end have no attached lifetime.
        if scope.prepared is None:
            return self._decline("prepared_unavailable", scope)
        if scope.db is None:
            return self._decline("active_source_unavailable", scope)
        if scope.prepared.projection.session_id != scope.session_id:
            return self._decline("scope_mismatch", scope)
        self._observe([], scope=scope)
        with self._lock:
            if not self._current(scope):
                return None
            if self._archive_error:
                return self._decline("archive_unavailable", scope)
        try:
            from agent.model_metadata import estimate_request_tokens_rough

            active = scope.db.get_messages(scope.session_id, include_inactive=False)
            with self._lock:
                if not self._current(scope):
                    return None
            decision = scope.prepared.select(
                request_messages,
                conversation_messages=conversation_messages,
                incoming_message=incoming_message,
                active_messages=active,
                budget_tokens=budget_tokens,
                measure_tokens=estimate_request_tokens_rough,
                reasoning_policy=scope.reasoning_policy,
            )
        except Exception:
            return self._decline("selection_unavailable", scope)
        with self._lock:
            if not self._current(scope):
                return None
            self._selection = {"status": decision.status, **decision.diagnostics}
            return decision.messages

    def get_tool_schemas(self):
        from .retrieval import tool_schemas

        return tool_schemas()

    def handle_tool_call(self, name, args, **kwargs):
        from .retrieval import TRUST_LABEL, handle_evidence_tool

        # Caller-supplied messages, session/home/checkpoint kwargs cannot set scope.
        scope = self._capture()
        store = scope.store if scope.home is not None else None
        if store is not None:
            self._observe([], scope=scope)
        with self._lock:
            stale = scope.generation != self._generation
            archive_error = self._archive_error
        if stale:
            result = {"trust": TRUST_LABEL, "ok": False, "error": "lifecycle_changed"}
        elif store is not None and archive_error:
            result = {"trust": TRUST_LABEL, "ok": False, "error": "archive_unavailable"}
        else:
            result = handle_evidence_tool(
                name,
                args,
                store=store,
                session_id=scope.session_id,
                kernel=scope.prepared.kernel if scope.prepared is not None else None,
                checkpoint_id=scope.prepared.checkpoint_id if scope.prepared is not None else None,
            )
        with self._lock:
            if scope.generation != self._generation:
                result = {"trust": TRUST_LABEL, "ok": False, "error": "lifecycle_changed"}
            return json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False)

    def get_status(self):
        with self._lock:
            return {**super().get_status(), "selection": dict(self._selection)}
