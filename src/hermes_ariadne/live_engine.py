"""Explicitly opt-in prepared-prefix engine; not automatic live compaction."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
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
        self._profile_home = None
        self._prepared = None
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
                if scope.generation != self._generation:
                    return
                outcome = scope.store.ingest(scope.session_id, messages, origin=origin)
                if scope.generation != self._generation:
                    return
                self._skipped = outcome["skipped"]
                self._archive_error = None
        except Exception as exc:
            self._record_scoped_error(scope, exc)

    def _record_scoped_error(self, scope, exc):
        error = type(exc).__name__
        with self._lock:
            if scope.generation != self._generation:
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

    def _decline(self, reason, scope):
        with self._lock:
            if scope.generation == self._generation:
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
            super().update_model(model, context_length, base_url, api_key, provider, api_mode)
            # Source identity is model-independent, but the preceding measurement isn't.
            if self._prepared is not None:
                self._selection = {
                    "status": "unmeasured",
                    "provider_fit_verified": False,
                    "host_fallback": "original_request_on_decline",
                }

    def select_context(
        self,
        request_messages,
        *,
        conversation_messages=None,
        incoming_message=None,
        budget_tokens=0,
    ):
        scope = self._capture()
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
            if scope.generation != self._generation:
                return None
            if self._archive_error:
                return self._decline("archive_unavailable", scope)
        try:
            from agent.model_metadata import estimate_request_tokens_rough

            active = scope.db.get_messages(scope.session_id, include_inactive=False)
            with self._lock:
                if scope.generation != self._generation:
                    return None
            decision = scope.prepared.select(
                request_messages,
                conversation_messages=conversation_messages,
                incoming_message=incoming_message,
                active_messages=active,
                budget_tokens=budget_tokens,
                measure_tokens=estimate_request_tokens_rough,
            )
        except Exception:
            return self._decline("selection_unavailable", scope)
        with self._lock:
            if scope.generation != self._generation:
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
