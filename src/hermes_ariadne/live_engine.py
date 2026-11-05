"""Explicitly opt-in prepared-prefix engine; not automatic live compaction."""

import json
from pathlib import Path

from .engine import AriadneEngine


class PreparedAriadneEngine(AriadneEngine):
    @property
    def name(self):
        return "ariadne-prepared"

    def __init__(self):
        super().__init__()
        self._prepared = None
        self._home = None
        self._selection = {
            "status": "not_initialized",
            "provider_fit_verified": False,
            "host_fallback": "original_request_on_decline",
        }

    def _clear_prepared(self):
        self._prepared = None
        self._home = None
        self._selection = {
            "status": "not_initialized",
            "provider_fit_verified": False,
            "host_fallback": "original_request_on_decline",
        }

    def bind_session_state(self, session_db=None, session_id=None):
        if session_id != self._session_id or session_db is not self._session_db:
            self._clear_prepared()
        super().bind_session_state(session_db, session_id)

    def on_session_reset(self):
        super().on_session_reset()
        self._clear_prepared()

    def on_session_end(self, session_id, messages):
        super().on_session_end(session_id, messages)
        if session_id == self._session_id:
            self._clear_prepared()

    def on_session_start(self, session_id, **kwargs):
        from hermes_constants import get_hermes_home

        from .prepared import PreparedView

        self._clear_prepared()
        super().on_session_start(session_id, **kwargs)
        self._home = Path(kwargs.get("hermes_home") or get_hermes_home())
        if self._store is not None and not self._archive_error:
            self._prepared = PreparedView.load(self._store, session_id, self._home)
        self._selection = {
            "status": "ready" if self._prepared is not None else "unavailable",
            "provider_fit_verified": False,
            "host_fallback": "original_request_on_decline",
        }

    def _decline(self, reason):
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
        super().update_model(model, context_length, base_url, api_key, provider, api_mode)
        # Source identity is model-independent, but the preceding measurement isn't.
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
        if self._home is None:
            return None  # Construction/reset/end have no attached lifetime.
        if self._prepared is None:
            return self._decline("prepared_unavailable")
        if self._session_db is None:
            return self._decline("active_source_unavailable")
        if self._prepared.projection.session_id != self._session_id:
            return self._decline("scope_mismatch")
        self._observe([])
        if self._archive_error:
            return self._decline("archive_unavailable")
        try:
            from agent.model_metadata import estimate_request_tokens_rough

            active = self._session_db.get_messages(self._session_id, include_inactive=False)
            decision = self._prepared.select(
                request_messages,
                conversation_messages=conversation_messages,
                incoming_message=incoming_message,
                active_messages=active,
                budget_tokens=budget_tokens,
                measure_tokens=estimate_request_tokens_rough,
            )
        except Exception:
            return self._decline("selection_unavailable")
        self._selection = {"status": decision.status, **decision.diagnostics}
        return decision.messages

    def get_tool_schemas(self):
        from .retrieval import tool_schemas

        return tool_schemas()

    def handle_tool_call(self, name, args, **kwargs):
        from .retrieval import TRUST_LABEL, handle_evidence_tool

        # Never use caller-supplied live messages as an alternative source/scope.
        store = self._store if self._home is not None else None
        if store is not None:
            self._observe([])
        if store is not None and self._archive_error:
            result = {"trust": TRUST_LABEL, "ok": False, "error": "archive_unavailable"}
        else:
            result = handle_evidence_tool(
                name,
                args,
                store=store,
                session_id=self._session_id,
                kernel=self._prepared.kernel if self._prepared is not None else None,
                checkpoint_id=self._prepared.checkpoint_id if self._prepared is not None else None,
            )
        return json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False)

    def get_status(self):
        return {**super().get_status(), "selection": dict(self._selection)}
