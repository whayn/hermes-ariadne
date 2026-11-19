"""The MVP context engine: rolling preparation under the plain name ``ariadne``.

In addition to the rolling engine's behaviour, it keeps the derived
conversation index (``notes.NoteStore.upsert_conversation``) that the memory
provider shows to new chats: a preview of each conversation after a turn, and
the latest published summary. Index writes are best-effort navigation data:
a failure is logged and never affects selection, archiving or publication.
"""

import logging

from .notes import NoteStore
from .rolling_engine import RollingAriadneEngine

logger = logging.getLogger(__name__)

ENGINE_NAME = "ariadne"


def _first_user_text(messages):
    for message in messages or ():
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text")
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                ]
                if any(parts):
                    return " ".join(p for p in parts if p)
    return None


class AriadneEngine(RollingAriadneEngine):
    @property
    def name(self):
        return ENGINE_NAME

    def _index(self, scope, **fields):
        if scope.home is None or not scope.session_id:
            return
        try:
            NoteStore(scope.home).upsert_conversation(scope.session_id, **fields)
        except Exception as exc:
            logger.warning("ariadne conversation index update failed: %s", type(exc).__name__)

    def _title(self, scope):
        getter = getattr(scope.db, "get_session", None)
        if getter is None:
            return None
        try:
            row = getter(scope.session_id)
        except Exception:
            return None
        title = row.get("title") if isinstance(row, dict) else getattr(row, "title", None)
        return title if isinstance(title, str) and title.strip() else None

    def on_turn_complete(self, messages, usage=None, **kwargs):
        result = super().on_turn_complete(messages, usage, **kwargs)
        scope = self._capture()
        self._index(scope, preview=_first_user_text(messages), title=self._title(scope))
        return result

    def _publish(self, scope, journal, target, text, observed, coverage):
        super()._publish(scope, journal, target, text, observed, coverage)
        self._index(scope, summary=text)
