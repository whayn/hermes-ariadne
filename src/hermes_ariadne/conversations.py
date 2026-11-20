"""Read-only, cross-session access to Ariadne's exact message archive.

Used by the memory provider to let a new chat look into past conversations.
It never creates or migrates the archive: a missing archive reads as empty.
Payloads are rehashed against their content-addressed ids before use, and
reasoning/replay carriers are removed from everything returned (the archive
itself stays exact). Results are historical data, never instructions.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from .notes import StorageRefused
from .retrieval import _require_supported_data_use, _UnsupportedDataUse
from .rolling_engine import _archive_path
from .store import APPLICATION_ID, carrier_free, search_text

MAX_PAGE = 50
MAX_MESSAGE_CHARS = 4000
MAX_ZOOM_CHARS = 16000
PREVIEW_CHARS = 240


def _text_of(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


class RestrictedEvidence(ValueError):
    """An archived record carries a machine data-use restriction or withheld text."""


def _check(event_id, payload):
    """Rehash, decode and admit one exact record before any projection."""
    if hashlib.sha256(payload).hexdigest() != event_id:
        raise ValueError("archived event payload does not match its identity")
    message = json.loads(payload.decode("utf-8", "surrogatepass"))
    try:
        _require_supported_data_use(message)  # same policy as evidence retrieval
    except _UnsupportedDataUse as exc:
        raise RestrictedEvidence("archived record is restricted") from exc
    return message


_LINKAGE_KEYS = ("tool_call_id", "name", "_tool_call_uid")


def _linkage(message):
    """Short identifiers tying a tool result to its call (bounded, strings only)."""
    return {
        key: message[key]
        for key in _LINKAGE_KEYS
        if isinstance(message.get(key), str) and len(message[key]) <= 200
    }


def _full_text(message):
    """Content plus tool calls, so zoom can page everything read() truncates."""
    text = _text_of(message)
    calls = message.get("tool_calls")
    if calls:
        encoded = calls if isinstance(calls, str) else json.dumps(calls, ensure_ascii=False)
        text += ("\n\n" if text else "") + "tool_calls: " + encoded
    return text


def _page_args(offset, limit, maximum):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError("invalid page")


class ArchiveReader:
    def __init__(self, home):
        self.home = Path(home)

    def _connect(self):
        try:
            archive = _archive_path(
                self.home, private_home=False
            )  # metadata-only admission, no creation
        except ValueError as exc:
            raise StorageRefused(str(exc)) from exc
        if not archive.is_file():
            return None
        conn = sqlite3.connect(archive.as_uri() + "?mode=ro", uri=True, timeout=10)
        if conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            conn.close()
            raise StorageRefused("not an ariadne archive")
        return conn

    def _rows(self, sql, params):
        conn = self._connect()
        if conn is None:
            return []
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def page(self, session_id, *, offset=0, limit=10):
        """Exact archived messages of one session in first-observed order.

        Edited messages appear as separate versions sharing a ``message_uid``.
        Long contents are clipped per message (``truncated``); use ``event``.
        """
        _page_args(offset, limit, MAX_PAGE)
        rows = self._rows(
            "SELECT e.event_id, e.payload_json, MIN(o.position) AS pos "
            "FROM events e JOIN observations o USING (event_id) WHERE o.session_id=? "
            "GROUP BY e.event_id ORDER BY pos, e.event_id LIMIT ? OFFSET ?",
            (session_id, limit + 1, offset),
        )
        messages = []
        for event_id, payload, position in rows[:limit]:
            message, omitted = carrier_free(_check(event_id, payload))
            text = _text_of(message)
            calls = message.get("tool_calls")
            calls_text = json.dumps(calls, ensure_ascii=False) if calls else ""
            messages.append(
                {
                    "event_id": event_id,
                    "message_uid": message.get("message_uid"),
                    "role": message.get("role"),
                    "position": position,
                    "content": text[:MAX_MESSAGE_CHARS],
                    "truncated": len(text) > MAX_MESSAGE_CHARS,
                    "chars": len(text),
                    "tool_calls": calls if len(calls_text) <= MAX_MESSAGE_CHARS else None,
                    "tool_calls_truncated": len(calls_text) > MAX_MESSAGE_CHARS,
                    "omitted_keys": omitted,
                    **_linkage(message),
                }
            )
        return {
            "session_id": session_id,
            "offset": offset,
            "messages": messages,
            "has_more": len(rows) > limit,
        }

    def event(self, session_id, event_id, *, char_offset=0, max_chars=MAX_ZOOM_CHARS):
        """One exact message (content, then tool calls) paged by characters; carriers omitted."""
        if type(char_offset) is not int or char_offset < 0:
            raise ValueError("invalid offset")
        if type(max_chars) is not int or not 1 <= max_chars <= MAX_ZOOM_CHARS:
            raise ValueError("invalid size")
        if not isinstance(event_id, str) or len(event_id) != 64:
            raise ValueError("invalid event id")
        rows = self._rows(
            "SELECT e.event_id, e.payload_json FROM events e WHERE e.event_id=? AND EXISTS "
            "(SELECT 1 FROM observations o WHERE o.event_id=e.event_id AND o.session_id=?)",
            (event_id, session_id),
        )
        if not rows:
            return None
        message, omitted = carrier_free(_check(*rows[0]))
        text = _full_text(message)
        chunk = text[char_offset : char_offset + max_chars]
        return {
            "session_id": session_id,
            "event_id": event_id,
            "role": message.get("role"),
            "char_offset": char_offset,
            "content": chunk,
            "chars": len(text),
            "has_more": char_offset + len(chunk) < len(text),
            "omitted_keys": omitted,
            **_linkage(message),
        }

    def search(self, query, *, limit=10, exclude=None):
        """Literal search across all archived sessions; bounded previews only."""
        if not isinstance(query, str) or not query or len(query) > 500:
            raise ValueError("invalid query")
        _page_args(0, limit, MAX_PAGE)
        rows = self._rows(
            "SELECT e.event_id, e.payload_json, o.session_id, MIN(o.position) "
            "FROM events e JOIN observations o USING (event_id) "
            "WHERE o.session_id != ? AND instr(e.search_text, ?) > 0 "
            "GROUP BY e.event_id, o.session_id ORDER BY o.session_id, MIN(o.position) LIMIT ?",
            (exclude or "", search_text(query), limit),
        )
        hits = []
        for event_id, payload, session_id, position in rows:
            message, _ = carrier_free(_check(event_id, payload))
            text = " ".join(_text_of(message).split())
            at = max(0, text.find(query) - PREVIEW_CHARS // 3)
            hits.append(
                {
                    "session_id": session_id,
                    "event_id": event_id,
                    "role": message.get("role"),
                    "position": position,
                    "preview": text[at : at + PREVIEW_CHARS],
                }
            )
        return hits
