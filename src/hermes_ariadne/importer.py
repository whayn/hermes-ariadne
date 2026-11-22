"""Archive conversations that the agent read from user-supplied files.

The agent interprets the files (any format) with Hermes's own tools and passes
plain ``role``/``content`` messages here in batches. Each imported conversation
gets a deterministic id derived from the agent-chosen ``key`` (for example the
file name plus the export's own conversation id), so re-importing the same
batch is a no-op. Messages land in the same exact archive as live chats and are
then appended to the profile memory tree's log (origin ``import``, in import
order) so they appear in every chat's view and open with ``ariadne_zoom``; the
summaries build in the background. The agent-written summary is index data
labelled as such, never canonical. The result carries an estimate of the summary
calls the batch adds (``tree_builder.estimate``, informational, never a cap).
"""

import hashlib
import json

from .conversations import RestrictedEvidence
from .notes import NoteStore, StorageRefused, conversation_fields
from .receipts import _mkdir_private
from .retrieval import _require_supported_data_use, _UnsupportedDataUse
from .rolling_engine import _archive_path
from .store import EventStore, canonical_json

IMPORT_PREFIX = "import-"
ROLES = ("user", "assistant", "system", "tool")
MAX_KEY_CHARS = 500
MAX_BATCH = 500
MAX_MESSAGE_BYTES = 65536
MAX_BATCH_BYTES = 2 * 1024 * 1024
MAX_POSITION = 1_000_000


def conversation_id(key):
    if not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY_CHARS:
        raise ValueError("invalid import key")
    return IMPORT_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _messages(messages, conv_id, offset):
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_BATCH:
        raise ValueError("messages must be a nonempty bounded list")
    if type(offset) is not int or not 0 <= offset <= MAX_POSITION - len(messages):
        raise ValueError("invalid offset")
    rows, total = [], 0
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or item.get("role") not in ROLES:
            raise ValueError("each message needs a supported role")
        try:  # refuse, never strip, machine data-use restrictions (same policy as retrieval)
            _require_supported_data_use(item)
        except _UnsupportedDataUse as exc:
            raise RestrictedEvidence("import message is restricted") from exc
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        size = len(content.encode("utf-8", "surrogatepass"))
        total += size
        if size > MAX_MESSAGE_BYTES or total > MAX_BATCH_BYTES:
            raise ValueError("import batch is too large")
        row = {
            "message_uid": hashlib.sha256(f"{conv_id}:{offset + index}".encode()).hexdigest()[:32],
            "role": item["role"],
            "content": content,
        }
        for optional in ("timestamp", "speaker"):
            value = item.get(optional)
            if value is not None:
                if not isinstance(value, str) or len(value) > 200:
                    raise ValueError(f"invalid {optional}")
                row[optional] = value
        rows.append(row)
    return rows


def _source_text(source):
    if source is None:
        return None
    return json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Importer:
    def __init__(self, home, node_bytes=None):
        self.home = home
        self.node_bytes = node_bytes

    def append_to_tree(self, conv_id, rows, *, title=None, source=None):
        """Append archived import rows to the profile log; returns appended + estimate."""
        from . import profile_tree as pt
        from .tree import utf8_size
        from .tree_builder import estimate
        from .tree_store import TreeStore, log_entry_for

        node_bytes = self.node_bytes or pt.NODE_BYTES
        store = TreeStore(self.home)
        before = store.count()
        entries = [log_entry_for(hashlib.sha256(canonical_json(r)).hexdigest(), r) for r in rows]
        positions = store.append(
            conv_id, entries, origin="import", title=title, source=_source_text(source)
        )
        new = [(p, e, r) for p, e, r in zip(positions, entries, rows) if p >= before]
        tag = store.sessions()[conv_id].tag
        sizes = [
            utf8_size(
                pt.leaf_source(
                    pt.LogEntry(
                        position=p,
                        session_id=conv_id,
                        event_id=e["event_id"],
                        message_uid=e["message_uid"],
                        role=e["role"],
                        date=pt.iso_or_none(e["date"]),
                        origin="import",
                        restricted=e["restricted"],
                    ),
                    r,
                    tag,
                )
            )
            for p, e, r in new
        ]
        existing = min((p for p, _, _ in new), default=before)
        return {
            "appended": len(new),
            "estimate": estimate(sizes, node_bytes, existing=existing),
            "summaries": "built in the background; the view shows new lines at once",
        }

    def import_batch(self, key, messages, *, offset=0, title=None, summary=None, source=None):
        conv_id = conversation_id(key)
        rows = _messages(messages, conv_id, offset)
        preview = next(
            (r["content"] for r in rows if r["role"] == "user" and isinstance(r["content"], str)),
            None,
        )
        preview = preview if offset == 0 else None
        # Every refusal happens before the archive is written: metadata, then storage.
        conversation_fields(title=title, preview=preview, summary=summary, source=source)
        notes = NoteStore(self.home)
        try:
            notes.prepare()
            archive = _archive_path(
                self.home, private_home=False
            )  # admission before any file is created
            _mkdir_private(archive.parent)
            archive = _archive_path(self.home, private_home=False)
        except ValueError as exc:
            raise StorageRefused(str(exc)) from exc
        outcome = EventStore(archive).ingest(conv_id, rows, origin="import", start=offset)
        notes.upsert_conversation(
            conv_id,
            title=title,
            preview=preview,
            summary=summary,
            origin="import",
            source=source,
        )
        try:
            tree = self.append_to_tree(conv_id, rows, title=title, source=source)
        except StorageRefused:
            tree = {"error": "storage_refused"}
        except Exception:  # noqa: BLE001 - archived; resending the batch appends it
            tree = {"error": "unavailable"}
        return {
            "id": conv_id,
            "archived": outcome["inserted"],
            "already_present": len(rows) - outcome["inserted"],
            "next_offset": offset + len(rows),
            "tree": tree,
        }
