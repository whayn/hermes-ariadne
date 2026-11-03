"""Explicit, immutable branch/revision projections over the canonical archive."""

import base64
import hashlib

from .store import canonical_json


class Projection:
    """Select exact versions, never infer chronology or latest revisions from hashes."""

    def __init__(self, archive, session_id, event_ids):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("an explicit session scope is required")
        if not isinstance(event_ids, (list, tuple)):
            raise ValueError("event identities must be an explicit ordered sequence")
        if any(not isinstance(item, str) for item in event_ids):
            raise ValueError("event identities must be strings")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("duplicate event identities in projection")
        identities = set()
        for item in event_ids:
            message = archive.get_event(item, session_id=session_id)
            if message is None:
                raise ValueError("source missing or outside selected session scope")
            uid = message.get("message_uid")
            if not isinstance(uid, str) or not uid or uid in identities:
                raise ValueError("ambiguous logical identity: explicitly choose one revision")
            identities.add(uid)
        self._archive = archive
        self._session_id = session_id
        self._event_ids = tuple(event_ids)
        self._projection_id = hashlib.sha256(
            canonical_json({"session_id": session_id, "event_ids": list(event_ids)})
        ).hexdigest()

    @property
    def event_ids(self):
        return self._event_ids

    @property
    def session_id(self):
        return self._session_id

    @property
    def projection_id(self):
        return self._projection_id

    def message(self, index):
        if type(index) is not int or not 0 <= index < len(self.event_ids):
            raise ValueError("source index is outside projection range")
        value = self._archive.get_event(self.event_ids[index], session_id=self.session_id)
        if value is None:
            raise ValueError("projected source missing or outside scope")
        return value

    def page(self, index, *, offset=0, limit=16384):
        """Page exact canonical bytes, not a lossy text serialization.

        Offsets/limits are BYTES. Reassemble decoded base64 chunks before decoding
        UTF-8/surrogatepass and JSON; chunks can split a multibyte codepoint.
        """
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 65536
        ):
            raise ValueError("offset/limit must be valid byte integers; limit is 1..65536")
        data = canonical_json(self.message(index))
        if offset > len(data):
            raise ValueError("offset outside source byte range")
        end = min(len(data), offset + limit)
        return {
            "projection_id": self.projection_id,
            "session_id": self.session_id,
            "event_id": self.event_ids[index],
            "encoding": "canonical-json:utf-8-surrogatepass:base64",
            "offset": offset,
            "total_bytes": len(data),
            "next_offset": end if end < len(data) else None,
            "data_base64": base64.b64encode(data[offset:end]).decode("ascii"),
        }
