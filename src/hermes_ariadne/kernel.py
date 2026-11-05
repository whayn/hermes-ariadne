"""Offline, archive-backed summary tree. No model client or live Hermes hooks.

The source projection is authoritative. This JSON snapshot is disposable derived
state, atomically saved at build/turn boundaries, not a concurrent job database.
"""

import heapq
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .store import canonical_json
from .tree import Part, View, advance_view, render_lines, utf8_size

REFERENCE_COMMIT = "0500bf6a8ea4bf0ce047488562022b1b6532cc30"
FORMAT = "hermes-ariadne-kernel"
VERSION = 1


@dataclass(frozen=True)
class SummaryRequest:
    part: Part
    inputs: tuple[str, ...]
    context_parts: tuple[Part, ...]
    context_lines: tuple[str, ...]
    context_end: int
    byte_limit: int


class MemoryKernel:
    """One explicit immutable projection, one single-writer derived snapshot.

    `summary_version` is a caller-managed cache version. Change it (and use a new
    state file) when changing the summary provider/prompt. A provider returns a
    nonempty UTF-8 summary within node_bytes, or None to leave work pending.
    """

    def __init__(
        self,
        projection,
        path,
        *,
        node_bytes=512,
        chat_high=128000,
        compactor_high=32000,
        summary_version="manual-v1",
    ):
        self._initialize(
            projection,
            path,
            node_bytes=node_bytes,
            chat_high=chat_high,
            compactor_high=compactor_high,
            summary_version=summary_version,
        )
        if self.path.exists():
            self._load()
        else:
            self._chat = advance_view(
                self._chat, len(projection.event_ids), self._nodes, next_message=True
            )
            self._save()

    @classmethod
    def from_snapshot(cls, projection, path, *, snapshot, **settings):
        """Validate captured decoded state without reading or creating its path.

        Path/archive separation and settings checks match ordinary construction.
        Invalid snapshots raise ValueError; there is no blank-state fallback.
        """
        kernel = cls.__new__(cls)
        kernel._initialize(projection, path, **settings)
        kernel._load_snapshot(snapshot)
        return kernel

    def _initialize(
        self,
        projection,
        path,
        *,
        node_bytes=512,
        chat_high=128000,
        compactor_high=32000,
        summary_version="manual-v1",
    ):
        for name, value in (
            ("node_bytes", node_bytes),
            ("chat_high", chat_high),
            ("compactor_high", compactor_high),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive byte integer")
        if not isinstance(summary_version, str) or not summary_version:
            raise ValueError("summary_version must be explicit")
        self.projection = projection
        self.path = Path(path)
        archive_path = projection._archive.path.resolve()
        archive_files = {archive_path} | {
            archive_path.with_name(archive_path.name + suffix)
            for suffix in ("-wal", "-shm", "-journal")
        }
        if self.path.is_symlink() or self.path.resolve() in archive_files:
            raise ValueError("derived state must be separate from the canonical archive")
        self.node_bytes = node_bytes
        self.summary_version = summary_version
        self._nodes = {}
        self._chat = View(high=chat_high)
        self._seen = View(high=compactor_high)
        count = len(projection.event_ids)
        self._all = tuple(
            Part(level, index)
            for level in range(count.bit_length())
            for index in range(count // (1 << level))
        )

    @property
    def parts(self):
        return self._chat.parts

    def _settings(self):
        return {
            "node_bytes": self.node_bytes,
            "chat_high": self._chat.high,
            "compactor_high": self._seen.high,
            "summary_version": self.summary_version,
        }

    def _manifest(self):
        return {
            "id": self.projection.projection_id,
            "session_id": self.projection.session_id,
            "event_ids": list(self.projection.event_ids),
        }

    @staticmethod
    def _view_data(view):
        return {
            "parts": [[p.level, p.index] for p in view.parts],
            "size": view.size,
            "batch": view.batch,
            "high": view.high,
        }

    def _save(self):
        data = {
            "format": FORMAT,
            "version": VERSION,
            "reference_commit": REFERENCE_COMMIT,
            "projection": self._manifest(),
            "settings": self._settings(),
            "nodes": [
                {"level": p.level, "index": p.index, "text": text}
                for p, text in sorted(
                    self._nodes.items(), key=lambda item: (item[0].level, item[0].index)
                )
            ],
            "chat": self._view_data(self._chat),
            "seen": self._view_data(self._seen),
        }
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical_json(data))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _load(self):
        try:
            data = json.loads(self.path.read_bytes().decode("utf-8", "surrogatepass"))
        except UnicodeError as error:
            raise ValueError("invalid derived state") from error
        self._load_snapshot(data)

    def _load_snapshot(self, data):
        try:
            if (
                data.get("format") != FORMAT
                or type(data.get("version")) is not int
                or data["version"] != VERSION
                or data.get("reference_commit") != REFERENCE_COMMIT
            ):
                raise ValueError("unrecognized derived state format/version")
            if data["projection"] != self._manifest() or data["settings"] != self._settings():
                raise ValueError("derived projection/settings mismatch; use a new state file")
            nodes = {}
            allowed = set(self._all)
            for item in data["nodes"]:
                part = Part(item["level"], item["index"])
                text = item["text"]
                if part not in allowed or part in nodes:
                    raise ValueError("invalid/duplicate derived node range")
                self._check_summary(text)
                nodes[part] = text
            for part in nodes:
                if part.level and any(child not in nodes for child in part.children):
                    raise ValueError("derived parent has missing children")
            self._nodes = nodes
            self._chat = self._load_view(data["chat"], self._chat.high, full=True)
            self._seen = self._load_view(data["seen"], self._seen.high, full=False)
        except (KeyError, TypeError, AttributeError, UnicodeError) as error:
            raise ValueError("invalid derived state") from error

    def _load_view(self, data, high, *, full):
        if (
            type(data["high"]) is not int
            or data["high"] != high
            or type(data["batch"]) is not bool
            or type(data["size"]) is not int
        ):
            raise ValueError("invalid derived view settings")
        parts = tuple(Part(*item) for item in data["parts"])
        end = 0
        for part in parts:
            if part.start != end or part.stop > len(self.projection.event_ids):
                raise ValueError("invalid derived view coverage")
            if part.level and part not in self._nodes:
                raise ValueError("derived view has missing internal node")
            end = part.stop
        size = sum(utf8_size(self._nodes.get(part, "")) for part in parts)
        if size != data["size"] or (full and end != len(self.projection.event_ids)):
            raise ValueError("invalid derived view size/coverage")
        return View(parts=parts, size=size, batch=data["batch"], high=high)

    def _check_summary(self, text):
        if not isinstance(text, str) or not text.strip() or utf8_size(text) > self.node_bytes:
            raise ValueError("summary must be nonempty UTF-8 within the node byte limit")

    def source_input(self, index):
        # Escapes make even unusual archived Unicode safe as model input. This is
        # NOT the verbatim source codec: zoom always reads canonical archive bytes.
        return json.dumps(
            self.projection.message(index),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _due(part):
        return (part.start if part.level == 0 else part.stop, -part.level, part.index)

    def build(self, compact):
        """Build ready jobs offline, checkpoint once, then publish a turn-boundary view.

        None leaves a job pending for a later invocation. No clipping, retries,
        network calls or fabricated fallback summaries. A process crash can lose
        this batch of derived work; it cannot lose canonical archive messages.
        """
        queue = []
        queued = set()

        def offer(part):
            if (
                part not in self._nodes
                and part not in queued
                and (part.level == 0 or all(c in self._nodes for c in part.children))
            ):
                heapq.heappush(queue, (self._due(part), part.level, part.index))
                queued.add(part)

        for part in self._all:
            offer(part)
        built = 0
        first = 0
        try:
            while queue:
                _, level, index = heapq.heappop(queue)
                part = Part(level, index)
                while first < len(self.projection.event_ids) and Part(0, first) in self._nodes:
                    first += 1
                cutoff = min(self._due(part)[0], first)
                self._seen = advance_view(self._seen, cutoff, self._nodes, next_message=True)
                if any(p not in self._nodes for p in self._seen.parts):
                    continue
                if self._seen.size > self._seen.high:
                    continue  # Explicit pending, not an over-budget provider request.
                inputs = (
                    (self.source_input(part.index),)
                    if level == 0
                    else tuple(self._nodes[child] for child in part.children)
                )
                text = inputs[0] if level == 0 and utf8_size(inputs[0]) <= self.node_bytes else None
                if text is None:
                    text = compact(
                        SummaryRequest(
                            part,
                            inputs,
                            self._seen.parts,
                            render_lines(self._seen, self._nodes),
                            cutoff,
                            self.node_bytes,
                        )
                    )
                if text is None:
                    continue
                self._check_summary(text)
                self._nodes[part] = text
                built += 1
                parent = Part(level + 1, index // 2)
                if parent.stop <= len(self.projection.event_ids):
                    offer(parent)
        finally:
            self._chat = advance_view(
                self._chat, len(self.projection.event_ids), self._nodes, next_message=True
            )
            # `seen` can contain a newly ready part whose previous size was zero.
            seen_end = self._seen.parts[-1].stop if self._seen.parts else 0
            self._seen = advance_view(self._seen, seen_end, self._nodes, next_message=True)
            self._save()
        return {
            "built": built,
            "pending": len(self._all) - len(self._nodes),
            "usable": self.view()["usable"],
        }

    def reference(self, part):
        return f"{self.projection.projection_id}:{part.start}+{part.count}"

    def _part(self, reference):
        if not isinstance(reference, str):
            raise ValueError("invalid node reference")
        match = re.fullmatch(r"([0-9a-f]{64}):(0|[1-9][0-9]*)\+([1-9][0-9]*)", reference)
        if not match or match[1] != self.projection.projection_id:
            raise ValueError("invalid or foreign projection reference")
        start, count = int(match[2]), int(match[3])
        if count & (count - 1) or start % count or start + count > len(self.projection.event_ids):
            raise ValueError("invalid reference range")
        return Part(count.bit_length() - 1, start // count)

    def _describe(self, part):
        return {
            "ref": self.reference(part),
            "start": part.start,
            "count": part.count,
            "ready": part in self._nodes,
            "text": self._nodes.get(part),
        }

    def view(self):
        missing = [self.reference(p) for p in self._chat.parts if p not in self._nodes]
        return {
            "projection_id": self.projection.projection_id,
            "summary_bytes": self._chat.size,
            "high": self._chat.high,
            "units": "raw-summary-utf8-bytes",
            "batch": self._chat.batch,
            "usable": not missing and self._chat.size <= self._chat.high,
            "pending": missing,
            "parts": [self._describe(p) for p in self._chat.parts],
            "lines": list(render_lines(self._chat, self._nodes)),
        }

    def prefix_view(self, end):
        """Compile a separate prefix for offline proposal simulation; do not mutate chat."""
        if type(end) is not int or not 0 <= end <= len(self.projection.event_ids):
            raise ValueError("prefix end outside source range")
        view = advance_view(self._chat, end, self._nodes, next_message=True)
        if view.size > view.high or any(p not in self._nodes for p in view.parts):
            raise ValueError("prefix view is pending or over its summary byte budget")
        return tuple({"start": p.start, "stop": p.stop, "text": self._nodes[p]} for p in view.parts)

    def zoom(self, reference, *, offset=0, limit=16384):
        part = self._part(reference)
        if part.level == 0:
            return {
                "kind": "source",
                **self._describe(part),
                "page": self.projection.page(part.start, offset=offset, limit=limit),
            }
        return {
            "kind": "node",
            **self._describe(part),
            "children": [self._describe(child) for child in part.children],
        }
