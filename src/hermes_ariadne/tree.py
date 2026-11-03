# Copyright 2026 Higher Order Company
# SPDX-License-Identifier: Apache-2.0
# Adapted from UniiChat Memory.ts, revision
# 0500bf6a8ea4bf0ce047488562022b1b6532cc30 (VictorTaelin/UniiChat).
# Changes: immutable Python values and a pure, single-view API; no app imports.
"""UniiChat's pure, single-view selection, not provider budgeting or live wiring.

Coordinates/count/end are integers in the source roots() domain: levels 0..30,
exclusive stops/count/end 0..2**31-1. Parts must tile a prefix starting at zero.
Sizes are nonnegative integers; high is any positive integer (small stress-test
budgets are supported). Booleans are rejected in integer fields.

The caller supplies all currently ready texts, omitting pending nodes. Advancing
recounts the reached parts from this snapshot, so newly ready leaves need no
caller-side size adjustment. Missing leaves remain in coverage at size zero;
rendering skips them. A view can remain above high if parents are unavailable
or a sufficiently large background batch cannot commit. The caller, not this
pure algorithm, must decide whether such a view is usable and enforce actual
provider budgets. No I/O, tree construction, summary calls, eviction, second
view, provider wrapper/cache marks or source/lifecycle handling lives here.
"""

import heapq
import re
from collections.abc import Mapping
from dataclasses import dataclass

# ECMAScript \s is not Python's \s (notably U+FEFF and U+0085).
_JS_SPACE = r"[\u0009-\u000d\u0020\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
_NEWLINES = re.compile(_JS_SPACE + r"*\n" + _JS_SPACE + "*")
# roots() in the pinned source scans bits 30..0. Reject, rather than silently
# truncate, values outside that complete representable prefix domain.
_MAX_COUNT = (1 << 31) - 1


def _integer(name: str, value: int, minimum: int = 0, maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool)")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside supported range")


def utf8_size(text: str) -> int:
    """Raw UTF-8 bytes, with Node Buffer's lone UTF-16 surrogate replacement."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError:
        well_formed = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
        return len(well_formed.encode("utf-8"))


def flat(text: str) -> str:
    """Collapse whitespace around LF only, exactly as upstream Memory.flat."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return text if "\n" not in text else _NEWLINES.sub(" ", text)


@dataclass(frozen=True)
class Part:
    """An aligned binary node; ``stop`` is exclusive, unlike upstream span()."""

    level: int
    index: int

    def __post_init__(self) -> None:
        _integer("level", self.level, maximum=30)
        _integer("index", self.index)
        if self.stop > _MAX_COUNT:
            raise ValueError("part exceeds the supported prefix range")

    @property
    def count(self) -> int:
        return 1 << self.level

    @property
    def start(self) -> int:
        return self.index * self.count

    @property
    def stop(self) -> int:
        return self.start + self.count

    @property
    def children(self) -> tuple["Part", ...]:
        if self.level == 0:
            return ()
        return (Part(self.level - 1, self.index * 2), Part(self.level - 1, self.index * 2 + 1))


def roots(count: int) -> tuple[Part, ...]:
    """The maximal binary partition of ``[0, count)``, largest first."""
    _integer("count", count, maximum=_MAX_COUNT)
    parts = []
    start = 0
    for level in range(30, -1, -1):
        if start + (1 << level) <= count:
            parts.append(Part(level, start >> level))
            start += 1 << level
    return tuple(parts)


@dataclass(frozen=True)
class View:
    """Immutable prefix partition; size counts raw ready summary bytes only."""

    parts: tuple[Part, ...] = ()
    size: int = 0
    batch: bool = False
    high: int = 128000

    def __post_init__(self) -> None:
        if not isinstance(self.parts, tuple):
            raise TypeError("parts must be an immutable tuple")
        _integer("size", self.size)
        _integer("high", self.high, minimum=1)
        if type(self.batch) is not bool:
            raise TypeError("batch must be bool")
        end = 0
        for part in self.parts:
            if not isinstance(part, Part):
                raise TypeError("parts must contain Part values")
            if part.start != end:
                raise ValueError("parts must cover a contiguous prefix starting at zero")
            end = part.stop


def _validate_inputs(view: View, nodes: Mapping[Part, str]) -> None:
    if not isinstance(view, View):
        raise TypeError("view must be View")
    if not isinstance(nodes, Mapping):
        raise TypeError("nodes must be a Mapping[Part, str]")
    for part, text in nodes.items():
        if not isinstance(part, Part) or not isinstance(text, str):
            raise TypeError("nodes must contain Part keys and str values; omit pending nodes")


def advance_view(
    view: View, end: int, nodes: Mapping[Part, str], *, next_message: bool = False
) -> View:
    """Reach ``end`` using current ready text, retaining pending leaves at zero bytes."""
    _validate_inputs(view, nodes)
    _integer("end", end, maximum=_MAX_COUNT)
    if type(next_message) is not bool:
        raise TypeError("next_message must be bool")
    parts = [part for part in view.parts if part.stop <= end]
    start = parts[-1].stop if parts else 0
    parts.extend(Part(0, index) for index in range(start, end))
    size = sum(utf8_size(nodes[part]) for part in parts if part in nodes)
    batch = view.batch
    if size > view.high or (next_message and batch):
        merged_parts, merged_size = _merge(tuple(parts), size, view.high, nodes)
        # Twice the comparison avoids rounding the reference's half-byte low.
        if next_message or 2 * merged_size <= max(view.high, 2 * (size - view.high)):
            parts, size = list(merged_parts), merged_size
        batch = 2 * size > view.high
    return View(tuple(parts), size, batch, view.high)


def _merge(
    parts: tuple[Part, ...], size: int, high: int, nodes: Mapping[Part, str]
) -> tuple[tuple[Part, ...], int]:
    """Upstream due-priority heap over a linked view; ready adjacent siblings only."""
    end = parts[-1].stop if parts else 0
    current: list[Part | None] = list(parts)
    following = list(range(1, len(parts) + 1))
    previous = list(range(-1, len(parts)))  # includes the end sentinel
    heap: list[tuple[float, int, Part, Part]] = []

    def size_of(part: Part) -> int:
        return utf8_size(nodes[part]) if part in nodes else 0

    def offer(k: int) -> None:
        if k < 0 or k >= len(current):
            return
        j = following[k]
        a, b = current[k], current[j] if j < len(current) else None
        if (
            a is None
            or b is None
            or a.index % 2 != 0
            or b.level != a.level
            or b.index != a.index + 1
            or Part(a.level + 1, a.index // 2) not in nodes
        ):
            return
        due = (end + 1) / a.count - a.index
        heapq.heappush(heap, (-due, k, a, b))

    for k in range(len(current)):
        offer(k)
    while heap and 2 * size > high:
        _, k, a, b = heapq.heappop(heap)
        j = following[k]
        if current[k] is not a or j >= len(current) or current[j] is not b:
            continue
        up = Part(a.level + 1, a.index // 2)
        size += size_of(up) - size_of(a) - size_of(b)
        current[k], current[j] = up, None
        following[k] = following[j]
        previous[following[j]] = k
        offer(previous[k])
        offer(k)
    return tuple(part for part in current if part is not None), size


def render_lines(
    view: View, nodes: Mapping[Part, str], *, end: int | None = None
) -> tuple[str, ...]:
    """Built lines wholly before exclusive ``end``; no wrapper or cache blocks."""
    _validate_inputs(view, nodes)
    if end is not None:
        _integer("end", end, maximum=_MAX_COUNT)
    return tuple(
        f"{part.start}+{part.count}|{flat(nodes[part])}"
        for part in view.parts
        if (end is None or part.stop <= end) and part in nodes
    )
