"""Validated prefix identity state is shared, not one cumulative set per generation (#19).

Each validated generation retains only its lineage (the ids of its validated
chain) while a memo-wide index maps every delta identity to the generations
that introduced it. A delta collides with a validated prefix exactly when one
of its identities has an introducer in that prefix's lineage, so sibling
branches never collide with each other and nothing is recorded for a failed
validation. Verdicts are compared with the unchanged full-prefix
``_canonical_rows`` and with independent fresh validation.
"""

import copy
import json

import pytest
from test_rolling_incremental import POLICY, REQUEST, ROUTE, reasoning_with_id, set_occurrence
from test_rolling_selection import exchange
from test_rolling_target_incremental import published_parent

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.prepared import _identity_value
from hermes_ariadne.store import EventStore

SETTINGS = rolling.RollingSettings()
COMMON = {
    "session_id": "s",
    "settings": SETTINGS,
    "policy": POLICY,
    "request_bytes": REQUEST,
    "route_labels": ROUTE,
}


# --- retained memory: O(distinct identities + G^2), not O(G x prefix) ---


def identity_state(memo):
    """Yield everything reachable from the memo's identity state.

    The archive row snapshot (``rows``), the accepted views and the memo's
    scalar scope are excluded: they are the source snapshot and the views
    themselves, not identity state. Each container is yielded once however many
    states share it; leaves are yielded each time they are met.
    """
    skip = {"rows", "accepted", "store", "policy", "session_id", "settings_digest"}
    stack = [value for key, value in vars(memo).items() if key not in skip]
    seen = set()
    while stack:
        item = stack.pop()
        if isinstance(item, rolling.RollingView):
            continue
        if isinstance(item, (str, bytes, int, float, type(None))):
            yield item
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        if isinstance(item, (set, frozenset, tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item)
            stack.extend(item.values())
        elif hasattr(item, "__dict__"):
            stack.extend(vars(item).values())


def retained_entries(memo):
    """Elements of every set/dict reachable from the memo's identity state."""
    return sum(
        len(item) for item in identity_state(memo) if isinstance(item, (set, frozenset, dict))
    )


def index_mentions(memo, generation_id):
    """Whether ``generation_id`` appears anywhere in the memo's retained identity state."""
    return any(item == generation_id for item in identity_state(memo) if isinstance(item, str))


def distinct_identities(rows):
    messages, reasoning, occurrences = set(), set(), set()
    for row in rows:
        messages.add(row["message_uid"])
        if row.get("codex_reasoning_items"):
            reasoning.update(item["id"] for item in json.loads(row["codex_reasoning_items"]))
        mapping = _identity_value(row, "tool_call_uids")
        if mapping:
            occurrences.update(uid for uid in mapping.values() if uid is not None)
    return len(messages) + len(reasoning) + len(occurrences)


def test_open_retains_identity_state_linear_in_distinct_identities(tmp_path):
    generations = 12
    exchanges = 4  # the default max_delta_exchanges
    chunks = [
        [row for j in range(exchanges) for row in exchange(f"g{i}x{j}")] for i in range(generations)
    ]
    store, head = published_parent(tmp_path, chunks, [], SETTINGS)
    rows = [row for chunk in chunks for row in chunk]
    distinct = distinct_identities(rows)
    with rolling_state.RollingJournal(
        tmp_path, "s", SETTINGS, store=store, policy=POLICY
    ) as journal:
        assert journal.load_view() == head
        retained = retained_entries(journal._memo)
    assert len(head.event_ids) == len(rows) and head.sequence == generations
    # Every generation's validated state is held (one per generation), yet the
    # identity entries are those of the distinct identities plus a G x G lineage,
    # not every generation's whole-prefix union.
    assert retained <= distinct + generations * generations, (retained, distinct)


# --- branches: helpers producing self-consistent generation bytes for any parent ---


def generation(settings, parent, chunk, sequence):
    """Bytes for a generation whose delta is ``chunk``; ``parent`` is (bytes, text, rows)."""
    limit = settings.max_checkpoint_bytes
    parent_bytes, parent_text, parent_rows = parent or (None, None, [])
    rows = parent_rows + chunk
    try:
        delta = rolling.readable_delta(copy.deepcopy(chunk), policy=POLICY)
    except ValueError:
        # Same fallback as forge_chain: a chunk the readable projection rejects
        # still becomes a self-consistent generation, so any verdict comes from
        # validate_generation. Every chunk in this file is readable (the
        # collisions are across generations), so this is not exercised here.
        delta = b'{"unusable":"delta"}'
    target = rolling.Target(
        session_id="s",
        settings_digest=settings.digest,
        parent_generation=None if parent is None else rolling.sha256(parent_bytes),
        parent_summary_sha256=None if parent is None else rolling.sha256(parent_text.encode()),
        event_ids=tuple(rolling.sha256(rolling.encode_json(row, limit)) for row in rows),
        message_uids=tuple(row["message_uid"] for row in rows),
        delta_start=len(parent_rows),
        delta_stop=len(rows),
        readable_input=rolling._summary_input(parent_text, delta, settings),
        request_bytes=REQUEST,
        route_labels=ROUTE,
    )
    text = f"summary {sequence} {rolling.sha256(repr(rows).encode())[:8]}"
    data = rolling.generation_bytes(target, text, sequence=sequence, settings=settings)
    return data, text, rows


class Tree:
    """A store plus generations forming a tree; all validated through ONE memo."""

    def __init__(self, tmp_path, settings=SETTINGS):
        self.settings = settings
        self.store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
        self.memo = rolling._ValidationMemo(self.store, "s", POLICY)
        self.by_id = {}

    def add(self, parent, chunk, sequence):
        self.store.ingest("s", chunk, origin="canonical")
        node = generation(self.settings, parent, chunk, sequence)
        self.by_id[rolling.sha256(node[0])] = node[0]
        return node

    def validate(self, node):
        """Verdict through the tree's one shared memo."""
        return self._verdict(node, self.memo)

    def validate_fresh(self, node):
        """Verdict through a call-local memo: independent of anything validated before."""
        return self._verdict(node, None)

    def _verdict(self, node, memo):
        try:
            view = rolling.validate_generation(
                node[0],
                self.store,
                "s",
                self.settings,
                policy=POLICY,
                parent_loader=self.by_id.__getitem__,
                _memo=memo,
            )
        except ValueError as exc:
            return str(exc)
        return view.generation_id


def full_verdict(rows):
    try:
        rolling._canonical_rows(copy.deepcopy(rows), policy=POLICY, settings=SETTINGS)
    except ValueError as exc:
        return str(exc)
    return "ok"


def agree(tree, node):
    """Shared-memo verdict == fresh verdict, and == full-prefix canonicalization."""
    shared, fresh = tree.validate(node), tree.validate_fresh(node)
    assert shared == fresh, (shared, fresh)
    full = full_verdict(node[2])
    if full == "ok":
        assert shared == rolling.sha256(node[0])
    else:
        assert shared == full
    return shared


def collide_reasoning(chunk, ident):
    reasoning_with_id(chunk[1], ident)


def test_sibling_branches_may_reuse_each_others_identities(tmp_path):
    tree = Tree(tmp_path)
    root = tree.add(None, exchange("a"), 1)
    left = tree.add(root, exchange("l"), 2)
    right_rows = exchange("r")
    collide_reasoning(right_rows, "rs_lowner0")  # the left sibling's reasoning id
    set_occurrence(right_rows[1], right_rows[2], "call_r", "occ_l")  # and its occurrence
    right = tree.add(root, right_rows, 2)
    for node in (root, left, right):
        assert agree(tree, node) == rolling.sha256(node[0])
    # Grandchildren: an identity introduced only by the OTHER branch is free ...
    free = exchange("x")
    collide_reasoning(free, "rs_lanswer1")  # introduced only by the left sibling
    node = tree.add(right, free, 3)
    assert agree(tree, node) == rolling.sha256(node[0])
    # ... while one introduced by this branch (shared with the sibling) collides.
    taken = exchange("y")
    collide_reasoning(taken, "rs_lowner0")  # introduced by BOTH siblings
    assert agree(tree, tree.add(right, taken, 3)) == "duplicate retained reasoning identity"
    taken_occurrence = exchange("z")
    set_occurrence(taken_occurrence[1], taken_occurrence[2], "call_z", "occ_l")
    assert agree(tree, tree.add(left, taken_occurrence, 3)) == (
        "occurrence identities must be unique"
    )


def test_delta_colliding_with_a_grandparent_identity_is_rejected(tmp_path):
    tree = Tree(tmp_path)
    root = tree.add(None, exchange("a"), 1)
    middle = tree.add(root, exchange("b"), 2)
    for name, mutate, message in (
        ("c", lambda c: collide_reasoning(c, "rs_aowner0"), "duplicate retained reasoning"),
        ("d", lambda c: set_occurrence(c[1], c[2], "call_d", "occ_a"), "occurrence identities"),
    ):
        chunk = exchange(name)
        mutate(chunk)
        assert message in agree(tree, tree.add(middle, chunk, 3))
    # The same identity directly from the parent is rejected too.
    chunk = exchange("e")
    collide_reasoning(chunk, "rs_bowner0")
    assert "duplicate retained reasoning" in agree(tree, tree.add(middle, chunk, 3))


def test_delta_colliding_with_a_distant_ancestor_identity_is_rejected(tmp_path):
    """The lineage is the whole chain, not just the parent and grandparent."""
    tree = Tree(tmp_path)
    node = tree.add(None, exchange("a"), 1)
    for sequence, name in enumerate("bcde", 2):
        node = tree.add(node, exchange(name), sequence)
    chunk = exchange("f")
    collide_reasoning(chunk, "rs_aowner0")  # introduced by the root, five links up
    assert "duplicate retained reasoning" in agree(tree, tree.add(node, chunk, 6))
    chunk = exchange("g")
    set_occurrence(chunk[1], chunk[2], "call_g", "occ_a")
    assert "occurrence identities" in agree(tree, tree.add(node, chunk, 6))


def test_failed_validation_records_nothing_and_a_valid_sibling_reuses_its_ids(tmp_path):
    tree = Tree(tmp_path)
    root = tree.add(None, exchange("a"), 1)
    assert agree(tree, root) == rolling.sha256(root[0])
    # A child whose delta identities are checked fine but which fails afterwards
    # (summary input binding), so a premature record would be visible.
    bad_rows = exchange("b")
    data, text, rows = generation(SETTINGS, root, bad_rows, 2)
    value = json.loads(data)
    value["summary_input_sha256"] = "0" * 64
    bad = (rolling.encode_json(value, SETTINGS.max_checkpoint_bytes), text, rows)
    tree.store.ingest("s", bad_rows, origin="canonical")
    tree.by_id[rolling.sha256(bad[0])] = bad[0]
    assert tree.validate(bad) == "summary input hash mismatch"
    assert not index_mentions(tree.memo, rolling.sha256(bad[0]))
    # A valid sibling reusing the failed delta's reasoning/occurrence identities.
    sibling_rows = exchange("c")
    collide_reasoning(sibling_rows, "rs_bowner0")
    set_occurrence(sibling_rows[1], sibling_rows[2], "call_c", "occ_b")
    sibling = tree.add(root, sibling_rows, 2)
    assert agree(tree, sibling) == rolling.sha256(sibling[0])
    assert index_mentions(tree.memo, rolling.sha256(sibling[0]))


# --- targets from a shared parent only check; they never record state ---


def test_targets_from_a_parent_with_validated_siblings(tmp_path):
    tree = Tree(tmp_path)
    root = tree.add(None, exchange("a"), 1)
    left = tree.add(root, exchange("l"), 2)
    for node in (root, left):
        assert agree(tree, node) == rolling.sha256(node[0])
    parent = tree.memo.prefix_states[rolling.sha256(root[0])][0]
    prefix = tree.memo.prefix_states[parent.generation_id][1]
    before = retained_entries(tree.memo)

    def both(delta):
        try:
            full = rolling.Target.create(copy.deepcopy(root[2] + delta), parent=parent, **COMMON)
        except ValueError as exc:
            full = str(exc)
        try:
            extension = rolling.Target._create_extension(
                copy.deepcopy(delta), parent=parent, prefix=prefix, **COMMON
            )
        except ValueError as exc:
            extension = str(exc)
        assert extension == full
        return full

    sibling_ids = exchange("t")
    collide_reasoning(sibling_ids, "rs_lowner0")  # only the validated sibling introduced it
    set_occurrence(sibling_ids[1], sibling_ids[2], "call_t", "occ_l")
    assert isinstance(both(sibling_ids), rolling.Target)
    parent_id = exchange("u")
    collide_reasoning(parent_id, "rs_aowner0")
    assert both(parent_id) == "duplicate retained reasoning identity"
    assert isinstance(both(exchange("v")), rolling.Target)
    # Checking targets retained nothing new.
    assert retained_entries(tree.memo) == before


def test_extension_rejects_a_prefix_state_from_another_lineage(tmp_path):
    tree = Tree(tmp_path)
    root = tree.add(None, exchange("a"), 1)
    left = tree.add(root, exchange("l"), 2)
    right = tree.add(root, exchange("r"), 2)
    for node in (root, left, right):
        tree.validate(node)
    left_view, left_state = tree.memo.prefix_states[rolling.sha256(left[0])]
    right_state = tree.memo.prefix_states[rolling.sha256(right[0])][1]
    assert right_state.length == left_state.length
    with pytest.raises(ValueError, match="target is not a parent prefix extension"):
        rolling.Target._create_extension(
            exchange("c"), parent=left_view, prefix=right_state, **COMMON
        )
    assert rolling.Target._create_extension(
        exchange("c"), parent=left_view, prefix=left_state, **COMMON
    )
