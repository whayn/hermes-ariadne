"""Incremental target rebuild on journal open (#19 S3): O(delta) per job, proven equivalent."""

import copy
import dataclasses
import sqlite3
import sys
from collections import Counter

import pytest
from rolling_prefix_helpers import empty_prefix
from test_rolling_incremental import (
    REQUEST,
    ROUTE,
    assistant_first,
    bad_companion,
    bad_role,
    dup_message_uid_first,
    dup_message_uid_middle,
    dup_occurrence,
    dup_reasoning_answer,
    dup_reasoning_owner,
    forge_chain,
    in_delta,
    message_text_mismatch,
    no_trailing_assistant,
    orphan_tool,
    restricted_data_use,
    reused_call_id,
    reused_native_ids,
    split_tool_result_into_parent_call,
    tool_first,
)
from test_rolling_journal_validation import chain
from test_rolling_selection import POLICY, exchange
from test_rolling_state import ready

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.store import EventStore


def spy_rebuild_canonicalization(monkeypatch):
    """Rows the target-rebuild path passes to the real ``_canonical_rows`` (by message uid).

    Calls made by ``validate_generation`` itself are the generation-validation
    cost, which test_rolling_validation_memo/test_rolling_incremental cover.

    Coupling: patches the private ``rolling._canonical_rows`` and filters on the
    caller's function name, as spy_full_validations in test_rolling_validation_memo does.
    """
    rebuilt = Counter()
    original = rolling._canonical_rows

    def spy(source, **kwargs):
        if sys._getframe(1).f_code.co_name != "validate_generation":
            rebuilt.update(row["message_uid"] for row in source)
        return original(source, **kwargs)

    monkeypatch.setattr(rolling, "_canonical_rows", spy)
    return rebuilt


def test_open_canonicalizes_each_target_row_at_most_once(tmp_path, monkeypatch):
    store, settings, views, pending, _ = chain(tmp_path, 6, pending=True)
    rebuilt = spy_rebuild_canonicalization(monkeypatch)
    # root targets use Target.create, which canonicalizes twice
    root_rows = set(views[0].message_uids)
    allowed = {uid: 2 if uid in root_rows else 1 for uid in pending.message_uids}
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        # Opening reconciles: every job's frozen target is rebuilt from the archive.
        # Seven jobs (six published + one ready); each job's delta rows are
        # canonicalized once, never again as part of a descendant's prefix.
        assert dict(rebuilt) == allowed
        rebuilt.clear()
        assert journal.pending_target() == pending  # a later operation: still linear
        assert set(rebuilt) <= set(allowed)
        assert all(count <= allowed[uid] for uid, count in rebuilt.items())


# --- differential oracle: Target.create(full rows) == Target._create_extension(delta only) ---

BAD_REQUESTS = {
    "request-not-canonical-json": b'[ {"content":"summarize","role":"user"} ]',
    "request-empty-list": b"[]",
    "request-unsupported-role": b'[{"content":"summarize","role":"tool"}]',
    "request-extra-key": b'[{"content":"summarize","name":"x","role":"user"}]',
    "request-not-json": b"summarize",
}


def case(name, build, **params):
    return pytest.param(build, params, id=name)


def wrong_session(c):
    c[1]["session_id"] = "other"


def drop_delta(a, b, c):
    c[:] = []  # a target that adds nothing to its parent


def noop(a, b, c):
    return None


# Mirrors CASES in test_rolling_incremental (same ids and builders), for the Target
# constructors; request/route admission cases are added because only Target has them.
CORPUS = [
    # accepted
    case("valid", noop),
    case("reused-call-id-across-exchanges", reused_call_id),
    case("reused-native-item-ids", reused_native_ids),
    case("bound-exact-length", noop, limits={"max_prefix_events": 12}),
    # violations spanning the parent/delta boundary
    case("dup-message-uid-first", dup_message_uid_first),
    case("dup-message-uid-middle", dup_message_uid_middle),
    case("dup-reasoning-id-owner", dup_reasoning_owner),
    case("dup-reasoning-id-answer", dup_reasoning_answer),
    case("dup-occurrence-uid", dup_occurrence),
    case("tool-result-for-parent-call", split_tool_result_into_parent_call),
    case("delta-starts-with-tool", tool_first),
    case("delta-starts-with-assistant", assistant_first),
    case("beyond-max-prefix-events", noop, limits={"max_prefix_events": 11}),
    # entirely inside the delta
    case("delta-bad-role", in_delta(bad_role)),
    case("delta-bad-companion", in_delta(bad_companion)),
    case("delta-text-mismatch", in_delta(message_text_mismatch)),
    case("delta-restricted-data-use", in_delta(restricted_data_use)),
    case("delta-orphan-tool", in_delta(orphan_tool)),
    case("delta-no-trailing-assistant", no_trailing_assistant),
    case("delta-wrong-session-row", in_delta(wrong_session)),
    case("non-advancing-target", drop_delta),
    # request and route admission
    *[case(name, noop, request=data) for name, data in BAD_REQUESTS.items()],
    case("route-labels-differ-from-policy", noop, route=("codex_backend", "other-model")),
    case("route-label-count", noop, route=("codex_backend",)),
    case("route-labels-not-a-tuple", noop, route=["codex_backend", "gpt-test"]),
]
SHAPES = {"chain": lambda a, b: [a, b], "root": lambda a, b: [a + b]}


def validated_parent(tmp_path, chunks, settings):
    """The real validate_generation over forged-but-valid generations: (view, prefix state)."""
    store, generations = forge_chain(tmp_path, chunks, settings)
    by_id = {rolling.sha256(data): data for data in generations}
    memo = rolling._ValidationMemo(store, "s", POLICY)
    view = rolling.validate_generation(
        generations[-1],
        store,
        "s",
        settings,
        policy=POLICY,
        parent_loader=by_id.__getitem__,
        _memo=memo,
    )
    recorded = memo.prefix_states[view.generation_id]
    assert recorded[0] is view
    return view, recorded[1]


def outcome(build):
    try:
        return "ok", build()
    except ValueError as exc:
        return "reject", str(exc)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("build,params", CORPUS)
def test_extension_target_equals_full_prefix_target(tmp_path, build, params, shape):
    settings = rolling.RollingSettings(**params.get("limits", {}))
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    build(a, b, c)
    parent, prefix = validated_parent(tmp_path, SHAPES[shape](a, b), settings)
    common = {
        "session_id": "s",
        "settings": settings,
        "policy": POLICY,
        "request_bytes": params.get("request", REQUEST),
        "route_labels": params.get("route", ROUTE),
    }
    full = outcome(lambda: rolling.Target.create(copy.deepcopy(a + b + c), parent=parent, **common))
    incremental = outcome(
        lambda: rolling.Target._create_extension(
            copy.deepcopy(c), parent=parent, prefix=prefix, **common
        )
    )
    assert incremental[0] == full[0], (full, incremental)
    if full[0] == "ok":
        assert incremental[1] == full[1]
        assert incremental[1].to_bytes(settings) == full[1].to_bytes(settings)
        assert incremental[1].delta_start == len(a) + len(b)
    else:
        # Same audited rejection where the full-prefix check reports a cross-row or
        # per-row fault; only a standalone delta starting with an assistant row words
        # the alternation fault differently.
        assert incremental[1] == full[1] or build is assistant_first, (
            full,
            incremental,
        )


def test_the_corpus_exercises_both_verdicts_for_every_kind_of_fault(tmp_path):
    verdicts = set()
    for param in CORPUS:
        build, params = param.values
        settings = rolling.RollingSettings(**params.get("limits", {}))
        a, b, c = exchange("a"), exchange("b"), exchange("c")
        build(a, b, c)
        parent, _ = validated_parent(tmp_path / param.id, [a, b], settings)
        full = outcome(
            lambda: rolling.Target.create(
                copy.deepcopy(a + b + c),
                session_id="s",
                settings=settings,
                policy=POLICY,
                parent=parent,
                request_bytes=params.get("request", REQUEST),
                route_labels=params.get("route", ROUTE),
            )
        )
        verdicts.add((param.id, full[0]))
    accepted = {name for name, verdict in verdicts if verdict == "ok"}
    assert accepted == {
        "valid",
        "reused-call-id-across-exchanges",
        "reused-native-item-ids",
        "bound-exact-length",
    }
    assert len(verdicts) == len(CORPUS)


# --- the real RollingJournal._validate_target, reached through reopen (reconcile) ---


def published_parent(tmp_path, chunks, delta, settings):
    """A journal with ``chunks`` published as a chain; ``delta`` rows archived, not frozen."""
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    store.ingest("s", [row for chunk in chunks for row in chunk] + delta, origin="canonical")
    rows, parent = [], None
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, policy=POLICY) as j:
        for sequence, chunk in enumerate(chunks, 1):
            rows = rows + chunk
            frozen = rolling.Target.create(
                copy.deepcopy(rows),
                session_id="s",
                settings=settings,
                policy=POLICY,
                parent=parent,
                request_bytes=REQUEST,
                route_labels=ROUTE,
            )
            ready(j, frozen, f"summary {sequence}")
            generation = rolling.generation_bytes(
                frozen, f"summary {sequence}", sequence=sequence, settings=settings
            )
            parent = j.publish_ready(frozen.job_id, generation)
    return store, parent


def forged_target(parent, rows, settings, request=REQUEST, route=ROUTE):
    """A self-consistent Target over ``rows`` (any validity) for a parent prefix."""
    limit = settings.max_checkpoint_bytes
    delta = rows[parent.frontier_count :]
    try:
        readable = rolling._summary_input(
            parent.summary_text,
            rolling.readable_delta(copy.deepcopy(delta), policy=POLICY, settings=settings),
            settings,
        )
    except ValueError:
        readable = b'{"unusable":"delta"}'
    return rolling.Target(
        session_id="s",
        settings_digest=settings.digest,
        parent_generation=parent.generation_id,
        parent_summary_sha256=parent.summary_sha256,
        event_ids=tuple(rolling.sha256(rolling.encode_json(row, limit)) for row in rows),
        message_uids=tuple(row["message_uid"] for row in rows),
        # a target that adds nothing can only claim an earlier start (it must stay constructible)
        delta_start=min(parent.frontier_count, len(rows) - 1),
        delta_stop=len(rows),
        readable_input=readable,
        request_bytes=request,
        route_labels=route,
    )


def store_without_validation(journal, frozen, monkeypatch):
    """Plant ``frozen`` through the real ``freeze_target``, minus its validation (a forged file)."""
    monkeypatch.setattr(journal, "_validate_target", lambda target: target)
    assert journal.freeze_target(frozen) == frozen


def count_extension_validations(monkeypatch):
    """Calls to the incremental ``RollingJournal._validate_extension`` (the real one runs)."""
    calls = []
    original = rolling_state.RollingJournal._validate_extension

    def spy(self, *args):
        calls.append(args)
        return original(self, *args)

    monkeypatch.setattr(rolling_state.RollingJournal, "_validate_extension", spy)
    return calls


def reopen(tmp_path, store, settings):
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, policy=POLICY) as j:
        return j.pending_target()


JOURNAL_CASES = [
    "valid",
    "reused-call-id-across-exchanges",
    "reused-native-item-ids",
    "dup-message-uid-first",
    "dup-reasoning-id-owner",
    "dup-reasoning-id-answer",
    "dup-occurrence-uid",
    "tool-result-for-parent-call",
    "delta-starts-with-tool",
    "delta-starts-with-assistant",
    "delta-bad-role",
    "delta-bad-companion",
    "delta-text-mismatch",
    "delta-restricted-data-use",
    "delta-orphan-tool",
    "delta-no-trailing-assistant",
    "non-advancing-target",
    "request-not-canonical-json",
    "request-unsupported-role",
    "route-labels-differ-from-policy",
]


@pytest.mark.parametrize("name", JOURNAL_CASES)
def test_reopen_validates_a_planted_target_like_the_full_rebuild(tmp_path, monkeypatch, name):
    param = next(p for p in CORPUS if p.id == name)
    build, params = param.values
    settings = rolling.RollingSettings(**params.get("limits", {}))
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    build(a, b, c)
    store, parent = published_parent(tmp_path, [a, b], c, settings)
    rows = copy.deepcopy(a + b + c)
    request, route = params.get("request", REQUEST), params.get("route", ROUTE)
    full = outcome(
        lambda: rolling.Target.create(
            copy.deepcopy(rows),
            session_id="s",
            settings=settings,
            policy=POLICY,
            parent=parent,
            request_bytes=request,
            route_labels=route,
        )
    )
    frozen = full[1] if full[0] == "ok" else forged_target(parent, rows, settings, request, route)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, policy=POLICY) as j:
        store_without_validation(j, frozen, monkeypatch)
    monkeypatch.undo()
    extensions = count_extension_validations(monkeypatch)
    if full[0] == "ok":
        assert reopen(tmp_path, store, settings) == frozen
    else:
        with pytest.raises(ValueError) as caught:
            reopen(tmp_path, store, settings)
        # for the audited reason (a standalone delta words an alternation fault differently)
        assert full[1] in str(caught.value) or build is assistant_first, (full, caught.value)
    assert len(extensions) >= 1  # the incremental path decided, not the full rebuild


def test_reopen_rejects_a_planted_target_with_an_unrelated_prefix(tmp_path, monkeypatch):
    settings = rolling.RollingSettings()
    a, b, c, other = exchange("a"), exchange("b"), exchange("c"), exchange("z")
    store, parent = published_parent(tmp_path, [a, b], c + other, settings)
    # Same delta, but the prefix is a different archived exchange than the parent's.
    swapped = forged_target(parent, other + b + c, settings)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, policy=POLICY) as j:
        store_without_validation(j, swapped, monkeypatch)
    monkeypatch.undo()
    extensions = count_extension_validations(monkeypatch)
    with pytest.raises(ValueError, match="not a parent prefix extension"):
        reopen(tmp_path, store, settings)
    assert len(extensions) >= 1


# Only fields that ``_validate_extension`` does not derive from the stored target itself
# can differ from the rebuilt one: request_bytes/route_labels are rebuilt FROM the target
# (their admission is covered by the request-*/route-* journal cases above), and
# parent_generation selects the parent. These identity fields are the ones that the final
# ``rebuilt != target`` comparison alone protects.
IDENTITY_TAMPERS = {
    "readable-input": lambda t: {"readable_input": t.readable_input + b" "},
    "delta-start": lambda t: {"delta_start": t.delta_start - 1},
    "parent-summary-sha256": lambda t: {"parent_summary_sha256": rolling.sha256(b"other summary")},
    "delta-message-uid": lambda t: {"message_uids": t.message_uids[:-1] + ("forged-uid",)},
}


@pytest.mark.parametrize("field", IDENTITY_TAMPERS)
def test_reopen_rejects_a_planted_target_with_a_tampered_identity_field(
    tmp_path, monkeypatch, field
):
    settings = rolling.RollingSettings()
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    store, parent = published_parent(tmp_path, [a, b], c, settings)
    good = forged_target(parent, a + b + c, settings)
    tampered = dataclasses.replace(good, **IDENTITY_TAMPERS[field](good))
    assert tampered != good and tampered.event_ids == good.event_ids  # valid prefix, valid rows
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, policy=POLICY) as j:
        store_without_validation(j, tampered, monkeypatch)
    monkeypatch.undo()
    extensions = count_extension_validations(monkeypatch)
    with pytest.raises(ValueError, match="frozen target source or identity mismatch"):
        reopen(tmp_path, store, settings)
    assert len(extensions) >= 1


def test_reopen_rejects_an_archive_row_changed_after_the_target_was_frozen(tmp_path):
    settings = rolling.RollingSettings()
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    store, parent = published_parent(tmp_path, [a, b], c, settings)
    frozen = forged_target(parent, a + b + c, settings)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, policy=POLICY) as j:
        j.freeze_target(frozen)
    assert reopen(tmp_path, store, settings) == frozen
    with sqlite3.connect(store.path) as conn:
        conn.execute("DELETE FROM observations WHERE event_id=?", (frozen.event_ids[-1],))
    with pytest.raises(ValueError, match="foreign canonical session"):
        reopen(tmp_path, store, settings)


def test_without_a_tied_prefix_state_the_full_rebuild_still_decides(tmp_path, monkeypatch):
    """No recorded state for the parent view -> today's full-prefix path, same verdicts."""
    settings = rolling.RollingSettings()
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    store, parent = published_parent(tmp_path, [a, b], c, settings)
    good = forged_target(parent, a + b + c, settings)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store, policy=POLICY) as j:
        j.freeze_target(good)

    class Untied(dict):  # hidden only from the journal's lookup, not validate_generation's
        def get(self, *args):
            if sys._getframe(1).f_code.co_name == "_validate_target":
                return None
            return super().get(*args)

    original = rolling._ValidationMemo.__init__

    def init(self, *args):
        original(self, *args)
        self.prefix_states = Untied()

    monkeypatch.setattr(rolling._ValidationMemo, "__init__", init)
    monkeypatch.setattr(
        rolling_state.RollingJournal,
        "_validate_extension",
        lambda *args: pytest.fail("incremental path used without a tied prefix state"),
    )
    assert reopen(tmp_path, store, settings) == good


def test_extension_rejects_a_prefix_state_that_is_not_the_parents(tmp_path):
    settings = rolling.RollingSettings()
    a, b, c = exchange("a"), exchange("b"), exchange("c")
    parent, prefix = validated_parent(tmp_path, [a, b], settings)
    common = {
        "session_id": "s",
        "settings": settings,
        "policy": POLICY,
        "parent": parent,
        "request_bytes": REQUEST,
        "route_labels": ROUTE,
    }
    for wrong in (empty_prefix(), dataclasses.replace(prefix, length=prefix.length + 1)):
        with pytest.raises(ValueError, match="target is not a parent prefix extension"):
            rolling.Target._create_extension(copy.deepcopy(c), prefix=wrong, **common)
    good = rolling.Target._create_extension(copy.deepcopy(c), prefix=prefix, **common)
    assert good.delta_start == parent.frontier_count
