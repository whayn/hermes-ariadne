"""Per-operation validation memo (#19): scope, copy isolation, depth parity, linear reads."""

import json
import sqlite3
import sys
from collections import Counter
from dataclasses import replace

import pytest
from test_rolling_journal_validation import chain, count_reads
from test_rolling_state import ready, target

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.rolling import _ValidationMemo
from hermes_ariadne.store import EventStore


def loader(views):
    by_id = {view.generation_id: view.snapshot_bytes for view in views}
    return by_id.__getitem__


def outcome(call):
    try:
        return call().sequence
    except ValueError as exc:
        return str(exc)


# --- memo scope and isolation ---


def test_memo_bound_to_store_and_session_is_enforced(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 2)
    other = EventStore(tmp_path / "other" / "events.sqlite3")
    args = (views[1].snapshot_bytes, store, "s", settings)
    kwargs = {"parent_loader": loader(views)}
    good = _ValidationMemo(store, "s", None)
    assert rolling.validate_generation(*args, **kwargs, _memo=good).sequence == 2
    for memo in (_ValidationMemo(other, "s", None), _ValidationMemo(store, "other", None)):
        with pytest.raises(ValueError, match="memo scope"):
            rolling.validate_generation(*args, **kwargs, _memo=memo)


def test_memo_rows_are_independent_copies_and_missing_is_not_cached(tmp_path, monkeypatch):
    store, settings, views, *_ = chain(tmp_path, 1)
    event = views[0].event_ids[0]
    pristine = store.get_event(event, session_id="s")
    memo = _ValidationMemo(store, "s", None)
    first = memo.get_event(event)
    first["content"] = "mutated after miss"
    first.setdefault("extra", []).append(1)
    second = memo.get_event(event)
    assert second == pristine
    second["content"] = "mutated after hit"
    assert memo.get_event(event) == pristine
    assert memo.rows[event] == pristine

    answers = iter([None, pristine])
    monkeypatch.setattr(store, "get_event", lambda *a, **k: next(answers))  # read at call time
    absent = _ValidationMemo(store, "s", None)
    assert absent.get_event(event) is None
    assert absent.get_event(event) == pristine  # the None was not remembered
    assert event in absent.rows


# --- depth-bound parity: a memo hit never extends lineage beyond the campaign bound ---


@pytest.mark.parametrize("depth", range(5))
@pytest.mark.parametrize("top,accepted", [(3, 1), (3, 2), (2, 1)])
def test_memo_hit_keeps_lineage_depth_bound(tmp_path, depth, top, accepted):
    store, settings, views, *_ = chain(tmp_path, 3, max_logical_jobs=3)
    parent_loader = loader(views)

    def validate(memo):
        return lambda: rolling.validate_generation(
            views[top - 1].snapshot_bytes,
            store,
            "s",
            settings,
            parent_loader=parent_loader,
            _depth=depth,
            _memo=memo,
        )

    memo = _ValidationMemo(store, "s", None)
    memo.accepted[views[accepted - 1].generation_id] = views[accepted - 1]
    assert outcome(validate(memo)) == outcome(validate(None))
    # Sanity: the bound is exercised on both sides of the limit.
    assert (outcome(validate(None)) == top) == (depth + top <= 3)


def test_memo_hit_requires_same_exact_bytes_and_settings(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 2)
    root = views[0]
    for stale in (
        replace(root, settings_digest="0" * 64),
        replace(root, snapshot_bytes=root.snapshot_bytes + b" "),
    ):
        memo = _ValidationMemo(store, "s", None)
        memo.accepted[root.generation_id] = stale
        with pytest.raises(ValueError, match="identity"):
            rolling.validate_generation(
                views[1].snapshot_bytes,
                store,
                "s",
                settings,
                parent_loader=loader(views),
                _memo=memo,
            )


# --- journal integration ---


def _forge(settings, view, text):
    """Structurally valid, self-consistent bytes for ``view`` with different summary text."""
    value = json.loads(view.snapshot_bytes)
    value["summary_text"] = text
    value["summary_sha256"] = rolling.sha256(text.encode())
    return rolling.encode_json(value, settings.max_checkpoint_bytes)


def test_journal_memo_lives_only_inside_an_operation(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 3)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal._memo is None
        with journal._operation():
            memo = journal._memo
            assert memo.store is store and memo.session_id == "s"
            journal._view(journal._parent_bytes(views[-1].generation_id))
            assert set(memo.accepted) == {view.generation_id for view in views}
        assert journal._memo is None
        with journal._operation():
            assert journal._memo is not memo and not journal._memo.accepted
    with pytest.raises(ValueError):
        with journal._operation():
            pass
    assert journal._memo is None


def test_operation_exception_resets_memo_and_busy_and_journal_stays_usable(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 3)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with pytest.raises(RuntimeError, match="boom"):
            with journal._operation():
                assert journal._memo is not None and journal._busy
                raise RuntimeError("boom")
        assert journal._memo is None
        assert journal._busy is False
        assert journal.load_view() == views[-1]
        assert journal._memo is None


def test_generation_accepted_only_after_journal_cross_check(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 2)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with journal._operation():
            # Structurally valid for validate_generation, but never journaled.
            forged = _forge(settings, views[-1], "forged head")
            with pytest.raises(ValueError, match="unjournaled or mismatching generation"):
                journal._view(forged)
            # The cross-checked root ancestor is accepted; the forged head never is.
            assert set(journal._memo.accepted) == {views[0].generation_id}
            assert journal._view(views[-1].snapshot_bytes) == views[-1]
            assert views[-1].generation_id in journal._memo.accepted
            assert rolling.sha256(forged) not in journal._memo.accepted


def test_accepted_parent_is_found_by_reread_bytes_not_by_name(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 3)
    root = views[0]
    forged = _forge(settings, root, "forged root")
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        original = journal._parent_bytes
        with journal._operation():
            assert journal._view(root.snapshot_bytes) == root
            assert set(journal._memo.accepted) == {root.generation_id}
            # The parent *name* is an accepted id, but the bytes reread under it differ.
            journal._parent_bytes = lambda digest: (
                forged if digest == root.generation_id else original(digest)
            )
            with pytest.raises(ValueError, match="lineage"):
                journal._view(views[1].snapshot_bytes)
            assert set(journal._memo.accepted) == {root.generation_id}


def test_memoized_reads_are_still_guarded_and_still_verified(tmp_path, monkeypatch):
    store, settings, views, *_ = chain(tmp_path, 2)
    memo = _ValidationMemo(store, "s", None)
    memo.accepted[views[0].generation_id] = views[0]
    kwargs = {"parent_loader": loader(views), "_memo": memo}
    rolling.validate_generation(views[1].snapshot_bytes, store, "s", settings, **kwargs)
    reads = count_reads(monkeypatch, store)
    calls = []
    rolling.validate_generation(
        views[1].snapshot_bytes, store, "s", settings, guard=lambda: calls.append(1), **kwargs
    )
    assert not reads  # served entirely from the memo ...
    # ... with a guard before/after the parent load and each of the 4 event rows
    assert len(calls) == 2 + 2 * len(views[1].event_ids)
    # A memoized row is re-verified on every use: poison the cached copy.
    memo.rows[views[1].event_ids[0]]["content"] = "poisoned"
    with pytest.raises(ValueError, match="source"):
        rolling.validate_generation(views[1].snapshot_bytes, store, "s", settings, **kwargs)


# --- sibling call site: revalidate_view reads each distinct event once per call ---


def test_revalidate_view_reads_each_chain_event_once_per_call(tmp_path, monkeypatch):
    store, settings, views, *_ = chain(tmp_path, 6)
    head = views[-1]
    reads = count_reads(monkeypatch, store)
    for call in range(2):
        reads.clear()
        checked = rolling.revalidate_view(head, store, "s", settings, parent_loader=loader(views))
        assert checked == head
        # Every referenced source is read and hash-checked, once, in each call (no cache
        # survives a call).
        assert set(reads) == set(head.event_ids)
        assert max(reads.values()) == 1, call


def test_revalidate_view_still_rejects_changed_shared_source(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 4)
    with sqlite3.connect(store.path) as conn:
        conn.execute("DELETE FROM observations WHERE event_id=?", (views[0].event_ids[0],))
    with pytest.raises(ValueError, match="source"):
        rolling.revalidate_view(views[-1], store, "s", settings, parent_loader=loader(views))


# --- each generation is fully validated at most once per journal open ---


def spy_full_validations(monkeypatch):
    """Count full validations: the real row canonicalizer, as called by the full path.

    Target.create calls the same helper for a different purpose, so only calls made
    by validate_generation itself (after every cheaper check) are counted.

    Coupling: this patches the private ``rolling._canonical_rows`` and filters on the
    caller's function name, so renaming either function breaks it. It is the only way
    to observe a full validation (as opposed to a memo hit) without a public hook.
    """
    full = Counter()
    original = rolling._canonical_rows

    def spy(source, **kwargs):
        if sys._getframe(1).f_code.co_name == "validate_generation":
            full[tuple(row["message_uid"] for row in source)] += 1
        return original(source, **kwargs)

    monkeypatch.setattr(rolling, "_canonical_rows", spy)
    return full


def test_open_fully_validates_each_generation_at_most_once(tmp_path, monkeypatch):
    store, settings, views, *_ = chain(tmp_path, 6)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        pending = target(store, settings, 7, views[-1])
        ready(journal, pending, "pending summary")  # ready, deliberately unpublished
    full = spy_full_validations(monkeypatch)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        # Opening reconciles in one operation, reaching every generation as a job parent,
        # as the manifest head and as an artifact; shared ancestors are validated once.
        # Canonicalization is incremental: each generation checks only its own delta.
        assert set(full) == {view.message_uids[view.delta_start :] for view in views}
        assert max(full.values()) == 1, dict(full)
        full.clear()
        # A later operation starts from nothing (no cross-operation state) but is
        # itself linear.
        assert journal.load_view() == views[-1]
        assert set(full) == {view.message_uids[view.delta_start :] for view in views}
        assert max(full.values()) == 1, dict(full)


def test_memo_bound_to_reasoning_policy_is_enforced(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 2)
    args = (views[1].snapshot_bytes, store, "s", settings)
    kwargs = {"parent_loader": loader(views)}
    policy = object()  # scope identity only: the check precedes any policy use
    for memo_policy, call_policy in ((object(), None), (None, policy), (object(), policy)):
        memo = _ValidationMemo(store, "s", memo_policy)
        with pytest.raises(ValueError, match="memo scope"):
            rolling.validate_generation(*args, **kwargs, policy=call_policy, _memo=memo)
    good = _ValidationMemo(store, "s", None)
    assert rolling.validate_generation(*args, **kwargs, _memo=good).sequence == 2
