"""Ancestor generation artifacts are reread and hash-bound once per journal operation (#19).

An accepted (journal-cross-checked) generation still has its exact head bytes
hashed on every call and every ancestor's exact bytes reread from disk and
SHA-256-bound in every journal operation. Within ONE operation an ancestor
already reread and bound is not reread again, so one operation's artifact work
is linear in the number of generations, not quadratic.
"""

import json
from collections import Counter
from dataclasses import replace

import pytest
from test_rolling_journal_validation import chain
from test_rolling_validation_memo import loader, outcome

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.rolling import _ValidationMemo


def spy_artifact_reads(monkeypatch):
    """Every bounded artifact read (``rolling_state._read``), keyed by file name."""
    reads = Counter()
    original = rolling_state._read

    def counting(path, limit):
        reads[path.name] += 1
        return original(path, limit)

    monkeypatch.setattr(rolling_state, "_read", counting)
    return reads


def spy_decodes(monkeypatch):
    """Every JSON decode in ``rolling``, keyed by the SHA-256 of the decoded bytes.

    ``decode_json`` and the canonical-check callers all go through ``_decode_json``.
    Use ``generation_decodes`` to keep only the decodes of generation snapshots.
    """
    decodes = Counter()
    original = rolling._decode_json

    def counting(data, limit):
        decodes[rolling.sha256(data)] += 1
        return original(data, limit)

    monkeypatch.setattr(rolling, "_decode_json", counting)
    return decodes


def generation_decodes(decodes, views):
    ids = {view.generation_id for view in views}
    return Counter({digest: n for digest, n in decodes.items() if digest in ids})


def names(views):
    return {view.generation_id + ".json" for view in views}


def recording_loader(views, loaded):
    """``loader(views)`` that appends every requested parent id to ``loaded``."""
    load = loader(views)

    def record(digest):
        loaded.append(digest)
        return load(digest)

    return record


@pytest.mark.parametrize("count", [8, 16])
def test_open_reads_each_generation_artifact_a_bounded_number_of_times(
    tmp_path, monkeypatch, count
):
    store, settings, views, _, _ = chain(tmp_path, count, pending=True)
    reads = spy_artifact_reads(monkeypatch)
    decodes = spy_decodes(monkeypatch)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
        pass
    generation_reads = {name: n for name, n in reads.items() if name != "manifest.json"}
    assert set(generation_reads) == names(views)
    # Reconcile reads each artifact as a job parent, as a directory artifact and
    # (head only) through the manifest, plus at most one ancestor reread: a
    # constant independent of the chain length, not one per descendant walk.
    assert max(generation_reads.values()) <= 3, generation_reads
    assert sum(generation_reads.values()) <= 3 * count
    # Each generation is decoded once (its one full validation); accepted
    # bytes are bound by hash before any decode.
    decoded = generation_decodes(decodes, views)
    assert set(decoded) == {view.generation_id for view in views}
    assert set(decoded.values()) == {1}, decoded


@pytest.mark.parametrize("count", [8, 16])
def test_each_operation_rereads_every_ancestor_exactly_once(tmp_path, monkeypatch, count):
    store, settings, views, _, _ = chain(tmp_path, count, pending=True)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        reads = spy_artifact_reads(monkeypatch)
        decodes = spy_decodes(monkeypatch)
        for _ in range(2):
            reads.clear()
            assert journal.load_view() == views[-1]
            # A later operation rereads and rebinds the whole chain again ...
            assert reads == Counter({"manifest.json": 1, **dict.fromkeys(names(views), 1)})
            reads.clear()
            assert (
                rolling.revalidate_view(views[-1], store, "s", settings, _journal=journal)
                == views[-1]
            )
            # ... and so does an independent revalidation operation.
            assert reads == Counter(dict.fromkeys(names(views), 1))
        # Accepted bytes never reach the JSON decoder.
        assert not generation_decodes(decodes, views)


def test_publication_operation_reads_each_ancestor_once(tmp_path, monkeypatch):
    store, settings, views, successor, _ = chain(tmp_path, 8, pending=True)
    data = rolling.generation_bytes(successor, "pending summary", sequence=9)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        reads = spy_artifact_reads(monkeypatch)
        assert journal.publish_ready(successor.job_id, data).sequence == 9
        # The published head is read twice (once as the successor target's
        # parent, once as the parent of the caller-supplied bytes); every other
        # ancestor is read once, since the second walk reuses the first's binding.
        generation_reads = {name: n for name, n in reads.items() if name in names(views)}
        assert set(generation_reads) == names(views)
        assert max(generation_reads.values()) <= 2, generation_reads


# --- the binding is exact-bytes, per operation, and populated only by a hash match ---


def _forge(settings, view, text):
    value = json.loads(view.snapshot_bytes)
    value["summary_text"] = text
    value["summary_sha256"] = rolling.sha256(text.encode())
    return rolling.encode_json(value, settings.max_checkpoint_bytes)


def test_failed_reread_never_binds_within_the_same_operation(tmp_path):
    """A caught mismatch inside one operation must not let a later walk skip that ancestor."""
    store, settings, views, *_ = chain(tmp_path, 4)
    root = views[0]
    forged = _forge(settings, root, "forged root")
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        original = journal._parent_bytes
        with journal._operation():
            journal._parent_bytes = lambda digest: (
                forged if digest == root.generation_id else original(digest)
            )
            for head in views[1:]:
                with pytest.raises(ValueError, match="lineage"):
                    journal._view(head.snapshot_bytes)
            assert root.generation_id not in journal._memo.bound


def test_bound_but_unaccepted_parent_is_still_reread(tmp_path):
    """Only a parent that is ALSO the tied accepted view may skip its reread."""
    store, settings, views, *_ = chain(tmp_path, 3)
    memo = _ValidationMemo(store, "s", None)
    by_id = {view.generation_id: view.snapshot_bytes for view in views}
    with memo.operation():
        rolling.validate_generation(
            views[1].snapshot_bytes,
            store,
            "s",
            settings,
            parent_loader=by_id.__getitem__,
            _memo=memo,
        )
        assert memo.bound == {views[0].generation_id} and not memo.accepted
        by_id[views[0].generation_id] = _forge(settings, views[0], "changed root")
        with pytest.raises(ValueError, match="lineage"):
            rolling.validate_generation(
                views[2].snapshot_bytes,
                store,
                "s",
                settings,
                parent_loader=by_id.__getitem__,
                _memo=memo,
            )


def test_binding_is_reset_between_operations_and_dropped_on_close(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 4)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal._memo.bound is None  # outside an operation nothing is bound
        with journal._operation():
            assert journal._memo.bound == set()
            journal._view(journal._parent_bytes(views[-1].generation_id))
            assert journal._memo.bound == {view.generation_id for view in views[:-1]}
        assert journal._memo.bound is None
        with journal._operation():
            assert journal._memo.bound == set()
        with pytest.raises(RuntimeError):
            with journal._operation():
                journal._view(journal._parent_bytes(views[-1].generation_id))
                raise RuntimeError("boom")
        assert journal._memo is None
    assert journal._memo is None


def test_nested_refusal_keeps_the_outer_operation_binding(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 3)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with journal._operation():
            journal._view(journal._parent_bytes(views[-1].generation_id))
            bound = set(journal._memo.bound)
            with pytest.raises(ValueError, match="reentrant"):
                journal.load_view()
            assert journal._memo.bound == bound and bound


def test_accepted_bytes_are_bound_by_exact_comparison_not_by_lookup(tmp_path):
    """A tied accepted entry under the bytes' hash is still compared byte-for-byte."""
    store, settings, views, *_ = chain(tmp_path, 2)
    memo = _ValidationMemo(store, "s", None)
    head = rolling.validate_generation(
        views[1].snapshot_bytes, store, "s", settings, parent_loader=loader(views), _memo=memo
    )
    lookalike = replace(head, snapshot_bytes=_forge(settings, head, "lookalike"))
    memo.accepted[head.generation_id] = lookalike
    memo.prefix_states[head.generation_id] = (lookalike, memo.prefix_states[head.generation_id][1])
    with pytest.raises(ValueError, match="rolling generation identity mismatch"):
        rolling.validate_generation(
            views[1].snapshot_bytes, store, "s", settings, parent_loader=loader(views), _memo=memo
        )


def test_accepted_path_refuses_non_bytes_like_the_decoder(tmp_path):
    store, settings, views, *_ = chain(tmp_path, 1)
    for bad in (views[0].snapshot_bytes.decode(), bytearray(views[0].snapshot_bytes)):
        with pytest.raises(ValueError, match="invalid or oversized JSON bytes"):
            rolling.validate_generation(bad, store, "s", settings)
    small = replace(settings, max_checkpoint_bytes=len(views[0].snapshot_bytes) - 1)
    with pytest.raises(ValueError, match="invalid or oversized JSON bytes"):
        rolling.validate_generation(views[0].snapshot_bytes, store, "s", small)


@pytest.mark.parametrize("depth", range(6))
@pytest.mark.parametrize("top", [2, 3])
def test_bound_parent_hit_keeps_lineage_depth_bound(tmp_path, depth, top):
    """A bound (skipped) parent still counts its whole chain against the campaign bound."""
    store, settings, views, *_ = chain(tmp_path, 3, max_logical_jobs=3)
    head = views[top - 1].snapshot_bytes

    def validate(memo, parent_loader):
        return lambda: rolling.validate_generation(
            head,
            store,
            "s",
            settings,
            parent_loader=parent_loader,
            _depth=depth,
            _memo=memo,
        )

    memo = _ValidationMemo(store, "s", None)
    loaded = []
    with memo.operation():
        # Validating the head in this operation reads and binds its ancestors ...
        rolling.validate_generation(
            head, store, "s", settings, parent_loader=loader(views), _memo=memo
        )
        for view in views[: top - 1]:
            memo.accepted[view.generation_id] = memo.prefix_states[view.generation_id][0]
        # ... so the revalidation below skips the parent's reread: the recording
        # loader proves the skip happened (a skip that silently fell back to the
        # reread would make the depth comparison vacuous).
        bound = outcome(validate(memo, recording_loader(views, loaded)))
    assert not loaded
    assert bound == outcome(validate(None, loader(views)))
    assert (outcome(validate(None, loader(views))) == top) == (depth + top <= 3)
