"""Bounded completed-generation source evidence, never a new acceptance path."""

import json
import shutil
import sqlite3
from collections import Counter
from dataclasses import replace

import pytest
import rolling_delta_reference as reference
from test_rolling_cold_validation import delta_chain, startup_costs

from hermes_ariadne import rolling, rolling_delta, rolling_state
from hermes_ariadne.store import EventStore


def observe(home, settings, cache):
    store = EventStore(home / "ariadne/events.sqlite3")
    try:
        with rolling_state.RollingJournal(home, "s", settings, store=store, cache=cache) as journal:
            view, pending = journal.load_view(), journal.pending_target()
            return (
                "ok",
                None if view is None else view.snapshot_bytes,
                None if pending is None else pending.to_bytes(settings),
                journal.report(),
                journal.charged_callback_ids(),
            )
    except Exception as exc:
        return type(exc), str(exc)


@pytest.mark.parametrize("count", [8, 16])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_completed_delta_canonicalization_is_shared(tmp_path, monkeypatch, count, cached, pending):
    store, settings, views, successor, _ = delta_chain(tmp_path, count, pending=pending)
    cache = rolling.RollingVerificationCache() if cached else None
    with startup_costs(monkeypatch, store) as costs:
        with rolling_state.RollingJournal(
            tmp_path, "s", settings, store=store, cache=cache
        ) as journal:
            assert journal.load_view() == views[-1]
            assert journal.pending_target() == successor
    # Pending binds every completed generation before reconciliation. Without
    # pending, the head target precedes its generation: keep that full fallback.
    assert sum(costs["canonical"].values()) == 2 * (count + 1), costs
    if pending:
        assert set(costs["canonical"].values()) == {1}
    else:
        assert sorted(costs["canonical"].values()) == [1] * (2 * count - 2) + [2, 2]
    assert set(costs["sources"].values()) == {1}
    assert sum(costs["records"].values()) <= count
    assert sum(costs["reconcile"].values()) <= 2 * count + 1


@pytest.mark.parametrize("records,byte_limit,admitted", [(0, 2**24, 0), (256, 1, 0), (2, 2**24, 2)])
def test_proof_pressure_falls_back_without_changing_results(
    tmp_path, monkeypatch, records, byte_limit, admitted
):
    store, settings, views, successor, _ = delta_chain(tmp_path, 8)
    monkeypatch.setattr(rolling_delta, "_PROOF_RECORDS", records)
    monkeypatch.setattr(rolling_delta, "_PROOF_BYTES", byte_limit)
    with startup_costs(monkeypatch, store) as costs:
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            assert journal.pending_target() == successor
            assert journal.load_view() == views[-1]
            proofs = journal._memo.source_proofs
            assert proofs.records == len(proofs.accepted) == admitted
            assert proofs.input_bytes == sum(
                len(p.readable_input) for p in proofs.accepted.values()
            )
            assert proofs.input_bytes <= byte_limit
            assert proofs.pressure >= 8 - admitted
            assert not journal._memo.deltas and not journal._memo.validated_targets
    assert sum(costs["canonical"].values()) == 34 - 2 * admitted
    assert set(costs["sources"].values()) == {1}


def test_proofs_borrow_existing_objects_and_do_not_cross_opens(tmp_path):
    store, settings, views, _, _ = delta_chain(tmp_path, 8)
    previous = None
    for _ in range(2):
        with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
            memo = journal._memo
            proofs = memo.source_proofs
            assert proofs is not previous
            assert proofs.records == len(proofs.accepted) == 8
            for key, proof in proofs.accepted.items():
                assert proof.view is memo.accepted[proof.view.generation_id]
                _, frozen = journal._job(proof.view.job_id)
                assert key == frozen.target_id
                assert proof.readable_input is frozen.readable_input
                if proof.parent_state is not None:
                    assert proof.parent_state is memo.prefix_states[proof.view.parent_generation][1]
            assert set(memo.accepted) == {v.generation_id for v in views}
            assert not memo.deltas and not memo.validated_targets
            previous = proofs
        assert journal._memo is None
        del memo
        assert previous.memo() is None


@pytest.mark.parametrize(
    "broken",
    ["accepted", "view", "input", "parent-state", "scope", "old-memo", "index"],
)
def test_incompatible_proof_does_not_bypass_the_full_checker(tmp_path, monkeypatch, broken):
    store, settings, views, _, _ = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with journal._operation():
            memo = journal._memo
            _, frozen = journal._job(views[5].job_id)
            parent = journal._parent_view(frozen.parent_generation)
            proofs = memo.source_proofs
            saved = proofs.accepted[frozen.target_id]
            assert proofs.find(frozen, parent, memo, store, "s", settings, None) is saved
            if broken == "accepted":
                memo.accepted.pop(saved.view.generation_id)
            elif broken == "view":
                proofs.accepted[frozen.target_id] = replace(saved, view=replace(saved.view))
            elif broken == "input":
                proofs.accepted[frozen.target_id] = replace(saved, readable_input=b"not expected")
            elif broken == "parent-state":
                pair = memo.prefix_states[parent.generation_id]
                memo.prefix_states[parent.generation_id] = (pair[0], replace(pair[1]))
            elif broken == "scope":
                proofs.settings = "0" * 64
            elif broken == "old-memo":
                import weakref

                other = rolling._ValidationMemo(store, "s", None, settings_digest=settings.digest)
                proofs.memo = weakref.ref(other)
            elif broken == "index":
                proofs.index = rolling._new_identity_index()
            assert proofs.find(frozen, parent, memo, store, "s", settings, None) is None
            calls = Counter()
            original = rolling._canonical_rows

            def full(rows, **kwargs):
                calls.update(row["message_uid"] for row in rows)
                return original(rows, **kwargs)

            monkeypatch.setattr(rolling, "_canonical_rows", full)
            assert (
                rolling_delta.validate_target(
                    frozen,
                    parent,
                    store,
                    "s",
                    settings,
                    policy=None,
                    memo=memo,
                    guard=journal._check,
                )
                == frozen
            )
            assert sum(calls.values()) == 2


def test_proof_hit_keeps_capture_checks_and_failed_operation_cleanup(tmp_path):
    store, settings, views, _, _ = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with pytest.raises(RuntimeError, match="capture revoked"):
            with journal._operation():
                _, frozen = journal._job(views[5].job_id)
                parent = journal._parent_view(frozen.parent_generation)
                memo = journal._memo
                proofs = memo.source_proofs
                before = proofs.hits
                calls = []

                def guard():
                    calls.append(1)
                    if len(calls) == 3:
                        raise RuntimeError("capture revoked")

                rolling_delta.validate_target(
                    frozen, parent, store, "s", settings, policy=None, memo=memo, guard=guard
                )
        assert proofs.hits == before + 1 and len(calls) == 3
        assert journal._memo is None and not journal._busy
        assert journal.load_view() == views[-1]
        assert journal._memo.source_proofs is not proofs


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "source",
        "source-session",
        "root",
        "extra",
        "manifest",
        "target-hash",
        "target-rehash",
        "charge",
        "job-text",
        "job-state",
    ],
)
@pytest.mark.parametrize("cache_mode", ["none", "cold", "warm"])
def test_sharing_matches_frozen_full_validator(tmp_path, monkeypatch, mutation, cache_mode):
    home = tmp_path / "fixture"
    store, settings, views, successor, directory = delta_chain(home, 8)
    pristine = tmp_path / "pristine"
    if cache_mode == "warm":
        shutil.copytree(home, pristine)
    if mutation.startswith("source"):
        with sqlite3.connect(store.path) as conn:
            event = views[0].event_ids[0]
            data = json.loads(
                conn.execute(
                    "SELECT payload_json FROM events WHERE event_id=?", (event,)
                ).fetchone()[0]
            )
            data["session_id" if mutation == "source-session" else "content"] = "changed"
            conn.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (rolling.encode_json(data, settings.max_checkpoint_bytes), event),
            )
    elif mutation == "root":
        path = directory / "generations" / (views[0].generation_id + ".json")
        rolling_state._atomic(path, b"{}")
    elif mutation == "extra":
        rolling_state._atomic(directory / "generations" / ("0" * 64 + ".json"), b"{}")
    elif mutation == "manifest":
        value = json.loads((directory / "manifest.json").read_bytes())
        value["generation_id"] = "0" * 64
        rolling_state._atomic(directory / "manifest.json", rolling.encode_json(value, 8192))
    elif mutation != "valid":
        with sqlite3.connect(directory / "journal.sqlite3") as conn:
            if mutation == "target-hash":
                conn.execute("UPDATE targets SET payload_sha256=?", ("0" * 64,))
            elif mutation == "target-rehash":
                value = json.loads(successor.to_bytes(settings))
                value["readable_input"] += " "
                payload = rolling.encode_json(value, settings.max_checkpoint_bytes)
                conn.execute(
                    "UPDATE targets SET payload=?,payload_sha256=? WHERE target_id=?",
                    (payload, rolling.sha256(payload), successor.target_id),
                )
            elif mutation == "charge":
                conn.execute("UPDATE campaign SET logical_reservations=0")
            elif mutation == "job-text":
                conn.execute(
                    "UPDATE jobs SET validated_text='changed' WHERE job_id=?", (views[3].job_id,)
                )
            elif mutation == "job-state":
                conn.execute("UPDATE jobs SET state='planned' WHERE job_id=?", (views[3].job_id,))
    candidate = tmp_path / "candidate"

    def run():
        cache = None if cache_mode == "none" else rolling.RollingVerificationCache()
        if candidate.exists():
            shutil.rmtree(candidate)
        if cache_mode == "warm":
            shutil.copytree(pristine, candidate)
            assert observe(candidate, settings, cache)[0] == "ok"
            assert cache.stats()["entries"] > 0
            shutil.rmtree(candidate)
        shutil.copytree(home, candidate)
        return observe(candidate, settings, cache)

    def full_generation(*args, _proofs=None, **kwargs):
        return reference.validate_generation(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(rolling_delta, "validate_generation", full_generation)
        patch.setattr(rolling_delta, "validate_target", reference.validate_target)
        expected = run()
    actual = run()
    assert actual == expected
    assert (actual[0] == "ok") == (mutation == "valid"), actual


def test_late_startup_failure_discards_candidates_and_never_promotes(tmp_path, monkeypatch):
    store, settings, _, _, _ = delta_chain(tmp_path, 8)
    cache = rolling.RollingVerificationCache()
    batches, journals = [], []
    accept = rolling_delta._ProofBatch.accept
    walk = rolling_state.RollingJournal._view

    def after_new_evidence(self, view, target):
        accept(self, view, target)
        if view.sequence == 3:
            assert len(self.owner.accepted) == 3
            assert self.pending  # failure after NEW validations, not just a hit
            batches.append(self)
            raise RuntimeError("late startup failure")

    def capture(self, data):
        journals.append(self)
        return walk(self, data)

    monkeypatch.setattr(rolling_delta._ProofBatch, "accept", after_new_evidence)
    monkeypatch.setattr(rolling_state.RollingJournal, "_view", capture)
    with pytest.raises(RuntimeError, match="late startup failure"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store, cache=cache)
    assert cache.stats()["entries"] == 0
    assert journals and all(j._memo is None and j._conn is None for j in journals)
    assert batches and all(not b.pending for b in batches)
    for batch in batches:
        assert batch.owner.records == len(batch.owner.accepted) == 3
        assert batch.owner.input_bytes == sum(
            len(p.readable_input) for p in batch.owner.accepted.values()
        )


def test_raw_generation_validation_does_not_retain_source_proofs(tmp_path):
    store, settings, views, _, _ = delta_chain(tmp_path, 1, pending=False)
    memo = rolling._ValidationMemo(store, "s", None, settings_digest=settings.digest)
    assert (
        rolling_delta.validate_generation(views[0].snapshot_bytes, store, "s", settings, _memo=memo)
        == views[0]
    )
    assert memo.source_proofs is None
    assert not memo.deltas
