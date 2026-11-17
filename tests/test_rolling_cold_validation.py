"""Cold delta startup must not repeat ancestor walks (#19, slices A/B only)."""

from collections import Counter
from contextlib import contextmanager

import pytest
from test_rolling_state import ready, setup, target

from hermes_ariadne import rolling, rolling_state


def delta_chain(home, count, *, pending=True):
    store, settings = setup(home)
    views = []
    with rolling_state.RollingJournal(
        home, "s", settings, store=store, storage_version=3
    ) as journal:
        parent = None
        for sequence in range(1, count + 1):
            frozen = journal.encode_target(target(store, settings, sequence, parent), parent)
            ready(journal, frozen, f"summary {sequence}")
            parent = journal.publish_ready(
                frozen.job_id,
                rolling.generation_bytes(
                    frozen, f"summary {sequence}", sequence=sequence, settings=settings
                ),
            )
            views.append(parent)
        successor = None
        if pending:
            successor = journal.encode_target(target(store, settings, count + 1, parent), parent)
            ready(journal, successor, "pending summary")
        directory = journal.directory
    return store, settings, views, successor, directory


@contextmanager
def startup_costs(monkeypatch, store):
    """Wrap actual readers/checkers: neither fixture data nor validation is mocked."""
    costs = {
        "records": Counter(),
        "reconcile": Counter(),
        "sources": Counter(),
        "canonical": Counter(),
        "encoded_bytes": 0,
        "owned": [],
    }
    active = {"phase": None, "journal": None}
    with monkeypatch.context() as patch:
        for name, phase in (("_validate_records", "records"), ("_reconcile", "reconcile")):
            original = getattr(rolling_state.RollingJournal, name)

            def phase_call(self, original=original, phase=phase):
                previous = dict(active)
                active.update(phase=phase, journal=self)
                try:
                    return original(self)
                finally:
                    active.update(previous)

            patch.setattr(rolling_state.RollingJournal, name, phase_call)
        read, get, canonical, encode = (
            rolling_state._read,
            store.get_event,
            rolling._canonical_rows,
            rolling.encode_json,
        )

        def artifact(path, limit):
            if path.parent.name == "generations" and active["phase"] is not None:
                costs[active["phase"]][path.name] += 1
                costs["owned"].append(active["journal"]._memo.bound is not None)
            return read(path, limit)

        def source(event, **kwargs):
            costs["sources"][event] += 1
            return get(event, **kwargs)

        def rows(values, **kwargs):
            costs["canonical"].update(row["message_uid"] for row in values)
            return canonical(values, **kwargs)

        def serialized(value, limit):
            data = encode(value, limit)
            costs["encoded_bytes"] += len(data)
            return data

        patch.setattr(rolling_state, "_read", artifact)
        patch.setattr(store, "get_event", source)
        patch.setattr(rolling, "_canonical_rows", rows)
        patch.setattr(rolling, "encode_json", serialized)
        patch.setattr(rolling_state, "encode_json", serialized)
        yield costs


@pytest.mark.parametrize("count", [8, 16])
@pytest.mark.parametrize("cached", [False, True])
def test_cold_delta_startup_binds_each_ancestor_once_per_audit(
    tmp_path, monkeypatch, count, cached
):
    store, settings, views, successor, _ = delta_chain(tmp_path, count)
    cache = rolling.RollingVerificationCache() if cached else None
    with startup_costs(monkeypatch, store) as costs:
        with rolling_state.RollingJournal(
            tmp_path, "s", settings, store=store, cache=cache
        ) as journal:
            assert journal.pending_target() == successor
            assert journal.load_view() == views[-1]
    records, reconcile = costs["records"], costs["reconcile"]
    print(
        {
            "generations": count,
            "cache": cached,
            "records": sum(records.values()),
            "reconcile": sum(reconcile.values()),
            "source_rows": sum(costs["sources"].values()),
            "canonical_rows": sum(costs["canonical"].values()),
            "encoded_bytes": costs["encoded_bytes"],
        }
    )
    # Audit 1 binds the pending target's complete parent chain once. Audit 2
    # binds job parents once, independently enumerates ALL artifacts and reads
    # the manifest head once. These are separate freshness/ownership scopes.
    assert sum(records.values()) <= count, records
    assert max(records.values()) <= 1
    assert sum(reconcile.values()) <= 2 * count + 1, reconcile
    assert max(reconcile.values()) <= 3
    assert set(records) == set(reconcile) == {view.generation_id + ".json" for view in views}
    assert costs["owned"] and all(costs["owned"])
    assert set(costs["sources"]) == set(successor.event_ids)
    assert set(costs["sources"].values()) == {1}
    # Canonical-delta sharing is deliberately NOT part of this slice.
    assert sum(costs["canonical"].values()) == 4 * count + 2
