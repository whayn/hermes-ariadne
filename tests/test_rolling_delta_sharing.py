"""Bounded completed-generation source evidence, never a new acceptance path."""

import pytest
from test_rolling_cold_validation import delta_chain, startup_costs

from hermes_ariadne import rolling, rolling_state


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
