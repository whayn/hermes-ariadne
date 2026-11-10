"""Focused regression tests for rolling facade limits configuration."""

import os

import pytest
from test_rolling_engine import complete, setup_engine
from test_rolling_selection import exchange

from hermes_ariadne.rolling import RollingSettings

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("HERMES_SOURCE required for ContextEngine ABC", allow_module_level=True)


@pytest.mark.parametrize(
    ("settings", "expected_max_tokens", "expected_timeout"),
    [
        (None, 2048, 60),
        (RollingSettings(trigger_bytes=1, max_tokens=17, timeout=3), 17, 3),
        (RollingSettings(trigger_bytes=1, max_tokens=128, timeout=7), 128, 7),
    ],
)
def test_facade_receives_configured_limits(
    tmp_path, monkeypatch, settings, expected_max_tokens, expected_timeout
):
    engine, db, facade = setup_engine(tmp_path, monkeypatch, settings=settings)
    db.rows = exchange("A")
    complete(engine, db, "A")
    assert not facade.calls
    db.rows += exchange("B")
    complete(engine, db, "B")
    assert len(facade.calls) == 1
    assert facade.calls[0]["max_tokens"] == expected_max_tokens
    assert facade.calls[0]["timeout"] == expected_timeout
