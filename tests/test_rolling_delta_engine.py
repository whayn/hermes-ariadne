"""Schema-3 engine wiring through the existing lifecycle oracle."""

import json

from test_rolling_engine import complete, setup_engine
from test_rolling_selection import exchange


def test_engine_creates_delta_artifacts(tmp_path, monkeypatch):
    engine, db, facade = setup_engine(tmp_path, monkeypatch)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    assert len(facade.calls) == 1
    view = engine._published
    assert view is not None
    value = json.loads(view.snapshot_bytes)
    assert value["version"] == 2
    assert "event_ids" not in value
