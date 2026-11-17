"""The committed benchmark must build/measure real homogeneous delta journals."""

import importlib.util
import json
import sqlite3
from pathlib import Path


def test_benchmark_delta_build_stores_each_identity_only_twice(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "rolling_bench", Path(__file__).parents[1] / "scripts/bench_rolling_open.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build(tmp_path / "home", 256, 64, storage_version=3)
    assert result["events"] == 256
    directory = next((tmp_path / "home/ariadne/rolling").iterdir())
    with sqlite3.connect(directory / "journal.sqlite3") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        references = sum(
            len(json.loads(row[0])["delta_event_ids"])
            for row in conn.execute("SELECT payload FROM targets")
        )
    references += sum(
        len(json.loads(path.read_bytes())["delta_event_ids"])
        for path in (directory / "generations").iterdir()
    )
    assert references == 2 * 256
    measured = module.measure(tmp_path / "home", 0, in_place=True, cache=True)
    assert measured["events"] == 256
    assert measured["stored_identity_references"] == 512
    assert measured["warm_source_checks"] == 256
