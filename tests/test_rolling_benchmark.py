"""The common A/B harness reports actual canonicalization and proof work."""

import runpy
from pathlib import Path


def test_benchmark_reports_source_sharing_and_warm_fallback(tmp_path):
    bench = runpy.run_path(str(Path(__file__).parents[1] / "scripts/bench_rolling_open.py"))
    home = tmp_path / "fixture"
    bench["build"](home, 32, 4, 3)
    result = bench["measure"](home, 0, in_place=True, cache=True)
    assert result["open_canonical_rows"] == 40  # N plus unmatched head delta
    assert result["open_canonical_calls"] == 5
    assert result["open_render_calls"] == result["open_summary_input_calls"] == 5
    assert result["open_source_proofs"]["hits"] == 3
    assert result["open_source_proofs"]["accepted"] == 4
    assert result["open_source_proofs"]["pressure"] == 0
    assert result["warm_canonical_rows"] == result["warm_canonical_calls"] == 0
    assert result["warm_source_proofs"]["accepted"] == 0
    assert result["open_source_checks"] == result["warm_source_checks"] == 32
