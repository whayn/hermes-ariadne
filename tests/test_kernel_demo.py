"""Two-process synthetic demo: recall cannot re-ingest/regenerate the source."""

import json
import os
import subprocess
import sys


def test_synthetic_demo_recovers_exact_code_after_process_restart(tmp_path):
    directory = tmp_path / "demo"

    def run(phase):
        process = subprocess.run(
            [sys.executable, "-m", "hermes_ariadne.demo", str(directory), "--phase", phase],
            env={**os.environ, "PYTHONPATH": str(__import__("pathlib").Path("src").resolve())},
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)

    built = run("build")
    recalled = run("recall")
    assert built["pid"] != recalled["pid"]
    assert built["pending"] == 0
    assert built["view_usable"]
    assert built["simulation_status"] == "fits"
    assert built["retained_suffix_exact"]
    assert recalled["exact_code_recovered"]
    assert recalled["recovered_sha256"] == built["expected_code_sha256"]
    assert recalled["pages"] > 1
    assert len(recalled["navigation"]) > 1
    assert recalled["archive_events_before"] == recalled["archive_events_after"]
