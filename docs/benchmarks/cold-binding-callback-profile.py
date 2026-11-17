"""Disposable engine-callback profile using existing synthetic test fixtures.

No AIAgent/native dispatch/provider calls: this exercises real engine public
callbacks against the existing injected DB/Facade. Set isolated HOME/HERMES_HOME,
HERMES_SOURCE and PYTHONPATH to the frozen package under test before invocation.
Arguments: REPO_FIXTURES OUTPUT_DIRECTORY. It writes only beneath that new scratch
output directory and creates profiles after a 32-generation fixture seed.
"""

import copy
import cProfile
import io
import json
import os
import pstats
import shutil
import sys
import time
from pathlib import Path

repo, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
home = Path(os.environ["HERMES_HOME"]).resolve()
if home == Path("/opt/data") or "/cache/scratch/" not in str(output) or output.exists():
    raise SystemExit("fresh scratch output and isolated home required")
output.mkdir(mode=0o700)
sys.path.insert(0, str(repo / "tests"))
sys.path.insert(0, os.environ["HERMES_SOURCE"])

import pytest

import hermes_ariadne
from hermes_ariadne import rolling
from test_rolling_engine import DB, complete, exchange, select, setup_engine

origin = str(Path(hermes_ariadne.__file__).resolve())
results = {"origin": origin, "claim": "synthetic-injected-facade-callbacks-not-provider-latency", "generations": 32}
with pytest.MonkeyPatch.context() as patch:
    engine, db, facade = setup_engine(output / "untraced", patch)
    for index in range(33):
        tag = f"turn-{index}"
        db.rows += exchange(tag, complete=False)[:1]
        select(engine, db, tag)
        db.rows = db.rows[:-1] + exchange(tag)
        complete(engine, db, tag)
    if engine._published.sequence != 32:
        raise RuntimeError("fixture did not reach32 publications")
    tag = "probe"
    db.rows += exchange(tag, complete=False)[:1]
    # No callbacks/journals are running while copying these synthetic files.
    shutil.copytree(output / "untraced", output / "profiled")
    other_db = DB()
    other_db.rows = copy.deepcopy(db.rows)
    profiled, _, _ = setup_engine(output / "profiled", patch, db=other_db)

    for label, candidate, source_db, trace in (("untraced", engine, db, False), ("profiled", profiled, other_db, True)):
        def selection():
            if select(candidate, source_db, tag) is None:
                raise RuntimeError("selection declined")

        def navigation():
            response = json.loads(candidate.handle_tool_call("ariadne_rolling_page", {"ref": "rolling:" + candidate._pinned.generation_id}))
            if not response["ok"]:
                raise RuntimeError("navigation declined")

        def completion():
            source_db.rows = source_db.rows[:-1] + exchange(tag)
            complete(candidate, source_db, tag)
            if candidate._published.sequence != 33:
                raise RuntimeError("completion did not publish generation33")

        for name, function in (("select", selection), ("navigation", navigation), ("completion", completion)):
            candidate._verification = rolling.RollingVerificationCache()
            profiler = cProfile.Profile()
            started = time.perf_counter()
            if trace:
                profiler.runcall(function)
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats("hermes_ariadne", 25)
                (output / (name + ".profile.txt")).write_text(stream.getvalue())
            else:
                function()
            results[label + "_" + name + "_s"] = round(time.perf_counter() - started, 6)
    results["final_publications"] = [engine._published.sequence, profiled._published.sequence]
print(json.dumps(results))
