import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from hermes_ariadne.store import EventStore


def test_committed_snapshot_survives_abrupt_exit_and_open_transaction_rolls_back(tmp_path):
    path = tmp_path / "events.sqlite3"
    message = {"message_uid": "crash-test", "role": "user", "content": "exact\n  café  "}
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    code = """
import json, os, sqlite3, sys
from hermes_ariadne.store import EventStore
path, payload = sys.argv[1:]
store = EventStore(path)
store.ingest('s', [json.loads(payload)], origin='hook')
conn = sqlite3.connect(path)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
conn.execute('INSERT INTO events VALUES (?, ?, ?, ?)',
             ('uncommitted', 'not-real', '{}', 'must not survive'))
os._exit(0)
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(path), json.dumps(message)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == 0, process.stderr
    store = EventStore(path)
    assert store.search("café", session_id="s")[0]["message"] == message
    assert store.ingest("s", [message], origin="hook")["inserted"] == 0
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
