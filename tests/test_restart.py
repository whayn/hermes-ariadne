"""Two actual Hermes processes; deterministic HTTP fixture, no live provider/key."""

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hermes_ariadne.install import install

EXACT = "ARIADNE-RESTART-MARKER café\n\t  preserve this spacing  "


def test_exact_search_after_process_restart_and_missed_hook(tmp_path):
    home = tmp_path / "isolated-home"
    home.mkdir()
    install(home)
    (home / "config.yaml").write_text(
        "context:\n  engine: ariadne\nmodel:\n  context_length: 128000\n"
        "auxiliary:\n  title_generation:\n    enabled: false\n",
        encoding="utf-8",
    )
    requests = []
    errors = []

    class ProviderFixture(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)

                assert all(
                    "message_uid" not in m and "_tool_call_uid" not in m for m in body["messages"]
                )
                last = body["messages"][-1]
                if last["role"] == "tool":
                    data = json.loads(last["content"])
                    stored = next(r for r in data["results"] if r["message"]["role"] == "user")
                    assert stored["message"]["content"] == EXACT
                    message = {"role": "assistant", "content": "retrieved exact message"}
                    finish = "stop"
                elif last["content"] == "SEARCH-ARCHIVE-NOW":
                    assert any(t["function"]["name"] == "ariadne_search" for t in body["tools"])
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "fixture_call",
                                "type": "function",
                                "function": {
                                    "name": "ariadne_search",
                                    "arguments": json.dumps({"query": "ARIADNE-RESTART-MARKER"}),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    # Hermes's existing request sanitizer strips trailing whitespace;
                    # Ariadne must still archive the exact DURABLE original.
                    assert last["content"] == EXACT.rstrip()
                    message = {"role": "assistant", "content": "stored"}
                    finish = "stop"
                answer = {
                    "id": "offline-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "ariadne/test",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                }
                streaming = bool(body.get("stream"))
                if streaming:
                    delta = dict(message)
                    if delta.get("tool_calls"):
                        delta["tool_calls"] = [
                            {**tc, "index": i} for i, tc in enumerate(delta["tool_calls"])
                        ]
                    chunks = [
                        {
                            **answer,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        },
                        {
                            **answer,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        },
                    ]
                    payload = (
                        "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                        + "data: [DONE]\n\n"
                    ).encode()
                else:
                    payload = json.dumps(answer).encode()
                self.send_response(200)
            except Exception as exc:
                errors.append(repr(exc))
                streaming = False
                payload = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderFixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    worker = Path(__file__).parent / "fixtures" / "hermes_worker.py"
    # Whitelist, not a copy of the gateway environment: no inherited auth/session/profile.
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "TMPDIR", "PYTHONPATH")
        if key in os.environ
    }
    env.update(
        HERMES_HOME=str(home),
        HERMES_MANAGED_DIR=tempfile.mkdtemp(prefix="managed-", dir=tmp_path),
        HOME=str(tmp_path),
        TZ="UTC",
        PYTHONPATH=os.environ["HERMES_SOURCE"] + os.pathsep + env.get("PYTHONPATH", ""),
    )
    reports = []
    try:
        for mode in ("seed", "search"):
            process = subprocess.run(
                [sys.executable, str(worker), mode, url],
                env=env,
                cwd=tmp_path,
                text=True,
                capture_output=True,
                timeout=90,
            )
            assert process.returncode == 0, process.stdout + process.stderr
            report = json.loads(process.stdout.splitlines()[-1])
            reports.append(report)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not errors, errors
    assert len(requests) == 3
    assert reports[0]["pid"] != reports[1]["pid"]
    assert reports[0]["uid"] == reports[1]["uid"]
    assert reports[1]["exact_content"] == EXACT
    assert reports[1]["archive_error"] is None
    assert reports[1]["recovered_missing_hook"] == 1
    print(json.dumps({"restart_evidence": reports}, ensure_ascii=False))
