"""Combined native-factory HTTP + durable kernel restart; scripted provider only."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def test_native_compaction_and_receipt_only_rebuild_across_processes(tmp_path):
    host = os.environ.get("HERMES_SOURCE")
    if not host:
        pytest.skip("explicit Hermes source/dependencies required for native transport")
    requests = []
    errors = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                payload = json.loads(raw)
                requests.append(payload)
                assert len(raw) <= 8000
                assert payload["max_tokens"] == 512
                target = json.loads(payload["messages"][1]["content"])["untrusted_data"]
                text = "NEEDLE increment code" if "NEEDLE" in target else "garden basil plan"
                answer = {
                    "id": "scripted-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
                }
                body = json.dumps(answer).encode()
                self.send_response(200)
            except Exception as error:
                errors.append(repr(error))
                body = b'{"error":"scripted fixture assertion failed"}'
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = tmp_path / "isolated-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    provider: custom\n    model: fixture\n"
        f"    base_url: http://127.0.0.1:{server.server_port}/v1\n"
        "    api_key: synthetic-fixture-key\n"
    )
    env = {
        key: os.environ[key]
        for key in ("PATH", "TMPDIR", "LANG", "SYSTEMROOT")
        if key in os.environ
    }
    env.update(
        HOME=str(home),
        HERMES_HOME=str(home),
        HERMES_SOURCE=host,
        PYTHONPATH=os.pathsep.join(
            [str(Path("src").resolve()), host, os.environ.get("PYTHONPATH", "")]
        ),
        NO_PROXY="127.0.0.1,localhost",
        TZ="UTC",
        PYTHONNOUSERSITE="1",
    )

    def run(phase):
        command = [
            sys.executable,
            "-m",
            "hermes_ariadne.compactor_demo",
            str(tmp_path / "case"),
            "--phase",
            phase,
        ]
        if phase == "build":
            command.append("--allow-model-calls")
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr + result.stdout
        return json.loads(result.stdout.strip().splitlines()[-1])

    try:
        built = run("build")
        assert built["pending"] == 0
        assert len(requests) == 3
        rebuilt = run("rebuild")
        assert built["pid"] != rebuilt["pid"]
        assert rebuilt["pending"] == 0
        assert rebuilt["exact_source_recovered"]
        assert rebuilt["model_dispatches"] == 0
        assert len(requests) == 3
        assert not errors
    finally:
        server.shutdown()
        server.server_close()
