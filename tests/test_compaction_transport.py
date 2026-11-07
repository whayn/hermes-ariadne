"""Combined native-factory HTTP + durable kernel restart; scripted provider only."""

import json
import os
import subprocess
import sys
import tempfile
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
    root = tmp_path / "isolated-root"
    home = root / "profiles" / "fixture-dev"
    home.mkdir(parents=True)
    config = home / "config.yaml"
    config.write_text(
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
        HOME=str(root),
        HERMES_HOME=str(root),
        HERMES_MANAGED_DIR=tempfile.mkdtemp(prefix="managed-", dir=tmp_path),
        HERMES_CONFIG=str(root / "config.yaml"),
        HERMES_ENV=str(root / ".env"),
        HERMES_PROFILE="default",
        HERMES_ACTIVE_PROFILE="default",
        HERMES_SOURCE=host,
        PYTHONPATH=os.pathsep.join(
            [str(Path("src").resolve()), host, os.environ.get("PYTHONPATH", "")]
        ),
        NO_PROXY="127.0.0.1,localhost",
        TZ="UTC",
        PYTHONNOUSERSITE="1",
    )

    # The CLI/factory/SDK remain real. These tripwires only observe selection and
    # prevent accidental external traffic or any host import during offline replay.
    child = """
import builtins, os, runpy, socket, sys
selected = sys.argv.pop(1)
rebuilding = sys.argv[sys.argv.index('--phase') + 1] == 'rebuild'
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {
        'agent', 'hermes_cli', 'hermes_constants', 'openai', 'httpx', 'native_compactor'
    }:
        assert not rebuilding, 'receipt-only rebuild imported host/native code'
        assert os.environ['HOME'] == os.environ['HERMES_HOME'] == selected
        assert not any(key in os.environ for key in (
            'HERMES_CONFIG', 'HERMES_ENV', 'HERMES_PROFILE', 'HERMES_ACTIVE_PROFILE'
        ))
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
original_connect = socket.socket.connect
def loopback_connect(sock, address):
    assert isinstance(address, tuple) and address[0] in {'127.0.0.1', '::1'}
    return original_connect(sock, address)
socket.socket.connect = loopback_connect
original_resolve = socket.getaddrinfo
def loopback_resolve(host, *args, **kwargs):
    assert host in {'127.0.0.1', '::1', 'localhost'}, 'non-loopback DNS forbidden'
    return original_resolve(host, *args, **kwargs)
socket.getaddrinfo = loopback_resolve
runpy.run_module('hermes_ariadne.compactor_demo', run_name='__main__')
"""
    before = config.read_bytes()

    def run(phase):
        command = [
            sys.executable,
            "-c",
            child,
            str(home),
            str(tmp_path / "case"),
            "--phase",
            phase,
        ]
        if phase == "build":
            command += ["--allow-model-calls", "--hermes-home", str(home)]
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
        assert config.read_bytes() == before
        assert not (home / "auth.json").exists()
        assert not (home / ".env").exists()
        assert not errors
    finally:
        server.shutdown()
        server.server_close()
