"""Real Git installer + plugin manager + native agent/tool loop in fresh profiles.

Scripted provider only: no model-generated extraction or semantic recall claim.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("HERMES_SOURCE"), reason="explicit Hermes source required")
def test_normal_git_install_agent_ingestion_new_conversation_restart(tmp_path):
    root = Path(__file__).resolve().parents[1]
    home = tmp_path / "library-test-home"
    home.mkdir(mode=0o700)
    managed = tmp_path / "managed"
    managed.mkdir(mode=0o700)
    fixture = tmp_path / "plugin-repo"
    fixture.mkdir()
    files = [
        "__init__.py",
        "plugin.yaml",
        "LICENSE",
        "NOTICE",
        "src/hermes_ariadne/__init__.py",
        "src/hermes_ariadne/store.py",
        "src/hermes_ariadne/library.py",
        "src/hermes_ariadne/library_plugin.py",
        "src/hermes_ariadne/skills/ariadne-import/SKILL.md",
        "src/hermes_ariadne/skills/ariadne-import/scripts/source_range.py",
    ]
    for name in files:
        source = root / name
        target = fixture / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    git_env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Synthetic Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "synthetic plugin snapshot",
        ],
    ):
        subprocess.run(["git", *args], cwd=fixture, env=git_env, check=True, capture_output=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=fixture,
        env=git_env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    raw = b'{"arbitrary_app": {"message": "I prefer jasmine tea", "excluded": "DO_NOT_RETURN_RAW_METADATA"}}'
    path = tmp_path / "supplied.json"
    path.write_bytes(raw)
    source = {
        "path": str(path),
        "start_bytes": 0,
        "length_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "label": "Synthetic supplied JSON",
    }
    note = {
        "title": "Tea preference",
        "text": "The source author prefers jasmine tea.",
        "kind": "summary",
        "tags": ["preference"],
        "sources": [0],
    }
    plan = {"git_url": fixture.as_uri(), "git_sha": sha, "source": source, "note": note}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    # Disposable synthetic profile only. No existing profile/isolation settings changed.
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [ariadne-library]\ncontext_engine:\n  provider: compressor\nmodel:\n  context_length: 128000\nagent:\n  api_max_retries: 0\n  auto_recovery_cycles: 0\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if "proxy" not in k.lower()
        and not k.endswith(("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD", "_CREDENTIALS"))
        and not k.startswith("HERMES_SESSION_")
    }
    env.update(
        HOME=str(tmp_path),
        HERMES_HOME=str(home),
        HERMES_MANAGED_DIR=str(managed),
        PYTHONPATH=os.environ["HERMES_SOURCE"] + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    requests, errors = [], []
    mode = ["prepare"]

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)

            if (
                body.get("response_format", {}).get("json_schema", {}).get("name")
                == "session_title"
            ):
                payload = json.dumps(
                    {
                        "id": "synthetic-title",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "synthetic/library",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps({"title": "Synthetic memory ingestion"}),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            available = {tool["function"]["name"] for tool in body.get("tools", [])}
            try:
                names = {
                    "ariadne_memory_prepare",
                    "ariadne_memory_search",
                    "ariadne_memory_read",
                    "ariadne_memory_review",
                }
                # This tracer intentionally covers the pinned host's deferred path.
                assert names.isdisjoint(available)
                assert {"tool_describe", "tool_call", "tool_search"} <= available
                assert all(name in json.dumps(body["tools"]) for name in names)
                assert "ariadne_memory_approve" not in available
                tools = [msg for msg in body["messages"] if msg["role"] == "tool"]
                step = len(tools) - 1
                if not tools:
                    tool_name, args = "tool_describe", {"names": sorted(names)}
                elif mode[0] == "prepare" and step == 0:
                    tool_name, args = (
                        "ariadne_memory_prepare",
                        {"sources": [source], "notes": [note]},
                    )
                elif mode[0] == "prepare" and step == 1:
                    prepared = json.loads(tools[-1]["content"])
                    assert prepared["ok"]
                    tool_name, args = (
                        "ariadne_memory_review",
                        {"ref": prepared["result"]["proposal_ref"]},
                    )
                elif mode[0] == "recall" and step == 0:
                    tool_name, args = "ariadne_memory_search", {"query": "jasmine"}
                elif mode[0] == "recall" and step == 1:
                    found = json.loads(tools[-1]["content"])
                    assert found["ok"] and len(found["result"]["results"]) == 1
                    tool_name, args = (
                        "ariadne_memory_read",
                        {"ref": found["result"]["results"][0]["ref"]},
                    )
                else:
                    assert step == 2 and "DO_NOT_RETURN_RAW_METADATA" not in json.dumps(tools)
                    if mode[0] == "prepare":
                        review = json.loads(tools[-1]["content"])
                        assert review["ok"]
                        assert review["result"]["note"]["text"] == note["text"]
                    tool_name = None
                if tool_name:
                    if tool_name not in available:
                        tool_name, args = (
                            "tool_call",
                            {"calls": [{"name": tool_name, "arguments": args}]},
                        )
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call" + str(len(tools)),
                                "type": "function",
                                "function": {"name": tool_name, "arguments": json.dumps(args)},
                            }
                        ],
                    }
                else:
                    message = {"role": "assistant", "content": "SYNTHETIC DONE"}
                response = {
                    "id": "synthetic",
                    "object": "chat.completion",
                    "model": "synthetic/library",
                    "created": 1,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool_name else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                }
                if body.get("stream"):
                    delta = {**message}
                    if delta.get("tool_calls"):
                        delta["tool_calls"] = [
                            {**call, "index": i} for i, call in enumerate(delta["tool_calls"])
                        ]
                    chunks = [
                        {
                            **response,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        },
                        {
                            **response,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": response["choices"][0]["finish_reason"],
                                }
                            ],
                        },
                    ]
                    payload = (
                        "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                        + "data: [DONE]\n\n"
                    ).encode()
                else:
                    payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/event-stream" if body.get("stream") else "application/json",
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except BaseException as exc:
                errors.append(type(exc).__name__)

                self.send_error(500, "synthetic fixture assertion failed")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    receipts = []
    try:
        for phase in ("prepare", "recall"):
            mode[0] = phase
            stderr = tmp_path / (phase + ".stderr")
            with stderr.open("wb") as output:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(root / "tests/fixtures/library_worker.py"),
                        phase,
                        f"http://127.0.0.1:{server.server_port}/v1",
                        str(plan_path),
                    ],
                    env=env,
                    cwd=tmp_path,
                    stdout=subprocess.PIPE,
                    stderr=output,
                    timeout=180,
                )
            code = process.returncode
            assert code == 0, f"library worker {phase} failed (exit {code}); see {stderr}"
            receipts.append(json.loads(process.stdout.decode().splitlines()[-1]))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not errors
    assert len(requests) == 10  # Eight main-loop requests plus two host-owned session titles.
    assert (
        sum(
            body.get("response_format", {}).get("json_schema", {}).get("name") == "session_title"
            for body in requests
        )
        == 2
    )
    assert receipts[0]["pid"] != receipts[1]["pid"]
    assert receipts[0]["note_refs"] == receipts[1]["note_refs"]
    assert (
        receipts[0]["source_sha256"]
        == receipts[1]["source_sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
    assert receipts[0]["tool_names"] == [
        "tool_describe",
        "ariadne_memory_prepare",
        "ariadne_memory_review",
    ]
    assert receipts[1]["tool_names"] == [
        "tool_describe",
        "ariadne_memory_search",
        "ariadne_memory_read",
    ]
    assert path.read_bytes() == raw
