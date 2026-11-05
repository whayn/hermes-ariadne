"""Installed engine → real AIAgent/sanitizers/SDK/HTTP/tools, synthetic only.

No provider-fit or semantic-recall claim: summaries and HTTP completions are
explicit fixtures. The host's unmodified SDK and tool dispatcher own the loop.
"""

import base64
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_ariadne.retrieval import tool_schemas


@pytest.fixture
def prepared_worker(tmp_path):
    source = os.environ.get("HERMES_SOURCE")
    if not source:
        pytest.skip("HERMES_SOURCE required for real Hermes transport")
    assert source is not None
    home = tmp_path / "isolated-home"
    home.mkdir()
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "TMPDIR", "PYTHONPATH")
        if key in os.environ
    }
    env.update(
        HOME=str(tmp_path),
        HERMES_HOME=str(home),
        TZ="UTC",
        PYTHONPATH=os.pathsep.join(
            (source, str(Path(__file__).resolve().parents[1] / "src"), env.get("PYTHONPATH", ""))
        ),
    )
    # Operator CLI plus actual directory discovery, outside the checkout.
    process = subprocess.run(
        [sys.executable, "-m", "hermes_ariadne.install", "--home", str(home), "--prepared"],
        text=True,
        capture_output=True,
        timeout=30,
        env=env,
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert (home / "plugins" / "ariadne-prepared" / "plugin.yaml").is_file()
    (home / "config.yaml").write_text(
        "context:\n  engine: ariadne-prepared\nmodel:\n  context_length: 128000\n"
        "auxiliary:\n  title_generation:\n    enabled: false\n",
        encoding="utf-8",
    )
    requests, errors = [], []
    trace = {}

    def tool_call(name, arguments):
        trace["tool_calls"].append(name)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"synthetic_call_{len(trace['tool_calls'])}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, sort_keys=True, separators=(",", ":")),
                    },
                }
            ],
        }

    def respond(body) -> dict:
        messages = body["messages"]
        # Persistence/replay sidecars must never reach the HTTP provider.
        for m in messages:
            assert set(m) <= {"role", "content", "tool_calls", "tool_call_id", "name"}
            if m["role"] == "tool":
                assert "name" not in m
            for call in m.get("tool_calls", []):
                assert set(call) == {"id", "type", "function"}
                assert set(call["function"]) == {"name", "arguments"}
        memory = next(
            (
                m["content"]
                for m in messages
                if isinstance(m.get("content"), str)
                and m["content"].startswith("Archived navigation")
            ),
            None,
        )
        if memory is not None:
            trace["navigation"].append(memory)
        last = messages[-1]
        if trace["script"] and memory and trace["step"] == 0:
            trace["step"] = 1
            trace["node_ref"] = json.loads(memory.splitlines()[1])["ref"]
            pattern = tool_schemas()[1]["parameters"]["properties"]["ref"]["pattern"]
            assert re.fullmatch(pattern, trace["node_ref"])
            return tool_call("ariadne_find", {"query": "SYNTHETIC-ARCHIVED-EXACT"})
        if trace["script"] and last["role"] == "tool" and trace["step"]:
            content = last["content"]
            assert len(content.encode("ascii")) <= 8192
            evidence = json.loads(content)
            assert evidence["ok"] is True, evidence
            assert "untrusted" in evidence["trust"]
            if trace["step"] == 1:
                assert evidence["kind"] == "find"
                assert evidence["count"] == len(evidence["results"])
                # Restart legitimately archives the earlier find output too.
                # Match the unique canonical USER source, not its lossy echo.
                sources = [hit for hit in evidence["results"] if hit["role"] == "user"]
                assert len(sources) == 1
                trace["find_counts"].append(evidence["count"])
                trace["event_ref"] = sources[0]["event_ref"]
                trace["step"] = 2
                return tool_call("ariadne_zoom", {"ref": trace["node_ref"]})
            if trace["step"] == 2:
                assert evidence["kind"] == "node"
                assert len(evidence["children"]) == 2
                assert evidence["node"]["ref"] == trace["node_ref"]
                trace["children"] = evidence["children"]
                trace["step"] = 3
                return tool_call("ariadne_zoom", {"ref": trace["event_ref"], "limit": 1024})
            assert evidence["kind"] == "source"
            page = evidence["page"]
            assert page["event_ref"] == trace["event_ref"]
            assert page["encoding"] == "canonical-json:utf-8-surrogatepass:base64"
            assert page["offset"] == len(trace["source"])
            trace["source"] += base64.b64decode(page["data_base64"], validate=True)
            trace["pages"] += 1
            if page["next_offset"] is not None:
                assert page["next_offset"] == len(trace["source"])
                return tool_call(
                    "ariadne_zoom",
                    {"ref": trace["event_ref"], "limit": 1024, "offset": page["next_offset"]},
                )
            assert page["total_bytes"] == len(trace["source"])
            trace["complete"] = True
        return {"role": "assistant", "content": "synthetic fixture complete"}

    class ProviderFixture(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append({"path": self.path, "body": body})
                assert self.path == "/v1/chat/completions"
                message = respond(body)
                finish = "tool_calls" if message.get("tool_calls") else "stop"
                answer = {
                    "id": "synthetic-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "ariadne/test",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                }
                streaming = bool(body.get("stream"))
                if streaming:
                    delta = copy.deepcopy(message)
                    if delta.get("tool_calls"):
                        delta["tool_calls"] = [
                            {**call, "index": i} for i, call in enumerate(delta["tool_calls"])
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

    def run(mode):
        trace.clear()
        trace.update(
            script=mode in {"live", "restart"},
            step=0,
            tool_calls=[],
            source=b"",
            pages=0,
            find_counts=[],
            navigation=[],
            complete=False,
        )
        before = len(requests)
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures" / "prepared_worker.py"),
                mode,
                f"http://127.0.0.1:{server.server_port}/v1",
            ],
            env=env,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert not errors, errors
        report = json.loads(process.stdout.splitlines()[-1])
        sent = requests[before:]
        assert report["api_calls"] == len(sent), "SDK physical HTTP count differs"
        print(
            json.dumps(
                {
                    "prepared_worker_trace": {
                        "mode": mode,
                        "pid": report["pid"],
                        "physical_posts": len(sent),
                        "selection": report["selection"],
                        "find_counts": trace["find_counts"],
                        "source_pages": trace["pages"],
                        "source_bytes": len(trace["source"]),
                        "original_rows_unchanged": report["original_rows_unchanged"],
                        "tools": report["tools"],
                        "operator_package_origin": report["operator_package_origin"],
                        "openai": report["openai_version"],
                        "httpx": report["httpx_version"],
                    }
                },
                sort_keys=True,
            )
        )
        return report, sent, copy.deepcopy(trace)

    try:
        yield run
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def assert_selected(report):
    assert report["selection"]["status"] == "selected", json.dumps(report["selection"])
    assert report["selection"]["provider_fit_verified"] is False
    assert report["original_rows_unchanged"] == 8
    assert report["archive_error"] is None


def assert_reassembled(report, trace):
    assert trace["complete"], trace
    expected = base64.b64decode(report["exact_source_base64"])
    assert trace["source"] == expected
    assert trace["event_ref"] == report["exact_event_ref"]
    assert "event:" + hashlib.sha256(trace["source"]).hexdigest() == trace["event_ref"]
    original = json.loads(trace["source"].decode("utf-8", "surrogatepass"))
    assert original["content"].endswith("  ")
    assert original["message_uid"] == "synthetic-message-0"
    assert trace["pages"] > 1
    assert trace["tool_calls"][:2] == ["ariadne_find", "ariadne_zoom"]
    assert set(trace["tool_calls"][2:]) == {"ariadne_zoom"}
    assert len(set(trace["navigation"])) == 1, "navigation changed within one turn"


def assert_complete_retained_loop(sent, trace):
    assert len(sent) == len(trace["tool_calls"]) + 1
    assert len(trace["navigation"]) == len(sent), "selection must survive EVERY tool iteration"
    previous = None
    for index, request in enumerate(sent):
        names = [tool["function"]["name"] for tool in request["body"].get("tools", [])]
        assert sorted(names) == ["ariadne_find", "ariadne_zoom"]
        messages = request["body"]["messages"]
        if previous is not None:
            assert messages[:-2] == previous, "an earlier protected tool/message changed"
            owner, result = messages[-2:]
            assert owner["role"] == "assistant" and result["role"] == "tool"
            assert len(owner["tool_calls"]) == 1
            call = owner["tool_calls"][0]
            assert call["function"]["name"] == trace["tool_calls"][index - 1]
            assert call["id"] == result["tool_call_id"]
        pending = set()
        for message in messages:
            for call in message.get("tool_calls", []):
                assert call["id"] not in pending
                pending.add(call["id"])
            if message["role"] == "tool":
                assert message["tool_call_id"] in pending
                pending.remove(message["tool_call_id"])
        assert not pending, "provider input lost a tool result"
        previous = messages


def test_installed_prepared_prefix_reaches_actual_sdk_transport_and_restart(prepared_worker):
    baseline, original, _ = prepared_worker("baseline")
    live, selected, trace = prepared_worker("live")
    assert_selected(live)
    assert len(original) == 1
    raw, wire = original[0]["body"]["messages"], selected[0]["body"]["messages"]
    assert wire[0] == raw[0], "host policy prefix changed"
    assert wire[2:] == raw[5:], "protected previous exchange/current wire suffix changed"
    assert "Archived navigation" in wire[1]["content"]
    assert "SYNTHETIC FIXTURE SUMMARY" in wire[1]["content"]
    assert "SYNTHETIC-ARCHIVED-EXACT" not in json.dumps(wire)
    assert "SYNTHETIC-OLD-SECOND" not in json.dumps(wire)
    assert raw[-1]["content"].endswith("protected whitespace")  # real host sanitizer
    assert [m["role"] for m in wire[2:]] == ["user", "assistant", "tool", "assistant", "user"]
    assert wire[3]["tool_calls"][0]["id"] == wire[4]["tool_call_id"]
    assert baseline["pid"] != live["pid"]
    assert_reassembled(live, trace)
    assert_complete_retained_loop(selected, trace)
    assert trace["find_counts"] == [1]
    restarted, resumed, after = prepared_worker("restart")
    assert_selected(restarted)
    assert_reassembled(restarted, after)
    assert_complete_retained_loop(resumed, after)
    assert after["find_counts"] == [2]
    assert resumed[0]["body"]["messages"][:-2] == selected[-1]["body"]["messages"]
    assert resumed[0]["body"]["messages"][-2:] == [
        {"role": "assistant", "content": "synthetic fixture complete"},
        raw[-1],
    ]
    assert restarted["pid"] not in {baseline["pid"], live["pid"]}
    assert resumed[0]["body"]["messages"][:2] == wire[:2]
    assert after["node_ref"] == trace["node_ref"]
    assert after["event_ref"] == trace["event_ref"]
    print(
        json.dumps(
            {
                "live_transport_evidence": {
                    "pids": [baseline["pid"], live["pid"], restarted["pid"]],
                    "physical_posts": [len(original), len(selected), len(resumed)],
                    "source_pages": [trace["pages"], after["pages"]],
                    "canonical_source_bytes": len(trace["source"]),
                    "unchanged_original_rows": live["original_rows_unchanged"],
                    "node_ref": trace["node_ref"],
                    "event_ref": trace["event_ref"],
                    "openai": live["openai_version"],
                    "httpx": live["httpx_version"],
                }
            },
            sort_keys=True,
        )
    )


def test_disabled_context_engine_toolset_is_not_injected(prepared_worker):
    _, _, _ = prepared_worker("baseline")
    report, sent, _ = prepared_worker("disabled")
    assert_selected(report)
    assert report["tools"] == []
    assert len(sent) == 1
    names = {tool["function"]["name"] for tool in sent[0]["body"].get("tools", [])}
    assert not names & {"ariadne_find", "ariadne_zoom", "ariadne_search"}
    assert "Archived navigation" in sent[0]["body"]["messages"][1]["content"]


def test_unmapped_prefill_declines_and_host_sends_original_not_veto(prepared_worker):
    baseline, original, _ = prepared_worker("baseline-fallback")
    report, sent, _ = prepared_worker("fallback")
    assert report["selection"]["status"] == "declined"
    assert report["selection"]["reason"] == "unsupported_alignment"
    assert report["selection"]["host_fallback"] == "original_request_on_decline"
    assert len(original) == len(sent) == report["api_calls"] == 1
    assert sent[0]["body"]["messages"] == original[0]["body"]["messages"]
    assert any(
        m.get("content") == "SYNTHETIC-UNMAPPED-PREFILL" for m in sent[0]["body"]["messages"]
    )
    assert "SYNTHETIC-ARCHIVED-EXACT" in json.dumps(sent[0]["body"]["messages"])
    assert report["original_rows_unchanged"] == baseline["original_rows_unchanged"] == 8
