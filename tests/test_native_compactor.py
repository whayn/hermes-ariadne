"""Synthetic loopback evidence for the offline native-factory compatibility adapter."""

import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def native_worker(tmp_path):
    source = os.environ.get("HERMES_SOURCE")
    if not source:
        pytest.skip("HERMES_SOURCE required for actual native factory tests")
    requests = []
    response = {"status": 200, "mode": "json"}

    class Fixture(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            decoded = raw
            if self.headers.get("Content-Encoding") == "zstd":
                from compression import zstd

                decoded = zstd.decompress(raw)
            requests.append(
                {
                    "path": self.path,
                    "raw": raw,
                    "decoded": decoded,
                    "body": json.loads(decoded),
                    "headers": dict(self.headers),
                }
            )
            answer = {
                "id": "synthetic-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "ariadne/test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "synthetic summary"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 32, "completion_tokens": 4, "total_tokens": 36},
            }
            answer.update(response.get("answer", {}))
            mode = response["mode"]
            if mode == "disconnect":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if mode == "stall":
                time.sleep(0.6)
                return
            if mode == "redirect":
                self.send_response(307)
                self.send_header("Location", self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = b"not-json" if mode == "malformed" else json.dumps(answer).encode()
            if mode == "sse":
                item = response.get(
                    "item",
                    {
                        "id": "msg_fixture",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "phase": "final_answer",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "synthetic Codex summary",
                                "annotations": [],
                            }
                        ],
                    },
                )
                terminal = {
                    "id": "resp_fixture",
                    "object": "response",
                    "created_at": 1,
                    "model": "ariadne/test",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "error": None,
                    "incomplete_details": None,
                    "status": "completed",
                    "output": None,
                    "usage": {"input_tokens": 32, "output_tokens": 4, "total_tokens": 36},
                    **response.get("terminal", {}),
                }
                if response.get("omit_usage"):
                    terminal.pop("usage", None)
                events = [
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                    {"type": "response." + terminal["status"], "response": terminal},
                ]
                if response.get("terminal_only"):
                    terminal["output"] = [item]
                    events = events[1:]
                delimiter = response.get("delimiter", "\n")
                separator = response.get("separator", delimiter * 2)
                payload = "".join("data: " + json.dumps(e) + separator for e in events).encode()
            if "sse_payload" in response:
                payload = response["sse_payload"]
            coding = response.get("content_encoding")
            if coding == "gzip":
                payload = gzip.compress(payload)
            elif coding == "corrupt-gzip":
                payload = b"invalid gzip stream"
            elif coding == "zstd":
                from compression import zstd

                payload = zstd.compress(payload)
            self.send_response(response["status"])
            if coding:
                self.send_header("Content-Encoding", "gzip" if coding == "corrupt-gzip" else coding)
            if not response.get("omit_content_type"):
                self.send_header(
                    "Content-Type", "text/event-stream" if mode == "sse" else "application/json"
                )
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("x-request-id", "synthetic-request")
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "isolated-home"
    home.mkdir()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    config = {
        "auxiliary": {
            "compression": {
                "provider": "custom",
                "model": "ariadne/test",
                "base_url": url,
                "api_key": "offline-synthetic-placeholder",
                "api_mode": "chat_completions",
            }
        }
    }

    def run(action="success", options=None, messages=None, route=None):
        if route:
            config["auxiliary"]["compression"].update(route)
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "TMPDIR", "PYTHONPATH")
            if key in os.environ
        }
        env.update(
            HOME=str(tmp_path),
            HERMES_HOME=str(home),
            HERMES_MANAGED_DIR=tempfile.mkdtemp(prefix="managed-", dir=tmp_path),
            TZ="UTC",
            PYTHONPATH=os.pathsep.join(
                (
                    source,
                    str(Path(__file__).resolve().parents[1] / "src"),
                    env.get("PYTHONPATH", ""),
                )
            ),
        )
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures" / "native_compactor_worker.py"),
                json.dumps({"action": action, "options": options or {}, "messages": messages}),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        report = json.loads(process.stdout.splitlines()[-1])
        assert report["responses_closed"], report
        assert report["response_streams_unchanged"], report
        assert report["response_observers_clean"], report
        return report

    yield run, requests, response
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.mark.parametrize("coding", [None, "gzip"])
def test_codex_missing_content_type_uses_validated_sse(native_worker, coding):
    run, requests, response = native_worker
    response.update(mode="sse", omit_content_type=True, content_encoding=coding)
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    assert report["completion"]["text"] == "synthetic Codex summary"
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"] and report["same_transport"]


def test_codex_missing_content_type_does_not_accept_json_as_completion(native_worker):
    run, requests, response = native_worker
    response.update(omit_content_type=True)
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert not report["ok"]
    assert report["outcome"] == "uncertain"
    assert len(requests) == len(report["reservations"]) == 1


def test_native_chat_exact_serialized_reservation_and_completion(native_worker):
    run, requests, _ = native_worker
    report = run()
    assert report["ok"], report
    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["max_tokens"] == 512
    assert request["body"]["model"] == "ariadne/test"
    assert report["reservations"] == [
        {
            "request_sha256": hashlib.sha256(request["raw"]).hexdigest(),
            "request_bytes": len(request["raw"]),
            "output_tokens": 512,
        }
    ]
    assert report["completion"] == {
        "text": "synthetic summary",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 32, "completion_tokens": 4, "total_tokens": 36},
        "provider": "custom",
        "model": "ariadne/test",
        "request_id": "synthetic-request",
        "refusal": False,
    }
    assert report["estimate"] == report["native_estimate"]
    assert report["estimate"] > 0
    assert report["fingerprint"]["estimator"] == "hermes_native_rough"
    assert len(report["fingerprint"]["host_fingerprint"]) == 64
    assert "127.0.0.1" not in json.dumps(report["fingerprint"])
    assert "offline-synthetic" not in json.dumps(report["fingerprint"])
    assert report["hooks_clean"]
    assert report["same_transport"]
    assert report["retries"] == 0


def test_malformed_completion_types_are_not_success(native_worker):
    run, requests, response = native_worker
    response["answer"] = {
        "choices": [{"index": 0, "message": {"content": 42}, "finish_reason": "stop"}]
    }
    report = run()
    assert not report["ok"], report
    assert report["outcome"] == "uncertain"
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"]


@pytest.mark.parametrize("status", [400, 429, 500])
def test_native_http_failures_have_no_hidden_retries(native_worker, status):
    run, requests, response = native_worker
    response["status"] = status
    response["answer"] = {"error": {"message": "synthetic-secret-not-for-reason"}}
    report = run(action="retry-config")
    assert not report["ok"]
    assert report["outcome"] == "failed"
    assert report["status_code"] == status
    assert len(requests) == len(report["reservations"]) == 1
    assert "synthetic-secret" not in report["reason"]
    assert report["hooks_clean"] and report["same_transport"]
    assert report["retries"] == 7


@pytest.mark.parametrize("mode", ["disconnect", "stall", "malformed", "redirect"])
def test_post_reservation_interruption_is_uncertain_without_resend(native_worker, mode):
    run, requests, response = native_worker
    response["mode"] = mode
    report = run(
        action="redirect-config" if mode == "redirect" else "success", options={"timeout": 0.2}
    )
    assert not report["ok"]
    assert report["outcome"] == "uncertain"
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"]


def test_overflow_prevents_callback_and_post(native_worker):
    run, requests, _ = native_worker
    report = run(messages=[{"role": "user", "content": "é" * 9000}])
    assert not report["ok"]
    assert report["outcome"] == "not_dispatched"
    assert not requests and not report["reservations"]
    assert report["hooks_clean"]


def test_exact_8000_serialized_bytes_allowed(native_worker):
    run, requests, _ = native_worker
    messages = [{"role": "user", "content": ""}]
    base = {"messages": messages, "model": "ariadne/test", "max_tokens": 512}
    overhead = len(json.dumps(base, ensure_ascii=False, separators=(",", ":")).encode())
    messages[0]["content"] = "x" * (8000 - overhead)
    report = run(messages=messages)
    assert report["ok"], report
    assert len(requests[0]["raw"]) == 8000
    assert report["reservations"][0]["request_bytes"] == 8000


def test_callback_budget_refusal_has_original_cause_and_no_post(native_worker):
    run, requests, _ = native_worker
    report = run(action="budget-refusal")
    assert not report["ok"]
    assert report["outcome"] == "not_dispatched"
    assert report["cause_type"] == "LedgerBudgetError"
    assert len(report["reservations"]) == 1 and not requests
    assert report["hooks_clean"]


@pytest.mark.parametrize(
    "action",
    [
        "wire-mutation",
        "wire-model-mutation",
        "cap-mutation",
        "transport-replaced",
        "no-send",
        "unsupported",
    ],
)
def test_fail_closed_before_any_reservation(native_worker, action):
    run, requests, _ = native_worker
    report = run(action=action)
    assert not report["ok"]
    assert report["outcome"] == "not_dispatched"
    assert not requests and not report["reservations"]
    if "hooks_clean" in report:
        assert report["hooks_clean"]


def test_nonisolated_adapter_is_refused(native_worker):
    run, requests, _ = native_worker
    report = run(options={"isolated_worker": False})
    assert not report["ok"] and report["outcome"] == "not_dispatched"
    assert not requests and not report["reservations"]


def test_reentrant_dispatch_does_not_touch_worker_transport(native_worker):
    run, requests, _ = native_worker
    report = run(action="concurrent")
    assert report["ok"], report
    assert report["concurrent_outcome"] == "not_dispatched"
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"]


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": None}, {"completion_tokens": 4}])
def test_usage_preserves_missing_null_and_partial_fields(native_worker, usage):
    run, requests, response = native_worker
    response["answer"] = {"usage": usage}
    report = run()
    assert report["ok"], report
    assert report["completion"]["usage"] == usage
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("text", "finish", "refusal"),
    [
        (None, "stop", None),
        ("", "stop", None),
        ("partial", "length", None),
        (None, "content_filter", "synthetic refusal"),
        (None, "tool_calls", None),
        ("unknown", None, None),
    ],
)
def test_completion_signals_returned_for_parent_validation(native_worker, text, finish, refusal):
    run, _, response = native_worker
    response["answer"] = {
        "choices": [
            {"index": 0, "message": {"content": text, "refusal": refusal}, "finish_reason": finish}
        ]
    }
    report = run()
    assert report["ok"], report
    assert report["completion"]["text"] == text
    assert report["completion"]["finish_reason"] == finish
    assert report["completion"]["refusal"] == bool(refusal)


def test_native_model_cap_predicate_preserves_completion_cap(native_worker):
    run, requests, _ = native_worker
    report = run(route={"model": "gpt-5.1"}, options={"max_output_tokens": 123})
    assert report["ok"], report
    assert requests[0]["body"]["max_completion_tokens"] == 123
    assert "max_tokens" not in requests[0]["body"]
    assert report["reservations"][0]["output_tokens"] == 123


@pytest.mark.parametrize("coding", [None, "gzip"])
def test_mixed_lf_cr_refusal_and_terminal_usage_native_loopback(native_worker, coding):
    run, requests, response = native_worker
    usage = {"input_tokens": 32, "output_tokens": 4, "total_tokens": 36}
    response.update(
        mode="sse",
        content_encoding=coding,
        separator="\n\r",
        item={
            "id": "msg_refusal",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "synthetic refusal"}],
        },
        terminal={"usage": usage},
    )
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    # The real native normalizer accepts this refusal text as a stopped completion.
    assert report["completion"]["text"] == "synthetic refusal"
    assert report["completion"]["finish_reason"] == "stop"
    assert report["completion"]["refusal"] is True, report
    assert report["completion"]["usage"] == usage
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"] and report["same_transport"]


@pytest.mark.parametrize("coding", [None, "gzip"])
@pytest.mark.parametrize("separator", ["\n\n", "\n\r"])
def test_terminal_only_refusal_native_loopback(native_worker, separator, coding):
    run, requests, response = native_worker
    response.update(
        mode="sse",
        content_encoding=coding,
        separator=separator,
        terminal_only=True,
        item={
            "id": "msg_refusal",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "synthetic refusal"}],
        },
    )
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    # This host assembles text from item events, not terminal response.output.
    assert report["completion"]["text"] is None
    assert report["completion"]["finish_reason"] == "stop"
    assert report["completion"]["refusal"] is True, report
    assert report["completion"]["usage"]["total_tokens"] == 36
    assert len(requests) == len(report["reservations"]) == 1


@pytest.mark.parametrize("coding", [None, "gzip"])
@pytest.mark.parametrize(
    ("separator", "refusal_location"),
    [
        ("\n\r", "item"),
        ("\n\n", "terminal_only"),
        ("\n\r", "terminal_only"),
        ("\n\n", "terminal_output"),
        ("\n\r", "terminal_output"),
    ],
)
def test_refusal_never_publishes_durable_success(
    native_worker, coding, separator, refusal_location
):
    run, requests, response = native_worker
    response.update(
        mode="sse",
        content_encoding=coding,
        separator=separator,
        terminal_only=refusal_location == "terminal_only",
        item={
            "id": "msg_refusal",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "synthetic refusal"}],
        },
    )
    if refusal_location == "terminal_output":
        # Text arrives normally, but the only refusal signal is in terminal output.
        response["terminal"] = {"output": [response["item"]]}
        response["item"] = {
            **response["item"],
            "content": [{"type": "output_text", "text": "synthetic summary", "annotations": []}],
        }
    report = run(
        action="durable",
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    assert report["build"]["pending"] == 1, report
    assert report["build"]["built"] == 0
    assert not report["usable"]
    assert report["receipt"] is None
    assert report["job"]["state"] == "FAILED"
    assert report["job"]["reason"] == "refused_or_incomplete_output"
    assert len(requests) == report["ledger"]["attempts"] == len(report["job"]["attempts"]) == 1
    attempt = report["job"]["attempts"][0]
    assert attempt["state"] == "FAILED"
    assert attempt["usage"]["total_tokens"] == 36
    assert attempt["request_bytes"] == len(requests[0]["raw"])
    assert attempt["request_sha256"] == hashlib.sha256(requests[0]["raw"]).hexdigest()
    assert report["ledger"]["limits"]["max_attempts_per_job"] == 1
    assert report["ledger"]["remaining"]["max_attempts"] == 0
    assert "SUCCEEDED" not in report["ledger"]["states"]
    assert report["hooks_clean"] and report["same_transport"]


@pytest.mark.parametrize("delimiter", ["\n", "\r\n", "\r"])
def test_native_codex_stream_keeps_wire_alias_and_provider_usage(native_worker, delimiter):
    run, requests, response = native_worker
    response["mode"] = "sse"
    response["delimiter"] = delimiter
    report = run(
        route={"api_mode": "codex_responses", "model": "gpt-5.4-900k"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    assert len(requests) == len(report["reservations"]) == 1
    assert requests[0]["path"] == "/v1/responses"
    body = requests[0]["body"]
    assert body["model"] == "gpt-5.4"
    assert body["stream"] is True
    assert not any(k in body for k in ("max_tokens", "max_output_tokens", "max_completion_tokens"))
    assert report["reservations"][0]["output_tokens"] is None
    assert report["completion"]["text"] == "synthetic Codex summary"
    assert report["completion"]["finish_reason"] == "stop"
    assert report["completion"]["usage"] == {
        "input_tokens": 32,
        "output_tokens": 4,
        "total_tokens": 36,
    }
    assert report["completion"]["request_id"] == "synthetic-request"
    assert report["fingerprint"]["model"] == "gpt-5.4-900k"
    assert report["hooks_clean"] and report["same_transport"]


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": None}, {"output_tokens": 4}])
def test_codex_usage_does_not_inherit_host_synthetic_zeroes(native_worker, usage):
    run, _, response = native_worker
    response.update(mode="sse", terminal={"usage": usage})
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    assert report["completion"]["usage"] == usage


@pytest.mark.parametrize("coding", [None, "gzip"])
@pytest.mark.parametrize("signal", ["incomplete", "refusal", "tool_calls"])
def test_codex_native_normalizer_retains_parent_rejection_signals(native_worker, signal, coding):
    run, requests, response = native_worker
    response.update(mode="sse", content_encoding=coding)
    if signal == "incomplete":
        response["terminal"] = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
    elif signal == "refusal":
        response["item"] = {
            "id": "msg_refusal",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "synthetic refusal"}],
        }
    else:
        response["item"] = {
            "id": "fc_fixture",
            "type": "function_call",
            "call_id": "call_fixture",
            "name": "synthetic_tool",
            "arguments": "{}",
            "status": "completed",
        }
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    assert len(requests) == 1
    if signal == "refusal":
        assert report["completion"]["refusal"] is True
    else:
        assert report["completion"]["finish_reason"] == (
            "length" if signal == "incomplete" else "tool_calls"
        )
    assert report["hooks_clean"]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"allow_unbounded_output": True},
        {"max_output_tokens": None},
        {"allow_unbounded_output": True, "max_output_tokens": 512},
    ],
)
def test_codex_never_claims_a_hard_output_cap(native_worker, options):
    run, requests, _ = native_worker
    report = run(route={"api_mode": "codex_responses"}, options=options)
    assert not report["ok"] and report["outcome"] == "not_dispatched"
    assert not requests and not report["reservations"]


@pytest.fixture
def sdk_framing_client(tmp_path, monkeypatch):
    """Supplemental deterministic chunk probes; positive proof uses native_worker."""
    if not os.environ.get("HERMES_SOURCE"):
        pytest.skip("HERMES_SOURCE required for inspected native wrapper")
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))

    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden in SDK framing unit fixture")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    import httpx
    import openai
    from agent import auxiliary_client as native

    from hermes_ariadne.native_compactor import NativeCompactorAdapter

    state = {"chunks": [], "responses": [], "read": 0, "closed": 0}

    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            for chunk in state["chunks"]:
                state["read"] += len(chunk)
                yield chunk

        def close(self):
            state["closed"] += 1

    def send(request):
        response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Chunks()
        )
        state["responses"].append(response)
        return response

    http = httpx.Client(transport=httpx.MockTransport(send))
    sdk = openai.OpenAI(
        base_url="http://127.0.0.1/v1",
        api_key="synthetic-placeholder",
        http_client=http,
        max_retries=7,
    )
    client = native.CodexAuxiliaryClient(sdk, "ariadne/test")
    monkeypatch.setattr(
        native, "get_text_auxiliary_client", lambda *a, **kw: (client, "ariadne/test")
    )
    adapter = NativeCompactorAdapter.from_hermes(
        isolated_worker=True, allow_unbounded_output=True, max_output_tokens=None
    )
    yield adapter, state
    assert http.event_hooks == {"request": [], "response": []}
    assert sdk.max_retries == 7
    assert all(response.is_closed for response in state["responses"])
    assert all("iter_bytes" not in response.__dict__ for response in state["responses"])
    http.close()


def refusal_events():
    return [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg_refusal",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "synthetic refusal"}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_fixture",
                "status": "completed",
                "output": None,
                "usage": {
                    "input_tokens": 32,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 11},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            },
        },
    ]


@pytest.mark.parametrize("coding", [None, "gzip"])
@pytest.mark.parametrize("ending", ["eof", "malformed", "duplicate"])
def test_text_then_eof_never_publishes_durable_success_or_retries(native_worker, coding, ending):
    run, requests, response = native_worker
    event = refusal_events()[0]
    event["item"]["content"] = [
        {"type": "output_text", "text": "synthetic complete-looking text", "annotations": []}
    ]
    events = [event]
    if ending == "malformed":
        events.append({"type": "response.completed", "response": {"id": "resp_fixture"}})
    elif ending == "duplicate":
        events.extend([refusal_events()[1], refusal_events()[1]])
    response.update(
        mode="sse",
        content_encoding=coding,
        sse_payload="".join("data: " + json.dumps(e) + "\n\n" for e in events).encode(),
    )
    for _ in range(2):  # New isolated process, same durable ledger/kernel on restart.
        report = run(
            action="durable-terminal",
            route={"api_mode": "codex_responses"},
            options={"allow_unbounded_output": True, "max_output_tokens": None},
        )
        assert report["ok"], report
        assert report["build"]["built"] == 0, report
        assert report["build"]["pending"] == 1
        assert not report["usable"] and report["published_nodes"] == 0
        assert report["receipt"] is None
        assert report["job"]["state"] == "UNCERTAIN"
        assert report["ledger"]["attempts"] == len(report["job"]["attempts"]) == len(requests) == 1
        attempt = report["job"]["attempts"][0]
        assert attempt["state"] == "UNCERTAIN" and attempt["usage"] is None
        assert attempt["request_bytes"] == len(requests[0]["raw"])
        assert attempt["request_sha256"] == hashlib.sha256(requests[0]["raw"]).hexdigest()
        assert report["ledger"]["remaining"]["max_attempts"] == 2
        assert report["hooks_clean"] and report["same_transport"]


@pytest.mark.parametrize("coding", [None, "gzip"])
def test_valid_terminal_publishes_and_reuses_durable_receipt(native_worker, coding):
    run, requests, response = native_worker
    response.update(mode="sse", content_encoding=coding, terminal={"usage": None})
    for _ in range(2):
        report = run(
            action="durable-terminal",
            route={"api_mode": "codex_responses"},
            options={"allow_unbounded_output": True, "max_output_tokens": None},
        )
        assert report["ok"], report
        assert report["build"]["pending"] == 0 and report["usable"]
        assert report["published_nodes"] == 1
        assert report["job"]["state"] == "SUCCEEDED"
        assert report["receipt"]["text"] == "synthetic Codex summary"
        assert report["receipt"]["usage"] is None
        assert len(requests) == report["ledger"]["attempts"] == 1
        assert report["fingerprint"]["completion_validation"] == "codex-terminal-v1"


@pytest.mark.parametrize(
    "case",
    [
        "done-only",
        "missing-response",
        "null-response",
        "array-response",
        "missing-status",
        "missing-output",
        "missing-id",
        "empty-id",
        "numeric-id",
        "in-progress",
        "failed-status",
        "error",
        "incomplete-details",
        "duplicate",
        "conflicting",
        "failed-completed-status",
        "incomplete-completed-status",
        "bad-output",
        "terminal-tool",
    ],
)
def test_unsafe_terminal_metadata_is_uncertain(sdk_framing_client, case):
    from hermes_ariadne.native_compactor import NativeDispatchError

    adapter, state = sdk_framing_client
    events = refusal_events()
    events[0]["item"]["content"] = [
        {"type": "output_text", "text": "synthetic text", "annotations": []}
    ]
    terminal = events[1]["response"]
    if case == "done-only":
        events = events[:1]
    elif case == "missing-response":
        events[1].pop("response")
    elif case in {"null-response", "array-response"}:
        events[1]["response"] = None if case == "null-response" else []
    elif case.startswith("missing-"):
        terminal.pop(case.removeprefix("missing-"))
    elif case in {"empty-id", "numeric-id"}:
        terminal["id"] = "" if case == "empty-id" else 17
    elif case in {"in-progress", "failed-status"}:
        terminal["status"] = "in_progress" if case == "in-progress" else "failed"
    elif case == "error":
        terminal["error"] = {"code": "server_error", "message": "synthetic"}
    elif case == "incomplete-details":
        terminal["incomplete_details"] = {"reason": "max_output_tokens"}
    elif case in {"duplicate", "conflicting"}:
        events.append(json.loads(json.dumps(events[1])))
        if case == "conflicting":
            events[2]["type"] = "response.failed"
            events[2]["response"]["status"] = "failed"
    elif case in {"failed-completed-status", "incomplete-completed-status"}:
        events[1]["type"] = "response." + case.split("-")[0]
    elif case == "bad-output":
        terminal["output"] = "not-an-output-array"
    elif case == "terminal-tool":
        terminal["output"] = [{"type": "function_call", "name": "synthetic", "arguments": "{}"}]
    state["chunks"] = [("data: " + json.dumps(event) + "\n\n").encode() for event in events]
    if case == "done-only":
        state["chunks"].append(b"data: [DONE]\n\n")
    reservations = []
    with pytest.raises(NativeDispatchError) as failure:
        adapter.dispatch(
            [{"role": "user", "content": "synthetic"}], before_send=reservations.append
        )
    assert failure.value.outcome == "uncertain"
    assert len(reservations) == 1


def test_completion_validation_version_binds_receipt_namespace(sdk_framing_client, tmp_path):
    from hermes_ariadne.compactor_demo import _ReceiptOnlyAdapter
    from hermes_ariadne.receipts import CompactionLedger, LedgerError

    adapter, _ = sdk_framing_client
    route = adapter.fingerprint
    assert route["completion_validation"] == "codex-terminal-v1"
    receipt_only = _ReceiptOnlyAdapter(
        {"route": route, "model": adapter.model, "output_tokens": None}
    )
    assert receipt_only.fingerprint == route
    old_route = {key: value for key, value in route.items() if key != "completion_validation"}
    limits = dict(
        max_attempts=3,
        max_request_bytes=8000,
        max_total_request_bytes=24000,
        max_attempts_per_job=3,
        max_jobs=8,
    )
    path = tmp_path / "old-receipts.sqlite"
    with CompactionLedger(path, namespace={"route": old_route}, limits=limits) as ledger:
        ledger.put_plan("old", {})
        attempt = ledger.reserve(
            "old", request_sha256="a" * 64, request_bytes=1, output_tokens=None
        )
        ledger.succeed("old", attempt, text="pre-fix EOF text", usage=None)
    with pytest.raises(LedgerError, match="namespace"):
        CompactionLedger(path, namespace={"route": route}, limits=limits)


@pytest.mark.parametrize("first", ["\n", "\r", "\r\n"])
@pytest.mark.parametrize("second", ["\n", "\r", "\r\n"])
@pytest.mark.parametrize("tail", ["", "\n", "\n\r"])
def test_sdk_framing_differential_chunk_matrix(
    sdk_framing_client, monkeypatch, first, second, tail
):
    from openai._streaming import SSEDecoder

    from hermes_ariadne.native_compactor import NativeDispatchError

    original_frames = SSEDecoder._iter_chunks
    traces = []

    def trace_frames(self, iterator):
        frames = []
        traces.append(frames)
        for frame in original_frames(self, iterator):
            frames.append(frame)
            yield frame

    monkeypatch.setattr(SSEDecoder, "_iter_chunks", trace_frames)
    adapter, state = sdk_framing_client
    events = refusal_events()
    payload = (
        "data: " + json.dumps(events[0]) + first + second + "data: " + json.dumps(events[1]) + tail
    ).encode()
    messages = [{"role": "user", "content": "synthetic"}]
    # Include cuts inside CRLF, JSON, immediately before EOF, and every byte.
    cuts = sorted({1, len(payload) // 2, len(payload) - 1, payload.index(b"data: ", 1) - 1})
    chunkings = [[payload], [payload[i : i + 1] for i in range(len(payload))]]
    chunkings.extend([payload[:cut], payload[cut:]] for cut in cuts)
    for chunks in chunkings:
        expected_frames = list(original_frames(SSEDecoder(), iter(chunks)))
        assert b"".join(expected_frames) == payload
        traces.clear()
        state["chunks"] = chunks
        baseline_error = None
        baseline = None
        try:
            baseline = adapter._client.chat.completions.create(
                model=adapter.model, messages=messages, timeout=5
            )
        except Exception as exc:
            baseline_error = exc
        reservations = []
        decoded = (
            [json.loads(event.data) for event in SSEDecoder().iter_bytes(iter(chunks))]
            if baseline_error is None
            else []
        )
        terminals = [event for event in decoded if event["type"] == "response.completed"]
        if baseline_error is not None or not terminals:
            with pytest.raises(NativeDispatchError) as failure:
                adapter.dispatch(messages, before_send=reservations.append)
            assert failure.value.outcome == "uncertain"
        else:
            result = adapter.dispatch(messages, before_send=reservations.append)
            assert baseline is not None
            choice = baseline.choices[0]
            assert result["text"] == choice.message.content
            assert result["finish_reason"] == choice.finish_reason
            assert result["usage"] == terminals[-1]["response"]["usage"]
            assert result["refusal"] is True
        assert len(reservations) == 1
        # Both observer and downstream SDK must see the exact authoritative frames,
        # or their consumed prefix if native stopped/failed before reading all bytes.
        assert traces
        assert all(frames == expected_frames[: len(frames)] for frames in traces)
        response = state["responses"][-1]
        assert response.is_closed and "iter_bytes" not in response.__dict__
        assert adapter._http.event_hooks == {"request": [], "response": []}
        assert adapter._sdk.max_retries == 7


def test_coalesced_eof_observes_refusal_after_completed_before_host_stops(sdk_framing_client):
    adapter, state = sdk_framing_client
    events = refusal_events()
    # At EOF the SDK coalesces these mixed delimiters into ONE frame. The host
    # stops at completed, so observing lazily one event at a time would lose refusal.
    state["chunks"] = [
        "".join("data: " + json.dumps(event) + "\n\r" for event in reversed(events)).encode()
    ]
    result = adapter.dispatch(
        [{"role": "user", "content": "synthetic"}], before_send=lambda data: None
    )
    assert result["refusal"] is True
    assert result["usage"] == events[1]["response"]["usage"]


def test_unterminated_metadata_is_bounded_before_sdk_buffering(sdk_framing_client, monkeypatch):
    from openai._streaming import SSEDecoder

    from hermes_ariadne.native_compactor import NativeDispatchError

    adapter, state = sdk_framing_client
    limit = 1024 * 1024
    state["chunks"] = [b"data: " + b"x" * (65536 - 6)] + [b"x" * 65536] * 32
    original = SSEDecoder._iter_chunks
    fed = []

    def tracked(self, iterator):
        sizes = []
        fed.append(sizes)

        def record():
            for chunk in iterator:
                sizes.append(len(chunk))
                yield chunk

        yield from original(self, record())

    monkeypatch.setattr(SSEDecoder, "_iter_chunks", tracked)
    reservations = []
    with pytest.raises(NativeDispatchError) as failure:
        adapter.dispatch(
            [{"role": "user", "content": "synthetic"}], before_send=reservations.append
        )
    assert failure.value.outcome == "uncertain"
    assert "metadata frame exceeds observation limit" in str(failure.value.__cause__)
    assert len(reservations) == 1
    assert max(map(sum, fed)) == limit  # Over-limit fragment never reaches SDK's buffer.
    assert state["read"] == limit + 65536
    assert state["closed"] == 1


@pytest.mark.parametrize("coding", [None, "gzip"])
def test_unterminated_metadata_loopback_is_uncertain_and_cleans_up(native_worker, coding):
    run, requests, response = native_worker
    response.update(
        mode="sse", content_encoding=coding, sse_payload=b"data: " + b"x" * (1024 * 1024)
    )
    report = run(
        action="retry-config",
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert not report["ok"], report
    assert report["outcome"] == "uncertain"
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"] and report["same_transport"]
    assert report["retries"] == 7


@pytest.fixture
def encoded_negative_client(tmp_path, monkeypatch):
    """Official-URL negative evidence only; no physical transport/network allowed."""
    if not os.environ.get("HERMES_SOURCE"):
        pytest.skip("HERMES_SOURCE required for inspected native wrapper")
    zstd = pytest.importorskip("compression.zstd")
    monkeypatch.setenv("HOME", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden in official-URL negative fixture")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    import httpx
    import openai
    from agent import auxiliary_client as native
    from agent.codex_headers import CODEX_AUX_BASE_URL, codex_request_compression_hooks

    from hermes_ariadne.native_compactor import NativeCompactorAdapter

    sends, reservations, encoded = [], [], []

    def send(request):
        sends.append(request)
        return httpx.Response(500, json={"error": {"message": "unexpected send"}})

    hooks = codex_request_compression_hooks(CODEX_AUX_BASE_URL)
    assert hooks is not None
    http = httpx.Client(transport=httpx.MockTransport(send), event_hooks=hooks)
    sdk = openai.OpenAI(
        base_url=CODEX_AUX_BASE_URL,
        api_key="offline-synthetic-placeholder",
        http_client=http,
        max_retries=7,
    )
    client = native.CodexAuxiliaryClient(sdk, "gpt-5.4-900k")
    monkeypatch.setattr(
        native, "get_text_auxiliary_client", lambda *a, **kw: (client, "gpt-5.4-900k")
    )
    adapter = NativeCompactorAdapter.from_hermes(
        isolated_worker=True, allow_unbounded_output=True, max_output_tokens=None
    )

    def remember(request):
        encoded.append(request.content)

    http.event_hooks["request"].append(remember)
    yield adapter, http, sends, reservations, encoded, zstd
    http.close()


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "stacked",
        "duplicate",
        "corrupt",
        "truncated",
        "trailing",
        "concatenated",
        "skippable",
        "model",
        "cap",
        "decoded-overflow",
        "unknown-size-overflow",
        "window-overflow",
        "wire-overflow",
        "decoder-unavailable",
    ],
)
def test_official_encoded_requests_fail_closed_before_reservation(
    encoded_negative_client, monkeypatch, case
):
    import builtins

    import httpx

    from hermes_ariadne.native_compactor import NativeDispatchError

    adapter, http, sends, reservations, encoded, zstd = encoded_negative_client
    expected = None

    def mutate(request):
        nonlocal expected
        body = request.content
        raw = zstd.decompress(body)
        if case in {"unknown", "stacked", "duplicate"}:
            request.headers["Content-Encoding"] = {
                "unknown": "gzip",
                "stacked": "gzip, zstd",
                "duplicate": "zstd, zstd",
            }[case]
            expected = "unsupported native request encoding"
        elif case == "corrupt":
            body = b"not-zstd"
            expected = "invalid native zstd frame"
        elif case == "truncated":
            body = body[:-1]
            expected = "incomplete or trailing native zstd frame"
        elif case == "trailing":
            body += b"trailing"
            expected = "incomplete or trailing native zstd frame"
        elif case == "concatenated":
            body += body
            expected = "incomplete or trailing native zstd frame"
        elif case == "skippable":
            body = b"\x50\x2a\x4d\x18\x00\x00\x00\x00" + body
            expected = "invalid native zstd frame"
        elif case in {"model", "cap"}:
            wire = json.loads(raw)
            wire["model" if case == "model" else "max_output_tokens"] = (
                "other-model" if case == "model" else 1
            )
            body = zstd.compress(json.dumps(wire).encode())
            expected = (
                "native wire model changed" if case == "model" else "unexpected Codex output cap"
            )
        elif case in {"decoded-overflow", "unknown-size-overflow"}:
            raw = b" " * 8001
            options = {zstd.CompressionParameter.window_log: 13}
            if case == "unknown-size-overflow":
                options[zstd.CompressionParameter.content_size_flag] = 0
            body = zstd.compress(raw, options=options)
            if case == "unknown-size-overflow":
                assert zstd.get_frame_info(body).decompressed_size is None
            expected = "decoded request exceeds byte limit"
        elif case == "window-overflow":
            compressor = zstd.ZstdCompressor(options={zstd.CompressionParameter.window_log: 20})
            body = compressor.compress(raw) + compressor.flush()
            assert zstd.get_frame_info(body).decompressed_size is None
            # The frame declares a 1 MiB window despite a small actual payload.
        elif case == "wire-overflow":
            body = zstd.compress(raw, level=-131072)
            adapter._max_request_bytes = len(raw)
            assert len(body) > len(raw)
            expected = "serialized request exceeds byte limit"
        elif case == "decoder-unavailable":
            original_import = builtins.__import__

            def without_zstd(name, *args, **kwargs):
                if name == "compression":
                    raise ImportError("simulated pre-3.14 stdlib")
                return original_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", without_zstd)
            expected = "native zstd decoder unavailable"
        request._content = body
        request.stream = httpx.ByteStream(body)
        request.headers["Content-Length"] = str(len(body))

    http.event_hooks["request"].append(mutate)
    original_hooks = {key: list(value) for key, value in http.event_hooks.items()}
    with pytest.raises(NativeDispatchError) as failure:
        adapter.dispatch(
            [{"role": "user", "content": "synthetic"}], before_send=reservations.append
        )
    assert failure.value.outcome == "not_dispatched"
    assert not sends and not reservations
    cause = failure.value
    chain = []
    while cause is not None:
        chain.append(str(cause))
        cause = cause.__cause__
    if expected:
        assert expected in chain
    else:
        assert any("too much memory" in item.lower() for item in chain), chain
    assert encoded
    assert http.event_hooks == original_hooks
    assert adapter._sdk.max_retries == 7


@pytest.mark.parametrize(
    "usage",
    [
        "absent",
        None,
        {},
        {"input_tokens": None},
        {"output_tokens": 4},
        {
            "input_tokens": 32,
            "output_tokens": 4,
            "input_tokens_details": {"cached_tokens": 11},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    ],
)
@pytest.mark.parametrize("coding", [None, "gzip"])
def test_gzip_codex_metadata_uses_native_http_decoding(native_worker, usage, coding):
    run, requests, response = native_worker
    response.update(
        mode="sse", content_encoding=coding, terminal={"usage": usage}, omit_usage=usage == "absent"
    )
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], json.dumps(report)
    assert report["completion"]["text"] == "synthetic Codex summary"
    assert report["completion"]["finish_reason"] == "stop"
    assert report["completion"]["usage"] == (None if usage == "absent" else usage)
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"] and report["same_transport"]


@pytest.mark.skipif(sys.version_info < (3, 14), reason="native zstd hook requires Python 3.14")
def test_actual_official_zstd_hook_native_factory_loopback_exact_bytes(native_worker):
    run, requests, response = native_worker
    response["mode"] = "sse"
    report = run(
        action="zstd-success",
        route={"api_mode": "codex_responses", "model": "gpt-5.4-900k"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], json.dumps({**report, "physical_posts": len(requests)})
    assert report["actual_compression_hook"]
    assert len(requests) == len(report["reservations"]) == 1
    request = requests[0]
    assert request["headers"]["Content-Encoding"] == "zstd"
    assert int(request["headers"]["Content-Length"]) == len(request["raw"])
    assert request["raw"] != request["decoded"]
    assert request["body"]["model"] == "gpt-5.4"
    assert not any(k in request["body"] for k in ("max_tokens", "max_output_tokens"))
    assert report["reservations"] == [
        {
            "request_sha256": hashlib.sha256(request["raw"]).hexdigest(),
            "request_bytes": len(request["raw"]),
            "output_tokens": None,
        }
    ]
    assert report["completion"]["text"] == "synthetic Codex summary"
    assert report["hooks_clean"] and report["same_transport"]


@pytest.mark.parametrize("failure", ["corrupt-gzip", "oversized-frame"])
def test_encoded_codex_failure_retains_uncertain_reservation(native_worker, failure):
    run, requests, response = native_worker
    response.update(
        mode="sse", content_encoding="corrupt-gzip" if failure == "corrupt-gzip" else "gzip"
    )
    if failure == "oversized-frame":
        response["terminal"] = {"synthetic_padding": "x" * (1024 * 1024)}
    report = run(
        action="retry-config",
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert not report["ok"], report
    assert report["outcome"] == "uncertain"
    assert len(requests) == len(report["reservations"]) == 1
    assert report["retries"] == 7
    assert report["hooks_clean"] and report["same_transport"]


@pytest.mark.skipif(sys.version_info < (3, 14), reason="stdlib zstd encoding requires Python 3.14")
def test_native_httpx_zstd_response_metadata_when_decoder_available(native_worker):
    pytest.importorskip("zstandard", reason="HTTPX 0.28 requires its optional zstandard dependency")
    run, requests, response = native_worker
    usage = {"input_tokens": None, "output_tokens_details": {"reasoning_tokens": 2}}
    response.update(mode="sse", content_encoding="zstd", terminal={"usage": usage})
    report = run(
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert report["ok"], report
    assert report["completion"]["usage"] == usage
    assert report["completion"]["text"] == "synthetic Codex summary"
    assert report["completion"]["finish_reason"] == "stop"
    assert len(requests) == len(report["reservations"]) == 1
    assert report["hooks_clean"]


@pytest.mark.skipif(sys.version_info < (3, 14), reason="native zstd hook requires Python 3.14")
@pytest.mark.parametrize("failure", ["budget-refusal", "byte-overflow"])
def test_actual_zstd_hook_refusal_or_overflow_has_no_post_or_reservation(native_worker, failure):
    run, requests, response = native_worker
    response["mode"] = "sse"
    report = run(
        action="zstd-budget-refusal" if failure == "budget-refusal" else "zstd-success",
        messages=[{"role": "user", "content": "x" * 9000}] if failure == "byte-overflow" else None,
        route={"api_mode": "codex_responses"},
        options={"allow_unbounded_output": True, "max_output_tokens": None},
    )
    assert not report["ok"] and report["outcome"] == "not_dispatched"
    assert report["actual_compression_hook"]
    assert not requests and not report["reservations"]
    if failure == "budget-refusal":
        assert report["cause_type"] == "LedgerBudgetError"
    assert report["hooks_clean"] and report["same_transport"]


def test_raw_request_works_without_stdlib_zstd(native_worker):
    run, requests, _ = native_worker
    report = run(action="raw-decoder-unavailable")
    assert report["ok"], report
    assert len(requests) == len(report["reservations"]) == 1
    assert "Content-Encoding" not in requests[0]["headers"]
    assert report["hooks_clean"]


def test_chat_cannot_opt_out_of_its_output_cap(native_worker):
    run, requests, _ = native_worker
    report = run(options={"allow_unbounded_output": True, "max_output_tokens": None})
    assert not report["ok"] and report["outcome"] == "not_dispatched"
    assert not requests and not report["reservations"]
