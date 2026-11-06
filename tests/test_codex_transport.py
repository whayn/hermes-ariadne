"""Installed prepared engine → actual AIAgent/native Codex SDK/Responses SSE.

All source, summaries, auth and responses are synthetic; only loopback is allowed.
This proves native replay/selection, not provider fit, model recall or paid routing.
"""

import ast
import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def completed_message(turn):
    return {
        "id": "msg_synthetic_" + turn,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": "synthetic Codex " + turn}],
    }


def codex_transport_summary(mode, report, sent, trace):
    return {
        "mode": mode,
        "pid": report["pid"],
        "posts": len(sent),
        "selections": [
            {"turn": entry["turn"], **entry["selection"]} for entry in report["selections"]
        ],
        "seed_sha256": report["seed_sha256"],
        "source_pages": trace["pages"],
        "openai": report["openai_version"],
        "httpx": report["httpx_version"],
        "operator_package_origin": report["operator_package_origin"],
        "engine_origin": report["engine_origin"],
    }


@pytest.mark.parametrize("mode", ["baseline", "text", "tools", "restart", "restarttools"])
def test_pure_transport_summary_includes_import_origins(mode):
    report = {
        "pid": 123,
        "selections": [],
        "seed_sha256": "synthetic-seed-digest",
        "openai_version": "synthetic-sdk-version",
        "httpx_version": "synthetic-httpx-version",
        "operator_package_origin": "/synthetic/wheel/site-packages/hermes_ariadne/__init__.py",
        "engine_origin": "/synthetic/isolated-home/plugins/ariadne-prepared/live_engine.py",
    }
    summary = codex_transport_summary(mode, report, [], {"pages": 0})
    for key in ("operator_package_origin", "engine_origin"):
        assert summary.get(key) == report[key]


@pytest.fixture
def codex_worker(tmp_path):
    source = os.environ.get("HERMES_SOURCE")
    if not source:
        pytest.skip("HERMES_SOURCE required for real Hermes Codex transport")
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
    installed = subprocess.run(
        [sys.executable, "-m", "hermes_ariadne.install", "--home", str(home), "--prepared"],
        text=True,
        capture_output=True,
        timeout=30,
        env=env,
        cwd=tmp_path,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    assert (home / "plugins" / "ariadne-prepared" / "plugin.yaml").is_file()
    (home / "config.yaml").write_text(
        "context:\n  engine: ariadne-prepared\nmodel:\n  context_length: 128000\n"
        "auxiliary:\n  title_generation:\n    enabled: false\n",
        encoding="utf-8",
    )
    requests, errors = [], []
    trace = {}

    def tool_call(name, arguments):
        index = len(trace["calls"]) + 1
        item = {
            "type": "function_call",
            "id": f"fc_native_DIFFERENT_{index}",
            "call_id": f"call_synthetic_tools_{index}",
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True, separators=(",", ":")),
            "status": "completed",
        }
        trace["calls"].append(copy.deepcopy(item))
        return item

    def respond(body, turn):
        if turn != "tools":
            return completed_message(turn)
        last = body["input"][-1]
        if last.get("type") != "function_call_output":
            assert not trace["calls"], "tool cycle restarted unexpectedly"
            return tool_call("ariadne_find", {"query": "SYNTHETIC-CODEX-ARCHIVED"})
        assert len(last["output"].encode("ascii")) <= 8192
        evidence = json.loads(last["output"])
        assert evidence["ok"] is True, evidence
        assert "untrusted" in evidence["trust"]
        if evidence["kind"] == "find":
            assert evidence["count"] == len(evidence["results"])
            sources = [hit for hit in evidence["results"] if hit["role"] == "user"]
            assert len(sources) == 1
            trace["event_ref"] = sources[0]["event_ref"]
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
        return completed_message(turn)

    class ProviderFixture(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append({"path": self.path, "body": body})
                assert self.path == "/v1/responses"
                assert body["stream"] is True
                assert "messages" not in body
                user = [item for item in body["input"] if item.get("role") == "user"][-1]
                content = user["content"]
                text = content if isinstance(content, str) else "".join(p["text"] for p in content)
                assert text.startswith("SYNTHETIC-CODEX-TURN-")
                turn = text.removeprefix("SYNTHETIC-CODEX-TURN-")
                item = respond(body, turn)
                terminal = {
                    "id": "resp_synthetic_" + turn,
                    "object": "response",
                    "created_at": 1,
                    "model": "ariadne/test",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "error": None,
                    "incomplete_details": None,
                    "status": "completed",
                    "output": [item],
                    "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
                }
                events = [
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                    {"type": "response.completed", "response": terminal},
                ]
                payload = "".join(
                    "data: " + json.dumps(event) + "\n\n" for event in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            except Exception as exc:
                errors.append(repr(exc))
                payload = json.dumps({"error": {"message": str(exc)}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderFixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def run(mode):
        trace.clear()
        trace.update(calls=[], source=b"", pages=0, complete=False)
        before = len(requests)
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures" / "codex_prepared_worker.py"),
                mode,
                f"http://127.0.0.1:{server.server_port}/v1",
            ],
            env=env,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert not errors, errors
        assert process.returncode == 0, process.stdout + process.stderr
        report = json.loads(process.stdout.splitlines()[-1])
        report["expected_steps"] = expected_history_steps(report, trace)
        sent = copy.deepcopy(requests[before:])
        assert len(report["expected_steps"]) == len(sent), (
            "every independently expected step must be sent"
        )
        assert sum(result["api_calls"] for result in report["results"]) == len(sent)
        assert len(report["selections"]) == len(sent), "every request must be observed"
        print(
            json.dumps(
                {"codex_transport": codex_transport_summary(mode, report, sent, trace)},
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


def assert_original_rows(report):
    assert report["archive_error"] is None
    assert report["original_rows_unchanged"] == 6
    assert report["original_rows_sha256"] == report["seed_sha256"]


def assert_protected_wire(wire, protected):
    assert len(protected) == 2
    for item in protected:
        assert wire.count(item) == 1, "protected native message missing, changed or duplicated"
    start = wire.index(protected[0])
    assert wire[start : start + 2] == protected, "protected native message order changed"


def history_messages(rows, *, loaded=False):
    """Project only transcript-bearing fields; ignore host row/usage metadata.

    Expected durable rows alone receive SessionDB's audited loaded-view strip.
    Observations are NEVER normalized. Native call dictionaries stay exact.
    """
    projected = []
    for row in rows:
        message = {"role": row["role"], "content": row.get("content")}
        if loaded and row["role"] in {"user", "assistant"} and isinstance(message["content"], str):
            message["content"] = message["content"].strip()
        for key in ("tool_calls", "tool_call_id", "codex_message_items"):
            value = row.get(key)
            if value:
                if key in {"tool_calls", "codex_message_items"} and isinstance(value, str):
                    value = json.loads(value)
                message[key] = copy.deepcopy(value)
        projected.append(message)
    return projected


def expected_native_input(rows):
    """Audited subset of custom Responses formatting, NOT the formatter under test.

    turn_request_assembly strips strings on the send copy. The Responses adapter
    replays ordinary message carriers, omits empty tool-owner text, and projects
    tool calls WITHOUT response_item_id. These fixtures have unique short call
    ids and no reasoning, structured content, cross-issuer or compaction items.
    """
    wire = []
    for row in rows:
        role, content = row["role"], row.get("content")
        if isinstance(content, str):
            content = content.strip()
        if role == "tool":
            wire.append(
                {
                    "type": "function_call_output",
                    "call_id": row["tool_call_id"],
                    "output": content or "",
                }
            )
            continue
        assert role in {"user", "assistant"}, "unsupported expected fixture role"
        carrier = row.get("codex_message_items")
        if isinstance(carrier, str):
            carrier = json.loads(carrier)
        if carrier:
            wire.extend(copy.deepcopy(carrier))
        elif role == "user" or content:
            wire.append({"type": "message", "role": role, "content": content})
        for call in row.get("tool_calls") or []:
            wire.append(
                {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["function"]["name"],
                    "arguments": json.dumps(
                        json.loads(call["function"]["arguments"]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    return wire


def assert_baseline(report, sent):
    assert_original_rows(report)
    protected_source = [
        {"role": "user", "content": "SYNTHETIC-PROTECTED-USER café\n\t trailing  "},
        {"role": "assistant", "content": "SYNTHETIC-PROTECTED-ANSWER\n\t trailing  "},
    ]
    assert [
        {"role": row["role"], "content": row["content"]} for row in report["durable"][4:6]
    ] == protected_source
    # Source-backed normalization: hermes_state_messages._loaded_view_content
    # strips loaded user/assistant strings; codex_responses_adapter's
    # _role_message_item keeps string content for this custom Responses route.
    protected = [
        {"type": "message", "role": row["role"], "content": row["content"].strip()}
        for row in protected_source
    ]
    assert report["selections"], "no baseline hook observations"
    assert len(report["selections"]) == len(sent)
    assert len(report["expected_steps"]) == len(sent), "incomplete independent baseline matrix"
    for entry, request, expected in zip(
        report["selections"], sent, report["expected_steps"], strict=True
    ):
        assert history_messages(entry["request"][1:]) == history_messages(expected, loaded=True), (
            "baseline hook history changed"
        )
        assert request["body"]["input"] == expected_native_input(expected), (
            "baseline complete history changed"
        )
        assert "exception" not in entry, entry
        assert entry["candidate"] is None
        assert entry["request_unchanged"] is True
        assert entry["selection"]["status"] == "declined", entry["selection"]
        assert entry["selection"]["reason"] == "prepared_unavailable", entry["selection"]
        assert_protected_wire(request["body"]["input"], protected)
        assert request["body"]["input"][4:6] == protected
    assert "SYNTHETIC-CODEX-ARCHIVED" in json.dumps(sent[0]["body"]["input"])


def pure_baseline_evidence():
    """Synthetic assertion inputs only: no worker, host imports, SDK or home."""
    rows = [
        {"role": "user", "content": "SYNTHETIC-CODEX-ARCHIVED"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "old second"},
        {"role": "assistant", "content": "old second answer"},
        {"role": "user", "content": "SYNTHETIC-PROTECTED-USER café\n\t trailing  "},
        {"role": "assistant", "content": "SYNTHETIC-PROTECTED-ANSWER\n\t trailing  "},
    ]
    report = {
        "archive_error": None,
        "original_rows_unchanged": 6,
        "original_rows_sha256": "synthetic-seed-digest",
        "seed_sha256": "synthetic-seed-digest",
        "durable": copy.deepcopy(rows),
        "expected_steps": [copy.deepcopy(rows)],
        "selections": [
            {
                "candidate": None,
                "request_unchanged": True,
                "selection": {"status": "declined", "reason": "prepared_unavailable"},
                "request": [{"role": "system", "content": "host policy"}]
                + [{"role": row["role"], "content": row["content"].strip()} for row in rows],
            }
        ],
    }
    sent = [
        {
            "body": {
                "instructions": "synthetic host policy",
                "tools": [{"name": "ariadne_find"}, {"name": "ariadne_zoom"}],
                "input": [
                    {"type": "message", "role": row["role"], "content": row["content"].strip()}
                    for row in rows
                ],
            }
        }
    ]
    return report, sent


def test_pure_baseline_accepts_noop_evidence():
    assert_baseline(*pure_baseline_evidence())


def test_pure_baseline_accepts_canonical_null_tool_calls():
    report, sent = pure_baseline_evidence()
    # Real SessionDB rows include SQL NULL here; restored request messages omit it.
    for row in report["durable"]:
        row["tool_calls"] = None
    for rows in report["expected_steps"]:
        for row in rows:
            row["tool_calls"] = None
    assert_baseline(report, sent)


def test_pure_baseline_accepts_complete_seed_incoming_and_replayed_tail():
    report, sent = pure_baseline_evidence()
    first = {"role": "user", "content": "SYNTHETIC-CODEX-TURN-first"}
    answer = {
        "role": "assistant",
        "content": "synthetic Codex first",
        "codex_message_items": [completed_message("first")],
    }
    second = {"role": "user", "content": "SYNTHETIC-CODEX-TURN-second"}
    source = {
        "checkpoints": [
            {"turn": "first", "durable": copy.deepcopy(report["durable"])},
            {"turn": "second", "durable": copy.deepcopy(report["durable"] + [first, answer])},
        ],
        "tool_results": [],
    }
    report["expected_steps"] = expected_history_steps(source, {"calls": []})
    initial_hook = copy.deepcopy(report["selections"][0])
    initial_wire = copy.deepcopy(sent[0])
    # Explicit source-shaped observations include native carrier, not plain text
    # reconstructed from the possibly broken request.
    report["selections"] = [copy.deepcopy(initial_hook), copy.deepcopy(initial_hook)]
    report["selections"][0]["request"] += [copy.deepcopy(first)]
    report["selections"][1]["request"] += copy.deepcopy([first, answer, second])
    sent = [copy.deepcopy(initial_wire), copy.deepcopy(initial_wire)]
    sent[0]["body"]["input"] += [{"type": "message", **first}]
    sent[1]["body"]["input"] += [
        {"type": "message", **first},
        completed_message("first"),
        {"type": "message", **second},
    ]
    assert_baseline(report, sent)


@pytest.mark.parametrize("damage", ["content", "replacement", "duplicate", "order", "missing_tail"])
@pytest.mark.parametrize("boundary", ["http", "hook", "all"])
def test_pure_baseline_rejects_complete_history_damage(damage, boundary):
    report, sent = pure_baseline_evidence()
    # Independent seeded source, not the HTTP input we are about to corrupt.
    report["expected_steps"] = [copy.deepcopy(report["durable"])]
    tail = {"role": "user", "content": "SYNTHETIC-CODEX-TURN-baseline"}
    report["expected_steps"][0].append(copy.deepcopy(tail))
    report["selections"][0]["request"].append(copy.deepcopy(tail))
    wire = sent[0]["body"]["input"]
    wire.append({"type": "message", **tail})
    targets = []
    if boundary in ("http", "all"):
        targets.append(wire)
    if boundary in ("hook", "all"):
        targets.append(report["selections"][0]["request"])
    for rows in targets:
        start = 1 if rows is report["selections"][0]["request"] else 0
        if damage == "content":
            rows[start + 1]["content"] = "CHANGED early assistant"
        elif damage == "replacement":
            rows[start + 1] = copy.deepcopy(rows[start])
        elif damage == "duplicate":
            rows.append(copy.deepcopy(rows[start + 1]))
        elif damage == "order":
            rows[start], rows[start + 1] = rows[start + 1], rows[start]
        else:
            rows.pop()
    with pytest.raises(AssertionError):
        assert_baseline(report, sent)


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate", []),
        ("request_unchanged", False),
        ("status", "selected"),
        ("reason", "unsupported_shape"),
        ("exception", "RuntimeError"),
        ("archive_error", "archive failed"),
        ("original_rows_unchanged", 5),
        ("original_rows_sha256", "changed-digest"),
    ],
)
def test_pure_baseline_rejects_bad_fallback(field, value):
    report, sent = pure_baseline_evidence()
    entry = report["selections"][0]
    if field in ("status", "reason"):
        entry["selection"][field] = value
    elif field in report:
        report[field] = value
    else:
        entry[field] = value
    with pytest.raises(AssertionError):
        assert_baseline(report, sent)


@pytest.mark.parametrize(
    "damage", ["omit_user", "omit_assistant", "omit_both", "reorder", "change", "duplicate"]
)
def test_pure_protected_wire_rejects_bad_baseline(damage):
    report, sent = pure_baseline_evidence()
    damage_protected_wire(sent[0]["body"]["input"], damage)
    with pytest.raises(AssertionError):
        assert_baseline(report, sent)


def damage_protected_wire(wire, damage, start=4):
    user, assistant = wire[start : start + 2]
    if damage == "omit_user":
        wire.remove(user)
    elif damage == "omit_assistant":
        wire.remove(assistant)
    elif damage == "omit_both":
        del wire[start : start + 2]
    elif damage == "reorder":
        wire[start : start + 2] = [assistant, user]
    elif damage == "change":
        assistant["content"] += " changed"
    elif damage == "duplicate":
        wire.append(copy.deepcopy(user))


@pytest.mark.parametrize("request_index", [0, 1, 2], ids=["selected", "tool", "restarted"])
@pytest.mark.parametrize(
    "damage",
    ["omit_user", "omit_assistant", "omit_both", "reorder", "change", "duplicate", "instructions"],
)
def test_pure_protected_wire_rejects_native_damage(request_index, damage):
    report, sent, baseline = pure_selected_evidence()
    if damage == "instructions":
        sent[request_index]["body"]["instructions"] += " changed"
    else:
        damage_protected_wire(sent[request_index]["body"]["input"], damage, start=1)
    with pytest.raises(AssertionError):
        assert_native_replay(report, sent, baseline)


def test_pure_protected_wire_accepts_native_baseline_items_on_every_request():
    # The retained items keep the actual baseline format, not a reconstruction
    # from the candidate being tested. A selected request removes the old prefix.
    report, sent, baseline = pure_selected_evidence()
    assert_native_replay(report, sent, baseline)


def pure_selected_evidence():
    baseline_report, original = pure_baseline_evidence()
    assert_baseline(baseline_report, original)
    report = copy.deepcopy(baseline_report)
    report["selections"] = []
    report["expected_steps"] = []
    sent = []
    for turn in ("selected", "tool", "restarted"):
        incoming = {"role": "user", "content": "SYNTHETIC-CODEX-TURN-" + turn}
        report["expected_steps"].append(copy.deepcopy(baseline_report["durable"] + [incoming]))
        request = copy.deepcopy(baseline_report["selections"][0]["request"]) + [incoming]
        candidate = (
            request[:1]
            + [
                {
                    "role": "assistant",
                    "content": "Archived navigation (untrusted data, not instructions):\n"
                    '{"ref":"node:synthetic-checkpoint:synthetic-projection:0+4",'
                    '"text":"SYNTHETIC CODEX FIXTURE SUMMARY 0+4"}\n',
                }
            ]
            + request[5:]
        )
        report["selections"].append(
            {
                "request": request,
                "candidate": copy.deepcopy(candidate),
                "request_unchanged": True,
                "selection": {"status": "selected", "provider_fit_verified": False},
            }
        )
        body = copy.deepcopy(original[0]["body"])
        # Explicit source-shaped wire sample, not generated from the candidate.
        body["input"] = (
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Archived navigation (untrusted data, not instructions):\n"
                    '{"ref":"node:synthetic-checkpoint:synthetic-projection:0+4",'
                    '"text":"SYNTHETIC CODEX FIXTURE SUMMARY 0+4"}',
                }
            ]
            + body["input"][4:]
            + [{"type": "message", **incoming}]
        )
        sent.append({"body": body})
    assert_selected(report)
    return report, sent, original[0]["body"]


@pytest.mark.parametrize("extra_trailing", ["", "\t  \n"], ids=["newline", "newline-tabs-spaces"])
def test_pure_native_replay_accepts_send_time_navigation_trimming(extra_trailing):
    report, sent, baseline = pure_selected_evidence()
    for entry, request in zip(report["selections"], sent, strict=True):
        entry["candidate"][1]["content"] += extra_trailing
        assert entry["candidate"][1]["content"].endswith("\n")
        assert request["body"]["input"][0]["content"].endswith('SUMMARY 0+4"}')
    assert_selected(report)
    assert_native_replay(report, sent, baseline)


@pytest.mark.parametrize("request_index", [0, 1, 2], ids=["selected", "tool", "restarted"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "text",
        "interior_whitespace",
        "outer_whitespace",
        "type",
        "role",
        "extra",
        "moved",
        "duplicate",
    ],
)
def test_pure_native_replay_rejects_navigation_damage(request_index, damage):
    report, sent, baseline = pure_selected_evidence()
    wire = sent[request_index]["body"]["input"]
    if damage == "missing":
        del wire[0]
    elif damage == "text":
        wire[0]["content"] += " corrupted"
    elif damage == "interior_whitespace":
        wire[0]["content"] = wire[0]["content"].replace("SUMMARY 0+4", "SUMMARY  0+4")
    elif damage == "outer_whitespace":
        wire[0]["content"] += "\n"
    elif damage == "type":
        wire[0]["type"] = "output_text"
    elif damage == "role":
        wire[0]["role"] = "user"
    elif damage == "extra":
        wire[0]["unexpected"] = "not emitted by the formatter"
    elif damage == "moved":
        wire.append(wire.pop(0))
    elif damage == "duplicate":
        wire.append(copy.deepcopy(wire[0]))
    # Hook evidence remains valid: only the captured ACTUAL HTTP body is damaged.
    assert_selected(report)
    with pytest.raises(AssertionError, match="prepared navigation"):
        assert_native_replay(report, sent, baseline)


def assert_selected(report):
    assert_original_rows(report)
    for entry in report["selections"]:
        assert entry["selection"]["status"] == "selected", entry["selection"]
        assert entry["selection"]["provider_fit_verified"] is False
        assert entry["request_unchanged"]
        request, candidate = entry["request"], entry["candidate"]
        assert candidate is not None
        assert candidate[:1] == request[:1], "host policy prefix changed"
        assert candidate[2:] == request[5:], "protected suffix/replay carrier changed"
        assert candidate[1]["content"].startswith("Archived navigation")
        assert "SYNTHETIC CODEX FIXTURE SUMMARY" in candidate[1]["content"]


def assert_native_replay(report, sent, baseline):
    # Use the observed no-op transport items, independently checked against the
    # seeded durable source by assert_baseline, never the candidate under test.
    protected = copy.deepcopy(baseline["input"][4:6])
    assert len(report["expected_steps"]) == len(sent), (
        "incomplete independent retained-history matrix"
    )
    for entry, request, expected in zip(
        report["selections"], sent, report["expected_steps"], strict=True
    ):
        assert history_messages(entry["request"][1:]) == history_messages(expected, loaded=True), (
            "complete hook history changed"
        )
        # Only the published four-row prefix may disappear; retained tool pairs
        # added after it must survive even if all observed boundaries agree to
        # omit the same pair. Native response-item ids are checked HERE, not on
        # HTTP, where the audited host intentionally omits them.
        assert history_messages(entry["candidate"][2:]) == history_messages(
            expected[4:], loaded=True
        ), "complete candidate history changed"
        wire = request["body"]["input"]
        # prepared.py emits a plain assistant summary without a replay carrier.
        # turn_request_assembly applies selection first (145-147), then strips
        # string content on the API copy only (191-193), keeping the hook candidate
        # and archived source intact. Model that trim only in expected navigation.
        # codex_responses_adapter._replay_message_items therefore yields nothing;
        # _chat_messages_to_responses_input falls back to _role_message_item:
        # exactly type/role/string content on this custom (not Codex backend) route.
        # ResponsesApiTransport.build_kwargs extracts the sole host system prefix
        # into instructions, so candidate[1] must become the FIRST native item.
        navigation = {
            "type": "message",
            "role": "assistant",
            "content": entry["candidate"][1]["content"].strip(),
        }
        assert wire[:1] == [navigation], "prepared navigation missing, changed or moved"
        assert wire.count(navigation) == 1, "prepared navigation duplicated"
        assert wire == [navigation] + expected_native_input(expected[4:]), (
            "complete HTTP retained history changed"
        )
        assert_protected_wire(wire, protected)
        assert request["body"]["instructions"] == baseline["instructions"]
        assert sorted(tool["name"] for tool in request["body"]["tools"]) == [
            "ariadne_find",
            "ariadne_zoom",
        ]
        assert "codex_message_items" not in json.dumps(request["body"])
        for message in entry["request"]:
            assert not message.get("codex_reasoning_items"), "ordinary carrier coverage only"
            for item in message.get("codex_message_items", []):
                assert item in wire, "id/status/phase/output_text lost in native formatter"
            for call in message.get("tool_calls", []):
                # The host intentionally emits call_id/name/arguments, NOT the
                # provider's function-call item id (adapter _replay_tool_call_items).
                assert {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                } in wire
            if message["role"] == "tool":
                assert {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                } in wire
        calls = [item["call_id"] for item in wire if item.get("type") == "function_call"]
        outputs = [item["call_id"] for item in wire if item.get("type") == "function_call_output"]
        assert calls == outputs, "native formatter lost a complete call/result pair"


def assert_durable_carrier(report, turn):
    expected = [completed_message(turn)]
    for key in ("durable", "replay"):
        owner = next(m for m in report[key] if m.get("content") == "synthetic Codex " + turn)
        carrier = owner["codex_message_items"]
        if key == "durable":
            # SessionDB.get_messages exposes this column as stored JSON text;
            # get_messages_as_conversation decodes the native replay carrier.
            assert isinstance(carrier, str)
            carrier = json.loads(carrier)
        assert carrier == expected


def test_completed_codex_carrier_selects_again_next_user_turn_and_restart(codex_worker):
    baseline, original, _ = codex_worker("baseline")
    assert_baseline(baseline, original)
    live, sent, _ = codex_worker("text")
    restarted, resumed, _ = codex_worker("restart")
    assert len({baseline["pid"], live["pid"], restarted["pid"]}) == 3
    assert len(original) == len(resumed) == 1 and len(sent) == 2
    assert "SYNTHETIC-CODEX-ARCHIVED" in json.dumps(original[0]["body"]["input"])
    assert "SYNTHETIC-CODEX-ARCHIVED" not in json.dumps(sent[0]["body"]["input"])
    assert sent[0]["body"]["instructions"] == original[0]["body"]["instructions"]
    second = live["selections"][1]
    owner = next(m for m in second["request"] if m.get("codex_message_items"))
    assert owner["codex_message_items"] == [completed_message("first")]
    for report, requests in ((live, sent), (restarted, resumed)):
        assert_native_replay(report, requests, original[0]["body"])
        assert_durable_carrier(report, "first")
        assert_durable_carrier(report, "second")
    for turn in ("first", "second"):
        assert completed_message(turn) in resumed[0]["body"]["input"]
    assert_selected(live)
    assert_selected(restarted)
    assert resumed[0]["body"]["input"][:-2] == sent[-1]["body"]["input"]


def test_pure_worker_tool_observer_preserves_return_and_snapshots_source():
    # Execute ONLY the observer definition, never the worker module or host SDK.
    path = Path(__file__).parent / "fixtures" / "codex_prepared_worker.py"
    tree = ast.parse(path.read_text())
    observer = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "observe_tool_result"
        ),
        None,
    )
    assert observer is not None, "worker needs an independent preassembly tool result observer"
    args, outputs = {"ref": "event:synthetic", "offset": 1024}, []
    calls = []

    def original(name, arguments, **kwargs):
        calls.append((name, copy.deepcopy(arguments), kwargs))
        return '{"ok":true,"synthetic":"EXACT café  "}'

    namespace = {
        "copy": copy,
        "original_tool": original,
        "tool_results": outputs,
        "current_turn": "tools",
    }
    exec(compile(ast.Module(body=[observer], type_ignores=[]), str(path), "exec"), namespace)
    result = namespace["observe_tool_result"]("ariadne_zoom", args, synthetic=True)
    assert result == '{"ok":true,"synthetic":"EXACT café  "}'
    assert calls == [("ariadne_zoom", args, {"synthetic": True})]
    assert outputs == [{"turn": "tools", "name": "ariadne_zoom", "args": args, "output": result}]
    args["offset"] = 9999
    assert outputs[0]["args"]["offset"] == 1024


def expected_history_steps(report, trace):
    """Independent matrix: BEFORE-run canonical checkpoints + server call constants.

    Tool output is observed at the engine handler return, before the host appends
    it or assembles the next request. Never use hook/conversation/candidate/HTTP
    observations to decide what should have survived.
    """
    calls = trace["calls"]
    results = report["tool_results"]
    assert len(calls) == len(results), "scripted call/result source mismatch"
    pairs = []
    for call, result in zip(calls, results, strict=True):
        assert result["turn"] == "tools"
        assert result["name"] == call["name"]
        assert result["args"] == json.loads(call["arguments"])
        pairs.append(
            [
                # _normalize_codex_response returns empty final_text for a call-only
                # response; build_assistant_message stores that empty string.
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call["call_id"],
                            "call_id": call["call_id"],
                            "response_item_id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call["call_id"], "content": result["output"]},
            ]
        )
    steps = []
    tool_turns = 0
    for checkpoint in report["checkpoints"]:
        turn = checkpoint["turn"]
        history = copy.deepcopy(checkpoint["durable"]) + [
            {
                "role": "user",
                "content": "SYNTHETIC-CODEX-TURN-" + turn,
            }
        ]
        steps.append(copy.deepcopy(history))
        if turn == "tools":
            tool_turns += 1
            for pair in pairs:
                history.extend(copy.deepcopy(pair))
                steps.append(copy.deepcopy(history))
    assert tool_turns == (1 if pairs else 0), "tool results outside scripted turn"
    return steps


def test_pure_expected_matrix_uses_preassembly_sources_not_observations():
    seed = pure_baseline_evidence()[0]["durable"]
    history, calls = pure_tool_history_evidence()
    source = {
        "checkpoints": [
            {"turn": "tools", "durable": copy.deepcopy(seed)},
            {"turn": "aftertools", "durable": copy.deepcopy(seed + history["durable"])},
        ],
        "tool_results": [
            {
                "turn": "tools",
                "name": call["name"],
                "args": json.loads(call["arguments"]),
                "output": f"source result {index}",
            }
            for index, call in enumerate(calls)
        ],
        # Poison every tempting circular source. The constructor must not read it.
        "selections": [{"request": [], "candidate": []}],
        "durable": [],
        "replay": [],
    }
    expected = expected_history_steps(source, {"calls": calls})
    assert len(expected) == 4
    assert expected[0] == seed + [{"role": "user", "content": "SYNTHETIC-CODEX-TURN-tools"}]
    assert expected[-1] == seed + history["durable"] + [
        {"role": "user", "content": "SYNTHETIC-CODEX-TURN-aftertools"}
    ]
    for index, call in enumerate(calls):
        owner, result = expected[index + 1][-2:]
        assert owner["tool_calls"] == history["durable"][index * 2]["tool_calls"]
        assert result == {
            "role": "tool",
            "tool_call_id": call["call_id"],
            "content": f"source result {index}",
        }
    assert expected[1][:-2] == expected[0]
    assert expected[2][:-2] == expected[1]
    expected[2][0]["content"] = "mutated derived expectation"
    assert source["checkpoints"][0]["durable"][0] == seed[0]
    assert expected[0][0] == seed[0]


def pure_retained_tool_evidence():
    """Expected post-prefix pairs come from a scripted source, not hook/HTTP."""
    report, sent, baseline = pure_selected_evidence()
    history, _ = pure_tool_history_evidence()
    pairs = copy.deepcopy(history["replay"])
    for index in range(2):
        pairs[index * 2]["content"] = None
        # Match the scripted server's already-canonical argument constants.
        function = pairs[index * 2]["tool_calls"][0]["function"]
        function["arguments"] = json.dumps(
            json.loads(function["arguments"]), sort_keys=True, separators=(",", ":")
        )
        pairs[index * 2 + 1]["content"] = "synthetic result " + str(index)
    report.update(copy.deepcopy(history))
    report["expected_steps"] = []
    for index, (entry, request, turn) in enumerate(
        zip(report["selections"], sent, ("selected", "tool", "restarted"), strict=True)
    ):
        retained = pairs[:2] if index == 0 else pairs
        incoming = {"role": "user", "content": "SYNTHETIC-CODEX-TURN-" + turn}
        # Seed source + explicit scripted calls/results, never the request being
        # checked. First step has one pair; subsequent/restarted steps have both.
        report["expected_steps"].append(
            copy.deepcopy(pure_baseline_evidence()[0]["durable"] + retained + [incoming])
        )
        for key in ("request", "candidate"):
            entry[key][-1:-1] = copy.deepcopy(retained)
        request["body"]["input"][-1:-1] = expected_native_input(retained)
    return report, sent, baseline


def test_pure_retained_tool_matrix_accepts_complete_inflight_and_restarted_history():
    report, sent, baseline = pure_retained_tool_evidence()
    assert_selected(report)
    assert_native_replay(report, sent, baseline)


@pytest.mark.parametrize("step", [1, 2], ids=["subsequent", "restarted"])
@pytest.mark.parametrize("boundary", ["hook", "candidate", "http", "all"])
@pytest.mark.parametrize("damage", ["omitted_pair", "duplicate_pair", "pair_order", "native_id"])
def test_pure_retained_tool_matrix_rejects_damage(step, boundary, damage):
    report, sent, baseline = pure_retained_tool_evidence()
    entry = report["selections"][step]
    for key in ("request", "candidate"):
        if boundary not in (key if key == "candidate" else "hook", "all"):
            continue
        rows = entry[key]
        start = next(i for i, row in enumerate(rows) if row.get("tool_calls"))
        if damage == "omitted_pair":
            del rows[start : start + 2]
        elif damage == "duplicate_pair":
            rows[start:start] = copy.deepcopy(rows[start : start + 2])
        elif damage == "pair_order":
            rows[start : start + 4] = rows[start + 2 : start + 4] + rows[start : start + 2]
        else:
            rows[start]["tool_calls"][0]["response_item_id"] = "fc_native_WRONG"
    if boundary in ("http", "all"):
        wire = sent[step]["body"]["input"]
        start = next(i for i, row in enumerate(wire) if row.get("type") == "function_call")
        if damage == "omitted_pair":
            del wire[start : start + 2]
        elif damage == "duplicate_pair":
            wire[start:start] = copy.deepcopy(wire[start : start + 2])
        elif damage == "pair_order":
            wire[start : start + 4] = wire[start + 2 : start + 4] + wire[start : start + 2]
        elif boundary == "http":
            # HTTP legitimately has NO response-item id. Alter its actual native
            # identity instead; hook/candidate native_id cases alter the fc id.
            wire[start]["call_id"] = wire[start + 1]["call_id"] = "call_WRONG"
    with pytest.raises(AssertionError):
        assert_native_replay(report, sent, baseline)


def pure_tool_history_evidence():
    calls, durable = [], []
    for index in range(2):
        call_id = f"call_synthetic_tools_{index}"
        item = {
            "type": "function_call",
            "id": f"fc_native_DIFFERENT_{index}",
            "call_id": call_id,
            "name": "ariadne_zoom",
            "arguments": '{ "ref": "event:synthetic", "offset": ' + str(index) + " }",
        }
        calls.append(item)
        durable.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "call_id": call_id,
                            "response_item_id": item["id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": item["arguments"]},
                        }
                    ],
                    "tool_call_uids": json.dumps({call_id: f"occurrence-{index}"}),
                },
                {"role": "tool", "tool_call_id": call_id, "tool_call_uid": f"occurrence-{index}"},
            ]
        )
    return {"durable": durable, "replay": copy.deepcopy(durable)}, calls


def replace_occurrence_uid(report, index, uid):
    owner, result = report["durable"][index * 2 : index * 2 + 2]
    owner["tool_call_uids"] = json.dumps({result["tool_call_id"]: uid})
    result["tool_call_uid"] = uid


def test_pure_native_uid_accepts_unique_stable_occurrences():
    live, calls = pure_tool_history_evidence()
    restarted = copy.deepcopy(live)
    assert assert_native_tool_history(live, calls) == assert_native_tool_history(restarted, calls)


@pytest.mark.parametrize("uid", [None, "", "   "])
def test_pure_native_uid_rejects_empty_occurrences(uid):
    report, calls = pure_tool_history_evidence()
    replace_occurrence_uid(report, 0, uid)
    with pytest.raises(AssertionError):
        assert_native_tool_history(report, calls)


def test_pure_native_uid_rejects_reused_occurrences():
    report, calls = pure_tool_history_evidence()
    replace_occurrence_uid(report, 1, "occurrence-0")
    with pytest.raises(AssertionError):
        assert_native_tool_history(report, calls)


def test_pure_native_uid_rejects_regenerated_restart_occurrences():
    live, calls = pure_tool_history_evidence()
    restarted = copy.deepcopy(live)
    replace_occurrence_uid(restarted, 0, "regenerated-after-restart")
    with pytest.raises(AssertionError):
        assert assert_native_tool_history(live, calls) == assert_native_tool_history(
            restarted, calls
        )


def assert_native_tool_history(report, calls):
    for key in ("durable", "replay"):
        owners = [m for m in report[key] if m.get("tool_calls")]
        assert len(owners) == len(calls)
        for owner, item in zip(owners, calls, strict=True):
            assert owner["tool_calls"] == [
                {
                    "id": item["call_id"],
                    "call_id": item["call_id"],
                    "response_item_id": item["id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
            ]
            assert item["id"] != "fc_" + item["call_id"].removeprefix("call_")
    # SessionDB must retain occurrence identity, not merely matching textual ids.
    occurrences, seen = [], set()
    for owner in (m for m in report["durable"] if m.get("tool_calls")):
        call = owner["tool_calls"][0]
        mapping = json.loads(owner["tool_call_uids"])
        uid = mapping[call["id"]]
        assert isinstance(uid, str) and uid.strip(), "missing durable tool occurrence UID"
        assert uid not in seen, "durable tool occurrence UID reused"
        seen.add(uid)
        results = [m for m in report["durable"] if m.get("tool_call_id") == call["id"]]
        assert len(results) == 1
        result_uid = results[0]["tool_call_uid"]
        assert uid == result_uid
        occurrences.append((call["id"], uid, result_uid))
    return tuple(occurrences)


def test_native_codex_tool_cycle_and_restart_keep_selection_and_exact_source(codex_worker):
    baseline, original, _ = codex_worker("baseline")
    assert_baseline(baseline, original)
    live, sent, trace = codex_worker("tools")
    restarted, resumed, _ = codex_worker("restarttools")
    assert len({baseline["pid"], live["pid"], restarted["pid"]}) == 3
    assert len(original) == len(resumed) == 1
    assert trace["complete"] and trace["pages"] > 1
    source = base64.b64decode(live["exact_source_base64"], validate=True)
    assert trace["source"] == source
    assert (
        trace["event_ref"]
        == live["exact_event_ref"]
        == "event:" + hashlib.sha256(source).hexdigest()
    )
    assert json.loads(source)["content"].endswith("  ")
    assert len(sent) == len(trace["calls"]) + 3
    assert trace["calls"][0]["name"] == "ariadne_find"
    assert {call["name"] for call in trace["calls"][1:]} == {"ariadne_zoom"}
    occurrence_uids = []
    for report, requests in ((live, sent), (restarted, resumed)):
        assert_native_replay(report, requests, original[0]["body"])
        occurrence_uids.append(assert_native_tool_history(report, trace["calls"]))
        for turn in ("first", "tools", "aftertools"):
            assert_durable_carrier(report, turn)
    assert occurrence_uids[0] == occurrence_uids[1], "restart regenerated existing call/result UIDs"
    # Every in-flight tool suffix, the complete subsequent exchange, and restart
    # must keep the original native response-item id, not derive fc_<call suffix>.
    for entry in live["selections"][2:] + restarted["selections"]:
        native = [call for m in entry["request"] for call in m.get("tool_calls", [])]
        assert native
        assert all(call["response_item_id"].startswith("fc_native_DIFFERENT_") for call in native)
    assert_selected(live)
    assert_selected(restarted)
    for request in sent + resumed:
        assert "SYNTHETIC-CODEX-ARCHIVED" not in json.dumps(
            [item for item in request["body"]["input"] if item.get("role") == "user"]
        )
