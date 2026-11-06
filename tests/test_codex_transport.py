"""Installed prepared engine → actual AIAgent/native Codex SDK/Responses SSE.

All source, summaries, auth and responses are synthetic; only loopback is allowed.
This proves native replay/selection, not provider fit, model recall or paid routing.
"""

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
        sent = copy.deepcopy(requests[before:])
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
    for entry, request in zip(report["selections"], sent, strict=True):
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
        "selections": [
            {
                "candidate": None,
                "request_unchanged": True,
                "selection": {"status": "declined", "reason": "prepared_unavailable"},
                "request": [{"role": "system", "content": "host policy"}] + rows,
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
    sent = []
    for turn in ("selected", "tool", "restarted"):
        incoming = {"role": "user", "content": "SYNTHETIC-CODEX-TURN-" + turn}
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
    for entry, request in zip(report["selections"], sent, strict=True):
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
