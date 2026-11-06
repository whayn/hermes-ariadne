"""Isolated real AIAgent/SessionDB/Codex SDK worker, synthetic loopback only."""

import base64
import copy
import hashlib
import ipaddress
import json
import os
import socket
import sqlite3
import sys
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
assert home.name == "isolated-home"
assert Path(os.environ["HOME"]) == home.parent
assert not any("proxy" in key.lower() for key in os.environ)
_connect, _getaddrinfo = socket.socket.connect, socket.getaddrinfo


def loopback_getaddrinfo(host, *args, **kwargs):
    host = "127.0.0.1" if host == "localhost" else host
    assert ipaddress.ip_address(host).is_loopback, "non-loopback DNS forbidden"
    return _getaddrinfo(host, *args, **kwargs)


def loopback_connect(sock, address):
    assert isinstance(address, tuple) and ipaddress.ip_address(address[0]).is_loopback
    return _connect(sock, address)


socket.socket.connect = loopback_connect
socket.getaddrinfo = loopback_getaddrinfo

# Bind the runner-selected dependencies before Hermes package-manager activation.
# isort: off
import httpx  # noqa: E402
import openai  # noqa: E402

import hermes_ariadne  # noqa: E402
from hermes_ariadne.kernel import MemoryKernel  # noqa: E402
from hermes_ariadne.prepared import PreparedView, publish_prepared  # noqa: E402
from hermes_ariadne.sources import Projection  # noqa: E402
from hermes_ariadne.store import EventStore, canonical_json  # noqa: E402
from hermes_state import SessionDB  # noqa: E402
from run_agent import AIAgent  # noqa: E402
# isort: on

MODE, BASE_URL = sys.argv[1:]
SESSION = "synthetic-codex-prepared"
EXACT = "SYNTHETIC-CODEX-ARCHIVED café 😀\n\t" + "λ exact  spacing\n" * 180 + "  "
seed_path = home / "codex-seed.json"
db = SessionDB(db_path=home / "state.db")
if MODE == "baseline":
    assert not seed_path.exists()
    db.create_session(SESSION, source="cli", model="ariadne/test")
    rows = [
        {"role": "user", "content": EXACT},
        {"role": "assistant", "content": "SYNTHETIC-OLD-ANSWER " + "fixture " * 200},
        {"role": "user", "content": "SYNTHETIC-OLD-SECOND " + "source " * 200},
        {"role": "assistant", "content": "SYNTHETIC-OLD-SECOND-ANSWER " + "answer " * 200},
        {"role": "user", "content": "SYNTHETIC-PROTECTED-USER café\n\t trailing  "},
        {"role": "assistant", "content": "SYNTHETIC-PROTECTED-ANSWER\n\t trailing  "},
    ]
    for index, row in enumerate(rows):
        row.update(timestamp=1700000000 + index, message_uid=f"synthetic-codex-{index}")
    assert db.append_messages_batch(SESSION, rows) == len(rows)
    seed = db.get_messages(SESSION)
    seed_path.write_bytes(canonical_json(seed))
    # Keep the no-selection baseline from adding responses to the live source DB.
    with sqlite3.connect(home / "state.db") as source:
        with sqlite3.connect(home / "baseline.db") as target:
            source.backup(target)
    db.close()
    db = SessionDB(db_path=home / "baseline.db")
else:
    seed = json.loads(seed_path.read_bytes())

agent = AIAgent(
    api_key="offline-synthetic-placeholder",
    base_url=BASE_URL,
    provider="custom",
    api_mode="codex_responses",
    model="ariadne/test",
    max_iterations=24,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    skip_background_review=True,
    enabled_toolsets=["context_engine"],
    session_db=db,
    session_id=SESSION,
    max_tokens=256,
)
assert agent.api_mode == "codex_responses"
assert isinstance(agent.client, openai.OpenAI), type(agent.client)
assert isinstance(agent.client._client, httpx.Client), type(agent.client._client)
engine = agent.context_compressor
assert engine.name == "ariadne-prepared"
engine_origin = str(sys.modules[type(engine).__module__].__file__)
assert "plugins/ariadne-prepared" in engine_origin
assert agent._context_engine_tool_names == {"ariadne_find", "ariadne_zoom"}

selections = []
original_select = engine.select_context


def observe_selection(request, **kwargs):
    """Observe EVERY real hook call without changing its arguments or result."""
    entry = {
        "turn": current_turn,
        "request": copy.deepcopy(request),
        "conversation": copy.deepcopy(kwargs.get("conversation_messages")),
        "incoming": copy.deepcopy(kwargs.get("incoming_message")),
    }
    selections.append(entry)
    try:
        candidate = original_select(request, **kwargs)
    except Exception as exc:
        entry["exception"] = type(exc).__name__
        raise
    entry.update(
        candidate=copy.deepcopy(candidate),
        request_unchanged=request == entry["request"],
        selection=copy.deepcopy(engine.get_status()["selection"]),
    )
    return candidate


engine.select_context = observe_selection
turns = {
    "text": ["first", "second"],
    "tools": ["first", "tools", "aftertools"],
}.get(MODE, [MODE])
results = []
for current_turn in turns:
    history = db.get_messages_as_conversation(SESSION, repair_alternation=True)
    result = agent.run_conversation(
        "SYNTHETIC-CODEX-TURN-" + current_turn,
        conversation_history=history,
        system_message="SYNTHETIC HOST POLICY: preserve host instructions.",
    )
    assert not result["failed"], result
    assert result["final_response"] == "synthetic Codex " + current_turn, result
    results.append({"turn": current_turn, "api_calls": result["api_calls"]})

original_rows = db.get_messages(SESSION)[: len(seed)]
assert original_rows == seed, "canonical original rows changed"
report = {
    "mode": MODE,
    "pid": os.getpid(),
    "results": results,
    "selections": selections,
    "archive_error": engine.get_status()["archive_error"],
    "original_rows_unchanged": len(seed),
    "original_rows_sha256": hashlib.sha256(canonical_json(original_rows)).hexdigest(),
    "seed_sha256": hashlib.sha256(canonical_json(seed)).hexdigest(),
    "durable": db.get_messages(SESSION),
    "replay": db.get_messages_as_conversation(SESSION, repair_alternation=True),
    "exact_source_base64": base64.b64encode(canonical_json(seed[0])).decode(),
    "exact_event_ref": "event:" + hashlib.sha256(canonical_json(seed[0])).hexdigest(),
    "operator_package_origin": hermes_ariadne.__file__,
    "engine_origin": engine_origin,
    "openai_version": openai.__version__,
    "httpx_version": httpx.__version__,
}
agent.close()
db.close()
if MODE == "baseline":
    store = EventStore(home / "ariadne" / "events.sqlite3")
    store.ingest(SESSION, seed[:4], origin="hermes_db")
    projection = Projection(
        store, SESSION, [hashlib.sha256(canonical_json(row)).hexdigest() for row in seed[:4]]
    )
    kernel = MemoryKernel(projection, home / "codex-kernel.json", node_bytes=128, chat_high=64)
    kernel.build(lambda req: f"SYNTHETIC CODEX FIXTURE SUMMARY {req.part.start}+{req.part.count}")
    publish_prepared(kernel, home=home)
    assert PreparedView.load(store, SESSION, home) is not None
print(json.dumps(report, ensure_ascii=True), flush=True)
