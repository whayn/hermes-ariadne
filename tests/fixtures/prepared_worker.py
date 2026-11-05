"""Synthetic prepared-prefix tracer through installed Hermes and its real SDK."""

import base64
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

# Bind the runner-selected real SDK and pure Ariadne package BEFORE Hermes's
# package-manager activation can replace sys.path. This also exercises installed
# wheels without depending on an editable source checkout surviving activation.
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
SESSION = "synthetic-prepared-transport"
CURRENT = "SYNTHETIC-CURRENT-TURN café\n\t protected whitespace  "
EXACT = "SYNTHETIC-ARCHIVED-EXACT café 😀\n\t" + "λ preserve  spacing\n" * 180 + "  "
seed_path = home / "seed.json"
db = SessionDB(db_path=home / "state.db")
if MODE.startswith("baseline"):
    assert not seed_path.exists()
    db.create_session(SESSION, source="cli", model="ariadne/test")
    rows = [
        {"role": "user", "content": EXACT},
        {"role": "assistant", "content": "SYNTHETIC-OLD-ANSWER " + "fixture " * 200},
        {"role": "user", "content": "SYNTHETIC-OLD-SECOND " + "source " * 200},
        {"role": "assistant", "content": "SYNTHETIC-OLD-SECOND-ANSWER " + "answer " * 200},
        {"role": "user", "content": "SYNTHETIC-PREVIOUS-USER café\n\t trailing  "},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "synthetic_previous_call",
                    "type": "function",
                    "function": {
                        "name": "ariadne_find",
                        "arguments": '{"query":"SYNTHETIC-NO-MATCH"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"ok":true,"results":[]}',
            "tool_name": "ariadne_find",
            "tool_call_id": "synthetic_previous_call",
        },
        {"role": "assistant", "content": "SYNTHETIC-PREVIOUS-ANSWER\n\t trailing  "},
    ]
    for index, row in enumerate(rows):
        row.update(timestamp=1700000000 + index, message_uid=f"synthetic-message-{index}")
    # The host batch writer persists call AND result occurrence identity. Lone
    # append_message calls cannot link their per-batch tool index across rows.
    assert db.append_messages_batch(SESSION, rows) == 8
    seed = db.get_messages(SESSION)
    assert (
        seed[6]["tool_call_uid"] == json.loads(seed[5]["tool_call_uids"])["synthetic_previous_call"]
    ), "fixture must persist complete tool occurrence identity"
    seed_path.write_bytes(canonical_json(seed))
    # A real SQLite snapshot gives the no-selection baseline identical durable history
    # without adding its synthetic response to the canonical DB used by the live tracer.
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
    api_mode="chat_completions",
    model="ariadne/test",
    max_iterations=24,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    skip_background_review=True,
    enabled_toolsets=[] if MODE == "disabled" else ["context_engine"],
    session_db=db,
    session_id=SESSION,
    max_tokens=256,
    prefill_messages=[{"role": "assistant", "content": "SYNTHETIC-UNMAPPED-PREFILL"}]
    if MODE in {"fallback", "baseline-fallback"}
    else None,
)
assert agent.context_compressor.name == "ariadne-prepared"
assert "plugins/ariadne-prepared" in str(
    sys.modules[type(agent.context_compressor).__module__].__file__
)
assert agent._context_engine_tool_names == (
    set() if MODE == "disabled" else {"ariadne_find", "ariadne_zoom"}
)
history = db.get_messages_as_conversation(SESSION, repair_alternation=True)
result = agent.run_conversation(
    CURRENT,
    conversation_history=history,
    system_message="SYNTHETIC HOST POLICY: preserve host instructions.",
)
assert not result["failed"], result
assert result["final_response"] == "synthetic fixture complete", result
status = agent.context_compressor.get_status()
assert db.get_messages(SESSION)[: len(seed)] == seed, "canonical original rows changed"
report = {
    "mode": MODE,
    "pid": os.getpid(),
    "api_calls": result["api_calls"],
    "selection": status["selection"],
    "archive_error": status["archive_error"],
    "tools": sorted(agent._context_engine_tool_names),
    "original_rows_unchanged": len(seed),
    "exact_source_base64": base64.b64encode(canonical_json(seed[0])).decode(),
    "exact_event_ref": "event:" + hashlib.sha256(canonical_json(seed[0])).hexdigest(),
    "operator_package_origin": hermes_ariadne.__file__,
    "openai_version": openai.__version__,
    "httpx_version": httpx.__version__,
}
agent.close()
db.close()
if MODE.startswith("baseline"):
    # Publish only after baseline, and BEFORE the next real AIAgent starts.
    store = EventStore(home / "ariadne" / "events.sqlite3")
    store.ingest(SESSION, seed[:4], origin="hermes_db")
    projection = Projection(
        store, SESSION, [hashlib.sha256(canonical_json(r)).hexdigest() for r in seed[:4]]
    )
    # A bounded chat view actually publishes internal navigation nodes; with
    # the default large high-water mark the four tiny fixture summaries stay
    # leaves, whose zoom correctly returns source rather than binary children.
    kernel = MemoryKernel(projection, home / "synthetic-kernel.json", node_bytes=128, chat_high=64)
    kernel.build(lambda req: f"SYNTHETIC FIXTURE SUMMARY {req.part.start}+{req.part.count}")
    first = kernel.prefix_view(4)[0]
    assert first["stop"] - first["start"] > 1
    manifest = publish_prepared(kernel, home=home)
    prepared = PreparedView.load(store, SESSION, home)
    assert prepared is not None
    report["manifest"] = str(manifest)
    report["projection_id"] = projection.projection_id
print(json.dumps(report, ensure_ascii=True), flush=True)
