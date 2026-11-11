"""Real Hermes worker for the opt-in native tracer (tests/test_rolling_transport.py).

Runs under a disposable HOME.

No facade replacement, manual completion callback, publication, seeded history or
provider SDK patch. The small wrappers only observe calls made by the native host.
All persisted message content in this fixture is synthetic and reasoning-free.
"""

import copy
import hashlib
import ipaddress
import json
import os
import socket
import sqlite3
import sys
from importlib.metadata import version
from pathlib import Path

assert os.environ.get("HERMES_ARIADNE_ROLLING_NATIVE") == "1", (
    "HERMES_ARIADNE_ROLLING_NATIVE=1 required"
)
home = Path(os.environ["HERMES_HOME"])
managed = Path(os.environ["HERMES_MANAGED_DIR"])
assert home.name == "isolated-home"
assert Path(os.environ["HOME"]) == home.parent
assert managed.is_dir() and not list(managed.iterdir())
assert not any("proxy" in key.lower() for key in os.environ)
assert not any(
    key in os.environ
    for key in (
        "HERMES_CONFIG",
        "HERMES_ENV",
        "HERMES_CONFIG_PATH",
        "HERMES_ENV_PATH",
        "HERMES_PROFILE",
        "HERMES_ACTIVE_PROFILE",
        "HERMES_PROFILE_NAME",
        "HERMES_CONFIG_BACKEND",
        "HERMES_REAL_HOME",
    )
)
assert not any(
    key.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS"))
    for key in os.environ
)


# Host code may catch the guard's AssertionError; this record cannot be swallowed.
credential_violations = []


def credential_read_guard(event, args):
    """Reject existing shared configuration/credentials before host bootstrap."""
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    requested = Path(os.fsdecode(args[0]))
    path = requested.resolve()
    protected = {"auth.json", ".env", ".op.env", "config.yaml", "credentials.json"}
    if requested.name in protected or path.name in protected:
        allowed = (
            not path.exists()
            or path.is_relative_to(home.parent.resolve())
            or path.is_relative_to(managed.resolve())
        )
        if not allowed:
            credential_violations.append(path.name)  # protected basename only, never a path
        assert allowed, "shared credential/config read forbidden"


sys.addaudithook(credential_read_guard)
_connect, _connect_ex, _getaddrinfo = (
    socket.socket.connect,
    socket.socket.connect_ex,
    socket.getaddrinfo,
)


def loopback_getaddrinfo(host, *args, **kwargs):
    host = "127.0.0.1" if host == "localhost" else host
    assert ipaddress.ip_address(host).is_loopback, "non-loopback DNS forbidden"
    return _getaddrinfo(host, *args, **kwargs)


def _loopback_address(address):
    assert isinstance(address, tuple) and ipaddress.ip_address(address[0]).is_loopback, (
        "non-loopback connect forbidden"
    )


def loopback_connect(sock, address):
    _loopback_address(address)
    return _connect(sock, address)


def loopback_connect_ex(sock, address):
    _loopback_address(address)
    return _connect_ex(sock, address)


socket.socket.connect = loopback_connect
socket.socket.connect_ex = loopback_connect_ex
socket.getaddrinfo = loopback_getaddrinfo

# Prebind runner-selected packages BEFORE native package-manager activation.
# isort: off
import httpx  # noqa: E402
import openai  # noqa: E402

import hermes_ariadne  # noqa: E402
from hermes_ariadne.install import install  # noqa: E402
from hermes_ariadne.store import canonical_json  # noqa: E402
# isort: on

assert Path(hermes_ariadne.__file__).resolve() == Path(os.environ["ARIADNE_EXPECT_ORIGIN"])
assert openai.__version__ == version("openai")
assert httpx.__version__ == version("httpx")
MODE, BASE_URL, PLAN_PATH = sys.argv[1:]
plan = json.loads(Path(PLAN_PATH).read_bytes())
SESSION = plan["session"]
BASELINE = MODE.startswith("baseline-")
assert MODE in {"advance", "restart", "baseline-C", "baseline-D", "baseline-E"}
# Fresh homes only for first process and independent no-view baselines. Restart
# uses the same previously installed bytes, not a fresh installation/publication.
if MODE != "restart":
    install(home, rolling=True)
plugin = home / "plugins" / "ariadne-rolling-plugin"
assert (plugin / "plugin.yaml").is_file()
assert not (home / "plugins" / "ariadne-rolling").exists(), "directory-loader name collision"

# isort: off
import run_agent  # noqa: E402
import hermes_state  # noqa: E402
from agent.plugin_llm import PluginLlm  # noqa: E402
from hermes_cli.plugins import get_plugin_context_engine  # noqa: E402
from hermes_state import SessionDB  # noqa: E402
from run_agent import AIAgent  # noqa: E402
# isort: on

host_source = Path(os.environ["ARIADNE_HOST_SOURCE"])
assert Path(run_agent.__file__).resolve() == host_source / "run_agent.py"
assert Path(hermes_state.__file__).resolve() == host_source / "hermes_state.py"
db = SessionDB(db_path=home / "state.db")
if MODE == "advance":
    assert db.get_messages(SESSION) == []
    db.create_session(SESSION, source="cli", model="ariadne/test")
else:
    assert db.get_messages(SESSION), "restart/baseline requires canonical prior history"

agent = AIAgent(
    api_key="offline-synthetic-placeholder",
    base_url=BASE_URL,
    provider="custom",
    api_mode="chat_completions",
    model="ariadne/test",
    credential_pool=None,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    skip_background_review=True,
    enabled_toolsets=["context_engine"],
    session_db=db,
    session_id=SESSION,
    max_tokens=256,
    max_iterations=32,
)
assert agent._credential_pool is None
assert isinstance(agent.client, openai.OpenAI)
assert isinstance(agent.client._client, httpx.Client)
assert agent.base_url.rstrip("/") == BASE_URL
engine = agent.context_compressor
assert engine.name == "ariadne-rolling"
engine_origin = Path(sys.modules[type(engine).__module__].__file__).resolve()
assert engine_origin.parent == plugin.resolve()
manager_engine = get_plugin_context_engine()
assert manager_engine is not engine, "manager must clone the engine per agent"
assert manager_engine._llm is engine._llm
assert isinstance(engine._llm, PluginLlm)
assert engine._llm._sync_caller is None and engine._llm._async_caller is None
assert agent._context_engine_tool_names == {"ariadne_find", "ariadne_zoom", "ariadne_rolling_page"}
expected_settings = plan["baseline_settings"] if BASELINE else plan["settings"]
assert {key: getattr(engine._settings, key) for key in expected_settings} == expected_settings


def checkpoint(rows):
    return {
        "event_ids": [hashlib.sha256(canonical_json(row)).hexdigest() for row in rows],
        "message_uids": [row["message_uid"] for row in rows],
    }


def durable_view():
    directory = home / "ariadne" / "rolling" / hashlib.sha256(SESSION.encode()).hexdigest()
    manifest = directory / "manifest.json"
    if not manifest.exists():
        return None
    published = json.loads(manifest.read_bytes())
    generation = published["generation_id"]
    content = (directory / "generations" / (generation + ".json")).read_bytes()
    assert hashlib.sha256(content).hexdigest() == generation
    view = json.loads(content)
    assert view["session_id"] == SESSION
    return {**view, "generation_id": generation}


def backup_before(turn):
    with sqlite3.connect(home / "state.db") as source:
        with sqlite3.connect(home / f"before-{turn}.db") as target:
            source.backup(target)


receipt = {
    "initial": engine.get_status(),
    "initial_rows": db.get_messages(SESSION),
    "initial_view": durable_view(),
    "turns": {},
    "views": {},
    "checkpoints": {},
    "tool_returns": {},
    "callbacks": {},
    "selections": {},
}
if MODE == "advance":
    assert receipt["initial_view"] is None, "automatic path must start without prepared artifacts"
    assert receipt["initial"]["rolling"]["logical_reservations"] == 0
elif MODE == "restart":
    # The parent derives these from the advance process's durable G3, written to
    # the plan only after that process exited; the fresh process must load them.
    expected = plan["restart_expected"]
    assert receipt["initial_view"]["generation_id"] == expected["generation_id"]
    assert receipt["initial_view"]["frontier_count"] == expected["frontier_count"]
    assert receipt["initial"]["rolling"]["logical_reservations"] == expected["logical_reservations"]
    assert receipt["initial"]["rolling"]["published_generation"] == expected["generation_id"]
else:
    assert receipt["initial_view"] is None

active_turn = None
native_select = engine.select_context
native_tool = engine.handle_tool_call
native_complete = engine.on_turn_complete


def observe_select(request_messages, **kwargs):
    before = copy.deepcopy((request_messages, kwargs))
    result = native_select(request_messages, **kwargs)
    assert (request_messages, kwargs) == before, "selection mutated host input"
    receipt["selections"][active_turn].append(
        {"selected": result is not None, "status": engine.get_status()["selection"]}
    )
    return result


def observe_tool(name, args, **kwargs):
    result = native_tool(name, args, **kwargs)
    receipt["tool_returns"][active_turn].append(
        {"name": name, "args": copy.deepcopy(args), "result": result}
    )
    return result


def observe_complete(messages, usage=None, **kwargs):
    # This wrapper NEVER calls the hook independently; normal native finalization
    # alone reaches it. Persist only allowlisted completion-gate observations.
    receipt["callbacks"][active_turn].append(
        {
            "usage_present": usage is not None,
            "interrupted": kwargs.get("interrupted"),
            "failed": kwargs.get("failed"),
            "turn_id_present": bool(kwargs.get("turn_id")),
            "turn_exit_reason": kwargs.get("turn_exit_reason"),
        }
    )
    return native_complete(messages, usage=usage, **kwargs)


engine.select_context = observe_select
engine.handle_tool_call = observe_tool
engine.on_turn_complete = observe_complete
turns = ["A", "B", "C", "D"] if MODE == "advance" else [MODE[-1] if BASELINE else "E"]
previous = copy.deepcopy(receipt["initial_rows"])
for active_turn in turns:
    if not BASELINE and active_turn in {"C", "D", "E"}:
        backup_before(active_turn)
    for field in ("tool_returns", "callbacks", "selections"):
        receipt[field][active_turn] = []
    history = db.get_messages_as_conversation(SESSION, repair_alternation=True)
    result = agent.run_conversation(
        plan["users"][active_turn], conversation_history=history, system_message=plan["policy"]
    )
    assert not result["failed"]
    assert result["final_response"] == plan["answers"][active_turn]
    rows = db.get_messages(SESSION)
    assert rows[: len(previous)] == previous, "canonical original rows changed"
    assert len(rows) > len(previous)
    assert len({row["message_uid"] for row in rows}) == len(rows)
    # Verify the independently persisted occurrence identities, including SQL
    # JSON TEXT ↔ restored dict. The HTTP formatter intentionally drops them.
    restored = db.get_messages_as_conversation(SESSION, repair_alternation=True)
    owners = {}
    restored_by_uid = {row["message_uid"]: row for row in restored}
    for row in rows:
        restored_row = restored_by_uid[row["message_uid"]]
        if row.get("tool_calls"):
            uids = json.loads(row["tool_call_uids"])
            assert uids and restored_row["_tool_call_uids"] == uids
            for call_id, uid in uids.items():
                assert uid and uid not in owners.values()
                owners[call_id] = uid
        if row["role"] == "tool":
            assert row["tool_call_uid"] == owners[row["tool_call_id"]]
            assert restored_row["_tool_call_uid"] == row["tool_call_uid"]
    # Read exact archive events; never rewrite canonical rows to make selection fit.
    if not BASELINE:
        for row in rows:
            event = hashlib.sha256(canonical_json(row)).hexdigest()
            assert engine._store.get_event(event, session_id=SESSION) == row
    receipt["turns"][active_turn] = engine.get_status()
    receipt["views"][active_turn] = durable_view()
    receipt["checkpoints"][active_turn] = checkpoint(rows)
    callbacks = receipt["callbacks"][active_turn]
    assert len(callbacks) == 1 and callbacks[0]["usage_present"]
    assert callbacks[0]["interrupted"] is False and callbacks[0]["failed"] is False
    assert callbacks[0]["turn_id_present"]
    assert callbacks[0]["turn_exit_reason"] == "text_response(finish_reason=stop)"
    selections = receipt["selections"][active_turn]
    assert selections, "native selection hook not reached"
    assert all(
        item["selected"] == (not BASELINE and active_turn in {"C", "D", "E"}) for item in selections
    )
    previous = copy.deepcopy(rows)
receipt["rows"] = previous
receipt["credential_violations"] = len(credential_violations)
# Full synthetic proof stays in the disposable fixture directory. No provider
# body, original text, ciphertext or readable reasoning is printed in receipts.
(home / f"receipt-{MODE}.json").write_bytes(canonical_json(receipt))
agent.close()
db.close()
# Host code may have swallowed the guard's AssertionError; the record cannot be.
assert not credential_violations, f"credential guard violations: {len(credential_violations)}"
origins = {
    name: str(Path(module.__file__).resolve())
    for name, module in (
        ("openai", openai),
        ("httpx", httpx),
        ("run_agent", run_agent),
        ("hermes_state", hermes_state),
    )
}
report = {
    "mode": MODE,
    "pid": os.getpid(),
    "operator_package_origin": str(Path(hermes_ariadne.__file__).resolve()),
    "engine_origin": str(engine_origin),
    "openai": openai.__version__,
    "httpx": httpx.__version__,
    "origins": origins,
    "native_source_sha256": {
        name: hashlib.sha256((host_source / name).read_bytes()).hexdigest()
        for name in (
            "agent/conversation_loop.py",
            "agent/turn_final_response.py",
            "agent/plugin_llm.py",
            "agent/auxiliary_client.py",
        )
    },
    "canonical_rows": len(previous),
    "credential_violations": len(credential_violations),
}
print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)
