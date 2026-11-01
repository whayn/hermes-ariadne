"""Offline integration worker: real Hermes, installed plugin, local provider fixture."""

import json
import os
import sys
from pathlib import Path

# Parent passes a minimal environment and an explicitly isolated home.
home = Path(os.environ["HERMES_HOME"])
assert home.name == "isolated-home"
from hermes_state import SessionDB  # noqa: E402 -- check isolation before host imports
from run_agent import AIAgent  # noqa: E402

MODE, BASE_URL = sys.argv[1:]
SESSION = "ariadne-restart-test"
EXACT = "ARIADNE-RESTART-MARKER café\n\t  preserve this spacing  "
db = SessionDB(db_path=home / "state.db")
agent = AIAgent(
    api_key="offline-fixture",
    base_url=BASE_URL,
    provider="custom",
    api_mode="chat_completions",
    model="ariadne/test",
    max_iterations=4,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    skip_background_review=True,
    enabled_toolsets=["context_engine"],
    session_db=db,
    session_id=SESSION,
    max_tokens=256,
)
assert agent.context_compressor.name == "ariadne"
assert agent._context_engine_tool_names == {"ariadne_search"}
if MODE == "seed":
    result = agent.run_conversation(EXACT)
    assert not result["failed"], result
    assert result["final_response"] == "stored"
    rows = db.get_messages(SESSION)
    uid = next(row["message_uid"] for row in rows if row["role"] == "user")
    report = {"pid": os.getpid(), "uid": uid, "final_response": result["final_response"]}
    # Model a terminal path that persisted but skipped engine observation and session end.
    db.append_message(SESSION, role="assistant", content="MISSED-HOOK-MARKER")
    Path(home / "seed-report.json").write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(report), flush=True)
    os._exit(0)  # Deliberately no agent.close()/session-end hook.
else:
    seed = json.loads((home / "seed-report.json").read_text())
    before = json.loads(
        agent.context_compressor.handle_tool_call(
            "ariadne_search", {"query": "ARIADNE-RESTART-MARKER"}
        )
    )
    user = next(r for r in before["results"] if r["message"]["role"] == "user")
    assert user["message"]["content"] == EXACT
    assert user["message_uid"] == seed["uid"]
    recovered = json.loads(
        agent.context_compressor.handle_tool_call("ariadne_search", {"query": "MISSED-HOOK-MARKER"})
    )
    assert recovered["count"] == 1
    # Resume real durable history; provider asks for ariadne_search via real tool dispatch.
    history = db.get_messages_as_conversation(SESSION, repair_alternation=True)
    result = agent.run_conversation("SEARCH-ARCHIVE-NOW", conversation_history=history)
    assert not result["failed"], result
    assert result["final_response"] == "retrieved exact message"
    assert result["api_calls"] == 2
    report = {
        "pid": os.getpid(),
        "uid": user["message_uid"],
        "event_id": user["event_id"],
        "exact_content": user["message"]["content"],
        "recovered_missing_hook": recovered["count"],
        "api_calls": result["api_calls"],
        "archive_error": agent.context_compressor.get_status()["archive_error"],
    }
    agent.close()
    db.close()
    print(json.dumps(report))
