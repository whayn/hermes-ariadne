"""Worker: one real AIAgent turn in the installed-plugin home (loopback provider only)."""

import json
import os
import sys
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
from hermes_state import SessionDB  # noqa: E402 -- isolation checked by the parent
from run_agent import AIAgent  # noqa: E402

SESSION, BASE_URL, PROMPT = sys.argv[1:]
db = SessionDB(db_path=home / "state.db")
agent = AIAgent(
    api_key="offline-fixture",
    base_url=BASE_URL,
    provider="custom",
    api_mode="chat_completions",
    model="ariadne/test",
    max_iterations=6,
    quiet_mode=True,
    skip_context_files=True,
    skip_background_review=True,
    enabled_toolsets=["memory", "context_engine"],
    session_db=db,
    session_id=SESSION,
    max_tokens=256,
)
engine = agent.context_compressor
result = agent.run_conversation(PROMPT)
report = {
    "pid": os.getpid(),
    "engine": getattr(engine, "name", None),
    "engine_module": type(engine).__module__,
    "failed": bool(result.get("failed")),
    "final_response": result.get("final_response"),
    "api_calls": result.get("api_calls"),
}
agent.close()
db.close()
print(json.dumps(report))
