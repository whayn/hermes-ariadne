"""Focused host guarantees: no context replacement and explicit toolset gating."""

import copy
import json
import logging
from types import SimpleNamespace

import pytest

from hermes_ariadne.engine import AriadneEngine


@pytest.mark.parametrize("enabled", [[], ["context_engine"]])
def test_host_gates_ariadne_tool_without_changing_existing_tools(enabled):
    from agent.agent_init import _inject_context_engine_tools

    existing = {"type": "function", "function": {"name": "existing", "parameters": {}}}
    agent = SimpleNamespace(
        context_compressor=AriadneEngine(),
        tools=[copy.deepcopy(existing)],
        enabled_toolsets=enabled,
        valid_tool_names={"existing"},
        session_id="gating-test",
        platform="cli",
        model="ariadne/test",
    )
    _inject_context_engine_tools(agent)
    assert agent.tools[0] == existing
    expected = {"ariadne_search"} if enabled else set()
    assert agent._context_engine_tool_names == expected
    assert agent.valid_tool_names == {"existing"} | expected


def test_real_agent_reset_only_path_rebinds_archive_to_new_session(tmp_path):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    try:
        for sid in ("old", "new"):
            db.create_session(sid, source="cli")
            db.append_message(sid, role="user", content=f"{sid}-only-content")
        engine = AriadneEngine()
        engine.bind_session_state(db, "old")
        engine.on_session_start("old", hermes_home=str(home))
        agent = object.__new__(AIAgent)
        agent.context_compressor = engine
        agent._session_db = db
        agent.session_id = "new"
        agent.model = "ariadne/test"
        agent.platform = "cli"
        agent.reset_session_state()  # no previous_messages: only reset + binding hook
        assert engine._session_id == "new"
        new = engine.handle_tool_call("ariadne_search", {"query": "new-only-content"})
        old = engine.handle_tool_call("ariadne_search", {"query": "old-only-content"})
        assert json.loads(new)["count"] == 1
        assert json.loads(old)["count"] == 0
    finally:
        db.close()


def test_host_skips_inherited_selector_and_keeps_request_object_identical():
    from agent.conversation_loop import _apply_context_engine_selection

    engine = AriadneEngine()
    request = [
        {"role": "system", "content": "unchanged prefix"},
        {"role": "user", "content": [{"type": "text", "text": "nested"}]},
    ]
    original = copy.deepcopy(request)
    selected = _apply_context_engine_selection(
        SimpleNamespace(context_compressor=engine, session_id="no-replacement"),
        request,
        request[1:],
        request[-1],
        logger=logging.getLogger(__name__),
    )
    assert selected is request
    assert request == original
