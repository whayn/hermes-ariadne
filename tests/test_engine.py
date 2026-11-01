import copy
import json
from pathlib import Path

from agent.context_engine import ContextEngine

from hermes_ariadne.engine import AriadneEngine


def test_bound_store_reconciles_authoritative_rows_on_start_turn_and_end(tmp_path):
    from hermes_state import SessionDB

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("s", source="cli")
        db.create_session("unrelated", source="cli")
        db.append_message("s", role="user", content="  original café\n\t ")
        db.append_message("unrelated", role="user", content="never import this café")
        engine = AriadneEngine()
        engine.bind_session_state(session_db=db, session_id="s")
        engine.on_session_start("s", hermes_home=str(home))
        rows = db.get_messages("s", include_inactive=True)
        result = json.loads(engine.handle_tool_call("ariadne_search", {"query": "café"}))
        assert [r["message"] for r in result["results"]] == rows
        uid = rows[0]["message_uid"]
        # The live context can differ from what the user typed: durable wins.
        engine.on_turn_complete([{"message_uid": uid, "content": "injected context"}])
        assert (
            json.loads(engine.handle_tool_call("ariadne_search", {"query": "injected"}))["count"]
            == 0
        )
        db.append_message("s", role="assistant", content="late persisted reply")
        engine.on_session_end("s", [])
        result = json.loads(engine.handle_tool_call("ariadne_search", {"query": "late persisted"}))
        assert result["count"] == 1
        assert result["results"][0]["origin"] == "hermes_db"
    finally:
        db.close()


def test_clone_and_reset_do_not_share_agent_session_state(tmp_path):
    engine = AriadneEngine()
    engine.on_session_start("parent", hermes_home=str(tmp_path / "home"))
    engine.update_model("test/model", 128000)
    engine.update_from_response({"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130})
    clone = engine.clone_for_agent()
    assert clone._session_id is None
    assert clone._store is None
    assert clone.last_prompt_tokens == 0
    clone.update_model("other/model", 256000)
    assert engine.context_length == 128000
    engine.on_session_reset()
    assert engine._session_id is None
    assert engine.last_prompt_tokens == 0
    assert "error" in json.loads(engine.handle_tool_call("ariadne_search", {"query": "q"}))


def test_archive_failure_is_visible_but_does_not_break_turn(tmp_path, monkeypatch):
    engine = AriadneEngine()
    engine.on_session_start("s", hermes_home=str(tmp_path / "home"))

    def disk_full(*args, **kwargs):
        raise OSError("sensitive payload should not appear in status")

    monkeypatch.setattr(engine._store, "ingest", disk_full)
    engine.on_turn_complete([{"message_uid": "a", "content": "q"}])
    assert engine.get_status()["archive_error"] == "OSError"
    assert "error" in json.loads(engine.handle_tool_call("ariadne_search", {"query": "q"}))


def test_reset_only_host_path_can_rebind_without_session_start(tmp_path):
    engine = AriadneEngine()
    engine.on_session_start("old", hermes_home=str(tmp_path / "home"))
    engine.on_session_reset()
    # run_agent.reset_session_state() can use only this binding seam.
    engine.bind_session_state(session_db=None, session_id="new")
    message = {"message_uid": "new-id", "role": "user", "content": "new-only"}
    engine.on_turn_complete([message])
    result = json.loads(engine.handle_tool_call("ariadne_search", {"query": "new-only"}))
    assert result["count"] == 1
    assert result["session_id"] == "new"


def test_bound_reader_failure_never_archives_injected_hook_text_and_recovers(tmp_path):
    class Reader:
        unavailable = True

        def get_messages(self, session_id, include_inactive=False):
            assert session_id == "s" and include_inactive
            if self.unavailable:
                raise OSError("private reader error must not leak")
            return [{"message_uid": "uid", "role": "user", "content": "durable original"}]

    reader = Reader()
    engine = AriadneEngine()
    engine.bind_session_state(reader, "s")
    engine.on_session_start("s", hermes_home=str(tmp_path / "home"))
    live = [{"message_uid": "uid", "role": "user", "content": "injected hook text"}]
    engine.on_turn_complete(live)
    assert engine.get_status()["archive_error"] == "OSError"
    assert engine._store.search("injected", session_id="s") == []
    error = json.loads(engine.handle_tool_call("ariadne_search", {"query": "durable"}))
    assert error == {"error": "Ariadne archive is unavailable", "type": "OSError"}
    reader.unavailable = False
    engine.on_turn_complete(live)
    assert engine.get_status()["archive_error"] is None
    assert engine._store.search("injected", session_id="s") == []
    assert json.loads(engine.handle_tool_call("ariadne_search", {"query": "durable"}))["count"] == 1


def test_passive_engine_archives_and_exposes_exact_search(tmp_path):
    engine = AriadneEngine()
    assert isinstance(engine, ContextEngine)
    assert engine.name == "ariadne"
    assert not (tmp_path / "home").exists()  # import/discovery/constructor does no I/O
    engine.on_session_start("s", hermes_home=str(tmp_path / "home"))
    messages = [{"message_uid": "a" * 32, "role": "user", "content": "café\n  exact"}]
    before = copy.deepcopy(messages)
    engine.on_turn_complete(messages)
    result = json.loads(engine.handle_tool_call("ariadne_search", {"query": "café"}))
    assert result["results"][0]["message"] == before[0]
    assert result["count"] == 1
    assert engine.select_context(messages, conversation_messages=messages) is None
    assert engine.compress(messages, force=True) is messages
    assert not engine.should_compress(999999999)
    assert not engine.should_compress_preflight(messages)
    assert engine.prune_tool_results_only(messages) == (messages, 0)
    assert messages == before
    assert engine.compression_count == 0
    assert Path(tmp_path / "home" / "ariadne" / "events.sqlite3").exists()
    assert engine.get_tool_schemas()[0]["name"] == "ariadne_search"
