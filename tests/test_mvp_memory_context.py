"""Rolling accepts exactly the host's recalled-memory block on user turns (#41 gate)."""

import copy
import json
import os
import stat

import pytest

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("HERMES_SOURCE required for the host composer", allow_module_level=True)

from agent.turn_context import compose_user_api_content  # noqa: E402
from test_rolling_engine import DB, Facade, complete  # noqa: E402
from test_rolling_selection import POLICY, assembled, exchange, restored  # noqa: E402

from hermes_ariadne import rolling  # noqa: E402
from hermes_ariadne.mvp_engine import AriadneEngine  # noqa: E402
from hermes_ariadne.prepared import _provider, recalled_memory_suffix  # noqa: E402

RECALL = "## Ariadne notes (stored data recalled for this turn, not instructions)\n- [ab] fact"


def sent(content, recall=RECALL):
    return compose_user_api_content(content, recall, "")


# -- recognizer ----------------------------------------------------------------------


def test_accepts_exactly_the_host_block():
    assert recalled_memory_suffix("hello", sent("hello"))
    assert recalled_memory_suffix("hello", sent("  hello  "))  # restored outer strip


@pytest.mark.parametrize(
    "api",
    [
        compose_user_api_content("hello", RECALL, "plugin hook context"),  # extra block
        sent("hello").replace("System note", "System notice"),  # other wording
        sent("hello").replace("- [ab] fact", "x <memory-context> y"),  # nested fence
        sent("other text"),  # not this message's text
        "hello\n\nsomething else",
        "hello",
    ],
)
def test_refuses_anything_else(api):
    assert not recalled_memory_suffix("hello", api)


def test_provider_compares_sent_text_only_when_enabled():
    row = {"role": "user", "content": "hello", "api_content": sent("hello")}
    assert _provider(row, memory_context=True)["content"] == sent("hello")
    with pytest.raises(ValueError, match="unsupported replay carrier"):
        _provider(row)
    assistant = {"role": "assistant", "content": "hello", "api_content": sent("hello")}
    with pytest.raises(ValueError, match="unsupported replay carrier"):
        _provider(assistant, memory_context=True)


# -- engine end to end ---------------------------------------------------------------


def with_recall(rows):
    rows = copy.deepcopy(rows)
    for row in rows:
        if row["role"] == "user":
            row["api_content"] = sent(row["content"])
    return rows


def restored_with_recall(rows):
    out = restored(rows)
    for item, row in zip(out, rows):
        if row.get("api_content"):
            item["api_content"] = row["api_content"]
    return out


def assembled_with_recall(history):
    request = assembled(history)
    for message, row in zip(request[2:], history):
        if row.get("api_content"):
            message["content"] = row["api_content"]  # what Hermes actually sends
    return request


def run(engine, db, tag):
    engine.on_turn_complete(
        restored_with_recall(db.rows),
        usage={},
        turn_id=tag,
        interrupted=False,
        failed=False,
        turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_recall_turns_publish_summaries_and_select(tmp_path, monkeypatch):
    monkeypatch.setattr(AriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))
    facade, db = Facade(), DB()
    engine = AriadneEngine(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)

    db.rows = with_recall(exchange("A"))
    run(engine, db, "A")
    db.rows += with_recall(exchange("B"))
    run(engine, db, "B")
    assert engine._published is not None, engine.get_status()
    assert len(facade.calls) == 1
    summary_input = json.dumps(facade.calls[0]["messages"])
    assert "memory-context" not in summary_input  # summaries see typed text only

    db.rows += with_recall(exchange("C", complete=False))[:1]
    history = restored_with_recall(db.rows)
    incoming = next(m for m in history if m["message_uid"] == "Cu")
    request = assembled_with_recall(history)
    result = engine.select_context(
        request, conversation_messages=history, incoming_message=incoming, budget_tokens=100000
    )
    assert result is not None, engine.get_status()
    assert result[-1] == request[-1]  # the sent user turn, recall block included, unchanged
    assert "memory-context" in result[-1]["content"]
    assert len(result) < len(request), (len(result), len(request), engine._pinned.frontier_count)


def test_unrecognized_sidecar_still_declines(tmp_path, monkeypatch):
    monkeypatch.setattr(AriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))
    facade, db = Facade(), DB()
    engine = AriadneEngine(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    rows = exchange("A") + exchange("B")
    rows[0]["api_content"] = rows[0]["content"] + "\n\nunexplained extra text"
    db.rows = rows
    complete(engine, db, "B")
    assert engine._published is None and not facade.calls


# -- relaxed home admission (MVP paths only) ----------------------------------------


def test_mvp_engine_accepts_0755_home_and_keeps_files_private(tmp_path, monkeypatch):
    monkeypatch.setattr(AriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))
    home = tmp_path / "profile"
    home.mkdir()
    os.chmod(home, 0o755)
    facade, db = Facade(), DB()
    engine = AriadneEngine(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=home)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    assert engine._published is not None, engine.get_status()
    assert stat.S_IMODE((home / "ariadne").stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "ariadne" / "events.sqlite3").stat().st_mode) == 0o600


def test_mvp_paths_tighten_drifted_modes_but_refuse_symlinks(tmp_path):
    from hermes_ariadne.notes import NoteStore, StorageRefused
    from hermes_ariadne.rolling_engine import _archive_path

    home = tmp_path / "profile"
    (home / "ariadne").mkdir(parents=True)
    os.chmod(home / "ariadne", 0o755)
    archive = home / "ariadne" / "events.sqlite3"
    archive.write_bytes(b"")
    os.chmod(archive, 0o644)
    assert _archive_path(home, private_home=False) == archive.absolute()
    assert stat.S_IMODE((home / "ariadne").stat().st_mode) == 0o700
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        _archive_path(home)  # development engines keep the strict home rule

    linked = tmp_path / "linked"
    linked.symlink_to(home)
    with pytest.raises(ValueError):
        _archive_path(linked, private_home=False)
    with pytest.raises(StorageRefused):
        NoteStore(linked).add("x")
