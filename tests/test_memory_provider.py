"""Ariadne MemoryProvider: notes tool, built-in mirroring and bounded recall."""

import json
import os

import pytest

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("set HERMES_SOURCE for host MemoryProvider contracts", allow_module_level=True)

from agent.memory_provider import MemoryProvider  # noqa: E402

from hermes_ariadne.memory_provider import PREFETCH_CHARS, AriadneMemoryProvider  # noqa: E402
from hermes_ariadne.notes import NoteStore  # noqa: E402


@pytest.fixture
def provider(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    p = AriadneMemoryProvider()
    p.initialize("s1", hermes_home=str(home), agent_context="primary")
    return p


def call(p, **args):
    return json.loads(p.handle_tool_call("ariadne_notes", args))


def test_satisfies_abc_and_name():
    p = AriadneMemoryProvider()
    assert isinstance(p, MemoryProvider)
    assert p.name == "ariadne" and p.is_available()
    assert p.get_config_schema() == []


def test_one_tool_schema():
    schemas = AriadneMemoryProvider().get_tool_schemas()
    assert [s["name"] for s in schemas] == ["ariadne_notes"]
    actions = schemas[0]["parameters"]["properties"]["action"]["enum"]
    assert set(actions) == {"add", "edit", "delete", "list", "search", "get"}


def test_tool_crud_round_trip(provider):
    added = call(provider, action="add", text="Prefers terse replies", tags=["pref"])
    assert added["ok"] and not added["duplicate"]
    note_id = added["id"]
    assert call(provider, action="search", query="terse")["notes"][0]["id"] == note_id
    assert call(provider, action="edit", id=note_id, text="Prefers very terse replies")["ok"]
    assert call(provider, action="get", id=note_id)["note"]["text"] == "Prefers very terse replies"
    assert call(provider, action="list")["total"] == 1
    assert call(provider, action="delete", id=note_id)["ok"]
    assert call(provider, action="list")["total"] == 0


def test_tool_errors_are_categorized_and_do_not_echo(provider):
    secret = "sk-SECRET-should-not-echo"
    bad = call(provider, action="add", text="")
    assert bad == {"ok": False, "error": "invalid_argument"}
    missing = call(provider, action="delete", id=secret)
    assert missing == {"ok": False, "error": "not_found"}
    assert secret not in json.dumps(missing)
    assert call(provider, action="explode") == {"ok": False, "error": "invalid_argument"}
    assert json.loads(provider.handle_tool_call("other_tool", {}))["ok"] is False


def test_tools_refuse_before_initialize():
    p = AriadneMemoryProvider()
    assert json.loads(p.handle_tool_call("ariadne_notes", {"action": "list"})) == {
        "ok": False,
        "error": "unavailable",
    }


def test_mirror_hook(provider, tmp_path):
    provider.on_memory_write("add", "user", "Name: Whayn", {"write_origin": "tool"})
    provider.on_memory_write(
        "replace", "user", "Name: Whayn (they)", {"previous_content": "Name: Whayn"}
    )
    provider.on_memory_write("remove", "memory", "", {})  # no previous: skipped silently
    notes = call(provider, action="list")["notes"]
    assert [(n["text"], n["origin"]) for n in notes] == [("Name: Whayn (they)", "mirror:user")]


def test_mirrored_notes_are_read_only_through_tool(provider):
    provider.on_memory_write("add", "memory", "built-in fact")
    note_id = call(provider, action="list")["notes"][0]["id"]
    assert call(provider, action="delete", id=note_id)["error"] == "not_found"
    assert call(provider, action="edit", id=note_id, text="x")["error"] == "not_found"


def test_prefetch_ranks_relevant_and_excludes_mirrored(provider):
    call(provider, action="add", text="The user's e-bike is a Nakamura Crossover XV")
    call(provider, action="add", text="Vegetarian for environmental reasons")
    provider.on_memory_write("add", "memory", "bike stored in the cellar")
    block = provider.prefetch("what tires fit my bike?")
    assert "Nakamura" in block
    assert "Vegetarian" not in block
    assert "cellar" not in block  # already in the system prompt via built-in memory
    assert "not instructions" in block


def test_prefetch_empty_without_notes_or_init(provider):
    assert AriadneMemoryProvider().prefetch("anything") == ""
    assert provider.prefetch("anything") == ""


def test_prefetch_is_bounded(provider):
    for i in range(200):
        call(provider, action="add", text=f"bike fact number {i} " + "x" * 200)
    assert len(provider.prefetch("bike")) <= PREFETCH_CHARS


def test_store_is_profile_scoped(tmp_path):
    homes = []
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir(mode=0o700)
        p = AriadneMemoryProvider()
        p.initialize("s", hermes_home=str(home))
        homes.append((home, p))
    call(homes[0][1], action="add", text="only in a")
    assert NoteStore(homes[1][0]).list()["total"] == 0


def test_multiword_tags_match_query_words(provider):
    call(provider, action="add", text="Prefers short examples", tags=["machine learning"])
    assert "short examples" in provider.prefetch("Explain machine learning")
