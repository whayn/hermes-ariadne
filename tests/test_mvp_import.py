"""Phase 4: agent-driven import into the exact archive + conversation index."""

import json
import os
import stat

import pytest

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("HERMES_SOURCE required", allow_module_level=True)

from hermes_ariadne.conversations import ArchiveReader  # noqa: E402
from hermes_ariadne.importer import MAX_MESSAGE_BYTES, Importer, conversation_id  # noqa: E402
from hermes_ariadne.memory_provider import AriadneMemoryProvider  # noqa: E402
from hermes_ariadne.notes import NoteStore  # noqa: E402


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    return home


def _chat(n, start=0):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(start, start + n)
    ]


def test_batches_are_ordered_idempotent_and_private(home):
    importer = Importer(home)
    first = importer.import_batch(
        "export.json#c1",
        _chat(3),
        title="Trip",
        summary="Planned a trip.",
        source={"file": "export.json"},
    )
    assert first["id"] == conversation_id("export.json#c1") and first["id"].startswith("import-")
    assert first == {**first, "archived": 3, "already_present": 0, "next_offset": 3}
    second = importer.import_batch("export.json#c1", _chat(2, 3), offset=3)
    again = importer.import_batch("export.json#c1", _chat(3))
    assert second["archived"] == 2 and again["archived"] == 0 and again["already_present"] == 3
    page = ArchiveReader(home).page(first["id"], limit=50)
    assert [m["content"] for m in page["messages"]] == [f"message {i}" for i in range(5)]
    for path in (home / "ariadne", home / "ariadne" / "events.sqlite3"):
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    row = NoteStore(home).get_conversation(first["id"])
    assert row["origin"] == "import" and row["title"] == "Trip"
    assert row["summary"] == "Planned a trip." and row["preview"] == "message 0"
    assert row["source"] == {"file": "export.json"}


def test_reimport_with_edited_text_adds_a_version_not_an_overwrite(home):
    importer = Importer(home)
    conv = importer.import_batch("k", [{"role": "user", "content": "v1"}])["id"]
    importer.import_batch("k", [{"role": "user", "content": "v2"}])
    contents = [m["content"] for m in ArchiveReader(home).page(conv)["messages"]]
    uids = {m["message_uid"] for m in ArchiveReader(home).page(conv)["messages"]}
    assert sorted(contents) == ["v1", "v2"] and len(uids) == 1


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "robot", "content": "x"}],
        [{"role": "user", "content": {"parts": ["x"]}}],
        [{"role": "user", "content": "x" * (MAX_MESSAGE_BYTES + 1)}],
        [{"role": "user", "content": "x", "timestamp": 5}],
        "not a list",
    ],
)
def test_rejects_invalid_batches_without_writing(home, messages):
    with pytest.raises(ValueError):
        Importer(home).import_batch("k", messages)
    assert not (home / "ariadne" / "events.sqlite3").exists()


@pytest.mark.parametrize("key", ["", "   ", "x" * 501, None])
def test_rejects_bad_keys(home, key):
    with pytest.raises(ValueError):
        Importer(home).import_batch(key, _chat(1))


def test_provider_import_then_new_chat_sees_and_reads_it(home):
    provider = AriadneMemoryProvider()
    provider.initialize("importing-chat", hermes_home=str(home))
    assert "ariadne_import" in [s["name"] for s in provider.get_tool_schemas()]
    result = json.loads(
        provider.handle_tool_call(
            "ariadne_import",
            {
                "key": "chatgpt.json#abc",
                "title": "E-bike tyres",
                "summary": "Discussed tyre sizes for a Nakamura e-bike.",
                "messages": [
                    {"role": "user", "content": "What tyres for my Nakamura?", "timestamp": "2024"},
                    {"role": "assistant", "content": "27.5x2.10"},
                ],
            },
        )
    )
    assert result["ok"] and result["archived"] == 2
    note = json.loads(
        provider.handle_tool_call(
            "ariadne_notes",
            {
                "action": "add",
                "text": "User rides a Nakamura e-bike",
                "source": {"kind": "import", "conversation": result["id"]},
            },
        )
    )
    assert note["ok"]
    bad = json.loads(provider.handle_tool_call("ariadne_import", {"key": "k", "messages": []}))
    assert bad == {"ok": False, "error": "invalid_argument"}

    fresh = AriadneMemoryProvider()
    fresh.initialize("later", hermes_home=str(home))
    recall = fresh.prefetch("tell me about my e-bike tyres", session_id="later")
    assert f"[{result['id']}]" in recall and "User rides a Nakamura e-bike" in recall
    read = json.loads(
        fresh.handle_tool_call("ariadne_conversations", {"action": "read", "id": result["id"]})
    )
    assert [m["content"] for m in read["messages"]] == ["What tyres for my Nakamura?", "27.5x2.10"]


def test_skill_is_registered_and_bundled():
    from test_mvp_plugin import FullContext, load_root

    class SkillContext(FullContext):
        def __init__(self):
            super().__init__()
            self.skills = []

        def register_skill(self, name, path, description=""):
            self.skills.append((name, path))

    ctx = SkillContext()
    load_root("_ariadne_root_skill").register(ctx)
    ((name, path),) = ctx.skills
    assert name == "import" and path.is_file()
    text = path.read_text()
    assert text.startswith("---\nname: import\n") and "ariadne_import" in text


def test_shared_home_permissions_report_storage_refused_not_bad_argument(tmp_path):
    # `hermes profile create` makes 0755 homes; the archive admission refuses them.
    home = tmp_path / "shared-home"
    home.mkdir()
    os.chmod(home, 0o755)
    provider = AriadneMemoryProvider()
    provider.initialize("s", hermes_home=str(home))
    batch = {"key": "k", "messages": [{"role": "user", "content": "hi"}]}
    assert json.loads(provider.handle_tool_call("ariadne_import", batch)) == {
        "ok": False,
        "error": "storage_refused",
    }
    read = {"action": "read", "id": "import-x"}
    assert json.loads(provider.handle_tool_call("ariadne_conversations", read)) == {
        "ok": False,
        "error": "storage_refused",
    }
    assert not (home / "ariadne" / "events.sqlite3").exists()
