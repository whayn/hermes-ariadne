"""Profile note store: CRUD, revisions, mirroring, bounds and private-path admission."""

import os
import stat

import pytest

from hermes_ariadne.notes import MAX_NOTE_CHARS, NoteStore


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    return home


def test_reads_never_initialize(home):
    store = NoteStore(home)
    assert store.list() == {"notes": [], "offset": 0, "total": 0}
    assert store.search("x") == []
    assert not (home / "ariadne").exists()


def test_add_creates_private_store_and_dedupes(home):
    store = NoteStore(home)
    first = store.add("User prefers terse replies.", tags=["pref"])
    again = store.add("User prefers terse replies.")
    assert first["id"] == again["id"] and again["duplicate"] is True
    directory, path = home / "ariadne", home / "ariadne" / "notes.sqlite3"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    listed = store.list()
    assert listed["total"] == 1
    assert listed["notes"][0]["text"] == "User prefers terse replies."
    assert listed["notes"][0]["tags"] == ["pref"]
    assert listed["notes"][0]["origin"] == "agent"


def test_edit_keeps_revisions_and_delete_removes_everything(home):
    store = NoteStore(home)
    note = store.add("Lives in Paris")
    store.edit(note["id"], "Lives in Boulogne")
    assert store.get(note["id"])["text"] == "Lives in Boulogne"
    assert [r["text"] for r in store.revisions(note["id"])] == [
        "Lives in Paris",
        "Lives in Boulogne",
    ]
    store.delete(note["id"])
    assert store.get(note["id"]) is None
    assert store.revisions(note["id"]) == []
    with pytest.raises(KeyError):
        store.delete(note["id"])


def test_search_is_case_insensitive_literal_and_bounded(home):
    store = NoteStore(home)
    for i in range(30):
        store.add(f"Project Ariadne fact {i}")
    store.add("unrelated")
    hits = store.search("ariadne", limit=5)
    assert len(hits) == 5 and all("Ariadne" in h["text"] for h in hits)
    assert store.search("%") == []  # LIKE wildcards are literal


def test_pagination(home):
    store = NoteStore(home)
    for i in range(7):
        store.add(f"n{i}")
    page = store.list(offset=5, limit=5)
    assert page["total"] == 7 and len(page["notes"]) == 2 and page["offset"] == 5


@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_NOTE_CHARS + 1), None, 3])
def test_rejects_invalid_text(home, text):
    with pytest.raises(ValueError):
        NoteStore(home).add(text)


def test_source_is_optional_bounded_mapping(home):
    store = NoteStore(home)
    note = store.add("fact", source={"kind": "conversation", "ref": "abc"})
    assert store.get(note["id"])["source"] == {"kind": "conversation", "ref": "abc"}
    with pytest.raises(ValueError):
        store.add("other", source="not a mapping")


def test_mirror_add_replace_remove(home):
    store = NoteStore(home)
    store.mirror("add", "user", "Name: Whayn")
    store.mirror("add", "user", "Name: Whayn")  # idempotent
    notes = store.list()["notes"]
    assert len(notes) == 1 and notes[0]["origin"] == "mirror:user"
    store.mirror("replace", "user", "Name: Whayn (they/them)", previous="Name: Whayn")
    assert [n["text"] for n in store.list()["notes"]] == ["Name: Whayn (they/them)"]
    store.mirror("remove", "user", "", previous="Name: Whayn (they/them)")
    assert store.list()["total"] == 0


def test_mirror_destructive_without_previous_is_skipped(home):
    store = NoteStore(home)
    store.mirror("add", "memory", "fact A")
    assert store.mirror("replace", "memory", "fact B", previous=None) == "skipped"
    assert store.mirror("remove", "memory", "", previous=None) == "skipped"
    assert [n["text"] for n in store.list()["notes"]] == ["fact A"]


def test_mirror_replace_of_unknown_entry_adds_it(home):
    store = NoteStore(home)
    store.mirror("replace", "memory", "new text", previous="never mirrored")
    assert [n["text"] for n in store.list()["notes"]] == ["new text"]


def test_mirror_does_not_touch_agent_notes(home):
    store = NoteStore(home)
    store.add("shared text")
    store.mirror("remove", "memory", "", previous="shared text")
    assert [n["origin"] for n in store.list()["notes"]] == ["agent"]


def test_refuses_symlinked_directory(home, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    os.symlink(elsewhere, home / "ariadne")
    with pytest.raises(ValueError):
        NoteStore(home).add("x")
    assert list(elsewhere.iterdir()) == []


def test_refuses_shared_permissions(home):
    (home / "ariadne").mkdir(mode=0o755)
    os.chmod(home / "ariadne", 0o755)
    with pytest.raises(ValueError):
        NoteStore(home).add("x")


def test_refuses_foreign_database(home):
    import sqlite3

    (home / "ariadne").mkdir(mode=0o700)
    path = home / "ariadne" / "notes.sqlite3"
    sqlite3.connect(path).execute("CREATE TABLE other(x)").connection.commit()
    os.chmod(path, 0o600)
    with pytest.raises(ValueError):
        NoteStore(home).list()


def test_leftover_temporary_does_not_block_initialization(home):
    (home / "ariadne").mkdir(mode=0o700)
    leftover = home / "ariadne" / ".notes-init-abc"
    leftover.write_bytes(b"partial")
    os.chmod(leftover, 0o600)
    NoteStore(home).add("x")
    assert not leftover.exists()


def test_persists_across_instances(home):
    NoteStore(home).add("durable")
    assert NoteStore(home).search("durable")[0]["text"] == "durable"
