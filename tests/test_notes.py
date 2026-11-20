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


def test_tightens_drifted_permissions_instead_of_refusing(home):
    (home / "ariadne").mkdir(mode=0o755)
    os.chmod(home / "ariadne", 0o755)
    NoteStore(home).add("x")
    assert stat.S_IMODE((home / "ariadne").stat().st_mode) == 0o700
    path = home / "ariadne" / "notes.sqlite3"
    os.chmod(path, 0o644)
    assert NoteStore(home).list()["total"] == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_refuses_hardlinked_store(home, tmp_path):
    NoteStore(home).add("x")
    os.link(home / "ariadne" / "notes.sqlite3", tmp_path / "alias")
    with pytest.raises(ValueError):
        NoteStore(home).list()


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


def test_concurrent_read_cannot_delete_an_active_initializer(home, monkeypatch):
    import threading

    import hermes_ariadne.notes as notes

    paused, release = threading.Event(), threading.Event()
    real_link = os.link

    def slow_link(src, dst):
        paused.set()
        assert release.wait(5)
        return real_link(src, dst)

    monkeypatch.setattr(notes.os, "link", slow_link)
    errors = []
    writer = threading.Thread(target=lambda: _try(lambda: NoteStore(home).add("x"), errors))
    writer.start()
    assert paused.wait(5)
    reader = threading.Thread(target=lambda: _try(lambda: NoteStore(home).list(), errors))
    reader.start()
    reader.join(0.3)  # the reader must wait for the initializer, not delete its file
    release.set()
    writer.join(5)
    reader.join(5)
    assert errors == []
    assert NoteStore(home).list()["total"] == 1


def _try(fn, errors):
    try:
        fn()
    except Exception as exc:  # pragma: no cover - reported by the assertion
        errors.append(repr(exc))


def test_crash_after_publication_link_recovers(home):
    NoteStore(home).add("kept")
    store = home / "ariadne" / "notes.sqlite3"
    os.link(store, home / "ariadne" / ".notes-init-crashed")  # residual publication link
    assert NoteStore(home).list()["total"] == 1
    assert not (home / "ariadne" / ".notes-init-crashed").exists()


def test_unrelated_hardlink_is_still_refused(home, tmp_path):
    NoteStore(home).add("x")
    os.link(home / "ariadne" / "notes.sqlite3", tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        NoteStore(home).list()


def test_unicode_tags_are_searchable_and_stored_readably(home):
    store = NoteStore(home)
    store.add("meeting preference", tags=["café"])
    assert [n["text"] for n in store.search("café")] == ["meeting preference"]
    note = store.add("other")
    store.edit(note["id"], "other", tags=["thé"])
    assert [n["text"] for n in store.search("thé")] == ["other"]


def test_count_and_page_share_one_snapshot(home, monkeypatch):
    import sqlite3

    store = NoteStore(home)
    note = store.add("only")
    real = sqlite3.Connection.execute
    connect = sqlite3.connect
    state = {"done": False}

    class Spy(sqlite3.Connection):
        def execute(self, sql, *args):
            if "LIMIT" in sql and "OFFSET" in sql and not state["done"]:
                state["done"] = True
                other = connect(home / "ariadne" / "notes.sqlite3", timeout=0.2)
                try:
                    other.execute("DELETE FROM notes WHERE id=?", (note["id"],))
                    other.commit()
                except sqlite3.OperationalError:
                    pass  # blocked by the reader's snapshot: acceptable
                finally:
                    other.close()
            return real(self, sql, *args)

    monkeypatch.setattr(
        "hermes_ariadne.notes.sqlite3.connect",
        lambda *a, **k: connect(*a, factory=Spy, **k),
    )
    page = store.list()
    assert page["total"] == len(page["notes"])


def test_tags_written_by_older_revision_remain_searchable(home):
    import sqlite3

    store = NoteStore(home)
    note = store.add("legacy row")
    with sqlite3.connect(home / "ariadne" / "notes.sqlite3") as conn:
        conn.execute("UPDATE notes SET tags=? WHERE id=?", ('["caf\\u00e9"]', note["id"]))
    assert [n["text"] for n in store.search("café")] == ["legacy row"]


def test_sidecar_disappearing_during_admission_is_tolerated(home, monkeypatch):
    import hermes_ariadne.notes as notes

    NoteStore(home).add("x")
    journal = home / "ariadne" / "notes.sqlite3-journal"
    journal.write_bytes(b"")
    journal.chmod(0o600)
    real = notes._check_private

    def racing(path, *, regular):
        if path == journal and path.exists():
            path.unlink()  # another session commits and SQLite removes its journal
        return real(path, regular=regular)

    monkeypatch.setattr(notes, "_check_private", racing)
    assert NoteStore(home).list()["total"] == 1
