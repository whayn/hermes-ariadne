"""Generic agent-authored notes over synthetic sources, not app converters."""

import hashlib
import importlib
import importlib.util
import json
import sqlite3

import pytest


def library(home):
    assert importlib.util.find_spec("hermes_ariadne.library") is not None, "generic library missing"
    return importlib.import_module("hermes_ariadne.library").Library(home)


def source(path, raw):
    path.write_bytes(raw)
    return {
        "path": str(path),
        "start_bytes": 0,
        "length_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "label": "Synthetic supplied file",
    }


def test_generic_prepare_review_approve_restart(tmp_path):
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    raw = b'{"arbitrary_format": ["prefers jasmine tea"]}'
    supplied = source(tmp_path / "input.json", raw)
    notes = [
        {
            "title": "Tea",
            "text": "Prefers jasmine tea",
            "kind": "summary",
            "tags": ["preferences"],
            "sources": [0],
        }
    ]
    store = library(home)
    proposal = store.prepare(sources=[supplied], notes=notes)
    assert store.search("jasmine")["results"] == []  # Drafts are not memory.
    reviewed = store.review(proposal["proposal_ref"])
    assert reviewed["note"]["text"] == "Prefers jasmine tea"
    receipt = store.approve(proposal["proposal_ref"])
    assert receipt["inserted"] == 1
    assert store.approve(proposal["proposal_ref"])["inserted"] == 0
    hits = library(home).search("jasmine")["results"]
    assert len(hits) == 1 and hits[0]["kind"] == "summary"
    note = store.read(hits[0]["ref"])
    assert note["text"] == notes[0]["text"]
    assert note["provenance"] == [
        {
            "ref": note["sources"][0],
            "label": supplied["label"],
            "start_bytes": 0,
            "length_bytes": len(raw),
        }
    ]
    assert store.original(note["sources"][0]) == raw
    assert (tmp_path / "input.json").read_bytes() == raw
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert json.loads(json.dumps(receipt))["note_refs"] == [hits[0]["ref"]]


def test_existing_empty_foreign_database_is_not_initialized(tmp_path):
    store = library(tmp_path)
    store.directory.mkdir(mode=0o700)
    store.path.touch(mode=0o600)
    before = store.path.read_bytes()
    supplied = source(tmp_path / "source.txt", b"source")
    with pytest.raises(ValueError, match="foreign|unsupported"):
        store.prepare(
            sources=[supplied],
            notes=[
                {
                    "title": "A",
                    "text": "source",
                    "kind": "quotation",
                    "tags": [],
                    "sources": [0],
                }
            ],
        )
    assert store.path.read_bytes() == before


def test_reads_do_not_initialize_and_unknown_proposal_cannot_publish(tmp_path):
    store = library(tmp_path)
    assert not store.directory.exists()
    assert store.search("anything")["results"] == []
    with pytest.raises(ValueError):
        store.approve("proposal:" + "0" * 64)
    assert not store.directory.exists()


def test_unreferenced_supplied_sources_are_not_retained(tmp_path):
    store = library(tmp_path)
    specs = [source(tmp_path / "a.txt", b"selected"), source(tmp_path / "b.txt", b"not selected")]
    with pytest.raises(ValueError, match="unreferenced"):
        store.prepare(
            sources=specs,
            notes=[
                {
                    "title": "A",
                    "text": "selected",
                    "kind": "quotation",
                    "tags": [],
                    "sources": [0],
                }
            ],
        )
    assert not store.directory.exists()


def test_bad_quotation_is_refused_before_initialization_and_does_not_poison_retry(tmp_path):
    store = library(tmp_path)
    supplied = source(tmp_path / "input.txt", b"source text")
    note = {"title": "A", "text": "not in source", "kind": "quotation", "tags": [], "sources": [0]}
    with pytest.raises(ValueError, match="quotation"):
        store.prepare(sources=[supplied], notes=[note])
    assert not store.directory.exists()
    note["text"] = "source text"
    assert store.prepare(sources=[supplied], notes=[note])["status"] == "draft"


def test_invalid_text_offset_is_not_a_silent_empty_page(tmp_path):
    store = library(tmp_path)
    supplied = source(tmp_path / "input.txt", b"abcd")
    proposal = store.prepare(
        sources=[supplied],
        notes=[
            {
                "title": "A",
                "text": "abcd",
                "kind": "quotation",
                "tags": [],
                "sources": [0],
            }
        ],
    )
    with pytest.raises(ValueError, match="page"):
        store.review(proposal["proposal_ref"], offset=5)


def prepared(tmp_path, *, text="A café\nstatement", count=1):
    store = library(tmp_path)
    supplied = source(tmp_path / "input.txt", text.encode())
    notes = [
        {"title": f"Note {i}", "text": text, "kind": "quotation", "tags": ["tag"], "sources": [0]}
        for i in range(count)
    ]
    return store, store.prepare(sources=[supplied], notes=notes), supplied, notes


def test_repeat_prepare_and_changed_interpretation_are_distinct(tmp_path):
    store, proposal, supplied, notes = prepared(tmp_path)
    assert store.prepare(sources=[supplied], notes=notes) == proposal
    first = store.approve(proposal["proposal_ref"])
    notes[0]["kind"] = "inference"
    notes[0]["text"] = "A possibly conflicting interpretation"
    changed = store.prepare(sources=[supplied], notes=notes)
    assert changed["proposal_ref"] != proposal["proposal_ref"]
    second = store.approve(changed["proposal_ref"])
    assert first["note_refs"] != second["note_refs"]
    assert len(store.search("Note")["results"]) == 2
    assert store.read(first["note_refs"][0])["kind"] == "quotation"


def test_unicode_character_pages_and_search_pagination(tmp_path):
    store, proposal, _, _ = prepared(tmp_path, count=7)
    receipt = store.approve(proposal["proposal_ref"])
    found, offset = [], 0
    while True:
        page = store.search("café", limit=2, offset=offset)
        found.extend(item["ref"] for item in page["results"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert len(set(found)) == 7 and set(found) == set(receipt["note_refs"])
    ref = found[0]
    text, offset = "", 0
    while True:
        page = store.read(ref, limit=3, offset=offset)
        text += page["text"]
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert text == "A café\nstatement"
    assert store.search("CAFÉ")["results"] == []


def test_search_verifies_shared_source_once_per_transaction(tmp_path, monkeypatch):
    store, proposal, _, _ = prepared(tmp_path, count=7)
    store.approve(proposal["proposal_ref"])
    reads = []
    original = store._source

    def capture(conn, ref):
        reads.append(ref)
        return original(conn, ref)

    monkeypatch.setattr(store, "_source", capture)
    assert store.search("not present", limit=1)["results"] == []
    assert len(reads) == 1
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE sources SET data=?", (b"changed",))
    with pytest.raises(ValueError, match="source changed"):
        store.search("café", limit=1)
    assert len(reads) == 2


def test_schema_rejects_unexpected_sqlite_like_names(tmp_path):
    store, proposal, _, _ = prepared(tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute("CREATE TABLE sqliteXextra (value TEXT)")
    with pytest.raises(ValueError, match="unsupported memory library schema"):
        store.approve(proposal["proposal_ref"])


def test_interrupted_first_initialization_leaves_retryable_state(tmp_path, monkeypatch):
    module = importlib.import_module("hermes_ariadne.library")
    store = library(tmp_path)
    supplied = source(tmp_path / "input.txt", b"source")
    note = {"title": "A", "text": "source", "kind": "quotation", "tags": [], "sources": [0]}
    original = module._sync

    def interrupted(path):
        if path == store.directory:
            raise OSError("synthetic initialization sync failure")
        original(path)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_sync", interrupted)
        with pytest.raises(OSError, match="initialization sync failure"):
            store.prepare(sources=[supplied], notes=[note])
    # An absent or fully initialized DB is safe. A blank final DB is not.
    assert store.prepare(sources=[supplied], notes=[note])["status"] == "draft"


def test_approval_has_no_fallible_directory_sync_after_commit(tmp_path, monkeypatch):
    store, proposal, _, _ = prepared(tmp_path)
    module = importlib.import_module("hermes_ariadne.library")

    def interrupted(path):
        raise OSError("synthetic unexpected directory sync")

    monkeypatch.setattr(module, "_sync", interrupted)
    receipt = store.approve(proposal["proposal_ref"])
    assert store.read(receipt["note_refs"][0])["text"] == "A café\nstatement"


def test_source_path_cannot_alias_the_library(tmp_path):
    store, _, _, notes = prepared(tmp_path)
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    raw = store.path.read_bytes()[:16]
    spec = {
        "path": str(sibling / ".." / store.directory.name / store.path.name),
        "start_bytes": 0,
        "length_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "label": "Not an external source",
    }
    notes[0]["kind"] = "summary"
    with pytest.raises(ValueError, match="external absolute path"):
        store.prepare(sources=[spec], notes=notes)


@pytest.mark.parametrize("target", ["home", "directory", "database", "wal", "shm", "journal"])
def test_unsafe_destination_permissions_refuse_before_source_capture(tmp_path, monkeypatch, target):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    store = library(home)
    supplied = source(tmp_path / "input.txt", b"source")
    unsafe = home
    if target != "home":
        store.directory.mkdir(mode=0o700)
        unsafe = store.directory
    if target in {"database", "wal", "shm", "journal"}:
        suffix = {"database": "", "wal": "-wal", "shm": "-shm", "journal": "-journal"}[target]
        unsafe = store.path.with_name(store.path.name + suffix)
        unsafe.touch(mode=0o600)
    unsafe.chmod(0o755 if unsafe.is_dir() else 0o644)
    called = []
    monkeypatch.setattr(store, "_capture", lambda spec: called.append(spec))
    before = (unsafe.stat().st_mode, unsafe.read_bytes() if unsafe.is_file() else None)
    with pytest.raises(ValueError, match="private"):
        store.prepare(
            sources=[supplied],
            notes=[
                {
                    "title": "A",
                    "text": "source",
                    "kind": "quotation",
                    "tags": [],
                    "sources": [0],
                }
            ],
        )
    assert called == []
    assert (unsafe.stat().st_mode, unsafe.read_bytes() if unsafe.is_file() else None) == before


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_source_links_and_destination_file_links_are_refused(tmp_path, link):
    store = library(tmp_path)
    supplied = source(tmp_path / "input.txt", b"source")
    linked = tmp_path / "linked.txt"
    if link == "symlink":
        linked.symlink_to(tmp_path / "input.txt")
    else:
        linked.hardlink_to(tmp_path / "input.txt")
    supplied["path"] = str(linked)
    note = {"title": "A", "text": "source", "kind": "quotation", "tags": [], "sources": [0]}
    with pytest.raises(ValueError):
        store.prepare(sources=[supplied], notes=[note])
    assert not store.directory.exists()
    store.directory.mkdir(mode=0o700)
    if link == "symlink":
        store.path.symlink_to(tmp_path / "input.txt")
    else:
        store.path.hardlink_to(tmp_path / "input.txt")
    with pytest.raises(ValueError):
        store.search("source")
    assert (tmp_path / "input.txt").read_bytes() == b"source"


@pytest.mark.parametrize(
    "table,column", [("sources", "data"), ("notes", "payload"), ("proposals", "payload")]
)
def test_corrupted_bytes_never_return_successful_evidence(tmp_path, table, column):
    store, proposal, _, _ = prepared(tmp_path)
    receipt = store.approve(proposal["proposal_ref"])
    with sqlite3.connect(store.path) as db:
        db.execute(f"UPDATE {table} SET {column}=?", (b"changed",))
    action = (
        (lambda: store.review(proposal["proposal_ref"]))
        if table == "proposals"
        else (lambda: store.read(receipt["note_refs"][0]))
    )
    with pytest.raises(ValueError, match="changed"):
        action()
    if table != "notes":
        with pytest.raises(ValueError):
            store.approve(proposal["proposal_ref"])


def test_selected_range_hash_change_and_profile_isolation(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(mode=0o700)
    two.mkdir(mode=0o700)
    store = library(one)
    path = tmp_path / "source.txt"
    supplied = source(path, b"BEFORE selected AFTER")
    supplied.update(start_bytes=7, length_bytes=8, sha256=hashlib.sha256(b"selected").hexdigest())
    notes = [{"title": "A", "text": "selected", "kind": "quotation", "tags": [], "sources": [0]}]
    proposal = store.prepare(sources=[supplied], notes=notes)
    ref = store.approve(proposal["proposal_ref"])["note_refs"][0]
    path.write_bytes(b"source changed")
    assert store.original(store.read(ref)["sources"][0]) == b"selected"
    with pytest.raises(ValueError, match="range|hash"):
        store.prepare(sources=[supplied], notes=notes)
    other = library(two)
    assert other.search("selected")["results"] == []
    with pytest.raises(ValueError):
        other.read(ref)
    assert not other.directory.exists()


@pytest.mark.parametrize(
    "mutation", ["unknown_key", "restriction", "source_index", "too_many", "too_long"]
)
def test_invalid_note_schema_is_refused_without_initialization(tmp_path, mutation):
    store = library(tmp_path)
    supplied = source(tmp_path / "input.txt", b"source")
    note = {"title": "A", "text": "source", "kind": "quotation", "tags": [], "sources": [0]}
    if mutation == "unknown_key":
        note["timestamp"] = "invented"
    elif mutation == "restriction":
        note["data_use"] = ["do-not-use"]
    elif mutation == "source_index":
        note["sources"] = [True]
    elif mutation == "too_long":
        note["text"] = "é" * 2049
    with pytest.raises(ValueError):
        store.prepare(sources=[supplied], notes=[note] * (17 if mutation == "too_many" else 1))
    assert not store.directory.exists()
