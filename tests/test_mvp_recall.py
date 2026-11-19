"""Phase 3: conversation index, read-only archive access and new-chat recall."""

import json
import os
import sqlite3
import stat

import pytest

from hermes_ariadne.notes import MAX_PREVIEW_CHARS, NoteStore

HOST = bool(os.environ.get("HERMES_SOURCE"))


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    return home


# -- index table (host-independent) ------------------------------------------------------


def test_index_preview_set_once_summary_replaced(home):
    store = NoteStore(home)
    store.upsert_conversation("s1", preview="first question", title="Bike")
    store.upsert_conversation("s1", preview="later question", summary="sum 1")
    store.upsert_conversation("s1", summary="sum 2")
    row = store.get_conversation("s1")
    assert (row["preview"], row["summary"], row["title"]) == ("first question", "sum 2", "Bike")
    assert row["origin"] == "live"


def test_index_list_excludes_current_and_orders_recent_first(home):
    store = NoteStore(home)
    for sid in ("a", "b", "c"):
        store.upsert_conversation(sid, preview=sid)
    store.upsert_conversation("a", summary="touched")
    page = store.conversations(exclude="b")
    assert [c["session_id"] for c in page["conversations"]] == ["a", "c"]
    assert page["total"] == 2


def test_index_search_and_bounds(home):
    store = NoteStore(home)
    store.upsert_conversation("a", preview="x" * 1000, summary="talked about Nakamura bikes")
    assert len(store.get_conversation("a")["preview"]) == MAX_PREVIEW_CHARS
    assert [c["session_id"] for c in store.search_conversations("nakamura")] == ["a"]
    assert store.search_conversations("nakamura", exclude="a") == []
    with pytest.raises(ValueError):
        store.upsert_conversation("", preview="x")
    with pytest.raises(ValueError):
        store.upsert_conversation("a", origin="other")


def test_index_reads_never_initialize(home):
    store = NoteStore(home)
    assert store.conversations()["total"] == 0
    assert store.get_conversation("a") is None
    assert not (home / "ariadne").exists()


# -- archive reader and provider (host) --------------------------------------------------

needs_host = pytest.mark.skipif(not HOST, reason="HERMES_SOURCE required")


def _archive(home, sessions):
    from hermes_ariadne.store import EventStore

    (home / "ariadne").mkdir(mode=0o700, exist_ok=True)
    store = EventStore(home / "ariadne" / "events.sqlite3")
    os.chmod(store.path, 0o600)
    for sid, messages in sessions.items():
        store.ingest(sid, messages, origin="hermes_db")
    for item in (home / "ariadne").iterdir():
        os.chmod(item, stat.S_IMODE(item.stat().st_mode) & 0o700)
    return store


def _msg(uid, role, content, **extra):
    return {"message_uid": uid, "role": role, "content": content, **extra}


@needs_host
def test_reader_pages_in_order_and_omits_carriers(home):
    from hermes_ariadne.conversations import MAX_MESSAGE_CHARS, ArchiveReader

    _archive(
        home,
        {
            "old": [
                _msg("u1", "user", "What tyre size for my bike?"),
                _msg("a1", "assistant", "27.5x2.10", reasoning="secret chain", tool_calls=None),
                _msg("u2", "user", "y" * (MAX_MESSAGE_CHARS + 10)),
            ]
        },
    )
    reader = ArchiveReader(home)
    page = reader.page("old", limit=2)
    assert [m["message_uid"] for m in page["messages"]] == ["u1", "a1"]
    assert page["has_more"] is True
    assert "reasoning" in page["messages"][1]["omitted_keys"]
    assert "secret chain" not in json.dumps(page)
    long = reader.page("old", offset=2)["messages"][0]
    assert long["truncated"] and long["chars"] == MAX_MESSAGE_CHARS + 10
    zoom = reader.event("old", long["event_id"], char_offset=MAX_MESSAGE_CHARS)
    assert zoom["content"] == "y" * 10 and zoom["has_more"] is False
    assert reader.event("other-session", long["event_id"]) is None


@needs_host
def test_reader_search_excludes_current_session(home):
    from hermes_ariadne.conversations import ArchiveReader

    _archive(
        home,
        {
            "old": [_msg("u1", "user", "Nakamura Crossover tyres")],
            "now": [_msg("u2", "user", "Nakamura again")],
        },
    )
    hits = ArchiveReader(home).search("Nakamura", exclude="now")
    assert [h["session_id"] for h in hits] == ["old"]
    assert "Nakamura" in hits[0]["preview"]


@needs_host
def test_reader_missing_archive_is_empty_and_not_created(home):
    from hermes_ariadne.conversations import ArchiveReader

    reader = ArchiveReader(home)
    assert reader.page("x")["messages"] == [] and reader.search("x") == []
    assert not (home / "ariadne").exists()


@needs_host
def test_reader_refuses_tampered_payload(home):
    from hermes_ariadne.conversations import ArchiveReader

    store = _archive(home, {"old": [_msg("u1", "user", "original")]})
    conn = sqlite3.connect(store.path)
    conn.execute("UPDATE events SET payload_json=?", (b'{"message_uid":"u1","content":"forged"}',))
    conn.commit()
    conn.close()
    with pytest.raises(ValueError):
        ArchiveReader(home).page("old")


@needs_host
def test_provider_index_on_first_turn_relevant_later(home):
    from hermes_ariadne.memory_provider import AriadneMemoryProvider

    store = NoteStore(home)
    store.upsert_conversation("old-1", preview="Planning the Lyon trip", summary="train times")
    store.upsert_conversation("old-2", preview="Bike tyre sizes", summary="27.5x2.10 tubes")
    provider = AriadneMemoryProvider()
    provider.initialize("new", hermes_home=str(home))
    first = provider.prefetch("hello there", session_id="new")
    assert "[old-1]" in first and "[old-2]" in first and "not instructions" in first
    second = provider.prefetch("which tyre tubes again?", session_id="new")
    assert "[old-2]" in second and "[old-1]" not in second
    assert provider.prefetch("unrelated words", session_id="new") == ""


@needs_host
def test_provider_conversation_tool(home):
    from hermes_ariadne.memory_provider import AriadneMemoryProvider

    _archive(
        home, {"old": [_msg("u1", "user", "Nakamura question"), _msg("a1", "assistant", "ok")]}
    )
    NoteStore(home).upsert_conversation("old", preview="Nakamura question", summary="bike")
    provider = AriadneMemoryProvider()
    provider.initialize("new", hermes_home=str(home))

    def call(**args):
        return json.loads(provider.handle_tool_call("ariadne_conversations", args))

    assert [c["session_id"] for c in call(action="list")["conversations"]] == ["old"]
    found = call(action="search", query="Nakamura")
    assert found["conversations"][0]["session_id"] == "old"
    assert found["messages"][0]["session_id"] == "old"
    assert call(action="open", id="old")["conversation"]["summary"] == "bike"
    read = call(action="read", id="old")
    assert [m["content"] for m in read["messages"]] == ["Nakamura question", "ok"]
    event = read["messages"][0]["event_id"]
    assert call(action="zoom", id="old", event_id=event)["content"] == "Nakamura question"
    assert call(action="read", id="missing") == {"ok": False, "error": "not_found"}
    assert call(action="zoom", id="old", event_id="bad")["error"] == "invalid_argument"
    assert call(action="open") == {"ok": False, "error": "invalid_argument"}


@needs_host
def test_engine_indexes_preview_and_published_summary(tmp_path, monkeypatch):
    from test_rolling_engine import DB, Facade, complete
    from test_rolling_selection import POLICY, exchange

    from hermes_ariadne import rolling
    from hermes_ariadne.memory_provider import AriadneMemoryProvider
    from hermes_ariadne.mvp_engine import AriadneEngine

    monkeypatch.setattr(AriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))
    facade, db = Facade(), DB()
    engine = AriadneEngine(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    assert engine.name == "ariadne"

    db.rows = exchange("A")
    complete(engine, db, "A")
    row = NoteStore(tmp_path).get_conversation("s")
    first_user = next(r["content"] for r in exchange("A") if r["role"] == "user")
    assert row["preview"] == " ".join(first_user.split())[:MAX_PREVIEW_CHARS]
    assert row["summary"] is None

    db.rows += exchange("B")
    complete(engine, db, "B")
    assert engine._published is not None and len(facade.calls) == 1
    assert NoteStore(tmp_path).get_conversation("s")["summary"] == "summary 1"

    # A brand-new chat in the same profile sees it and can read the exact originals.
    provider = AriadneMemoryProvider()
    provider.initialize("fresh", hermes_home=str(tmp_path))
    recall = provider.prefetch("hi", session_id="fresh")
    assert "[s]" in recall and "summary 1" in recall
    read = json.loads(
        provider.handle_tool_call("ariadne_conversations", {"action": "read", "id": "s"})
    )
    assert read["messages"][0]["content"] == first_user


@needs_host
def test_engine_index_failure_never_breaks_turn(tmp_path, monkeypatch):
    from test_rolling_engine import DB, Facade, complete
    from test_rolling_selection import POLICY, exchange

    from hermes_ariadne import rolling
    from hermes_ariadne.mvp_engine import AriadneEngine

    monkeypatch.setattr(AriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk")

    monkeypatch.setattr(NoteStore, "upsert_conversation", boom)
    facade, db = Facade(), DB()
    engine = AriadneEngine(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    db.rows = exchange("A") + exchange("B")
    complete(engine, db, "B")
    assert engine._published is not None
