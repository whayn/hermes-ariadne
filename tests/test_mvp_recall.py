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
def test_provider_no_longer_recalls_the_conversation_index(home):
    # #54: the `ariadne` engine's memory view shows every conversation; prefetch
    # recalls notes only (the index stays as data, it is not injected any more).
    from hermes_ariadne.memory_provider import AriadneMemoryProvider

    store = NoteStore(home)
    store.upsert_conversation("old-1", preview="Planning the Lyon trip", summary="train times")
    store.upsert_conversation("old-2", preview="Bike tyre sizes", summary="27.5x2.10 tubes")
    provider = AriadneMemoryProvider()
    provider.initialize("new", hermes_home=str(home))
    assert provider.prefetch("hello there", session_id="new") == ""
    assert provider.prefetch("which tyre tubes again?", session_id="new") == ""
    store.add("Tyre tubes are 27.5x2.10")
    recall = provider.prefetch("which tyre tubes again?", session_id="new")
    assert "27.5x2.10" in recall and "[old-2]" not in recall and "not instructions" in recall


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

    # A brand-new chat can still read the exact originals (the index is no longer
    # recalled: #54 shows conversations in the engine's memory view instead).
    provider = AriadneMemoryProvider()
    provider.initialize("fresh", hermes_home=str(tmp_path))
    assert provider.prefetch("hi", session_id="fresh") == ""
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


# -- review fixes ------------------------------------------------------------------------


def test_store_from_previous_schema_is_migrated_in_place(home):
    import hermes_ariadne.notes as notes

    (home / "ariadne").mkdir(mode=0o700)
    path = home / "ariadne" / "notes.sqlite3"
    conn = sqlite3.connect(path)
    for sql in notes.SCHEMA[:3]:  # schema v1: notes, revisions, index only
        conn.execute(sql)
    conn.execute(f"PRAGMA application_id={notes.APPLICATION_ID}")
    conn.execute("PRAGMA user_version=1")
    conn.execute("INSERT INTO notes VALUES('n1','kept','[]',NULL,'agent',1,1)")
    conn.commit()
    conn.close()
    os.chmod(path, 0o600)
    store = NoteStore(home)
    assert store.conversations()["total"] == 0  # a read migrates instead of failing
    store.upsert_conversation("s1", preview="hello")
    assert store.get("n1")["text"] == "kept"
    assert store.get_conversation("s1")["preview"] == "hello"


@needs_host
@pytest.mark.parametrize(
    "restriction",
    [
        {"data_use": ["no_training"]},
        {"display_metadata": json.dumps({"data_use": ["no_training"]})},
        {"display_metadata": json.dumps({"withheld_prompt": "secret original"})},
    ],
)
def test_reader_refuses_restricted_records(home, restriction):
    from hermes_ariadne.conversations import ArchiveReader, RestrictedEvidence

    _archive(home, {"old": [_msg("u1", "user", "restricted text", **restriction)]})
    reader = ArchiveReader(home)
    with pytest.raises(RestrictedEvidence):
        reader.page("old")
    with pytest.raises(RestrictedEvidence):
        reader.search("restricted")
    event = (
        sqlite3.connect(home / "ariadne" / "events.sqlite3")
        .execute("SELECT event_id FROM events")
        .fetchone()[0]
    )
    with pytest.raises(RestrictedEvidence):
        reader.event("old", event)


@needs_host
def test_provider_reports_restricted_without_content(home):
    from hermes_ariadne.memory_provider import AriadneMemoryProvider

    _archive(home, {"old": [_msg("u1", "user", "restricted text", data_use=["no_training"])]})
    provider = AriadneMemoryProvider()
    provider.initialize("new", hermes_home=str(home))
    out = provider.handle_tool_call("ariadne_conversations", {"action": "read", "id": "old"})
    assert json.loads(out) == {"ok": False, "error": "restricted"}


@needs_host
def test_zoom_recovers_tool_calls_that_read_truncates(home):
    from hermes_ariadne.conversations import MAX_MESSAGE_CHARS, ArchiveReader

    calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "terminal", "arguments": json.dumps({"cmd": "x" * 5000})},
        }
    ]
    _archive(home, {"old": [_msg("a1", "assistant", None, tool_calls=calls)]})
    reader = ArchiveReader(home)
    row = reader.page("old")["messages"][0]
    assert row["tool_calls"] is None and row["tool_calls_truncated"]
    text, offset = "", 0
    while True:
        part = reader.event("old", row["event_id"], char_offset=offset)
        text += part["content"]
        offset += len(part["content"])
        if not part["has_more"]:
            break
    assert "x" * 5000 in text and '"terminal"' in text
    assert len(text) > MAX_MESSAGE_CHARS


@needs_host
def test_preview_skips_restricted_first_message():
    from hermes_ariadne.mvp_engine import _first_user_text

    assert _first_user_text([{"role": "user", "content": "ok"}]) == "ok"
    assert _first_user_text([{"role": "user", "content": "x", "data_use": ["no_training"]}]) is None


@needs_host
def test_engine_index_ignores_completion_from_a_replaced_session(tmp_path, monkeypatch):
    from test_rolling_engine import DB, Facade, complete
    from test_rolling_selection import POLICY, exchange

    from hermes_ariadne import rolling
    from hermes_ariadne.mvp_engine import AriadneEngine

    monkeypatch.setattr(AriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))
    facade, db = Facade(), DB()
    engine = AriadneEngine(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    db.rows = exchange("A") + exchange("B")

    def rebind():
        engine.on_session_reset()
        engine.bind_session_state(DB(), "other")

    facade.hook = rebind  # session replaced while the summary call is in flight
    complete(engine, db, "B")
    assert NoteStore(tmp_path).get_conversation("other") is None


@needs_host
def test_read_and_zoom_keep_tool_result_linkage(home):
    from hermes_ariadne.conversations import ArchiveReader

    calls = [
        {"id": f"c{i}", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        for i in (1, 2)
    ]
    _archive(
        home,
        {
            "old": [
                _msg("a1", "assistant", None, tool_calls=calls),
                _msg("t1", "tool", "first result", tool_call_id="c1", name="t"),
                _msg("t2", "tool", "second result", tool_call_id="c2", name="t"),
            ]
        },
    )
    reader = ArchiveReader(home)
    rows = reader.page("old")["messages"]
    assert [(r.get("tool_call_id"), r.get("name")) for r in rows[1:]] == [("c1", "t"), ("c2", "t")]
    zoom = reader.event("old", rows[2]["event_id"])
    assert (zoom["tool_call_id"], zoom["name"]) == ("c2", "t")


@needs_host
@pytest.mark.parametrize("name", ["profile?test", "pro#file", "50%25"])
def test_reader_handles_uri_reserved_characters_in_home(tmp_path, name):
    from hermes_ariadne.conversations import ArchiveReader

    home = tmp_path / name
    home.mkdir(mode=0o700)
    _archive(home, {"old": [_msg("u1", "user", "hello")]})
    before = sorted(p.name for p in tmp_path.iterdir())
    assert ArchiveReader(home).page("old")["messages"][0]["content"] == "hello"
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # read-only: no stray files


@needs_host
@pytest.mark.parametrize(
    "metadata",
    [{"data_use": ["no_training"]}, {"withheld_prompt": "original"}],
)
def test_engine_preview_admits_the_canonical_record(tmp_path, monkeypatch, metadata):
    from test_rolling_engine import DB, Facade, complete
    from test_rolling_selection import POLICY, exchange

    from hermes_ariadne import rolling
    from hermes_ariadne.mvp_engine import AriadneEngine

    monkeypatch.setattr(AriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY))
    facade, db = Facade(), DB()
    engine = AriadneEngine(llm=facade, settings=rolling.RollingSettings(trigger_bytes=1))
    engine.update_model("gpt-test", 100000, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "s")
    engine.on_session_start("s", hermes_home=tmp_path)
    db.rows = exchange("A")
    first = next(r for r in db.rows if r["role"] == "user")
    first["display_metadata"] = json.dumps(metadata)  # durable only, not in the callback
    complete(engine, db, "A")
    db.rows += exchange("B")
    complete(engine, db, "B")
    row = NoteStore(tmp_path).get_conversation("s")
    assert row is None or row["preview"] is None


def test_v1_store_that_already_has_the_index_migrates(home):
    import hermes_ariadne.notes as notes

    (home / "ariadne").mkdir(mode=0o700)
    path = home / "ariadne" / "notes.sqlite3"
    conn = sqlite3.connect(path)
    for sql in notes.SCHEMA:  # pre-release revisions created the table at version 1
        conn.execute(sql)
    conn.execute(f"PRAGMA application_id={notes.APPLICATION_ID}")
    conn.execute("PRAGMA user_version=1")
    conn.execute(
        "INSERT INTO conversations (session_id, title, preview, summary, origin, created, "
        "updated) VALUES ('old','t','p','s','live',1,1)"
    )
    conn.commit()
    conn.close()
    os.chmod(path, 0o600)
    assert NoteStore(home).get_conversation("old")["preview"] == "p"
