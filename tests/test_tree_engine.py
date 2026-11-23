"""The ``ariadne`` context engine on the profile-wide memory tree (issue #54, phase D)."""

import copy
import hashlib
import json
import os

import pytest

if not os.environ.get("HERMES_SOURCE"):
    pytest.skip("HERMES_SOURCE required for the ContextEngine ABC", allow_module_level=True)

from agent.turn_context import compose_user_api_content  # noqa: E402
from test_rolling_selection import POLICY, assembled, exchange, restored  # noqa: E402
from test_tree_builder import FakeLlm  # noqa: E402

from hermes_ariadne import profile_tree as pt  # noqa: E402
from hermes_ariadne import tree_engine  # noqa: E402
from hermes_ariadne.store import EventStore, canonical_json  # noqa: E402
from hermes_ariadne.tree import Part, View  # noqa: E402
from hermes_ariadne.tree_builder import TreeBuilder, TreeSettings  # noqa: E402
from hermes_ariadne.tree_engine import TreeAriadneEngine  # noqa: E402
from hermes_ariadne.tree_store import TreeStore  # noqa: E402

BUDGET = 100000


@pytest.fixture(autouse=True)
def _isolated_registry():
    tree_engine._reset_registry()
    yield
    tree_engine._reset_registry()


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o755)
    os.chmod(home, 0o755)
    return home


class DB:
    """SessionDB shape the engine reads: messages, plus ``get_session`` (source/title)."""

    def __init__(self):
        self.rows = []
        self.sessions = {}

    def get_messages(self, session_id, *, include_inactive):
        rows = copy.deepcopy([r for r in self.rows if r.get("session_id") == session_id])
        return rows if include_inactive else [r for r in rows if r.get("active", 1)]

    def get_session(self, session_id):
        found = self.sessions.get(session_id)
        return dict(found) if found else None


def rows_for(session, tag, *, complete=True, **extra):
    """``exchange(tag)`` rows moved to ``session`` (uids stay unique by tag)."""
    out = exchange(tag, complete=complete)
    for row in out:
        row["session_id"] = session
        row.update(extra)
    return out


class World:
    def __init__(self, home, monkeypatch, settings=None, llm=None):
        monkeypatch.setattr(
            TreeAriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY)
        )
        self.home, self.db = home, DB()
        self.llm = llm or FakeLlm()
        self.settings = settings or TreeSettings()

    def engine(self, session, source="cli", title=None, *, base=None):
        if source is not None:  # None: the session is unknown to SessionDB
            self.db.sessions.setdefault(session, {"source": source, "title": title})
        engine = (
            base.clone_for_agent()
            if base is not None
            else TreeAriadneEngine(llm=self.llm, tree_settings=self.settings, background=False)
        )
        engine.update_model("gpt-test", BUDGET, "https://invalid.test", "", "fixture", "fixture")
        engine.bind_session_state(self.db, session)
        engine.on_session_start(session, hermes_home=self.home)
        return engine

    def turn(self, engine, session, tag, **extra):
        self.db.rows += rows_for(session, tag, **extra)
        self.complete(engine, session, tag)

    def complete(self, engine, session, tag):
        engine.on_turn_complete(
            restored(self.history_rows(session)),
            usage={},
            turn_id=tag,
            interrupted=False,
            failed=False,
            turn_exit_reason="text_response(finish_reason=stop)",
        )

    def history_rows(self, session):
        return [r for r in self.db.rows if r["session_id"] == session]

    def build(self, engine):
        builder = engine._shared.builder
        for _ in range(200):
            if not builder.step()["progress"]:
                return
        raise AssertionError("tree did not converge")

    def ask(self, session, tag):
        """Persist the incoming user message of exchange ``tag``."""
        self.db.rows += rows_for(session, tag, complete=False)[:1]

    def select(self, engine, session, tag, budget=BUDGET, *, request=None, history=None):
        history = history or restored(self.history_rows(session))
        incoming = next(m for m in history if m["message_uid"] == tag + "u")
        request = request or assembled(history)
        before = copy.deepcopy(request)
        frozen = copy.deepcopy(history)
        result = engine.select_context(
            request, conversation_messages=history, incoming_message=incoming, budget_tokens=budget
        )
        assert request == before and history == frozen  # never mutated
        if result is not None:
            result[-1]["content"] = "mutated copy"
            assert request == before
        return result, request

    def store(self):
        return TreeStore(self.home)


def retired(result, request):
    """How many conversation messages the candidate replaced (prefix is 2 messages)."""
    return len(request) - (len(result) - 1)


def assert_shape(result, request):
    assert result[:2] == request[:2]
    view = result[2]
    assert view["role"] == "assistant" and set(view) == {"role", "content"}
    assert view["content"].startswith(pt.VIEW_DOC)
    count = retired(result, request)
    tail = request[2 + count :]
    assert result[3:-1] == tail[:-1]
    assert result[3]["role"] == "user"  # the view is always followed by a user turn
    return count


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_fresh_chat_with_history_gets_the_view_after_the_host_prefix(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a", title="Bikes")
    world.turn(a, "chat-a", "A")
    world.turn(a, "chat-a", "B")
    world.build(a)
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    result, request = world.select(b, "chat-b", "N")
    assert result is not None, b.get_status()
    assert assert_shape(result, request) == 0
    text = result[2]["content"]
    tag = world.store().sessions()["chat-a"].tag
    assert f"{tag} = Bikes" in text and "question A" in text and "answer B" in text
    assert len(result) == len(request) + 1
    status = b.get_status()["selection"]
    assert status["status"] == "selected" and status["retired_count"] == 0
    assert status["provider_fit_verified"] is False


def test_empty_profile_declines_and_the_host_sends_the_original(home, monkeypatch):
    world = World(home, monkeypatch)
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    result, _ = world.select(b, "chat-b", "N")
    assert result is None
    assert b.get_status()["selection"]["reason"] == "empty_tree"


def test_own_messages_retire_only_when_built_and_at_exchange_boundaries(home, monkeypatch):
    world = World(home, monkeypatch, TreeSettings(tail_bytes=0))
    a = world.engine("chat-a")
    for tag in "ABCD":
        world.turn(a, "chat-a", tag)
    world.ask("chat-a", "E")
    result, request = world.select(a, "chat-a", "E")
    # Nothing built: nothing can be replaced, and every line would repeat a verbatim
    # message of this chat, so there is nothing to show.
    assert result is None
    assert a.get_status()["selection"]["reason"] == "nothing_to_show"

    world.build(a)
    result, request = world.select(a, "chat-a", "E")
    count = assert_shape(result, request)
    history = restored(world.history_rows("chat-a"))
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    assert count == starts[-2]  # previous and current exchange stay verbatim
    assert result[3:-1] == request[2 + count : -1]
    assert "question A" in result[2]["content"]  # the retired exchange is in the view
    assert "question D" not in result[2]["content"]  # the verbatim one is not repeated


def test_a_partly_built_conversation_retires_only_the_covered_prefix(home, monkeypatch):
    world = World(home, monkeypatch, TreeSettings(tail_bytes=0))
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    world.turn(a, "chat-a", "B")
    world.build(a)
    world.turn(a, "chat-a", "C")  # appended, not built: WAIT
    world.turn(a, "chat-a", "D")
    world.ask("chat-a", "E")
    result, request = world.select(a, "chat-a", "E")
    count = assert_shape(result, request)
    assert count == 8  # A and B are built; unbuilt C stops retirement at its start
    store = world.store()
    positions = store.positions(
        "chat-a",
        [
            hashlib.sha256(canonical_json(r)).hexdigest()
            for r in world.history_rows("chat-a")[:count]
        ],
    )
    nodes = store.nodes()
    assert all(Part(0, p) in nodes for p in positions.values())


def test_a_tool_round_in_the_current_exchange_keeps_its_complete_group(home, monkeypatch):
    world = World(home, monkeypatch, TreeSettings(tail_bytes=0))
    a = world.engine("chat-a")
    for tag in "ABC":
        world.turn(a, "chat-a", tag)
    world.build(a)
    world.db.rows += rows_for("chat-a", "D", complete=False)  # user, call, result
    result, request = world.select(a, "chat-a", "D")
    count = assert_shape(result, request)
    assert count == 8 and request[-1]["role"] == "tool"
    assert [m["role"] for m in result[-3:]] == ["user", "assistant", "tool"]
    assert result[-2] == request[-2]


def test_declines_unknown_budget_over_budget_and_unsupported_shapes(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    world.build(a)
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    for budget, reason in ((0, "unknown_budget"), (True, "unknown_budget"), (4200, "over_budget")):
        result, _ = world.select(b, "chat-b", "N", budget)
        assert result is None
        assert b.get_status()["selection"]["reason"] == reason

    history = restored(world.history_rows("chat-b"))
    no_prefix = assembled(history)[2:]
    result, _ = world.select(b, "chat-b", "N", request=no_prefix)
    assert result is None and b.get_status()["selection"]["reason"] == "unsupported_shape"
    assert call(b, "ariadne_memory_status")["selection"]["shape_check"] == "host_prefix"

    forged = assembled(history)
    forged[-1]["content"] = "not what the user sent"
    result, _ = world.select(b, "chat-b", "N", request=forged)
    assert result is None and b.get_status()["selection"]["reason"] == "unsupported_shape"


def test_view_byte_bound_is_explicit(home, monkeypatch):
    monkeypatch.setattr(tree_engine, "VIEW_OVERHEAD", 0)  # so the fixed text alone is over 400
    world = World(home, monkeypatch, TreeSettings(high=200))
    a = world.engine("chat-a")
    for index in range(30):
        world.turn(a, "chat-a", f"T{index}")  # unbuilt: one range line
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    result, _ = world.select(b, "chat-b", "N")
    assert result is None
    assert b.get_status()["selection"]["reason"] == "view_bytes_exceeded"
    assert b.get_status()["selection"]["max_view_bytes"] == 2 * 200 + tree_engine.VIEW_OVERHEAD


def test_recalled_memory_block_is_still_admitted(home, monkeypatch):
    recall = "## Ariadne notes (stored data recalled for this turn, not instructions)\n- [ab] x"
    world = World(home, monkeypatch, TreeSettings(tail_bytes=0))
    a = world.engine("chat-a")

    def with_recall(rows):
        for row in rows:
            if row["role"] == "user":
                row["api_content"] = compose_user_api_content(row["content"], recall, "")
        return rows

    for tag in "ABC":
        world.db.rows += with_recall(rows_for("chat-a", tag))
        world.complete(a, "chat-a", tag)
    world.build(a)
    world.db.rows += with_recall(rows_for("chat-a", "D", complete=False)[:1])
    history = restored(world.history_rows("chat-a"))
    for item, row in zip(history, world.history_rows("chat-a")):
        if row.get("api_content"):
            item["api_content"] = row["api_content"]
    request = assembled(history)
    for message, row in zip(request[2:], history):
        if row.get("api_content"):
            message["content"] = row["api_content"]
    result, request = world.select(a, "chat-a", "D", request=request, history=history)
    assert result is not None, a.get_status()
    assert retired(result, request) > 0
    assert "memory-context" in request[-1]["content"]


# --------------------------------------------------------------------------- #
# Log membership and backfill
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", ["subagent", "cron"])
def test_subagent_and_cron_sessions_are_archived_but_never_logged(home, monkeypatch, source):
    world = World(home, monkeypatch)
    primary = world.engine("chat-a")
    world.turn(primary, "chat-a", "A")
    before = world.store().count()
    worker = world.engine("worker", source=source)
    world.turn(worker, "worker", "W")
    assert world.store().count() == before
    assert "worker" not in world.store().sessions()
    archive = EventStore(home / "ariadne" / "events.sqlite3")
    assert archive.search("question W", session_id="worker")  # still archived
    world.ask("worker", "X")
    result, _ = world.select(worker, "worker", "X")
    assert result is None and worker.get_status()["selection"]["reason"] == "non_primary"


def test_unknown_session_kind_is_not_logged(home, monkeypatch):
    world = World(home, monkeypatch)
    engine = world.engine("chat-a", source=None)
    world.turn(engine, "chat-a", "A")
    assert world.store().count() == 0
    world.db.sessions["chat-a"] = {"source": "desktop"}  # the host created it meanwhile
    world.complete(engine, "chat-a", "A")
    assert world.store().count() == 4


def test_first_start_backfills_the_archive_without_subagent_or_cron(home, monkeypatch):
    archive = EventStore(home / "ariadne" / "events.sqlite3")
    archive.ingest("old-chat", rows_for("old-chat", "O"), origin="hermes_db")
    archive.ingest("old-sub", rows_for("old-sub", "S"), origin="hermes_db")
    archive.ingest("old-cron", rows_for("old-cron", "C"), origin="hermes_db")
    archive.ingest("import-abc", [{"message_uid": "i1", "role": "user", "content": "imp"}],
                   origin="import")  # fmt: skip
    archive.ingest("deleted", rows_for("deleted", "D"), origin="hermes_db")  # unknown: out
    world = World(home, monkeypatch)
    world.db.sessions.update(
        {
            "old-chat": {"source": "telegram", "title": "Old", "parent_session_id": "x"},
            "old-sub": {"source": "subagent", "parent_session_id": "old-chat"},
            "old-cron": {"source": "cron"},
        }
    )
    world.engine("new-chat")
    store = world.store()
    assert set(store.sessions()) == {"old-chat", "import-abc"}
    assert store.backfilled()
    assert [e.origin for e in store.entries(0, store.count())].count("import") == 1
    assert store.sessions()["old-chat"].title == "Old"
    count = store.count()
    world.engine("another")  # never twice
    assert store.count() == count


def test_a_restart_recovers_archived_rows_that_were_never_appended(home, monkeypatch):
    world = World(home, monkeypatch)
    world.engine("chat-a")  # first start: backfill done, marker persisted
    assert world.store().backfilled() and world.store().count() == 0
    # a crash between archiving and appending left an abandoned primary session behind
    world.db.sessions["lost"] = {"source": "telegram", "title": "Lost"}
    world.db.sessions["lost-sub"] = {"source": "subagent"}
    archive = EventStore(home / "ariadne" / "events.sqlite3")
    archive.ingest("lost", rows_for("lost", "L"), origin="hermes_db")
    archive.ingest("lost-sub", rows_for("lost-sub", "M"), origin="hermes_db")
    tree_engine._reset_registry()  # a new process
    world.engine("chat-b")
    store = world.store()
    assert set(store.sessions()) == {"lost"}
    assert [e.message_uid for e in store.entries(0, store.count())] == ["Lu", "Lo", "Lt", "La"]
    assert store.sessions()["lost"].title == "Lost"
    count = store.count()
    tree_engine._reset_registry()
    world.engine("chat-c")  # reconciling again appends nothing
    assert store.count() == count


def test_reconcile_does_not_skip_a_session_unknown_at_the_time(home, monkeypatch):
    world = World(home, monkeypatch)
    world.engine("chat-a")
    archive = EventStore(home / "ariadne" / "events.sqlite3")
    archive.ingest("late", rows_for("late", "L"), origin="hermes_db")  # state.db lags
    tree_engine._reset_registry()
    world.engine("chat-b")
    assert world.store().count() == 0  # not positively classified: not admitted
    world.db.sessions["late"] = {"source": "cli"}
    tree_engine._reset_registry()
    world.engine("chat-c")
    assert set(world.store().sessions()) == {"late"}  # the next start recovers it


def test_a_session_archived_during_classification_is_never_admitted(home, monkeypatch):
    archive = EventStore(home / "ariadne" / "events.sqlite3")
    archive.ingest("old-chat", rows_for("old-chat", "O"), origin="hermes_db")
    world = World(home, monkeypatch)
    world.db.sessions["old-chat"] = {"source": "cli"}
    original = tree_engine._classify_archive

    def racing(*args):
        out = original(*args)
        world.db.sessions["worker"] = {"source": "subagent"}
        archive.ingest("worker", rows_for("worker", "W"), origin="hermes_db")  # after the snapshot
        return out

    monkeypatch.setattr(tree_engine, "_classify_archive", racing)
    world.engine("new-chat")
    store = world.store()
    assert set(store.sessions()) == {"old-chat"} and store.backfilled()
    tree_engine._reset_registry()
    world.engine("chat-b")  # a later reconcile classifies it (non-primary) and still skips it
    assert "worker" not in store.sessions()


def test_appends_wait_for_the_backfill_then_catch_up(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    shared = a._shared
    shared.backfill = "running"  # as while the background backfill thread works
    world.turn(a, "chat-a", "A")
    assert world.store().count() == 0
    shared.backfill = "done"
    world.turn(a, "chat-a", "B")
    assert [e.message_uid for e in world.store().entries(0, 8)][:4] == ["Au", "Ao", "At", "Aa"]


def test_the_backfill_catches_up_sessions_deferred_while_it_ran(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    shared = a._shared
    shared.backfill = "running"
    world.turn(a, "chat-a", "A")  # archived, deferred (the session may never speak again)
    assert world.store().count() == 0 and shared.deferred == {"chat-a"}
    shared._backfill(world.db)
    assert shared.backfill == "done" and not shared.deferred
    assert [e.message_uid for e in world.store().entries(0, 4)] == ["Au", "Ao", "At", "Aa"]


def test_turns_are_appended_in_order_and_idempotently(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    world.complete(a, "chat-a", "A")  # same rows again: nothing new
    store = world.store()
    entries = store.entries(0, store.count())
    assert [e.message_uid for e in entries] == ["Au", "Ao", "At", "Aa"]
    assert {e.origin for e in entries} == {"live"} and len(entries) == 4


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def test_import_is_appended_with_origin_import_and_returns_an_estimate(home, monkeypatch):
    from hermes_ariadne.memory_provider import AriadneMemoryProvider

    world = World(home, monkeypatch)
    world.engine("chat-a")
    provider = AriadneMemoryProvider()
    provider.initialize("chat-a", hermes_home=str(home), agent_context="primary")
    messages = [
        {"role": "user", "content": "short question", "timestamp": "2024-01-02T03:04:05Z"},
        {"role": "assistant", "content": "long answer " * 100, "timestamp": "2024"},
        {"role": "user", "content": "thanks"},
    ]
    result = json.loads(
        provider.handle_tool_call(
            "ariadne_import",
            {"key": "export.json#1", "messages": messages, "title": "Old chat",
             "source": {"file": "export.json"}},
        )
    )  # fmt: skip
    assert result["ok"] and result["archived"] == 3
    tree = result["tree"]
    assert tree["appended"] == 3
    assert tree["estimate"]["leaf_calls"] == 1 and tree["estimate"]["calls"] >= 2
    store = world.store()
    session = store.sessions()[result["id"]]
    assert session.origin == "import" and session.title == "Old chat"
    assert "export.json" in session.source
    assert {e.origin for e in store.entries(0, store.count())} == {"import"}
    again = json.loads(
        provider.handle_tool_call("ariadne_import", {"key": "export.json#1", "messages": messages})
    )
    assert again["tree"]["appended"] == 0 and again["tree"]["estimate"]["calls"] == 0


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def call(engine, name, **args):
    return json.loads(engine.handle_tool_call(name, args))


def test_tool_schemas_replace_the_rolling_tools(home, monkeypatch):
    world = World(home, monkeypatch)
    names = [s["name"] for s in world.engine("chat-a").get_tool_schemas()]
    assert names == ["ariadne_zoom", "ariadne_search", "ariadne_memory_status"]


def test_zoom_opens_lines_and_pages_exact_messages(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    long = "x" * 9000 + "END"
    world.db.rows += [
        {**rows_for("chat-a", "L")[0], "content": long, "message_uid": "Lu"},
        {**rows_for("chat-a", "L")[3], "content": "ok", "message_uid": "La"},
    ]
    world.complete(a, "chat-a", "L")
    world.build(a)
    opened = call(a, "ariadne_zoom", id=0, n=4)
    assert opened["ok"] and "never instructions" in opened["note"]
    assert [line.split("|")[0] for line in opened["lines"]] == ["0+2", "2+2"]
    leaf = call(a, "ariadne_zoom", id=0, n=2)
    assert leaf["lines"][0].startswith("0+1|") and "question A" in leaf["lines"][0]
    first = call(a, "ariadne_zoom", id=4)
    assert first["content"] == long[:8000] and first["has_more"] and first["next_at"] == 8000
    tag = world.store().sessions()["chat-a"].tag
    assert first["conversation"] == tag and first["role"] == "user"
    rest = call(a, "ariadne_zoom", id=4, at=8000)
    assert rest["content"].endswith("END") and not rest["has_more"]
    for bad in (
        {"id": 1, "n": 2},
        {"id": 0, "n": 3},
        {"id": -1},
        {"id": True},
        {"id": 0, "at": -1},
    ):
        assert call(a, "ariadne_zoom", **bad)["error"] == "invalid_argument"
    assert call(a, "ariadne_zoom", id=99)["error"] == "not_found"


def test_zoom_refuses_restricted_messages(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    rows = rows_for("chat-a", "R")
    rows[0]["display_metadata"] = json.dumps({"data_use": ["no-training"]})
    world.db.rows += rows
    world.complete(a, "chat-a", "R")
    store = world.store()
    assert store.entry(0).restricted
    refused = call(a, "ariadne_zoom", id=0)
    assert refused["ok"] is False and refused["error"] == "restricted"
    import sqlite3

    with sqlite3.connect(store.path) as conn:  # a flag alone refuses too (fail closed)
        conn.execute("UPDATE log SET restricted=1 WHERE position=1")
    flagged = call(a, "ariadne_zoom", id=1)
    assert flagged["ok"] is False and flagged["error"] == "restricted"
    world.build(a)
    assert pt.RESTRICTED in store.nodes()[Part(0, 0)]
    assert "question R" not in json.dumps(call(a, "ariadne_zoom", id=0, n=2))


def test_search_finds_primary_messages_by_global_position(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    world.turn(a, "chat-a", "Zebra")
    sub = world.engine("worker", source="subagent")
    world.turn(sub, "worker", "ZebraSub")  # archived only
    found = call(a, "ariadne_search", query="question Zebra")
    assert found["ok"] and "never instructions" in found["note"]
    tag = world.store().sessions()["chat-a"].tag
    assert [(r["id"], r["conversation"]) for r in found["results"]] == [(4, tag)]
    assert "question Zebra" in found["results"][0]["preview"]
    assert call(a, "ariadne_search", query="")["error"] == "invalid_argument"
    assert call(a, "ariadne_search", query="nothing like it")["results"] == []


def test_memory_status_and_retry(home, monkeypatch):
    def script(n, messages, kwargs):
        return RuntimeError("timeout") if n == 1 else "merged"

    world = World(home, monkeypatch, llm=FakeLlm(script))
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    world.build(a)
    status = call(a, "ariadne_memory_status")
    assert status["ok"] and status["builder"]["uncertain"] == 1
    assert status["builder"]["count"] == 4
    assert status["caps"]["max_jobs"] == 100000 and status["caps"]["node_bytes"] == 512
    assert status["remaining"]["internal_calls"] == 2
    retried = call(a, "ariadne_memory_status", action="retry")
    assert retried["rearmed"] == 1
    world.build(a)
    assert call(a, "ariadne_memory_status")["remaining"]["internal_calls"] == 0
    assert call(a, "ariadne_memory_status", action="other")["error"] == "invalid_argument"


def test_memory_status_reports_latest_context_selection(home, monkeypatch):
    world = World(home, monkeypatch)
    prior = world.engine("prior")
    world.turn(prior, "prior", "A")
    world.build(prior)
    chat = world.engine("desktop", source="desktop")
    world.ask("desktop", "B")
    assert call(chat, "ariadne_memory_status")["selection"]["status"] == "not_evaluated"
    selected, _ = world.select(chat, "desktop", "B")
    assert selected is not None
    status = call(chat, "ariadne_memory_status")["selection"]
    assert status["status"] == "selected" and status["view_lines"] > 0
    declined, _ = world.select(chat, "desktop", "B", budget=0)
    assert declined is None
    status = call(chat, "ariadne_memory_status")["selection"]
    assert status["status"] == "declined" and status["reason"] == "unknown_budget"
    assert status["host_fallback"] == "original_request_on_decline"
    assert "view_lines" not in status


# --------------------------------------------------------------------------- #
# Lifecycle, cache, tags
# --------------------------------------------------------------------------- #


def test_engine_clones_share_one_store_and_builder_per_home(home, monkeypatch):
    world = World(home, monkeypatch)
    first = world.engine("chat-a")
    second = world.engine("chat-b", base=first)
    assert second is not first and second.name == "ariadne"
    assert second._shared is first._shared
    assert isinstance(first._shared.builder, TreeBuilder)
    assert tree_engine.shared_home(home) is first._shared


def test_background_builder_runs_after_a_turn_without_blocking(home, monkeypatch):
    import threading

    gate = threading.Event()

    def script(n, messages, kwargs):
        gate.wait(5)
        return "merged"

    monkeypatch.setattr(
        TreeAriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY)
    )
    db = DB()
    db.sessions["chat-a"] = {"source": "cli"}
    llm = FakeLlm(script)
    engine = TreeAriadneEngine(llm=llm, tree_settings=TreeSettings())
    engine.update_model("gpt-test", BUDGET, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "chat-a")
    engine.on_session_start("chat-a", hermes_home=home)
    import time

    for _ in range(100):  # the one-time backfill runs on its own thread
        if engine._shared.backfill == "done":
            break
        time.sleep(0.05)
    assert engine._shared.backfill == "done"
    db.rows += rows_for("chat-a", "A")
    started = time.monotonic()
    engine.on_turn_complete(restored(db.rows), usage={}, turn_id="A")  # returns at once
    assert time.monotonic() - started < 2.0  # the summary call is still blocked
    assert TreeStore(home).count() == 4
    gate.set()
    builder = engine._shared.builder
    for _ in range(100):
        if Part(2, 0) in TreeStore(home).nodes():
            break
        time.sleep(0.05)
    assert Part(2, 0) in TreeStore(home).nodes()
    engine.on_session_end("chat-a", [])
    assert builder._stopped


def test_cache_prefix_is_stable_across_turns_between_merge_batches(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    world.turn(a, "chat-a", "B")
    world.build(a)
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    first, _ = world.select(b, "chat-b", "N")
    world.db.rows += rows_for("chat-b", "N")[1:]
    world.complete(b, "chat-b", "N")
    world.turn(a, "chat-a", "C")  # another conversation moves on meanwhile
    world.ask("chat-b", "M")
    second, _ = world.select(b, "chat-b", "M")
    assert second is not None, b.get_status()
    assert b.get_status()["selection"]["cache_checked"] is True
    old, new = pt.history_lines(first[2]["content"]), pt.history_lines(second[2]["content"])
    assert new[: len(old)] == old and len(new) > len(old)


def test_lengthened_tags_are_the_same_in_leaves_and_legend(home, monkeypatch):
    world = World(home, monkeypatch)
    store = world.store()
    store.prepare()
    import sqlite3

    clash = pt.conversation_tag("chat-a")
    with sqlite3.connect(store.path) as conn:  # an earlier conversation already owns the tag
        conn.execute(
            "INSERT INTO sessions VALUES('other', ?, 'Other', NULL, 'live', 0, 0)", (clash,)
        )
    store.mark_backfilled()
    a = world.engine("chat-a", title="Mine")
    world.turn(a, "chat-a", "A")
    world.build(a)
    tag = store.sessions()["chat-a"].tag
    assert len(tag) == 12 and tag.startswith(clash)
    assert store.nodes()[Part(0, 0)].startswith(tag + " ")
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    result, _ = world.select(b, "chat-b", "N")
    text = result[2]["content"]
    assert f"\n{tag} = Mine (" in text


def test_leaf_source_takes_the_stored_tag():
    entry = pt.LogEntry(0, "s", "e", "u", "user", None, "live", False)
    message = {"role": "user", "content": "hi"}
    assert pt.leaf_source(entry, message).startswith(pt.conversation_tag("s") + " ")
    assert pt.leaf_source(entry, message, "abcdef0123456789").startswith("abcdef0123456789 ?")
    for bad in ("ABCDEF01", "abc", "abcdef012", 5):
        with pytest.raises(ValueError):
            pt.leaf_source(entry, message, bad)
    legend = {"abcdef0123456789": pt.LegendEntry("T", None, None)}
    nodes = {Part(0, 0): "abcdef0123456789 ? user: hi"}
    out = pt.render_view(View((Part(0, 0),), 0), nodes, legend, exclude_positions=frozenset())
    assert out.endswith("abcdef0123456789 = T (? .. ?)\n" + pt.CONVERSATIONS_CLOSE)


# --------------------------------------------------------------------------- #
# Unbuilt history stays selectable: runs collapse into one range line
# --------------------------------------------------------------------------- #


def _range_lines(result):
    return [line for line in pt.history_lines(result[2]["content"]) if ".." in line.split("|")[0]]


def test_unbuilt_history_collapses_into_a_range_line_and_selection_succeeds(home, monkeypatch):
    world = World(home, monkeypatch, TreeSettings(high=200))
    a = world.engine("chat-a")
    for index in range(30):
        world.turn(a, "chat-a", f"T{index}")  # 120 messages, none built
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    result, _ = world.select(b, "chat-b", "N")
    assert result is not None, b.get_status()
    lines = pt.history_lines(result[2]["content"])
    assert len(lines) <= 2 and _range_lines(result) == [
        "0..119|(120 messages not summarized yet: search, or zoom an id)"
    ]
    selection = b.get_status()["selection"]
    assert selection["status"] == "selected" and selection["view_parts"] == 120
    assert selection["rendered_bytes"] <= selection["max_view_bytes"]


def test_zoom_accepts_any_position_inside_a_collapsed_run(home, monkeypatch):
    world = World(home, monkeypatch, TreeSettings(high=200))
    a = world.engine("chat-a")
    for index in range(8):
        world.turn(a, "chat-a", f"T{index}")
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    result, _ = world.select(b, "chat-b", "N")
    assert _range_lines(result) == ["0..31|(32 messages not summarized yet: search, or zoom an id)"]
    exact = call(b, "ariadne_zoom", id=12)  # one message, in full
    assert exact["ok"] and exact["id"] == 12 and "question T3" in exact["content"]
    pair = call(b, "ariadne_zoom", id=12, n=2)  # an aligned part inside the run
    assert [line.split("|")[0] for line in pair["lines"]] == ["12+1", "13+1"]
    assert all(line.endswith(pt.WAIT) for line in pair["lines"])
    deep = call(b, "ariadne_zoom", id=16, n=16)
    assert [line.split("|")[0] for line in deep["lines"]] == ["16+8", "24+8"]
    assert call(b, "ariadne_zoom", id=31)["ok"]
    assert call(b, "ariadne_zoom", id=32)["error"] == "not_found"
    found = call(b, "ariadne_search", query="question T3")
    assert found["ok"] and [r["id"] for r in found["results"]] == [12]


def test_building_inside_a_run_changes_the_view_at_that_run_only(home, monkeypatch):
    world = World(home, monkeypatch, TreeSettings(high=10**6))
    a = world.engine("chat-a")
    for tag in "ABC":
        world.turn(a, "chat-a", tag)
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    first, _ = world.select(b, "chat-b", "N")
    assert _range_lines(first) == ["0..11|(12 messages not summarized yet: search, or zoom an id)"]
    world.build(a)
    again, _ = world.select(b, "chat-b", "N")
    assert again is not None and _range_lines(again) == []
    assert b.get_status()["selection"]["cache_checked"] is False  # the run itself changed


def test_the_view_stays_within_its_bound_while_ten_thousand_messages_are_unbuilt(home, monkeypatch):
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "A")
    count = 10_000
    entries = [
        {
            "event_id": hashlib.sha256(f"imp-{i}".encode()).hexdigest(),
            "message_uid": f"imp-{i}",
            "role": "user" if i % 2 == 0 else "assistant",
            "date": "2026-01-01",
            "restricted": False,
        }
        for i in range(count)
    ]
    world.store().append("imported-chat", entries, origin="import", title="Imported")
    b = world.engine("chat-b")
    world.ask("chat-b", "N")
    result, _ = world.select(b, "chat-b", "N")
    assert result is not None, b.get_status()
    total = world.store().count()
    assert total == count + 4
    assert _range_lines(result) == [
        f"0..{total - 1}|({total} messages not summarized yet: search, or zoom an id)"
    ]
    selection = b.get_status()["selection"]
    assert selection["rendered_bytes"] <= selection["max_view_bytes"]
    again, _ = world.select(b, "chat-b", "N")  # nothing changed: same view, prefix holds
    assert again is not None and again[2] == result[2]
    assert b.get_status()["selection"]["cache_checked"] is True


def test_the_retirement_count_is_pinned_within_one_tool_loop(home, monkeypatch):
    world = World(home, monkeypatch, TreeSettings(tail_bytes=0))
    a = world.engine("chat-a")
    for tag in "AB":
        world.turn(a, "chat-a", tag)
    world.build(a)
    for tag in "CD":
        world.turn(a, "chat-a", tag)  # appended, not built yet
    world.ask("chat-a", "E")
    first, request = world.select(a, "chat-a", "E")
    assert assert_shape(first, request) == 8  # A and B retire; C and D wait for their summaries
    world.build(a)  # the builder catches up between two calls of the same tool loop
    second, request = world.select(a, "chat-a", "E")
    assert assert_shape(second, request) == 8  # same incoming message: the count does not move
    assert second[2] == first[2]  # the very same view: the cache prefix did not move
    assert a.get_status()["selection"]["cache_checked"] is True
    # a tool round of the same exchange keeps it too
    world.db.rows += rows_for("chat-a", "E", complete=False)[1:]
    history = restored(world.history_rows("chat-a"))
    third, request = world.select(a, "chat-a", "E", history=history)
    assert assert_shape(third, request) == 8
    # the next user message is a new loop: now C and D may retire
    world.db.rows += rows_for("chat-a", "E")[3:]
    world.complete(a, "chat-a", "E")
    world.build(a)
    world.ask("chat-a", "F")
    fourth, request = world.select(a, "chat-a", "F")
    assert assert_shape(fourth, request) > 8


# --------------------------------------------------------------------------- #
# Waiting for in-flight summaries: process exit only
# --------------------------------------------------------------------------- #


def _live_engine(home, monkeypatch, llm, settings):
    """A background engine (the real worker) whose one-time backfill has finished."""
    import time

    monkeypatch.setattr(
        TreeAriadneEngine, "_bound_reasoning_policy", staticmethod(lambda *a: POLICY)
    )
    db = DB()
    db.sessions["chat-a"] = {"source": "cli"}
    engine = TreeAriadneEngine(llm=llm, tree_settings=settings)
    engine.update_model("gpt-test", BUDGET, "https://invalid.test", "", "fixture", "fixture")
    engine.bind_session_state(db, "chat-a")
    engine.on_session_start("chat-a", hermes_home=home)
    for _ in range(100):
        if engine._shared.backfill == "done":
            break
        time.sleep(0.05)
    assert engine._shared.backfill == "done"
    return engine, db


def _record_stops(monkeypatch):
    stops = []
    real = TreeBuilder.stop

    def recording(self, timeout=5.0):
        stops.append(timeout)
        return real(self, timeout)

    monkeypatch.setattr(TreeBuilder, "stop", recording)
    return stops


def _wait(check, timeout=5):
    import time

    deadline = time.time() + timeout
    while not check() and time.time() < deadline:
        time.sleep(0.01)
    assert check()


def test_turns_selection_tools_and_session_end_never_wait_for_the_builder(home, monkeypatch):
    import threading

    gate = threading.Event()
    stops = _record_stops(monkeypatch)
    llm = FakeLlm(lambda n, m, k: gate.wait(10) and "merged")
    engine, db = _live_engine(home, monkeypatch, llm, TreeSettings(exit_wait=30))
    db.rows += rows_for("chat-a", "A")
    engine.on_turn_complete(restored(db.rows), usage={}, turn_id="A")
    _wait(lambda: engine._shared.builder.status()["running"] >= 1)
    db.rows += rows_for("chat-a", "B", complete=False)[:1]
    history = restored(db.rows)
    request = assembled(history)
    engine.select_context(
        request,
        conversation_messages=history,
        incoming_message=next(m for m in history if m["message_uid"] == "Bu"),
        budget_tokens=BUDGET,
    )
    engine.handle_tool_call("ariadne_memory_status", {})
    assert stops == []  # a turn never stops, let alone waits for, the builder
    engine.on_session_end("chat-a", [])  # a real session boundary in a long-lived process
    assert stops == [0]  # the last release stops scheduling and waits for nothing
    gate.set()
    _wait(lambda: Part(1, 0) in TreeStore(home).nodes())


def test_process_exit_waits_for_the_inflight_call_up_to_exit_wait(home, monkeypatch):
    import threading
    import time

    gate = threading.Event()
    stops = _record_stops(monkeypatch)
    llm = FakeLlm(lambda n, m, k: gate.wait(10) and "merged")
    engine, db = _live_engine(home, monkeypatch, llm, TreeSettings(exit_wait=10))
    db.rows += rows_for("chat-a", "A")
    engine.on_turn_complete(restored(db.rows), usage={}, turn_id="A")
    builder = engine._shared.builder
    _wait(lambda: builder.status()["running"] >= 1)
    engine.on_session_end("chat-a", [])  # agent close: stops scheduling, no wait
    threading.Timer(0.3, gate.set).start()
    started = time.monotonic()
    tree_engine._shutdown()  # the atexit path
    assert 0.2 < time.monotonic() - started < 5  # waited for the call, not the whole 10 s
    assert max(stops) > 1 and 8 < max(stops) <= 10  # exit_wait, minus time already spent
    assert builder.status()["running"] == 0
    jobs = TreeStore(home).jobs()
    assert jobs and all(job.state == "done" for job in jobs.values())


def test_process_exit_gives_up_after_exit_wait_and_leaves_the_job_reserved(home, monkeypatch):
    import threading
    import time

    gate = threading.Event()
    llm = FakeLlm(lambda n, m, k: gate.wait(10) and "merged")
    engine, db = _live_engine(home, monkeypatch, llm, TreeSettings(exit_wait=0.3))
    db.rows += rows_for("chat-a", "A")
    engine.on_turn_complete(restored(db.rows), usage={}, turn_id="A")
    builder = engine._shared.builder
    _wait(lambda: builder.status()["running"] >= 1)
    started = time.monotonic()
    tree_engine._shutdown()
    assert 0.25 < time.monotonic() - started < 3
    states = {j.state for j in TreeStore(home).jobs().values()}
    assert states == {"reserved"}
    gate.set()
    _wait(lambda: Part(1, 0) in TreeStore(home).nodes())


def test_process_exit_with_exit_wait_zero_does_not_wait(home, monkeypatch):
    import threading
    import time

    gate = threading.Event()
    stops = _record_stops(monkeypatch)
    llm = FakeLlm(lambda n, m, k: gate.wait(10) and "merged")
    engine, db = _live_engine(home, monkeypatch, llm, TreeSettings(exit_wait=0))
    db.rows += rows_for("chat-a", "A")
    engine.on_turn_complete(restored(db.rows), usage={}, turn_id="A")
    _wait(lambda: engine._shared.builder.status()["running"] >= 1)
    started = time.monotonic()
    tree_engine._shutdown()
    assert time.monotonic() - started < 1 and set(stops) == {0}
    states = {j.state for j in TreeStore(home).jobs().values()}
    assert states == {"reserved"}
    gate.set()
    _wait(lambda: Part(1, 0) in TreeStore(home).nodes())


def test_shutdown_stops_every_home_before_waiting_and_shares_one_deadline(monkeypatch):
    import time
    import types

    calls = []

    class Fake:
        def __init__(self, name):
            self.settings = TreeSettings(exit_wait=0.4)

            def stop(timeout=5.0):
                calls.append((name, round(timeout, 1)))
                time.sleep(timeout)  # an in-flight call that never finishes

            self.builder = types.SimpleNamespace(stop=stop)

        def wait_backfill(self, timeout):
            calls.append(("backfill", timeout))

    _, homes = tree_engine._registry()
    homes["a"], homes["b"] = Fake("a"), Fake("b")
    try:
        tree_engine._shutdown()
    finally:
        homes.pop("a"), homes.pop("b")
    assert calls[:2] == [("a", 0.0), ("b", 0.0)]  # everything stops scheduling first
    assert calls[2:4] == [("backfill", 10.0)] * 2
    assert calls[4] == ("a", 0.4) and calls[5] == ("b", 0.0)  # b gets what is left: nothing


def test_search_scan_limit_counts_only_tree_messages(home, monkeypatch):
    # Archived worker messages are not in the tree; they must not use up the scan.
    monkeypatch.setattr(tree_engine, "SEARCH_SCAN", 3)
    world = World(home, monkeypatch)
    a = world.engine("chat-a")
    world.turn(a, "chat-a", "Zebra")
    sub = world.engine("worker", source="subagent")
    for i in range(5):
        world.turn(sub, "worker", f"ZebraSub{i}")
    found = call(a, "ariadne_search", query="question Zebra")
    assert [r["id"] for r in found["results"]] == [0]
    assert found["truncated"] is False
