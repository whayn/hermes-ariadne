"""Search pages reach older archive evidence without a fixed history horizon."""

from test_tree_engine import World, call
from test_tree_engine import _isolated_registry as _isolated_registry
from test_tree_engine import home as home

from hermes_ariadne import tree_engine


def test_search_pages_reach_matches_beyond_scan_window(home, monkeypatch):
    monkeypatch.setattr(tree_engine, "SEARCH_SCAN", 3)
    world = World(home, monkeypatch)
    engine = world.engine("chat")
    for i in range(5):
        world.turn(engine, "chat", f"Zebra{i}")
    first = call(engine, "ariadne_search", query="question Zebra", limit=2)
    assert [r["id"] for r in first["results"]] == [16, 12]
    second = call(
        engine, "ariadne_search", query="question Zebra", limit=2, cursor=first["next_cursor"]
    )
    third = call(
        engine, "ariadne_search", query="question Zebra", limit=2, cursor=second["next_cursor"]
    )
    assert [r["id"] for r in second["results"]] == [8, 4]
    assert [r["id"] for r in third["results"]] == [0]
    assert third["next_cursor"] is None


def test_cursor_is_query_bound_and_new_rows_do_not_shift_pages(home, monkeypatch):
    world = World(home, monkeypatch)
    engine = world.engine("chat")
    for i in range(3):
        world.turn(engine, "chat", f"Zebra{i}")
    first = call(engine, "ariadne_search", query="question Zebra", limit=1)
    cursor = first["next_cursor"]
    assert (
        call(engine, "ariadne_search", query="other", cursor=cursor)["error"] == "invalid_argument"
    )
    assert (
        call(engine, "ariadne_search", query="question Zebra", cursor="bad")["error"]
        == "invalid_argument"
    )
    world.turn(engine, "chat", "ZebraNew")
    rest = call(engine, "ariadne_search", query="question Zebra", cursor=cursor)
    assert [r["id"] for r in rest["results"]] == [4, 0]
    assert rest["next_cursor"] is None
