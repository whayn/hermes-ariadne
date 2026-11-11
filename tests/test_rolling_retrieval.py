"""Bounded pinned-generation paging and unchanged exact-event evidence."""

import base64
import copy
import json
from dataclasses import replace

import pytest
from test_rolling_selection import POLICY, fixture

from hermes_ariadne import rolling
from hermes_ariadne.store import canonical_json


def call(view, store, settings, args, *, name="ariadne_rolling_page", session_id="s", **kwargs):
    from hermes_ariadne.rolling_retrieval import handle_rolling_tool

    return handle_rolling_tool(
        name,
        args,
        view=view,
        store=store,
        session_id=session_id,
        settings=settings,
        reasoning_policy=POLICY,
        **kwargs,
    )


def test_generation_bound_ordered_pages_default_and_terminal(tmp_path):
    store, settings, view, source, _, _ = fixture(tmp_path)
    args = {"ref": "rolling:" + view.generation_id}
    default = call(view, store, settings, args)
    assert default["ok"] is True and default["total"] == 4
    assert default["next_offset"] is None
    assert "untrusted" in default["trust"]
    all_rows = []
    offset = 0
    while True:
        page = call(view, store, settings, {**args, "offset": offset, "limit": 2})
        assert page["ok"] and page["total"] == len(view.event_ids)
        assert (
            len(json.dumps(page, ensure_ascii=True, separators=(",", ":")).encode("ascii")) <= 8192
        )
        all_rows.extend(page["results"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert all_rows == [
        {"index": i, "event_ref": "event:" + event, "role": source[i]["role"]}
        for i, event in enumerate(view.event_ids)
    ]
    assert default["results"] == all_rows
    assert call(view, store, settings, {**args, "offset": 4})["results"] == []
    assert args == {"ref": "rolling:" + view.generation_id}


def test_expired_generation_never_loads_or_retargets(tmp_path):
    store, settings, first, source, _, _ = fixture(tmp_path)
    target = rolling.Target.create(
        source[:8],
        session_id="s",
        settings=settings,
        policy=POLICY,
        parent=first,
        request_bytes=b'[{"content":"next","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    second = rolling.validate_generation(
        rolling.generation_bytes(target, "next summary", sequence=2),
        store,
        "s",
        settings,
        policy=POLICY,
        parent_loader=lambda _: first.snapshot_bytes,
    )
    loads = []
    expired = call(
        second,
        store,
        settings,
        {"ref": "rolling:" + first.generation_id},
        parent_loader=lambda ref: loads.append(ref) or first.snapshot_bytes,
    )
    assert expired["error"] == "generation_expired" and loads == []
    current = call(
        second,
        store,
        settings,
        {"ref": "rolling:" + second.generation_id},
        parent_loader=lambda _: first.snapshot_bytes,
    )
    assert current["ok"] and current["total"] == 8


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"ref": "rolling:" + "A" * 64},
        {"ref": "rolling:" + "a" * 63},
        {"ref": "event:" + "a" * 64},
        {"ref": "rolling:" + "a" * 64, "path": "arbitrary"},
        {"ref": "rolling:" + "a" * 64, "offset": -1},
        {"ref": "rolling:" + "a" * 64, "offset": True},
        {"ref": "rolling:" + "a" * 64, "limit": True},
        {"ref": "rolling:" + "a" * 64, "limit": 0},
        {"ref": "rolling:" + "a" * 64, "limit": 21},
    ],
)
def test_strict_page_arguments(tmp_path, args):
    store, settings, view, _, _, _ = fixture(tmp_path)
    assert call(view, store, settings, args)["error"] == "invalid_arguments"


@pytest.mark.parametrize(
    "mutation", ["hash", "restricted", "withheld", "missing", "uid", "foreign"]
)
def test_whole_dependency_check_precedes_even_empty_or_first_page(tmp_path, monkeypatch, mutation):
    store, settings, view, source, _, _ = fixture(tmp_path)
    original = store.get_event

    def changed(event, *, session_id):
        row = original(event, session_id=session_id)
        if event == view.event_ids[-1]:
            if mutation == "missing":
                return None
            row = copy.deepcopy(row)
            key, value = {
                "hash": ("content", "changed"),
                "restricted": ("data_use", ["no_training"]),
                "withheld": ("display_metadata", {"withheld_prompt": ""}),
                "uid": ("message_uid", "different"),
                "foreign": ("session_id", "other"),
            }[mutation]
            row[key] = value
        return row

    monkeypatch.setattr(store, "get_event", changed)
    for offset in (0, 4):
        result = call(
            view,
            store,
            settings,
            {"ref": "rolling:" + view.generation_id, "offset": offset, "limit": 1},
        )
        assert result["ok"] is False and "results" not in result and "total" not in result


def test_scope_settings_and_forged_view_are_not_trusted(tmp_path):
    store, settings, view, _, _, _ = fixture(tmp_path)
    args = {"ref": "rolling:" + view.generation_id}
    assert call(view, store, settings, args, session_id="foreign")["ok"] is False
    assert call(view, store, replace(settings, reserve_tokens=5000), args)["ok"] is False
    assert call(replace(view, summary_text="forged"), store, settings, args)["ok"] is False
    assert call(None, store, settings, args)["ok"] is False


def test_existing_exact_event_pages_reconstruct_unchanged_archive_and_no_binary_reinterpretation(
    tmp_path,
):
    store, settings, view, source, _, _ = fixture(tmp_path)
    event = "event:" + view.event_ids[-1]
    chunks = []
    offset = 0
    while True:
        result = call(
            view,
            store,
            settings,
            {"ref": event, "offset": offset, "limit": 97},
            name="ariadne_zoom",
        )
        assert result["ok"] and result["kind"] == "source"
        chunks.append(base64.b64decode(result["page"]["data_base64"]))
        offset = result["page"]["next_offset"]
        if offset is None:
            break
    assert b"".join(chunks) == canonical_json(source[3])
    assert b"OPAQUE" in b"".join(chunks)
    found = call(view, store, settings, {"query": "answer old"}, name="ariadne_find")
    assert found["ok"] and found["results"][0]["event_ref"] == event
    assert (
        call(view, store, settings, {"ref": event}, name="ariadne_zoom", session_id="other")["ok"]
        is False
    )
    node = "node:" + "a" * 64 + ":" + "b" * 64 + ":0+1"
    assert (
        call(view, store, settings, {"ref": node}, name="ariadne_zoom")["error"]
        == "kernel_unavailable"
    )


def test_schemas_are_fixed_and_scope_free():
    from hermes_ariadne.rolling_retrieval import tool_schemas

    first = tool_schemas()
    page = first[-1]
    assert page["name"] == "ariadne_rolling_page"
    assert page["parameters"]["required"] == ["ref"]
    assert page["parameters"]["additionalProperties"] is False
    assert set(page["parameters"]["properties"]) == {"ref", "offset", "limit"}
    assert page["parameters"]["properties"]["limit"]["maximum"] == 20
    first[-1]["parameters"]["properties"]["limit"]["maximum"] = 99
    assert tool_schemas()[-1]["parameters"]["properties"]["limit"]["maximum"] == 20


def test_twenty_descriptor_cap_and_full_index_totals(tmp_path):
    from test_rolling_selection import exchange

    from hermes_ariadne.store import EventStore

    source = [row for i in range(6) for row in exchange("p" + str(i))]
    settings = rolling.RollingSettings(max_delta_exchanges=6)
    store = EventStore(tmp_path / "ariadne" / "events.sqlite3")
    store.ingest("s", source, origin="canonical")
    target = rolling.Target.create(
        source,
        session_id="s",
        settings=settings,
        policy=POLICY,
        parent=None,
        request_bytes=b'[{"content":"input","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    view = rolling.validate_generation(
        rolling.generation_bytes(target, "summary", sequence=1), store, "s", settings, policy=POLICY
    )
    first = call(view, store, settings, {"ref": "rolling:" + view.generation_id, "limit": 20})
    last = call(
        view, store, settings, {"ref": "rolling:" + view.generation_id, "offset": 20, "limit": 20}
    )
    assert len(first["results"]) == 20 and first["next_offset"] == 20
    assert len(last["results"]) == len(source) - 20 and last["next_offset"] is None
    assert first["total"] == last["total"] == len(source)
    assert [row["event_ref"] for row in first["results"] + last["results"]] == [
        "event:" + event for event in view.event_ids
    ]
    for page in (first, last):
        assert (
            len(json.dumps(page, ensure_ascii=True, separators=(",", ":")).encode("ascii")) <= 8192
        )


def test_current_ref_with_missing_or_changed_parent_never_returns_partial_index(tmp_path):
    store, settings, first, source, _, _ = fixture(tmp_path)
    target = rolling.Target.create(
        source[:8],
        session_id="s",
        settings=settings,
        policy=POLICY,
        parent=first,
        request_bytes=b'[{"content":"next","role":"user"}]',
        route_labels=("codex_backend", "gpt-test"),
    )
    second = rolling.validate_generation(
        rolling.generation_bytes(target, "next summary", sequence=2),
        store,
        "s",
        settings,
        policy=POLICY,
        parent_loader=lambda _: first.snapshot_bytes,
    )
    for loader in (None, lambda _: first.snapshot_bytes + b" "):
        result = call(
            second,
            store,
            settings,
            {"ref": "rolling:" + second.generation_id, "offset": 4, "limit": 1},
            parent_loader=loader,
        )
        assert result["ok"] is False and "results" not in result and "total" not in result
