"""Profile-wide tree planning and rendering (issue #54, phase A): pure, no I/O."""

import ast
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import pytest

from hermes_ariadne import profile_tree as pt
from hermes_ariadne.tree import Part, View, advance_view, flat, roots, utf8_size

SRC = Path(pt.__file__)
SESSION = "20261121_101500_abcdef"
TAG = hashlib.sha256(SESSION.encode()).hexdigest()[:8]


def entry(position=0, *, session=SESSION, role="user", date="2026-11-21", restricted=False, **kw):
    return pt.LogEntry(
        position=position,
        session_id=session,
        event_id=kw.pop("event_id", f"e{position}"),
        message_uid=kw.pop("message_uid", f"u{position}"),
        role=role,
        date=date,
        origin=kw.pop("origin", "live"),
        restricted=restricted,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Module contract
# --------------------------------------------------------------------------- #


def test_module_is_the_worktree_source_and_pure():
    assert SRC.resolve().is_relative_to(Path(__file__).resolve().parents[1])  # this checkout
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add("." * node.level + (module if node.level else module.split(".")[0]))
    allowed = {
        "hashlib", "json", "re", "bisect", "heapq", "dataclasses", "collections",
        "typing", "types", "datetime", ".tree",
    }  # fmt: skip
    assert imported <= allowed, imported - allowed
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"open", "sqlite3", "threading", "print", "input"}


def test_public_values_are_frozen():
    for value in (entry(), pt.LegendEntry("t", None, None)):
        field = dataclasses.fields(value)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, 1)


# --------------------------------------------------------------------------- #
# LogEntry / conversation_tag
# --------------------------------------------------------------------------- #


def test_log_entry_accepts_valid_values():
    assert entry(3, date=None, origin="import", role="tool").position == 3


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"position": True}, TypeError),
        ({"position": 1.0}, TypeError),
        ({"position": -1}, ValueError),
        ({"session_id": ""}, ValueError),
        ({"session_id": 5}, TypeError),
        ({"event_id": ""}, ValueError),
        ({"message_uid": ""}, ValueError),
        ({"role": "narrator"}, ValueError),
        ({"role": None}, TypeError),
        ({"date": 5}, TypeError),
        ({"date": ""}, ValueError),
        ({"date": "not-an-iso-date"}, ValueError),
        ({"date": "2026-11-21 10:00"}, ValueError),
        ({"origin": "other"}, ValueError),
        ({"restricted": 1}, TypeError),
    ],
)
def test_log_entry_rejects_invalid_values(kwargs, exception):
    base = dict(
        position=0, session_id="s", event_id="e", message_uid="u", role="user",
        date=None, origin="live", restricted=False,
    )  # fmt: skip
    with pytest.raises(exception):
        pt.LogEntry(**{**base, **kwargs})


def test_conversation_tag_is_eight_hex_of_sha256():
    assert pt.conversation_tag(SESSION) == TAG
    assert len(TAG) == 8 and int(TAG, 16) >= 0
    assert pt.conversation_tag("a") != pt.conversation_tag("b")
    with pytest.raises(TypeError):
        pt.conversation_tag(1)
    with pytest.raises(ValueError):
        pt.conversation_tag("")


# --------------------------------------------------------------------------- #
# leaf_source / leaf_text_if_fits
# --------------------------------------------------------------------------- #


def test_leaf_source_user_assistant_system_text():
    msg = {"role": "user", "content": "hello\nworld"}
    assert pt.leaf_source(entry(), msg) == f"{TAG} 2026-11-21 user: hello\nworld"
    assert pt.leaf_source(
        entry(date=None, role="assistant"), {"role": "assistant", "content": "x"}
    ) == (f"{TAG} ? assistant: x")
    assert pt.leaf_source(entry(role="system"), {"role": "system", "content": "s"}).endswith(
        " system: s"
    )


def test_leaf_source_content_parts_and_missing_content():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "one"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "two"},
        ],
    }
    text = pt.leaf_source(entry(), msg)
    assert text == f"{TAG} 2026-11-21 user: one\n[image_url]\ntwo"
    assert pt.leaf_source(entry(role="assistant"), {"role": "assistant", "content": None}) == (
        f"{TAG} 2026-11-21 assistant: "
    )


def test_leaf_source_assistant_tool_calls_and_tool_results():
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
    }
    msg = {"role": "assistant", "content": "looking", "tool_calls": [call, call]}
    assert pt.leaf_source(entry(role="assistant"), msg) == (
        f'{TAG} 2026-11-21 assistant: looking\ncall read_file({{"path":"a.txt"}})'
        f'\ncall read_file({{"path":"a.txt"}})'
    )
    only_calls = {"role": "assistant", "content": "", "tool_calls": [call]}
    assert pt.leaf_source(entry(role="assistant"), only_calls) == (
        f'{TAG} 2026-11-21 assistant: call read_file({{"path":"a.txt"}})'
    )
    bad_args = {"function": {"name": "f", "arguments": "not json"}}
    assert pt.leaf_source(
        entry(role="assistant"), {"role": "assistant", "content": "", "tool_calls": [bad_args]}
    ).endswith("assistant: call f(not json)")
    result = {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": "data"}
    assert pt.leaf_source(entry(role="tool"), result) == (
        f"{TAG} 2026-11-21 tool: result read_file: data"
    )
    nameless = {"role": "tool", "tool_call_id": "c1", "content": "data"}
    assert pt.leaf_source(entry(role="tool"), nameless).endswith("tool: result ?: data")


def test_leaf_source_never_carries_reasoning_or_replay_state():
    secret = "SECRET-REASONING"
    call = {
        "id": "c1",
        "extra_content": {"google": secret},
        "function": {"name": "f", "arguments": "{}", "thought_signature": secret},
    }
    msg = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": secret},
            {"type": "text", "text": "visible"},
            {"type": "text", "text": "x", "extra_content": secret},
        ],
        "reasoning": secret,
        "reasoning_content": secret,
        "reasoning_details": [secret],
        "codex_reasoning_items": [secret],
        "tool_calls": [call],
    }
    text = pt.leaf_source(entry(role="assistant"), msg)
    assert secret not in text
    assert "visible" in text and "call f({})" in text
    carrying = {
        "role": "assistant",
        "content": "a",
        "tool_calls": '[{"thinking": "' + secret + '"}]',
    }
    assert secret not in pt.leaf_source(entry(role="assistant"), carrying)


def test_leaf_source_keeps_real_tool_arguments_and_maps_legacy_roles():
    # An argument NAMED like a carrier is call content, not provider state.
    call = {"function": {"name": "f", "arguments": '{"reasoning": "why"}'}}
    msg = {"role": "assistant", "content": "a", "tool_calls": [call]}
    assert pt.leaf_source(entry(role="assistant"), msg).endswith('call f({"reasoning":"why"})')
    assert pt.leaf_source(entry(role="developer"), {"role": "developer", "content": "d"}).endswith(
        " system: d"
    )
    assert pt.leaf_source(entry(role="function"), {"role": "function", "content": "r"}).endswith(
        " tool: result ?: r"
    )


def test_message_restriction_matches_the_retrieval_gate():
    from hermes_ariadne.retrieval import _require_supported_data_use

    plain = {"role": "user", "content": "x"}
    cases = [
        plain,
        {**plain, "data_use": []},
        {**plain, "data_use": ["no_train"]},
        {**plain, "data_use": "no_train"},
        {**plain, "withheld_prompt": "original"},
        {**plain, "withheld_prompt": ""},
        {**plain, "display_metadata": None},
        {**plain, "display_metadata": {"data_use": []}},
        {**plain, "display_metadata": {"data_use": ["x"]}},
        {**plain, "display_metadata": '{"data_use": ["x"]}'},
        {**plain, "display_metadata": json.dumps(json.dumps({"data_use": ["x"]}))},
        {**plain, "display_metadata": json.dumps({"withheld_prompt": "p"})},
        {**plain, "display_metadata": "not json"},
        {**plain, "display_metadata": '{"a": 1, "a": 2}'},
        {**plain, "display_metadata": "[1]"},
        {**plain, "display_metadata": '{"a": NaN}'},
        {**plain, "display_metadata": 5},
        "message",
        None,
    ]
    for message in cases:
        try:
            _require_supported_data_use(message)
        except ValueError:
            expected = True
        else:
            expected = False
        assert pt.message_restricted(message) is expected, message


def test_a_restricted_message_is_withheld_even_when_the_entry_flag_is_stale():
    message = {"role": "user", "content": "TOP SECRET", "data_use": ["no_train"]}
    out = pt.leaf_source(entry(restricted=False), message)
    assert out == f"{TAG} 2026-11-21 user: [restricted message: not summarized]"


class _Exploding(dict):
    def __getitem__(self, key):
        raise AssertionError("restricted message was read")

    get = items = keys = values = __iter__ = __getitem__


@pytest.mark.parametrize("message", [None, {"role": "user", "content": "TOP SECRET"}, _Exploding()])
def test_restricted_message_text_never_reaches_the_leaf(message):
    out = pt.leaf_source(entry(restricted=True), message)
    assert out == f"{TAG} 2026-11-21 user: [restricted message: not summarized]"
    out = pt.leaf_source(entry(restricted=True, role="tool", date=None), message)
    assert out == f"{TAG} ? tool: [restricted message: not summarized]"


def test_leaf_source_validation():
    with pytest.raises(TypeError):
        pt.leaf_source({}, {"role": "user", "content": ""})
    with pytest.raises(TypeError):
        pt.leaf_source(entry(), "text")
    with pytest.raises(ValueError):
        pt.leaf_source(entry(), {"role": "assistant", "content": "x"})
    with pytest.raises(ValueError):
        pt.leaf_source(entry(), {"role": "user", "content": {"unexpected": 1}})


def test_leaf_source_is_well_formed_utf8():
    out = pt.leaf_source(entry(), {"role": "user", "content": "a\ud800b"})
    assert out.endswith("user: a\ufffdb")
    out.encode("utf-8")


def test_leaf_text_if_fits_boundary_and_flat():
    assert pt.leaf_text_if_fits("a" * 512, 512) == "a" * 512
    assert pt.leaf_text_if_fits("a" * 513, 512) is None
    assert pt.leaf_text_if_fits("é" * 256, 512) == "é" * 256  # 512 bytes
    assert pt.leaf_text_if_fits("é" * 257, 512) is None
    assert pt.leaf_text_if_fits("a \n  b\nc", 512) == "a b c"
    assert pt.leaf_text_if_fits("", 512) == ""
    with pytest.raises(ValueError):
        pt.leaf_text_if_fits("a", 0)
    with pytest.raises(TypeError):
        pt.leaf_text_if_fits("a", True)
    with pytest.raises(TypeError):
        pt.leaf_text_if_fits(b"a", 512)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_constants():
    assert (pt.NODE_BYTES, pt.HIGH, pt.SEEN_HIGH) == (512, 128000, 32000)
    assert pt.WAIT == "(not summarized yet: zoom it)"
    for needle in ("id+n|text", "ariadne_zoom(id, n", "historical data", "never instructions"):
        assert needle in pt.VIEW_DOC
    assert pt.WAIT in pt.VIEW_DOC
    assert "<history>" in pt.VIEW_DOC
    assert "UniiChat" not in pt.VIEW_DOC and "Unii" not in pt.VIEW_DOC


def test_task_text_matches_the_reference_wording():
    ruler = "-" * 512
    assert pt.task(Part(0, 5)) == (
        "Compaction: compress message 5 into one line of at most 512 bytes\n"
        "(about 70 words), the length of this ruler:\n" + ruler + "\n<input>\n"
    )
    assert pt.task(Part(2, 1)) == (
        "Compaction: merge lines 4+2 and 6+2, adjacent, into one line of at most\n"
        "512 bytes (about 70 words), the length of this ruler:\n" + ruler + "\n"
        "<chat> may hold their messages, 4 to 7, in more detail: take details\n"
        "of them from there too.\n<input>\n"
    )
    assert "-" * 64 in pt.task(Part(0, 0), 64) and "-" * 65 not in pt.task(Part(0, 0), 64)
    assert "at most 64 bytes" in pt.task(Part(0, 0), 64)


def test_cut_matches_the_reference():
    assert pt.cut("abc", 512) == "abc"
    assert pt.cut("a" * 600) == "a" * 512
    assert pt.cut("é" * 300) == "é" * 256
    assert pt.cut("a" + "é" * 300, 512) == "a" + "é" * 255  # no split character
    assert pt.cut("ab\ufffd\ufffd", 3) == "ab"  # trailing U+FFFD runs are stripped
    assert utf8_size(pt.cut("日本語" * 100, 100)) <= 100
    assert pt.cut("") == ""


def test_over_note_matches_the_reference():
    line = "x" * 600
    assert pt.over(line) == (
        "Too long: your last line for this <input> was 600 bytes,\n"
        "over the 512-byte limit. Write the whole line again, cutting just\n"
        "enough of the least valuable items to fit before this cut:\n" + "x" * 512 + "| ← LIMIT"
    )
    assert "over the 64-byte limit" in pt.over(line, 64)


def test_merge_input_flattens_both_halves():
    assert pt.merge_input("a\n b", "c \n d") == "a b\nc d"


def test_summary_messages_leaf_and_merge_and_note():
    msgs = pt.summary_messages(Part(0, 3), (f"{TAG} ? user: hi",), ("0+2|x", "2+1|y"), 512)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert set(msgs[0]) == set(msgs[1]) == {"role", "content"}
    assert msgs[0]["content"] == pt.COMPACTION_RULES + "\n\n" + pt.VIEW_LINES
    assert msgs[1]["content"] == (
        "<chat>\n0+2|x\n2+1|y\n</chat>\n" + pt.task(Part(0, 3)) + f"{TAG} ? user: hi\n</input>"
    )
    merged = pt.summary_messages(Part(1, 1), ("a\nb", "c"), (), 512)
    assert merged[1]["content"] == ("<chat>\n</chat>\n" + pt.task(Part(1, 1)) + "a b\nc\n</input>")
    noted = pt.summary_messages(Part(0, 3), ("s",), (), 512, note=pt.over("y" * 600))
    assert noted[1]["content"].endswith("</input>\n" + pt.over("y" * 600))
    assert "ariadne_zoom" not in msgs[0]["content"]  # a compaction has no tools


@pytest.mark.parametrize(
    "part, inputs, context",
    [
        (Part(0, 0), (), ()),
        (Part(0, 0), ("a", "b"), ()),
        (Part(1, 0), ("a",), ()),
        (Part(0, 0), ("a",), "lines"),
        (Part(0, 0), ("a",), (1,)),
        (Part(0, 0), (1,), ()),
        (Part(0, 0), ["a"], ()),
    ],
)
def test_summary_messages_rejects_invalid_shapes(part, inputs, context):
    with pytest.raises((TypeError, ValueError)):
        pt.summary_messages(part, inputs, context, 512)


# --------------------------------------------------------------------------- #
# due / ready_jobs / job_context
# --------------------------------------------------------------------------- #


def all_parts(count):
    return [Part(level, i) for level in range(count.bit_length()) for i in range(count >> level)]


def test_ready_jobs_examples():
    assert pt.ready_jobs(0, set(), set()) == []
    assert pt.ready_jobs(3, set(), set()) == [Part(0, 0), Part(0, 1), Part(0, 2)]
    built = {Part(0, 0), Part(0, 1)}
    assert pt.ready_jobs(3, built, set()) == [Part(1, 0), Part(0, 2)]
    assert pt.ready_jobs(3, built, {Part(1, 0)}) == [Part(0, 2)]
    # (1,1) is incomplete at count 3: its second child does not exist yet.
    assert Part(1, 1) not in pt.ready_jobs(3, built | {Part(0, 2)}, set())
    assert pt.ready_jobs(4, built | {Part(0, 2), Part(0, 3)}, set()) == [
        Part(1, 0),
        Part(1, 1),
    ]
    assert pt.ready_jobs(4, {Part(1, 0), Part(1, 1)}, set()) == [Part(2, 0)]
    done = {Part(0, i) for i in range(4)} | {Part(1, 0), Part(1, 1), Part(2, 0)}
    assert pt.ready_jobs(4, done, set()) == []


def test_ready_jobs_leaves_that_fit_are_left_to_the_caller():
    fits = {Part(0, 0), Part(0, 2)}
    assert pt.ready_jobs(3, set(), set(), fits=fits) == [Part(0, 1)]
    assert pt.ready_jobs(3, set(), set(), limit=2) == [Part(0, 0), Part(0, 1)]


def test_ready_jobs_works_with_a_nodes_mapping_and_validates():
    assert pt.ready_jobs(2, {Part(0, 0): "a"}, {}) == [Part(0, 1)]
    for bad in (-1, True, 2**31):
        with pytest.raises((TypeError, ValueError)):
            pt.ready_jobs(bad, set(), set())
    with pytest.raises(TypeError):
        pt.ready_jobs(2, [Part(0, 0)], set())
    with pytest.raises(ValueError):
        pt.ready_jobs(2, set(), set(), limit=0)


def test_ready_jobs_property_matches_brute_force():
    rng = random.Random(54)
    for _ in range(300):
        count = rng.randint(0, 40)
        built = {p for p in all_parts(count) if rng.random() < 0.4}
        blocked = {p for p in all_parts(count) if rng.random() < 0.15}
        fits = {Part(0, i) for i in range(count) if rng.random() < 0.3}
        expected = [
            p
            for p in all_parts(count)
            if p not in built
            and p not in blocked
            and not (p.level == 0 and p in fits)
            and (p.level == 0 or all(c in built for c in p.children))
            # nothing below a built node is offered again
            and not any(_covers(a, p) for a in built)
        ]
        assert pt.ready_jobs(count, built, blocked, fits=fits) == sorted(expected, key=pt.due)


def _covers(ancestor, part):
    return (
        ancestor.level > part.level and ancestor.start <= part.start and part.stop <= ancestor.stop
    )


def test_first_unbuilt():
    assert pt.first_unbuilt(0, set()) == 0
    assert pt.first_unbuilt(3, {Part(0, 0), Part(0, 1)}) == 2
    assert pt.first_unbuilt(2, {Part(0, 0), Part(0, 1)}) == 2
    assert pt.first_unbuilt(3, {Part(0, 1)}) == 0


def texts(*pairs):
    return {Part(level, index): text for (level, index), text in pairs}


def test_job_context_cutoff_rule_and_none_cases():
    nodes = {Part(0, i): f"t{i}" for i in range(6)}
    seen = View(high=1000)
    # leaf 4: cutoff = min(4, first_unbuilt=6) = 4
    view, lines = pt.job_context(seen, Part(0, 4), nodes, 6)
    assert [p.stop for p in view.parts][-1] == 4
    assert lines == ("0+1|t0", "1+1|t1", "2+1|t2", "3+1|t3")
    # merge (1,1) covers 2..3: cutoff = stop 4
    view, lines = pt.job_context(seen, Part(1, 1), nodes, 6)
    assert len(lines) == 4
    # first unbuilt leaf limits the context
    nodes.pop(Part(0, 3))
    assert pt.job_context(seen, Part(0, 5), nodes, 3) is not None
    view, lines = pt.job_context(seen, Part(0, 5), nodes, 3)
    assert lines == ("0+1|t0", "1+1|t1", "2+1|t2")
    # an unbuilt part inside the seen view -> None
    assert pt.job_context(View(high=1000), Part(0, 5), nodes, 6) is None
    # over its high -> None
    big = {Part(0, i): "x" * 40 for i in range(4)}
    assert pt.job_context(View(high=10), Part(0, 3), big, 4) is None


def test_job_context_validation():
    with pytest.raises(TypeError):
        pt.job_context("view", Part(0, 0), {}, 0)
    with pytest.raises(TypeError):
        pt.job_context(View(), (0, 0), {}, 0)
    with pytest.raises(ValueError):
        pt.job_context(View(), Part(0, 0), {}, -1)


# --------------------------------------------------------------------------- #
# render_view
# --------------------------------------------------------------------------- #

T1, T2 = "aaaaaaaa", "bbbbbbbb"


def legend(*tags):
    return {t: pt.LegendEntry(f"title {t}", "2026-01-01", "2026-02-02") for t in tags}


def test_render_view_exact_layout_wait_and_legend():
    view = View(parts=(Part(1, 0), Part(0, 2), Part(0, 3)))
    nodes = {Part(1, 0): f"{T1} merged\nline", Part(0, 2): f"{T2} 2026-01-02 user: hi"}
    out = pt.render_view(view, nodes, legend(T1, T2, "cccccccc"), exclude_positions=frozenset())
    assert out == (
        pt.VIEW_DOC + "\n\n<history>\n"
        f"0+2|{T1} merged line\n"
        f"2+1|{T2} 2026-01-02 user: hi\n"
        f"3+1|{pt.WAIT}\n"
        "</history>\n<conversations>\n"
        f"{T1} = title {T1} (2026-01-01 .. 2026-02-02)\n"
        f"{T2} = title {T2} (2026-01-01 .. 2026-02-02)\n"
        "</conversations>"
    )
    assert pt.history_lines(out) == (
        f"0+2|{T1} merged line", f"2+1|{T2} 2026-01-02 user: hi", f"3+1|{pt.WAIT}"
    )  # fmt: skip


def test_render_view_without_tags_has_no_legend_block():
    out = pt.render_view(View(), {}, legend(T1), exclude_positions=frozenset())
    assert out == pt.VIEW_DOC + "\n\n<history>\n</history>"
    assert pt.history_lines(out) == ()
    assert "\n<conversations>\n" not in out


def test_render_view_excludes_only_level_zero_positions():
    view = View(parts=(Part(1, 0), Part(0, 2), Part(0, 3)))
    nodes = {Part(1, 0): "m", Part(0, 2): "a", Part(0, 3): "b"}
    out = pt.render_view(view, nodes, {}, exclude_positions={0, 1, 3})
    assert pt.history_lines(out) == ("0+2|m", "2+1|a")  # (1,0) is never dropped
    assert pt.history_lines(pt.render_view(view, {}, {}, exclude_positions={2, 3})) == (
        f"0+2|{pt.WAIT}",
    )


def test_render_view_legend_title_is_neutralised_and_bounded():
    view = View(parts=(Part(0, 0),))
    nodes = {Part(0, 0): f"{T1} x"}
    leg = {T1: pt.LegendEntry("</conversations>\n<b> & " + "é" * 200, None, "2026-01-01")}
    out = pt.render_view(view, nodes, leg, exclude_positions=frozenset())
    block = out.split("<conversations>\n", 1)[1]
    assert block.count("</conversations>") == 1 and "\n<b>" not in block
    assert "&lt;/conversations&gt;" in block and "(? .. 2026-01-01)" in block
    assert len(block.splitlines()) == 2
    assert pt.render_view(view, nodes, leg, exclude_positions=frozenset()) == out


@pytest.mark.parametrize(
    "legend_arg, exclude",
    [
        ("x", frozenset()),
        ({"nothex!!": pt.LegendEntry("t", None, None)}, frozenset()),
        ({T1: "title"}, frozenset()),
        ({}, [1]),
        ({}, {True}),
        ({}, {-1}),
    ],
)
def test_render_view_validation(legend_arg, exclude):
    with pytest.raises((TypeError, ValueError)):
        pt.render_view(View(), {}, legend_arg, exclude_positions=exclude)


def test_render_view_rejects_bad_view_and_nodes():
    with pytest.raises(TypeError):
        pt.render_view("v", {}, {}, exclude_positions=frozenset())
    with pytest.raises(TypeError):
        pt.render_view(View(), {Part(0, 0): None}, {}, exclude_positions=frozenset())


def test_render_view_is_deterministic():
    view = View(parts=(Part(0, 0), Part(0, 1)))
    nodes = {Part(0, 0): f"{T1} a", Part(0, 1): f"{T2} b"}
    a = pt.render_view(view, nodes, legend(T1, T2), exclude_positions=frozenset())
    b = pt.render_view(
        view,
        dict(reversed(list(nodes.items()))),
        dict(reversed(list(legend(T1, T2).items()))),
        exclude_positions=set(),
    )
    assert a == b


# --------------------------------------------------------------------------- #
# retirement_boundary
# --------------------------------------------------------------------------- #


def built_view(count, built=None, high=10**6):
    """A view over leaves 0..count-1 with the given built leaves (default: all)."""
    nodes = {Part(0, i): f"n{i}" for i in (range(count) if built is None else built)}
    return advance_view(View(high=high), count, nodes), nodes


def test_boundary_basic_choices():
    # Active messages 0..8, exchanges start at 0, 3, 6; positions equal indexes.
    own = tuple(range(9))
    starts = (0, 3, 6)
    sizes = (10,) * 9
    view, nodes = built_view(9)
    # Tail budget 90 fits everything: retire nothing.
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 90) == 0
    # 60 -> smallest start with suffix <= 60 is 3 (suffix 60).
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 60) == 3
    # 30 -> start 6 is past starts[-2]=3? No: [-2] is 3, so never beyond 3.
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 30) == 3
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 0) == 3
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 10**9) == 0


def test_boundary_requires_two_exchanges():
    view, nodes = built_view(4)
    own, sizes = tuple(range(4)), (5,) * 4
    assert pt.retirement_boundary(own, (), sizes, view, nodes, 0) == 0
    assert pt.retirement_boundary(own, (0,), sizes, view, nodes, 0) == 0
    assert pt.retirement_boundary((), (), (), View(), {}, 0) == 0


def test_boundary_moves_earlier_until_covered_by_built_parts():
    own = tuple(range(9))
    starts = (0, 3, 6)
    sizes = (10,) * 9
    # Leaf 4 is unbuilt but sits in the exchange that stays verbatim: boundary 3 holds.
    view, nodes = built_view(9, built=[0, 1, 2, 3, 5, 6, 7, 8])
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 0) == 3
    # Leaf 1 unbuilt: nothing can be retired.
    view, nodes = built_view(9, built=[0, 2, 3, 4, 5, 6, 7, 8])
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 0) == 0
    # A built ancestor covering the unbuilt leaf restores coverage ...
    parts = (Part(1, 0), *(Part(0, i) for i in range(2, 9)))
    nodes = {Part(1, 0): "m01", **{k: v for k, v in nodes.items() if k.level == 0}}
    assert pt.retirement_boundary(own, starts, sizes, View(parts=parts), nodes, 0) == 3
    # ... but a view part that is not built (an unbuilt ancestor) covers nothing.
    del nodes[Part(1, 0)]
    assert pt.retirement_boundary(own, starts, sizes, View(parts=parts), nodes, 0) == 0


def test_boundary_positions_beyond_the_view_or_missing_are_not_covered():
    starts = (0, 2, 4)
    sizes = (1,) * 6
    view, nodes = built_view(1)  # the view covers log position 0 only
    own = (0, 1, 2, 3, None, None)
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 0) == 0
    view, nodes = built_view(4)
    own = (0, 1, 2, 3, None, None)
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 0) == 2
    own = (0, None, 2, 3, 4, 5)  # message 1 is not in the log: nothing can retire past it
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 0) == 0


def test_boundary_with_interleaved_conversations():
    # Our messages sit at sparse global positions; other conversations fill the gaps.
    own = (1, 4, 5, 9, 10, 12)
    starts = (0, 2, 4)
    sizes = (10,) * 6
    view, nodes = built_view(13)
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 20) == 2
    view, nodes = built_view(13, built=[i for i in range(13) if i != 4])
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 20) == 0
    view, nodes = built_view(13, built=[i for i in range(13) if i != 5])
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 20) == 2  # 5 stays verbatim


def test_boundary_leading_messages_before_the_first_exchange():
    own = tuple(range(6))
    starts = (1, 3, 5)  # message 0 precedes any exchange start
    sizes = (10,) * 6
    view, nodes = built_view(6)
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 10**6) == 0
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 50) == 1
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 40) == 3
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 60) == 0
    assert pt.retirement_boundary(own, starts, sizes, view, nodes, 0) == 3


@pytest.mark.parametrize(
    "own, starts, sizes, tail",
    [
        ((0, 1), (0, 1), (1,), 0),  # length mismatch
        ((0, 1), (1, 0), (1, 1), 0),  # not increasing
        ((0, 1), (0, 0), (1, 1), 0),
        ((0, 1), (0, 2), (1, 1), 0),  # out of range
        ((0, 1), (-1, 0), (1, 1), 0),
        ((0, 1), (0, 1), (1, -1), 0),
        ((0, 1), (0, 1), (1, 1), -1),
        ((0, 1), (0, 1), (1, 1), True),
        ((0, -1), (0, 1), (1, 1), 0),
        ((0, True), (0, 1), (1, 1), 0),
        ((0, 1), (0, True), (1, 1), 0),
        ((0, 1), (0, 1), (1, True), 0),
        ([0, 1], (0, 1), (1, 1), 0),
    ],
)
def test_boundary_validation(own, starts, sizes, tail):
    view, nodes = built_view(2)
    with pytest.raises((TypeError, ValueError)):
        pt.retirement_boundary(own, starts, sizes, view, nodes, tail)


# --------------------------------------------------------------------------- #
# select_view / check_selection (fixed scenario + mutations)
# --------------------------------------------------------------------------- #


def scenario(*, built=None, tail_bytes=40, count=12, own=None, previous_count=None):
    """One conversation owning positions 2..11 of a 12-entry log (two others before)."""
    positions = tuple(range(2, count)) if own is None else own
    n = len(positions)
    request = tuple({"role": "user", "content": f"m{i}"} for i in range(n))
    starts = tuple(range(0, n, 3))
    sizes = (10,) * n
    nodes = {
        Part(0, i): f"{T1 if i < 2 else T2} t{i}"
        for i in (range(count) if built is None else built)
    }
    view = advance_view(View(high=10**6), count, nodes)
    leg = legend(T1, T2)
    return dict(
        view=view, nodes=nodes, legend=leg, own_positions=positions, exchange_starts=starts,
        sizes=sizes, tail_bytes=tail_bytes, request=request,
    )  # fmt: skip


def test_select_view_and_check_selection_accept_a_valid_selection():
    sel = pt.select_view(**scenario())
    assert 0 < sel.boundary <= sel.exchange_starts[-2]
    assert sel.tail == sel.request[sel.boundary :]
    assert sel.exclude_positions == frozenset(sel.own_positions[sel.boundary :])
    report = pt.check_selection(sel)
    assert report.violations == () and report.ok
    assert report.rendered_bytes == utf8_size(sel.rendered)
    assert report.summary_bytes == sum(utf8_size(t) for t in sel.nodes.values())
    assert report.summary_bytes <= sel.view.high
    # Retired messages are in the render; verbatim ones are not duplicated.
    lines = pt.history_lines(sel.rendered)
    for position in sel.own_positions[: sel.boundary]:
        assert any(line.startswith(f"{position}+1|") for line in lines)
    for position in sel.own_positions[sel.boundary :]:
        assert not any(line.startswith(f"{position}+1|") for line in lines)


def replace(sel, **changes):
    return dataclasses.replace(sel, **changes)


def test_checker_detects_coverage_loss_by_a_dropped_line():
    sel = pt.select_view(**scenario())
    lines = pt.history_lines(sel.rendered)
    victim = f"{sel.own_positions[0]}+1|"
    kept = [line for line in lines if not line.startswith(victim)]
    mutated = sel.rendered.replace("\n".join(lines) + "\n", "\n".join(kept) + "\n")
    assert mutated != sel.rendered
    report = pt.check_selection(replace(sel, rendered=mutated))
    assert not report.ok
    assert any(v.startswith("coverage") for v in report.violations)


def test_checker_detects_retiring_an_unbuilt_or_unlogged_message():
    params = scenario(built=[i for i in range(12) if i != 5])
    sel = pt.select_view(**params)
    assert pt.check_selection(sel).ok
    # Force a boundary that retires message index 3 (position 5, unbuilt).
    forced = replace(
        sel, boundary=6, tail=sel.request[6:], exclude_positions=frozenset(sel.own_positions[6:])
    )
    forced = replace(
        forced,
        rendered=pt.render_view(
            sel.view, sel.nodes, sel.legend, exclude_positions=forced.exclude_positions
        ),
    )
    report = pt.check_selection(forced)
    assert any(v.startswith("coverage") for v in report.violations)
    unlogged = scenario(own=(2, 3, 4, 5, 6, 7, 8, 9, 10, None))
    sel2 = pt.select_view(**unlogged)
    forced2 = replace(sel2, boundary=len(sel2.request), tail=(), exclude_positions=frozenset())
    assert any(v.startswith("coverage") for v in pt.check_selection(forced2).violations)


def test_checker_detects_wrongly_excluded_positions():
    sel = pt.select_view(**scenario())
    bad = frozenset({0})  # another conversation's message must never be omitted
    mutated = replace(
        sel,
        exclude_positions=bad,
        rendered=pt.render_view(sel.view, sel.nodes, sel.legend, exclude_positions=bad),
    )
    assert any(v.startswith("coverage") for v in pt.check_selection(mutated).violations)


def test_checker_detects_a_broken_tail_and_boundary():
    sel = pt.select_view(**scenario())
    cases = [
        replace(sel, tail=sel.tail[1:]),
        replace(sel, tail=sel.tail + ({"role": "user", "content": "extra"},)),
        replace(sel, tail=(*sel.tail[:-1], {"role": "user", "content": "altered"})),
        replace(sel, boundary=sel.exchange_starts[-1], tail=sel.request[sel.exchange_starts[-1] :]),
        replace(sel, boundary=sel.boundary + 1, tail=sel.request[sel.boundary + 1 :]),
    ]
    for mutated in cases:
        report = pt.check_selection(mutated)
        assert not report.ok, mutated.boundary
        assert any(v.startswith(("identity", "boundary")) for v in report.violations)


def test_checker_detects_a_render_that_is_not_the_deterministic_render():
    sel = pt.select_view(**scenario())
    report = pt.check_selection(replace(sel, rendered=sel.rendered.replace("t5", "T5")))
    assert any(v.startswith("render") for v in report.violations)


def test_checker_detects_a_budget_overrun():
    params = scenario()
    params["view"] = advance_view(View(high=20), 12, params["nodes"])
    sel = pt.select_view(**params)
    report = pt.check_selection(sel)
    assert any(v.startswith("budget") for v in report.violations)


def test_checker_cache_prefix_and_its_mutation():
    first = pt.select_view(**scenario(count=10, own=tuple(range(2, 10))))
    second = pt.select_view(**scenario(count=14, own=tuple(range(2, 14))))
    report = pt.check_selection(second, previous=first)
    assert report.ok and report.cache_checked, report.violations
    lines_before = pt.history_lines(first.rendered)
    lines_after = pt.history_lines(second.rendered)
    assert lines_after[: len(lines_before)] == lines_before
    assert pt.check_selection(second).cache_checked is False
    # Mutation: any perturbation of an earlier line breaks the exact prefix.
    first_line = lines_after[0]
    broken = second.rendered.replace(first_line, first_line + " ", 1)
    report = pt.check_selection(replace(second, rendered=broken), previous=first)
    assert any(v.startswith("cache") for v in report.violations)


def test_checker_skips_the_cache_check_after_a_merge_batch():
    nodes = {Part(0, i): f"{T1} t{i}" for i in range(8)}
    for p in (Part(1, 0), Part(1, 1), Part(1, 2), Part(1, 3), Part(2, 0), Part(2, 1), Part(3, 0)):
        nodes[p] = f"m{p.level}{p.index}"
    params = dict(legend=legend(T1), own_positions=tuple(range(8)), exchange_starts=(0, 3, 6),
                  sizes=(1,) * 8, tail_bytes=0, request=tuple(range(8)))  # fmt: skip
    before = pt.select_view(view=advance_view(View(high=10**6), 8, nodes), nodes=nodes, **params)
    merged_view = advance_view(View(high=40), 8, nodes, next_message=True)
    assert merged_view.size <= merged_view.high
    assert merged_view.parts != before.view.parts
    after = pt.select_view(view=merged_view, nodes=nodes, **params)
    report = pt.check_selection(after, previous=before)
    assert report.cache_checked is False and report.ok


def test_checker_validates_its_arguments():
    sel = pt.select_view(**scenario())
    with pytest.raises(TypeError):
        pt.check_selection("sel")
    with pytest.raises(TypeError):
        pt.check_selection(sel, previous="old")


# --------------------------------------------------------------------------- #
# Randomized (seeded) invariants
# --------------------------------------------------------------------------- #

NODE = 80


class World:
    """A profile log with several conversations, a chat view, and background builds."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        rng = self.rng
        self.sessions = [f"session-{seed}-{k}" for k in range(rng.randint(1, 4))]
        self.tags = [pt.conversation_tag(s) for s in self.sessions]
        self.log: list[tuple[int, str, str]] = []  # (session index, role, content)
        self.nodes: dict[Part, str] = {}
        self.view = View(high=rng.choice([300, 800, 2500, 20000]))
        self.seen = View(high=rng.choice([200, 600, 4000]))

    def entry(self, position):
        k, role, _content = self.log[position]
        return pt.LogEntry(
            position=position, session_id=self.sessions[k], event_id=f"e{position}",
            message_uid=f"u{position}", role=role, date=f"2026-11-{1 + position % 28:02d}",
            origin="live", restricted=False,
        )  # fmt: skip

    def message(self, position):
        _, role, content = self.log[position]
        return {"role": role, "content": content}

    def source(self, position):
        return pt.leaf_source(self.entry(position), self.message(position))

    def append_exchange(self, k):
        rng = self.rng
        roles = ["user"] + [
            rng.choice(["assistant", "tool", "assistant"]) for _ in range(rng.randint(1, 3))
        ]
        starts = []
        for role in roles:
            size = rng.choice([5, 30, 120, 400])
            self.log.append((k, role, "".join(rng.choice("abc é\n") for _ in range(size))))
            starts.append(len(self.log) - 1)
        self.view = advance_view(self.view, len(self.log), self.nodes, next_message=True)

    def build(self, fraction):
        rng = self.rng
        count = len(self.log)
        for _ in range(100):
            fits = {
                Part(0, i)
                for i in range(count)
                if pt.leaf_text_if_fits(self.source(i), NODE) is not None
            }
            for leaf in fits:
                if leaf not in self.nodes and rng.random() < max(fraction, 0.5):
                    self.nodes[leaf] = pt.leaf_text_if_fits(self.source(leaf.index), NODE)
            jobs = [
                p
                for p in pt.ready_jobs(count, self.nodes, set(), fits=fits)
                if rng.random() < fraction
            ]
            if not jobs:
                break
            for part in jobs:
                if part.level == 0:
                    text = pt.cut(flat(self.source(part.index)), NODE)
                else:
                    a, b = (self.nodes[c] for c in part.children)
                    text = pt.cut(pt.merge_input(a, b).replace("\n", " | "), NODE)
                self.nodes[part] = text
        self.view = advance_view(self.view, count, self.nodes)

    def legend(self):
        out = {}
        for k, tag in enumerate(self.tags):
            dates = [self.entry(i).date for i, (s, _, _) in enumerate(self.log) if s == k]
            if dates:
                out[tag] = pt.LegendEntry(f"chat {k}", min(dates), max(dates))
        return out

    def selection(self, k, pending, tail_bytes):
        rng = self.rng
        mine = [i for i, (s, _, _) in enumerate(self.log) if s == k]
        own = [*mine] + [None] * pending
        request = tuple(
            self.message(i) if i is not None else {"role": "user", "content": f"new{j}"}
            for j, i in enumerate(own)
        )
        starts = [
            j
            for j, i in enumerate(own)
            if (self.message(i)["role"] if i is not None else "user") == "user"
        ]
        sizes = tuple(utf8_size(m["content"]) + rng.randint(0, 20) for m in request)
        return pt.select_view(
            view=self.view, nodes=dict(self.nodes), legend=self.legend(), own_positions=tuple(own),
            exchange_starts=tuple(starts), sizes=sizes, tail_bytes=tail_bytes, request=request,
        )  # fmt: skip


def oracle_boundary(sel):
    """Independent re-statement of the boundary rule, by brute force."""
    starts = sel.exchange_starts
    if len(starts) < 2:
        return 0
    cands = sorted({0, *starts[:-1]})
    preferred = next((c for c in cands if sum(sel.sizes[c:]) <= sel.tail_bytes), cands[-1])

    def covered(position):
        if position is None:
            return False
        for part in sel.view.parts:
            if part.start <= position < part.stop:
                return part in sel.nodes
        return False

    for c in sorted((c for c in cands if c <= preferred), reverse=True):
        if all(covered(p) for p in sel.own_positions[:c]):
            return c
    return 0


def walk(seed):
    """Yield (world, conversation, selection, previous selection) over a random history."""
    world = World(seed)
    rng = world.rng
    previous = {}
    for _ in range(rng.randint(4, 14)):
        world.append_exchange(rng.randrange(len(world.sessions)))
        world.build(rng.choice([0.0, 0.3, 0.7, 1.0]))
        for k in range(len(world.sessions)):
            if not any(s == k for s, _, _ in world.log):
                continue
            sel = world.selection(k, rng.randint(0, 3), rng.choice([0, 50, 300, 2000, 10**6]))
            yield world, k, sel, previous.get(k)
            previous[k] = sel


@pytest.mark.parametrize("seed", range(60))
def test_random_worlds_satisfy_the_selection_invariants(seed):
    for _world, _k, sel, previous in walk(seed):
        assert sel.boundary == oracle_boundary(sel)
        report = pt.check_selection(sel, previous=previous)
        # A view can sit over its high while parents are unbuilt: that is the one
        # violation a legal history may show, and it must be reported exactly then.
        assert [v for v in report.violations if not v.startswith("budget")] == []
        assert any(v.startswith("budget") for v in report.violations) == (
            report.summary_bytes > sel.view.high
        )
        assert report.rendered_bytes == utf8_size(sel.rendered)
        assert sel.tail == sel.request[sel.boundary :]
        assert sel.exclude_positions == {
            p for p in sel.own_positions[sel.boundary :] if p is not None
        }


def test_random_worlds_exercise_every_branch():
    totals = {"cache": 0, "retired": 0, "wait": 0, "merged": 0, "clean": 0}
    for seed in range(60):
        for world, _k, sel, previous in walk(seed):
            report = pt.check_selection(sel, previous=previous)
            totals["cache"] += report.cache_checked
            totals["retired"] += sel.boundary > 0
            totals["wait"] += pt.WAIT in sel.rendered
            totals["merged"] += any(p.level > 0 for p in sel.view.parts)
            totals["clean"] += report.ok
    assert all(value > 20 for value in totals.values()), totals


def test_random_mutations_are_always_detected():
    detected = {"coverage": 0, "tail": 0, "current-exchange": 0, "cache": 0}
    for seed in range(60):
        for _world, _k, sel, previous in walk(2000 + seed):
            report = pt.check_selection(sel, previous=previous)
            if not report.ok:
                continue  # only mutate a selection that was valid to begin with
            lines = pt.history_lines(sel.rendered)
            if sel.boundary > 0:  # coverage loss: drop the line of a retired message
                covering = next(
                    p
                    for p in sel.view.parts
                    if p.start <= sel.own_positions[sel.boundary - 1] < p.stop
                )
                victim = f"{covering.start}+{covering.count}|"
                kept = [line for line in lines if not line.startswith(victim)]
                assert len(kept) < len(lines)
                text = sel.rendered.replace("\n".join(lines) + "\n", "\n".join(kept) + "\n", 1)
                bad = pt.check_selection(replace(sel, rendered=text))
                assert any(v.startswith("coverage") for v in bad.violations)
                detected["coverage"] += 1
            if sel.tail:  # a broken tail
                assert not pt.check_selection(replace(sel, tail=sel.tail[1:])).ok
                detected["tail"] += 1
            if len(sel.exchange_starts) >= 2 and sel.boundary < sel.exchange_starts[-1]:
                at = sel.exchange_starts[-1]  # retiring the current exchange
                assert not pt.check_selection(replace(sel, boundary=at, tail=sel.request[at:])).ok
                detected["current-exchange"] += 1
            if report.cache_checked:  # a non-prefix render
                old = pt.history_lines(previous.rendered)
                if old:
                    text = sel.rendered.replace(old[0], old[0] + "~", 1)
                    bad = pt.check_selection(replace(sel, rendered=text), previous=previous)
                    assert any(v.startswith("cache") for v in bad.violations)
                    detected["cache"] += 1
    assert all(value > 10 for value in detected.values()), detected


def test_appends_without_a_merge_keep_the_rendered_lines_as_a_prefix():
    """Cache stability stated directly, independent of the checker."""
    rng = random.Random(7)
    nodes: dict[Part, str] = {}
    view = View(high=10**9)
    previous_lines = ()
    for step in range(1, 60):
        nodes[Part(0, step - 1)] = f"{T1 if rng.random() < 0.5 else T2} line {step}"
        old_parts = view.parts
        view = advance_view(view, step, nodes, next_message=True)
        assert view.parts[: len(old_parts)] == old_parts
        out = pt.render_view(view, nodes, legend(T1, T2), exclude_positions=frozenset())
        lines = pt.history_lines(out)
        assert lines[: len(previous_lines)] == previous_lines
        assert out.startswith(
            pt.VIEW_DOC + "\n\n<history>\n" + "".join(line + "\n" for line in previous_lines)
        )
        previous_lines = lines


def test_roots_cover_log_positions_after_every_advance():
    world = World(3)
    for _ in range(10):
        world.append_exchange(0)
        world.build(1.0)
        assert tuple(p for p in world.view.parts)[-1].stop == len(world.log)
        assert sum(p.count for p in roots(len(world.log))) == len(world.log)


# --------------------------------------------------------------------------- #
# Collapsed runs of unbuilt parts (bounded view while building)
# --------------------------------------------------------------------------- #


def leaves(count):
    return View(parts=tuple(Part(0, i) for i in range(count)))


def run_line(first, last):
    return (
        f"{first}..{last}|("
        + f"{last - first + 1} messages not summarized yet: search, or zoom an id)"
    )


def test_run_line_format_is_documented_and_explained_to_the_model():
    assert pt.wait_run(5) == "(5 messages not summarized yet: search, or zoom an id)"
    assert run_line(3, 7) == f"3..7|{pt.wait_run(5)}"
    for text in (pt.VIEW_LINES, pt.VIEW_DOC):
        assert "first..last|" in text and "not summarized yet" in text
        assert "range line" in text
    assert "ariadne_search" in pt.VIEW_DOC and "ariadne_zoom(id, 1)" in pt.VIEW_DOC
    assert "inside a range" in pt.VIEW_DOC
    assert pt.WAIT in pt.VIEW_LINES  # a single unbuilt line keeps its own form


def test_consecutive_unbuilt_parts_collapse_into_one_range_line():
    nodes = {Part(0, 1): f"{T1} one", Part(0, 4): f"{T2} four"}
    out = pt.render_view(leaves(8), nodes, legend(T1, T2), exclude_positions=frozenset())
    assert pt.history_lines(out) == (
        f"0+1|{pt.WAIT}",
        f"1+1|{T1} one",
        run_line(2, 3),
        f"4+1|{T2} four",
        run_line(5, 7),
    )
    assert pt.render_view(leaves(8), nodes, {}, exclude_positions=frozenset()) == pt.render_view(
        leaves(8), nodes, {}, exclude_positions=set()
    )


def test_a_run_spans_parts_of_every_level_and_a_lone_part_keeps_its_line():
    view = View(parts=(Part(2, 0), Part(1, 2), Part(0, 6), Part(0, 7)))
    assert pt.history_lines(pt.render_view(view, {}, {}, exclude_positions=frozenset())) == (
        run_line(0, 7),
    )
    nodes = {Part(1, 2): "m"}
    assert pt.history_lines(pt.render_view(view, nodes, {}, exclude_positions=frozenset())) == (
        f"0+4|{pt.WAIT}",
        "4+2|m",
        run_line(6, 7),
    )


def test_a_verbatim_position_ends_a_run_and_ids_stay_exact():
    out = pt.render_view(leaves(6), {}, {}, exclude_positions={2, 5})
    assert pt.history_lines(out) == (run_line(0, 1), run_line(3, 4))
    out = pt.render_view(leaves(4), {}, {}, exclude_positions={1})
    assert pt.history_lines(out) == (f"0+1|{pt.WAIT}", run_line(2, 3))


def test_line_span_reads_both_line_forms():
    assert pt.line_span(run_line(10, 4999)) == (10, 4999)
    assert pt.line_span("8+4|text") == (8, 11)
    assert pt.line_span(f"3+1|{pt.WAIT}") == (3, 3)
    for bad in ("", "3|x", "a..b|x", "7..3|x", "0+3|x", "1+2|x", "x"):
        assert pt.line_span(bad) is None, bad


def unbuilt_selection(count=40, built=(), **kw):
    nodes = {Part(0, i): f"{T1} t{i}" for i in built}
    return pt.select_view(
        view=advance_view(View(high=10**6), count, nodes),
        nodes=nodes,
        legend=legend(T1),
        own_positions=kw.get("own_positions", ()),
        exchange_starts=kw.get("exchange_starts", ()),
        sizes=kw.get("sizes", ()),
        tail_bytes=0,
        request=kw.get("request", ()),
    )


def test_checker_accepts_run_lines_as_exact_coverage_and_counts_their_bytes():
    sel = unbuilt_selection(built=[3, 4])
    report = pt.check_selection(sel)
    assert report.ok, report.violations
    assert pt.history_lines(sel.rendered) == (
        run_line(0, 2),
        f"3+1|{T1} t3",
        f"4+1|{T1} t4",
        run_line(5, 39),
    )
    spans = [pt.line_span(line) for line in pt.history_lines(sel.rendered)]
    assert spans[0] == (0, 2) and spans[-1] == (5, 39)
    assert report.rendered_bytes == utf8_size(sel.rendered)
    assert report.wait_bytes == utf8_size(run_line(0, 2) + "\n") + utf8_size(run_line(5, 39) + "\n")
    assert report.summary_bytes == sum(utf8_size(t) for t in sel.nodes.values())


def test_checker_rejects_a_run_that_does_not_cover_exactly_its_positions():
    sel = unbuilt_selection(built=[3])
    lines = pt.history_lines(sel.rendered)
    body = "\n".join(lines) + "\n"
    forged = {
        "last too far": lines[0].replace("0..2", "0..3"),
        "count wrong": lines[0].replace("(3 messages", "(4 messages"),
        "swallows a built part": run_line(0, 3),
        "split run": run_line(0, 1) + "\n" + run_line(2, 2),
    }
    for name, line in forged.items():
        text = sel.rendered.replace(body, body.replace(lines[0], line, 1))
        report = pt.check_selection(replace(sel, rendered=text))
        assert any(v.startswith("coverage") for v in report.violations), name
    # a run line standing where a built part must be shown
    wrong = body.replace(lines[1], run_line(3, 3).replace("3..3", "3+1"), 1)
    report = pt.check_selection(replace(sel, rendered=sel.rendered.replace(body, wrong)))
    assert not report.ok


def test_checker_rejects_two_adjacent_runs_instead_of_one_maximal_run():
    sel = unbuilt_selection(count=6)
    body = "\n".join(pt.history_lines(sel.rendered)) + "\n"
    text = sel.rendered.replace(body, run_line(0, 2) + "\n" + run_line(3, 5) + "\n")
    report = pt.check_selection(replace(sel, rendered=text))
    assert any(v.startswith("coverage") for v in report.violations)


def test_a_retired_message_inside_a_run_is_not_covered():
    sel = unbuilt_selection(
        count=6,
        own_positions=(0, 1, 2, 3),
        exchange_starts=(0, 2),
        sizes=(10,) * 4,
        request=tuple({"role": "user", "content": str(i)} for i in range(4)),
    )
    assert sel.boundary == 0  # nothing built: nothing may be retired
    forced = replace(sel, boundary=2, tail=sel.request[2:], exclude_positions=frozenset({2, 3}))
    forced = replace(
        forced, rendered=pt.render_view(sel.view, {}, sel.legend, exclude_positions={2, 3})
    )
    report = pt.check_selection(forced)
    assert any("not built" in v for v in report.violations)


def test_cache_prefix_holds_when_nothing_changes_and_when_a_trailing_run_grows():
    first = unbuilt_selection(count=1)
    assert pt.history_lines(first.rendered) == (f"0+1|{pt.WAIT}",)
    same = unbuilt_selection(count=1)
    report = pt.check_selection(same, previous=first)
    assert report.ok and report.cache_checked
    grown = unbuilt_selection(count=2)  # the lone WAIT line becomes a run
    assert pt.history_lines(grown.rendered) == (run_line(0, 1),)
    report = pt.check_selection(grown, previous=first)
    assert report.ok and report.cache_checked, report.violations
    more = unbuilt_selection(count=9)
    report = pt.check_selection(more, previous=grown)
    assert report.ok and report.cache_checked, report.violations
    # earlier lines are still an exact prefix when the run is not the first line
    base = unbuilt_selection(count=5, built=[0, 1])
    after = unbuilt_selection(count=8, built=[0, 1])
    report = pt.check_selection(after, previous=base)
    assert report.ok and report.cache_checked, report.violations
    assert pt.history_lines(after.rendered)[:3] == pt.history_lines(base.rendered)[:2] + (
        run_line(2, 7),
    )


def test_a_run_may_only_extend_never_rewrite_what_came_before():
    base = unbuilt_selection(count=5, built=[0, 1])
    after = unbuilt_selection(count=8, built=[0, 1])
    old = pt.history_lines(base.rendered)
    for text in (
        after.rendered.replace(old[0], old[0] + "~", 1),  # an earlier line moved
        after.rendered.replace(run_line(2, 7), run_line(3, 7).replace("5 mess", "6 mess"), 1),
        after.rendered.replace(run_line(2, 7), run_line(2, 7)[:-1] + " ", 1),
    ):
        report = pt.check_selection(replace(after, rendered=text), previous=base)
        assert any(v.startswith("cache") for v in report.violations), text[-200:]


def test_building_inside_a_run_changes_the_view_there_and_skips_the_prefix_check():
    before = unbuilt_selection(count=9, built=[0])
    after = unbuilt_selection(count=9, built=[0, 4])  # a leaf inside the run is built
    assert pt.history_lines(after.rendered)[1:] == (
        run_line(1, 3), f"4+1|{T1} t4", run_line(5, 8)
    )  # fmt: skip
    report = pt.check_selection(after, previous=before)
    assert report.ok and not report.cache_checked


def test_a_fresh_backfill_of_ten_thousand_messages_fits_the_view_bound():
    count, high = 10_000, 128000
    sel = unbuilt_selection(count=count)
    report = pt.check_selection(sel)
    assert report.ok, report.violations
    assert pt.history_lines(sel.rendered) == (run_line(0, count - 1),)
    assert report.rendered_bytes <= 2 * high + 4096
    assert report.wait_bytes < 100 and report.summary_bytes == 0
    # a conversation that owns the tail: its positions are verbatim and split the run
    own = tuple(range(count - 6, count))
    sel = unbuilt_selection(
        count=count,
        own_positions=own,
        exchange_starts=(0, 3),
        sizes=(10,) * 6,
        request=tuple({"role": "user", "content": str(i)} for i in range(6)),
    )
    assert pt.check_selection(sel).ok and sel.boundary == 0
    assert pt.history_lines(sel.rendered) == (run_line(0, count - 7),)


def test_runs_are_bounded_by_the_built_parts_and_verbatim_positions_between_them():
    built = list(range(1, 400, 2))  # every other leaf: the worst alternation
    sel = unbuilt_selection(count=400, built=built)
    lines = pt.history_lines(sel.rendered)
    assert pt.check_selection(sel).ok
    assert len(lines) == 400  # alternation is the bound: one line per built part or hole
    assert all(pt.line_span(line) is not None for line in lines)
    runs = [line for line in lines if ".." in line.split("|", 1)[0]]
    assert runs == []  # single unbuilt parts keep their own form


def test_select_view_pin_keeps_an_earlier_boundary_only_while_it_is_valid():
    params = scenario(tail_bytes=0)  # fresh boundary: the last retirable exchange start
    fresh = pt.select_view(**params)
    allowed = [0, *fresh.exchange_starts[:-1]]
    assert fresh.boundary == fresh.exchange_starts[-2] > 0
    pinned = pt.select_view(**params, pin_boundary=allowed[1])
    assert pinned.boundary == allowed[1] < fresh.boundary
    assert pt.check_selection(pinned).ok
    assert pinned.tail == pinned.request[allowed[1] :]
    # not an exchange start, past the fresh boundary, or None: the fresh boundary wins
    for pin in (1, fresh.boundary + 3, None, 10**6):
        assert pt.select_view(**params, pin_boundary=pin).boundary == fresh.boundary
    unbuilt = pt.select_view(**scenario(tail_bytes=0, built=[0, 1, 2, 3, 4, 5]), pin_boundary=6)
    assert unbuilt.boundary < 6 and pt.check_selection(unbuilt).ok  # coverage beats the pin
    with pytest.raises((TypeError, ValueError)):
        pt.select_view(**params, pin_boundary=True)
    with pytest.raises((TypeError, ValueError)):
        pt.select_view(**params, pin_boundary=-1)
