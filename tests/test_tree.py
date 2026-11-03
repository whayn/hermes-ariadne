"""Pure tree tests; all chat text and summaries are synthetic."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

_GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "tree_golden.json").read_text())


@pytest.mark.parametrize("case", _GOLDEN["primitives"])
def test_text_primitives_match_pinned_reference(case):
    from hermes_ariadne.tree import flat, utf8_size

    assert utf8_size(case["text"]) == case["bytes"]
    assert flat(case["text"]) == case["flat"]


def _parts(rows):
    from hermes_ariadne.tree import Part

    return tuple(Part(*row) for row in rows)


def _check_scenario(scenario):
    from hermes_ariadne.tree import Part, View, advance_view, render_lines, utf8_size

    view = View(parts=_parts(scenario.get("parts", [])), high=scenario["high"])
    nodes = {Part(level, index): text for level, index, text in scenario["nodes"]}
    for step in scenario["steps"]:
        nodes.update({Part(level, index): text for level, index, text in step.get("add", [])})
        previous, unchanged_nodes = view, dict(nodes)
        view = advance_view(view, step["end"], nodes, next_message=step.get("next_message", False))
        expected = step["expected"]
        assert view == View(
            _parts(expected["parts"]), expected["size"], expected["batch"], expected["high"]
        )
        assert render_lines(view, nodes) == tuple(step["lines"])
        for render in step["render_ends"]:
            assert render_lines(view, nodes, end=render["end"]) == tuple(render["lines"])
        assert previous is not view
        assert nodes == unchanged_nodes
        assert sum(part.count for part in view.parts) == step["end"]
        assert view.size == sum(utf8_size(nodes[part]) for part in view.parts if part in nodes)
        assert all(part.level == 0 or part in nodes for part in view.parts)


def test_append_and_render_match_reference():
    _check_scenario(_GOLDEN["scenarios"][0])


@pytest.mark.parametrize("scenario", _GOLDEN["scenarios"], ids=lambda case: case["name"])
def test_view_transitions_match_pinned_reference(scenario):
    _check_scenario(scenario)


@pytest.mark.parametrize("case", _GOLDEN["roots"])
def test_roots_match_pinned_reference(case):
    from hermes_ariadne.tree import roots

    assert roots(case["count"]) == _parts(case["parts"])


@pytest.mark.parametrize("case", _GOLDEN["parts"])
def test_part_properties_match_pinned_reference(case):
    from hermes_ariadne.tree import Part

    part = Part(*case["part"])
    assert (part.start, part.stop, part.count) == (case["start"], case["stop"], case["count"])
    assert part.children == _parts(case["children"])


def test_view_defaults_are_frozen():
    from hermes_ariadne.tree import View

    view = View()
    assert (view.parts, view.size, view.batch, view.high) == ((), 0, False, 128000)
    with pytest.raises(FrozenInstanceError):
        view.size = 9


@pytest.mark.parametrize("level,index", [(True, 0), (0, False), (1.0, 0), (0, "1")])
def test_part_rejects_non_integer_coordinates(level, index):
    from hermes_ariadne.tree import Part

    with pytest.raises(TypeError):
        Part(level, index)


@pytest.mark.parametrize("level,index", [(-1, 0), (31, 0), (0, -1), (30, 1), (0, 2**31 - 1)])
def test_part_rejects_unsupported_ranges(level, index):
    from hermes_ariadne.tree import Part

    with pytest.raises(ValueError):
        Part(level, index)


@pytest.mark.parametrize("count", [-1, 2**31, 2**100])
def test_roots_rejects_unsupported_counts(count):
    from hermes_ariadne.tree import roots

    with pytest.raises(ValueError):
        roots(count)


@pytest.mark.parametrize("count", [True, False, 1.5, "2", None])
def test_roots_rejects_non_integer_counts(count):
    from hermes_ariadne.tree import roots

    with pytest.raises(TypeError):
        roots(count)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"parts": []}, TypeError),
        ({"parts": ("not a part",)}, TypeError),
        ({"size": -1}, ValueError),
        ({"size": True}, TypeError),
        ({"size": 1.5}, TypeError),
        ({"batch": 1}, TypeError),
        ({"high": 0}, ValueError),
        ({"high": -10}, ValueError),
        ({"high": True}, TypeError),
        ({"high": 1.5}, TypeError),
    ],
)
def test_view_rejects_invalid_field_values(kwargs, exception):
    from hermes_ariadne.tree import View

    with pytest.raises(exception):
        View(**kwargs)


@pytest.mark.parametrize("rows", [[[0, 1]], [[0, 0], [0, 2]], [[1, 0], [0, 1]], [[0, 0], [0, 0]]])
def test_view_requires_contiguous_nonoverlapping_prefix_coverage(rows):
    from hermes_ariadne.tree import View

    with pytest.raises(ValueError):
        View(parts=_parts(rows))


@pytest.mark.parametrize(
    "end,exception", [(True, TypeError), (1.0, TypeError), (-1, ValueError), (2**31, ValueError)]
)
def test_public_view_operations_reject_invalid_end(end, exception):
    from hermes_ariadne.tree import View, advance_view, render_lines

    for operation in (advance_view, render_lines):
        with pytest.raises(exception):
            if operation is advance_view:
                operation(View(), end, {})
            else:
                operation(View(), {}, end=end)


@pytest.mark.parametrize("next_message", [0, 1, None, "yes"])
def test_advance_requires_explicit_boolean_next_message(next_message):
    from hermes_ariadne.tree import View, advance_view

    with pytest.raises(TypeError):
        advance_view(View(), 0, {}, next_message=next_message)


@pytest.mark.parametrize("which", ["advance", "render"])
@pytest.mark.parametrize("nodes", [[], {"0+1": "text"}, {0: "text"}, {None: "text"}])
def test_public_view_operations_reject_invalid_node_mappings(which, nodes):
    from hermes_ariadne.tree import View, advance_view, render_lines

    with pytest.raises(TypeError):
        if which == "advance":
            advance_view(View(), 0, nodes)
        else:
            render_lines(View(), nodes)


@pytest.mark.parametrize("text", [None, 1, b"text", True])
def test_pending_nodes_are_absent_not_non_text_values(text):
    from hermes_ariadne.tree import Part, View, advance_view, render_lines

    for operation in (advance_view, render_lines):
        with pytest.raises(TypeError):
            if operation is advance_view:
                operation(View(), 1, {Part(0, 0): text})
            else:
                operation(View(), {Part(0, 0): text})


@pytest.mark.parametrize("text", [None, 1, b"text", True])
def test_text_helpers_require_strings(text):
    from hermes_ariadne.tree import flat, utf8_size

    for operation in (flat, utf8_size):
        with pytest.raises(TypeError):
            operation(text)


@pytest.mark.parametrize("which", ["advance", "render"])
def test_public_view_operations_require_a_view(which):
    from hermes_ariadne.tree import advance_view, render_lines

    with pytest.raises(TypeError):
        if which == "advance":
            advance_view(None, 0, {})
        else:
            render_lines(None, {})


def test_parts_and_roots_cover_a_prefix_with_exclusive_stops():
    from hermes_ariadne.tree import Part, roots

    part = Part(3, 2)
    assert (part.start, part.stop, part.count) == (16, 24, 8)
    assert part.children == (Part(2, 4), Part(2, 5))
    assert Part(0, 0).children == ()
    assert roots(13) == (Part(3, 0), Part(2, 2), Part(0, 12))
    assert roots(0) == ()
    assert hash(part) == hash(Part(3, 2))
    with pytest.raises(FrozenInstanceError):
        part.index = 9
