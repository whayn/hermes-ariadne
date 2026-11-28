"""The data-use refusal matrix of ``retrieval._require_supported_data_use``.

Ported from the deleted ``test_retrieval.py`` (#70): every message reaching an
import, the memory tree or a request is admitted by this predicate.
"""

import json

import pytest

from hermes_ariadne.retrieval import _require_supported_data_use, _UnsupportedDataUse

REFUSED = [
    {"data_use": ["no_training"]},
    {"data_use": None},
    {"data_use": False},
    {"data_use": ""},
    {"data_use": {}},
    {"data_use": "no_training"},
    {"data_use": [17]},
    {"display_metadata": {"data_use": ["no_training"]}},
    {"display_metadata": {"data_use": None}},
    {"display_metadata": {"data_use": False}},
    {"display_metadata": {"data_use": {}}},
    {"display_metadata": '{"data_use":["no_training"]}'},
    {"display_metadata": json.dumps('{"data_use":["no_training"]}')},
    {"display_metadata": '{"data_use":[] ,"data_use":["no_training"]}'},
    {"display_metadata": '{"data_use":["no_training"],"data_use":[]}'},
    {"display_metadata": '{"data_use":NaN}'},
    {"display_metadata": '{"data_use":[]'},
    {"display_metadata": []},
    {"display_metadata": False},
    {"display_metadata": ""},
    {"display_metadata": "null"},
    {"display_metadata": json.dumps(json.dumps('{"data_use":[]}'))},
    {"data_use": [], "display_metadata": {"data_use": ["unknown_requirement"]}},
    {"data_use": ["unknown_requirement"], "display_metadata": {"data_use": []}},
]

ADMITTED = [
    {},
    {"data_use": []},
    {"display_metadata": None},
    {"display_metadata": {}},
    {"display_metadata": {"data_use": []}},
    {"display_metadata": '{"data_use":[]}'},
    {"display_metadata": json.dumps('{"data_use":[]}')},
    {"data_use": [], "display_metadata": {"data_use": [], "other": 1}},
]


def message(**metadata):
    return {"message_uid": "m", "role": "tool", "content": "needle PRIVATE", **metadata}


@pytest.mark.parametrize("metadata", REFUSED)
def test_restricted_unknown_or_malformed_data_use_is_refused(metadata):
    original = message(**metadata)
    copy = json.loads(json.dumps(original))
    with pytest.raises(_UnsupportedDataUse) as refused:
        _require_supported_data_use(original)
    assert str(refused.value) == "unsupported_data_use"  # fixed text, never content
    assert original == copy  # refused, never stripped


@pytest.mark.parametrize("metadata", ADMITTED)
def test_absent_or_empty_data_use_is_admitted(metadata):
    _require_supported_data_use(message(**metadata))


@pytest.mark.parametrize("value", ["PRIVATE typed prompt", "", None, False, [], {}, 17])
@pytest.mark.parametrize("carrier", ["top-level", "mapping", "json", "double-json"])
def test_withheld_prompt_presence_is_refused_whatever_its_value(value, carrier):
    metadata = {"withheld_prompt": value, "data_use": []}
    if carrier == "top-level":
        fields = metadata
    elif carrier == "mapping":
        fields = {"display_metadata": metadata}
    elif carrier == "json":
        fields = {"display_metadata": json.dumps(metadata)}
    else:
        fields = {"display_metadata": json.dumps(json.dumps(metadata))}
    with pytest.raises(_UnsupportedDataUse):
        _require_supported_data_use(message(**fields))


@pytest.mark.parametrize("value", [None, "text", ["role"], 7])
def test_a_non_mapping_message_is_refused(value):
    with pytest.raises(_UnsupportedDataUse):
        _require_supported_data_use(value)
