"""Journal open reuses verified bytes instead of re-deriving them (#22).

An open reads and hashes every stored target payload and generation artifact,
but decodes each exact payload once and compares derived identities instead of
re-encoding whole prefixes for every job. Each shortcut must give exactly the
verdict (and, where it returns bytes, exactly the bytes) of the code it
replaces; the reference implementations below are the pre-#22 code paths.
"""

import dataclasses
import json
import random
import sqlite3
from collections import Counter

import pytest
from test_rolling_journal_validation import chain
from test_rolling_state import target

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.rolling import sha256

HEX = "0123456789abcdef"


def reference_require_digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in HEX for c in value):
        raise ValueError("invalid SHA-256 identity")
    return value


def outcome(function, *args):
    try:
        return ("ok", function(*args))
    except Exception as exc:  # the exact exception type and message must match
        return (type(exc), str(exc))


def digest(rng):
    return "".join(rng.choice(HEX) for _ in range(64))


DIGEST_LIKE = [
    "a" * 64,
    "A" * 64,  # uppercase is not a lowercase digest
    "a" * 63,
    "a" * 65,
    "",
    "g" + "a" * 63,
    "a" * 63 + "\n",
    " " + "a" * 63,
    "a" * 32 + '"' + "a" * 31,
    "a" * 32 + "\\" + "a" * 31,
    "é" * 64,
    "\ud800" + "a" * 63,  # lone surrogate
    "０" * 64,  # fullwidth digit
    1,
    None,
    b"a" * 64,
]


@pytest.mark.parametrize("value", DIGEST_LIKE)
def test_require_digest_matches_the_reference(value):
    assert outcome(rolling.require_digest, value) == outcome(reference_require_digest, value)


def test_require_digest_matches_the_reference_on_random_strings():
    rng = random.Random(22)
    alphabet = HEX + 'ABCDEFghxyz \t\n"\\é\u2028\x00'
    for _ in range(5000):
        value = "".join(rng.choice(alphabet if rng.random() < 0.2 else HEX) for _ in range(64))
        assert outcome(rolling.require_digest, value) == outcome(reference_require_digest, value)


def digest_lists():
    rng = random.Random(7)
    lists = [[], [digest(rng)], [digest(rng) for _ in range(50)]]
    for odd in DIGEST_LIKE:
        lists.append([digest(rng), odd, digest(rng)])
        lists.append([odd])
    return lists


@pytest.mark.parametrize("values", digest_lists())
@pytest.mark.parametrize("as_tuple", [False, True])
def test_digest_list_encoding_is_encode_json(values, as_tuple):
    values = tuple(values) if as_tuple else values
    for limit in (1, 2, 66, 67, 68, 1024, 67 * max(1, len(values)), 67 * len(values) + 1, 2**20):
        assert outcome(rolling._encode_digest_list, values, limit) == outcome(
            rolling.encode_json, list(values), limit
        ), (values, limit)


def reference_from_bytes(data, settings):
    """``Target.from_bytes`` before #22: decode, rebuild, re-encode, compare."""
    value = rolling.decode_json(data, settings.max_checkpoint_bytes)
    if type(value) is not dict or set(value) != {
        f.name for f in dataclasses.fields(rolling.Target)
    }:
        raise ValueError("invalid target schema")
    for key in ("event_ids", "message_uids", "route_labels"):
        if type(value[key]) is not list:
            raise ValueError("invalid target sequence")
        value[key] = tuple(value[key])
    for key in ("readable_input", "request_bytes"):
        if type(value[key]) is not str:
            raise ValueError("invalid target input")
        value[key] = value[key].encode("utf-8")
    result = rolling.Target(**value)
    if result.to_bytes(settings) != data:
        raise ValueError("noncanonical target")
    return result


def target_payload_variants(frozen, settings):
    data = frozen.to_bytes(settings)
    value = json.loads(data)

    def encoded(mapping, **kwargs):
        return json.dumps(mapping, **kwargs).encode("utf-8")

    def replaced(**changes):
        return rolling.encode_json({**value, **changes}, 2**31 - 1)

    yield data
    yield data + b" "
    yield b" " + data
    yield encoded(value)  # spaces after separators
    yield encoded(value, separators=(",", ":"))  # unsorted keys
    yield encoded(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    yield data.replace(b'"delta_start":', b'"delta_start" :')
    yield data.replace(b"summarize", b"summ\\u0061rize")
    yield data.replace(b"summarize", b"summ\\/arize")
    yield replaced(delta_start=float(value["delta_start"]))
    yield replaced(delta_start=True)
    yield replaced(delta_stop=value["delta_stop"] + 1)
    yield replaced(event_ids=value["event_ids"][:-1])
    yield replaced(event_ids="not a list")
    yield replaced(readable_input=7)
    yield replaced(extra=1)
    yield replaced(route_labels=["codex_backend"])
    yield replaced(session_id="")
    yield replaced(event_ids=[1, *value["event_ids"][1:]])
    yield replaced(message_uids=list(reversed(value["message_uids"])))
    yield data.replace(b"summarize", b"summ\\ud800rize")
    yield b'{"a":1,"a":1}'
    yield b"[]"
    yield b"\xff"
    yield "not bytes"


def test_target_from_bytes_matches_the_reference(tmp_path):
    store, settings, views, _, _ = chain(tmp_path, 2)
    frozen = target(store, settings, 3, views[-1])
    tight = dataclasses.replace(settings, max_prefix_events=len(frozen.event_ids) - 1)
    verdicts = Counter()
    for config in (settings, tight):
        for data in target_payload_variants(frozen, settings):
            expected = outcome(reference_from_bytes, data, config)
            actual = outcome(rolling.Target.from_bytes, data, config)
            assert actual == expected, data
            verdicts[actual[0] == "ok"] += 1
            if actual[0] == "ok":
                # The identity primed from the verified bytes is the derived one:
                # a fresh copy (no cached properties) derives the same values.
                fresh = dataclasses.replace(actual[1])
                assert actual[1].target_id == fresh.target_id == sha256(data)
                assert actual[1].job_id == fresh.job_id
    # Both verdicts are exercised (canonical variants that change identity are accepted).
    assert verdicts[True] >= 2 and verdicts[False] >= 20, verdicts


def generation_variants(view, frozen):
    """(target, summary text) pairs that differ from ``frozen``/its text in one field."""
    text = view.summary_text
    other = "0" * 64
    yield frozen, text
    yield frozen, text + " "
    for name, changed in (
        ("session_id", "t"),
        ("settings_digest", other),
        ("parent_generation", other if frozen.parent_generation else None),
        ("parent_summary_sha256", other if frozen.parent_summary_sha256 else None),
        ("parent_generation", None),
        ("event_ids", frozen.event_ids[:-1] + (other,)),
        ("event_ids", frozen.event_ids[:-2]),
        ("message_uids", frozen.message_uids[:-1] + ("other",)),
        ("delta_start", frozen.delta_start + 1),
        ("delta_start", 0),
        ("readable_input", frozen.readable_input + b" "),
        ("request_bytes", b'[{"content":"other","role":"user"}]'),
        ("route_labels", ("codex_backend", "other")),
    ):
        try:
            yield dataclasses.replace(frozen, **{name: changed}), text
        except ValueError:
            pass  # not a constructible Target, so never a journal's target


def view_variants(view, settings):
    """``view`` with one field changed, its snapshot re-encoded to match.

    A target change always changes its job id; these keep the job id and change
    the generation side instead, so every compared field is needed on its own.
    """
    other = "0" * 64
    value = json.loads(view.snapshot_bytes)
    for name, changed in (
        ("session_id", "t"),
        ("settings_digest", other),
        ("parent_generation", other),
        ("parent_summary_sha256", other),
        ("event_ids", [*value["event_ids"][:-1], other]),
        ("message_uids", [*value["message_uids"][:-1], "other"]),
        ("frontier_count", value["frontier_count"] + 1),
        ("prefix_sha256", other),
        ("delta_start", value["delta_start"] + 1),
        ("delta_stop", value["delta_stop"] + 1),
        ("summary_text", value["summary_text"] + " "),
        ("summary_sha256", other),
        ("summary_input_sha256", other),
        ("job_id", other),
    ):
        forged = {**value, name: changed}
        data = rolling.encode_json(forged, settings.max_checkpoint_bytes)
        fields = {
            key: forged[key] for key in value if key in rolling.RollingView.__dataclass_fields__
        }
        fields["event_ids"] = tuple(fields["event_ids"])
        fields["message_uids"] = tuple(fields["message_uids"])
        yield rolling.RollingView(data, sha256(data), **fields)


def test_generation_comparison_matches_reencoding(tmp_path):
    store, settings, views, successor, _ = chain(tmp_path, 3, pending=True)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        targets = [journal._job(view.job_id)[1] for view in views]
    checked = Counter()
    for view, frozen in zip(views, targets):
        for candidate, text in generation_variants(view, frozen):
            expected = outcome(
                lambda: rolling.generation_bytes(
                    candidate, text, sequence=view.sequence, settings=settings
                )
            )
            reencoded = expected[0] == "ok" and expected[1] == view.snapshot_bytes
            assert rolling._generation_matches(view, candidate, text) == reencoded
            checked[reencoded] += 1
        for forged in view_variants(view, settings):
            # The journal only compares once the job text equals the view's text.
            text = forged.summary_text
            expected = outcome(
                lambda: rolling.generation_bytes(
                    frozen, text, sequence=forged.sequence, settings=settings
                )
            )
            reencoded = expected[0] == "ok" and expected[1] == forged.snapshot_bytes
            assert not reencoded
            assert rolling._generation_matches(forged, frozen, text) is False
            checked["forged"] += 1
    assert checked[True] >= len(views) and checked[False] >= 10 * len(views), checked
    assert checked["forged"] == 14 * len(views)


def spy(monkeypatch, owner, name):
    calls = Counter()
    original = getattr(owner, name)

    def counting(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counting)
    return calls


def spy_decodes(monkeypatch):
    decodes = Counter()
    original = rolling._decode_json

    def counting(data, limit):
        decodes[sha256(data)] += 1
        return original(data, limit)

    monkeypatch.setattr(rolling, "_decode_json", counting)
    return decodes


@pytest.mark.parametrize("count", [4, 12])
def test_open_decodes_each_target_payload_once_and_reencodes_no_prefix(
    tmp_path, monkeypatch, count
):
    store, settings, views, successor, _ = chain(tmp_path, count, pending=True)
    decodes = spy_decodes(monkeypatch)
    rebuilt = spy(monkeypatch, rolling.Target, "_canonical_bytes")
    regenerated = spy(monkeypatch, rolling, "generation_bytes")
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        # Read back through the open's cache: no further decode happens here.
        target_ids = {journal._job(view.job_id)[1].target_id for view in views}
        target_ids.add(successor.target_id)
        assert {digest: decodes[digest] for digest in target_ids} == dict.fromkeys(target_ids, 1)
        # No job re-serializes its target or regenerates a generation's bytes.
        assert not rebuilt and not regenerated
        decodes.clear()
        # Later operations in the same open reuse the verified payloads.
        assert journal.pending_target() == successor
        assert journal.load_view() == views[-1]
        assert journal.inspect(successor.job_id)["state"] == "ready"
        assert not set(decodes) & target_ids
        assert not rebuilt and not regenerated


def test_reused_target_is_still_rehashed_on_every_use(tmp_path):
    store, settings, views, successor, paths = chain(tmp_path, 3, pending=True)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.pending_target() == successor
        value = json.loads(successor.to_bytes(settings))
        value["route_labels"] = ["codex_backend", "forged"]
        forged = rolling.encode_json(value, settings.max_checkpoint_bytes)
        conn = sqlite3.connect(paths[0])
        try:
            with conn:
                conn.execute(
                    "UPDATE targets SET payload=? WHERE target_id=?",
                    (forged, successor.target_id),
                )
        finally:
            conn.close()
        # The payload's new bytes no longer hash to the stored digest or its id.
        with pytest.raises(ValueError, match="frozen target hash mismatch"):
            journal.pending_target()
    with pytest.raises(ValueError, match="frozen target hash mismatch"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)


def test_validated_views_share_their_parent_identities(tmp_path):
    store, settings, views, _, _ = chain(tmp_path, 4)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        head = journal.load_view()
        accepted = journal._memo.accepted
        chain_views = [head]
        while chain_views[-1].parent_generation is not None:
            chain_views.append(accepted[chain_views[-1].parent_generation])
        for child, parent in zip(chain_views, chain_views[1:]):
            count = parent.frontier_count
            assert child.event_ids[:count] == parent.event_ids
            assert all(a is b for a, b in zip(child.event_ids, parent.event_ids))
            assert all(a is b for a, b in zip(child.message_uids, parent.message_uids))
    assert [view.generation_id for view in reversed(chain_views)] == [
        view.generation_id for view in views
    ]
