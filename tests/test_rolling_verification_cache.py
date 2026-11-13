"""Cross-open verification cache: same verdicts, fewer recomputations.

A RollingVerificationCache lets a later journal open skip decoding and
canonicalizing bytes an earlier open in the same scope fully validated. Every
artifact, target payload and archive row is still reread and hashed on each
open, and journal state is always rechecked, so every tamper below must give
exactly the verdict an open without a cache gives.
"""

import json
import sqlite3
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from test_rolling_journal_validation import chain

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.rolling import RollingVerificationCache, sha256
from hermes_ariadne.store import EventStore


def open_journal(tmp_path, store, settings, cache=None, **kwargs):
    return rolling_state.RollingJournal(tmp_path, "s", settings, store=store, cache=cache, **kwargs)


def observe(tmp_path, store, settings, cache=None):
    """What an open (and its first operations) yields: values, or the exact refusal."""
    try:
        with open_journal(tmp_path, store, settings, cache) as journal:
            return (
                "ok",
                journal.load_view(),
                journal.pending_target(),
                journal.report(),
                journal.charged_callback_ids(),
            )
    except Exception as exc:
        return (type(exc), str(exc))


def warm(tmp_path, store, settings):
    cache = RollingVerificationCache()
    with open_journal(tmp_path, store, settings, cache) as journal:
        journal.load_view()
    assert cache.stats()["entries"] > 0
    return cache


def spy_decodes(monkeypatch):
    decodes = Counter()
    original = rolling._decode_json

    def counting(data, limit):
        decodes[sha256(data)] += 1
        return original(data, limit)

    monkeypatch.setattr(rolling, "_decode_json", counting)
    return decodes


def spy(monkeypatch, owner, name):
    calls = Counter()
    original = getattr(owner, name)

    def counting(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counting)
    return calls


def test_warm_open_rereads_everything_but_recomputes_nothing(tmp_path, monkeypatch):
    store, settings, views, successor, paths = chain(tmp_path, 6, pending=True)
    cache = warm(tmp_path, store, settings)
    decodes = spy_decodes(monkeypatch)
    canonicalized = spy(monkeypatch, rolling, "_canonical_rows")
    reads = Counter()
    read = rolling_state._read

    def counting_read(path, limit):
        reads[path.name] += 1
        return read(path, limit)

    monkeypatch.setattr(rolling_state, "_read", counting_read)
    decoded_rows = spy(monkeypatch, store, "get_event")
    checked = Counter()
    verify = store.verify_events

    def counting_verify(event_ids, *, session_id):
        found = verify(event_ids, session_id=session_id)
        checked.update(found)
        return found

    monkeypatch.setattr(store, "verify_events", counting_verify)
    payloads = Counter()
    target = rolling_state.RollingJournal._target

    def counting_target(self, target_id):
        payloads[target_id] += 1
        return target(self, target_id)

    monkeypatch.setattr(rolling_state.RollingJournal, "_target", counting_target)
    with open_journal(tmp_path, store, settings, cache) as journal:
        assert journal.load_view() == views[-1]
        assert journal.pending_target() == successor
    # No artifact or target payload decoded, no row canonicalized again ...
    payload_ids = {view.generation_id for view in views} | set(payloads)
    assert not payload_ids & set(decodes)
    assert not canonicalized and not decoded_rows
    # ... yet every artifact is reread, every target payload is reread from the
    # journal, and every referenced archive row is rehashed by the store once.
    assert {view.generation_id + ".json" for view in views} <= set(reads)
    assert set(payloads) >= {successor.target_id}
    referenced = set(successor.event_ids)
    assert set(checked) == referenced and set(checked.values()) == {1}


def test_warm_and_cold_opens_return_equal_values(tmp_path):
    store, settings, views, successor, _ = chain(tmp_path, 5, pending=True)
    cold = observe(tmp_path, store, settings)
    cache = RollingVerificationCache()
    first = observe(tmp_path, store, settings, cache)
    second = observe(tmp_path, store, settings, cache)
    assert cold == first == second
    assert cold[1] == views[-1] and cold[2] == successor


def rewrite_artifact(paths, view, settings, **changes):
    value = json.loads(view.snapshot_bytes)
    value.update(changes)
    path = paths[1] / (view.generation_id + ".json")
    path.write_bytes(rolling.encode_json(value, settings.max_checkpoint_bytes))
    path.chmod(0o600)


def archive_payload(store, event, payload):
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE events SET payload_json=CAST(? AS BLOB) WHERE event_id=?", (payload, event)
        )


def archive_row(store, event):
    with sqlite3.connect(store.path) as conn:
        return json.loads(
            conn.execute("SELECT payload_json FROM events WHERE event_id=?", (event,)).fetchone()[0]
        )


def journal_sql(paths, sql, *args):
    with sqlite3.connect(paths[0]) as conn:
        conn.execute(sql, args)


def mutate(name, store, settings, views, successor, paths):
    root, middle, head = views[0], views[2], views[-1]
    pending_event = successor.event_ids[-1]
    if name == "none":
        return
    if name == "root_row_payload":
        archive_payload(store, root.event_ids[0], b'{"message_uid":"x","role":"user"}')
    elif name == "pending_row_payload":
        archive_payload(store, pending_event, b'{"message_uid":"x","role":"user"}')
    elif name == "root_row_unobserved":
        with sqlite3.connect(store.path) as conn:
            conn.execute("DELETE FROM observations WHERE event_id=?", (root.event_ids[0],))
    elif name == "pending_row_unobserved":
        with sqlite3.connect(store.path) as conn:
            conn.execute("DELETE FROM observations WHERE event_id=?", (pending_event,))
    elif name == "row_foreign_session":
        with sqlite3.connect(store.path) as conn:
            conn.execute(
                "UPDATE observations SET session_id='foreign' WHERE event_id=?",
                (middle.event_ids[-1],),
            )
    elif name == "row_uid_reencoded":
        row = archive_row(store, middle.event_ids[-1])
        row["message_uid"] = "foreign"
        archive_payload(
            store, middle.event_ids[-1], rolling.encode_json(row, settings.max_checkpoint_bytes)
        )
    elif name == "ancestor_summary":
        rewrite_artifact(
            paths, middle, settings, summary_text="forged", summary_sha256=sha256(b"forged")
        )
    elif name == "head_summary":
        rewrite_artifact(
            paths, head, settings, summary_text="forged", summary_sha256=sha256(b"forged")
        )
    elif name == "ancestor_removed":
        (paths[1] / (middle.generation_id + ".json")).unlink()
    elif name == "manifest_older":
        value = json.loads(paths[2].read_bytes())
        value.update(generation_id=middle.generation_id, job_id=middle.job_id)
        paths[2].write_bytes(rolling.encode_json(value, 8192))
    elif name == "job_text":
        journal_sql(
            paths,
            "UPDATE jobs SET validated_text='forged',text_sha256=? WHERE job_id=?",
            sha256(b"forged"),
            middle.job_id,
        )
    elif name == "job_failed":
        journal_sql(paths, "UPDATE jobs SET state='failed' WHERE job_id=?", middle.job_id)
    elif name == "target_payload":
        value = json.loads(successor.to_bytes(settings))
        value["route_labels"] = ["codex_backend", "forged"]
        journal_sql(
            paths,
            "UPDATE targets SET payload=? WHERE target_id=?",
            rolling.encode_json(value, settings.max_checkpoint_bytes),
            successor.target_id,
        )
    elif name == "target_payload_rehashed":
        # Payload and its stored digest rewritten consistently; only the target
        # id (the hash its job refers to) still names the original bytes.
        value = json.loads(successor.to_bytes(settings))
        value["route_labels"] = ["codex_backend", "forged"]
        forged = rolling.encode_json(value, settings.max_checkpoint_bytes)
        journal_sql(
            paths,
            "UPDATE targets SET payload=?,payload_sha256=? WHERE target_id=?",
            forged,
            sha256(forged),
            successor.target_id,
        )
    elif name == "published_target_payload":
        with sqlite3.connect(paths[0]) as conn:
            target_id = conn.execute(
                "SELECT target_id FROM jobs WHERE job_id=?", (middle.job_id,)
            ).fetchone()[0]
            payload = conn.execute(
                "SELECT payload FROM targets WHERE target_id=?", (target_id,)
            ).fetchone()[0]
            conn.execute(
                "UPDATE targets SET payload=? WHERE target_id=?",
                (payload.replace(b"summarize", b"summarise"), target_id),
            )
    elif name == "accounting":
        journal_sql(paths, "UPDATE campaign SET logical_reservations=logical_reservations-1")
    elif name == "unjournaled_artifact":
        data = rolling.generation_bytes(successor, "forged summary", sequence=len(views) + 1)
        artifact = paths[1] / (sha256(data) + ".json")
        artifact.write_bytes(data)
        artifact.chmod(0o600)
    else:
        raise AssertionError(name)


MUTATIONS = [
    "none",
    "root_row_payload",
    "pending_row_payload",
    "root_row_unobserved",
    "pending_row_unobserved",
    "row_foreign_session",
    "row_uid_reencoded",
    "ancestor_summary",
    "head_summary",
    "ancestor_removed",
    "manifest_older",
    "job_text",
    "job_failed",
    "target_payload",
    "target_payload_rehashed",
    "published_target_payload",
    "accounting",
    "unjournaled_artifact",
]


@pytest.mark.parametrize("name", MUTATIONS)
def test_warm_cache_gives_the_uncached_verdict_after_any_change(tmp_path, name):
    store, settings, views, successor, paths = chain(tmp_path, 5, pending=True)
    cache = warm(tmp_path, store, settings)
    before = cache.stats()
    mutate(name, store, settings, views, successor, paths)
    cached = observe(tmp_path, store, settings, cache)
    uncached = observe(tmp_path, store, settings)
    assert cached == uncached
    assert (cached[0] == "ok") == (name == "none")
    if name != "none":
        # A refused open records nothing new.
        assert cache.stats() == before


def test_cache_is_not_reused_across_scopes(tmp_path, monkeypatch):
    store, settings, views, _, _ = chain(tmp_path, 3)
    cache = warm(tmp_path, store, settings)
    ids = {view.generation_id for view in views}
    decodes = spy_decodes(monkeypatch)
    # Same journal and bytes under another reasoning policy: validated in full.
    policy = rolling.ReasoningReplayPolicy("codex_backend", "gpt-test")
    with open_journal(tmp_path, store, settings, cache, policy=policy) as journal:
        assert journal.load_view() == views[-1]
    assert ids <= set(decodes)
    decodes.clear()
    # Binding that scope discarded the first one: it starts over in full too.
    with open_journal(tmp_path, store, settings, cache) as journal:
        assert journal.load_view() == views[-1]
    assert ids <= set(decodes)
    decodes.clear()
    with open_journal(tmp_path, store, settings, cache) as journal:
        assert journal.load_view() == views[-1]
    assert not ids & set(decodes)


def test_scope_key_separates_every_component(tmp_path):
    store, settings, *_ = chain(tmp_path, 1)
    other_store = EventStore(tmp_path / "other" / "events.sqlite3")
    policy = rolling.ReasoningReplayPolicy("codex_backend", "gpt-test")
    key = RollingVerificationCache.scope_key
    base = key(store, "s", settings.digest, None)
    variants = [
        key(other_store, "s", settings.digest, None),
        key(store, "t", settings.digest, None),
        key(store, "s", replace(settings, max_summary_bytes=4097).digest, None),
        key(store, "s", settings.digest, policy),
        key(store, "s", settings.digest, rolling.ReasoningReplayPolicy("codex_backend", "m")),
    ]
    assert len({base, *variants}) == len(variants) + 1
    assert key(store, "s", settings.digest, policy) == key(
        store, "s", settings.digest, rolling.ReasoningReplayPolicy("codex_backend", "gpt-test")
    )


def test_failed_open_records_nothing(tmp_path):
    store, settings, views, _, paths = chain(tmp_path, 4)
    journal_sql(paths, "UPDATE campaign SET next_sequence=next_sequence+1")
    cache = RollingVerificationCache()
    with pytest.raises(ValueError, match="corrupt rolling sequence"):
        open_journal(tmp_path, store, settings, cache)
    assert cache.stats() == {"entries": 0, "source_ids": 0}


def test_open_failing_after_validating_records_nothing(tmp_path):
    store, settings, views, successor, paths = chain(tmp_path, 4, pending=True)
    # Reconciliation fully validates every job and generation before it reaches
    # the unjournaled artifact and fails: none of that work may be recorded.
    mutate("unjournaled_artifact", store, settings, views, successor, paths)
    cache = RollingVerificationCache()
    with pytest.raises(ValueError, match="unjournaled or mismatching generation"):
        open_journal(tmp_path, store, settings, cache)
    assert cache.stats() == {"entries": 0, "source_ids": 0}


def test_failed_operation_keeps_only_earlier_completed_operations(tmp_path):
    store, settings, views, _, paths = chain(tmp_path, 4)
    cache = RollingVerificationCache()
    with open_journal(tmp_path, store, settings, cache) as journal:
        recorded = cache.stats()
        assert recorded["entries"] > 0
        with pytest.raises(RuntimeError):
            with journal._operation():
                journal._load_view()
                raise RuntimeError("boom")
        assert cache.stats() == recorded


def test_bounds_evict_least_recently_used_without_changing_results(tmp_path, monkeypatch):
    store, settings, views, successor, _ = chain(tmp_path, 6, pending=True)
    expected = observe(tmp_path, store, settings)
    for bounds in ({"max_entries": 3}, {"max_source_ids": 20}, {"max_source_ids": 1}):
        cache = RollingVerificationCache(**bounds)
        assert observe(tmp_path, store, settings, cache) == expected
        stats = cache.stats()
        assert stats["entries"] <= bounds.get("max_entries", stats["entries"])
        assert stats["source_ids"] <= bounds.get("max_source_ids", stats["source_ids"])
        assert observe(tmp_path, store, settings, cache) == expected


def test_eviction_drops_the_least_recently_used_entry():
    scope = ("archive", "s", "0" * 64, type(None), None)

    def entries(cache):
        return [digest for _, digest in cache._entries]

    cache = RollingVerificationCache(max_entries=3)
    cache.put_all(scope, [("generation", name, name, 1) for name in "abc"])
    assert cache.get(scope, "generation", "a") == "a"  # a is now most recent
    cache.put_all(scope, [("generation", "d", "d", 1)])
    assert entries(cache) == ["c", "a", "d"]  # b, the least recently used, went
    assert cache.get(scope, "generation", "b") is None

    cache = RollingVerificationCache(max_source_ids=10)
    cache.put_all(scope, [("generation", "a", "a", 4), ("generation", "b", "b", 4)])
    assert cache.get(scope, "generation", "a") == "a"
    cache.put_all(scope, [("generation", "c", "c", 4)])
    assert entries(cache) == ["a", "c"] and cache.stats()["source_ids"] == 8
    cache.put_all(scope, [("generation", "e", "e", 11)])  # heavier than the bound
    assert cache.get(scope, "generation", "e") is None and entries(cache) == ["a", "c"]


def test_put_all_never_downgrades_a_validated_target():
    cache = RollingVerificationCache()
    scope = ("archive", "s", "0" * 64, type(None), None)
    target = object()
    validated = rolling._KnownTarget(target, ((), (), ()))
    cache.put_all(scope, [("target", "t", validated, 1)])
    cache.put_all(scope, [("target", "t", rolling._KnownTarget(target, None), 1)])
    assert cache.get(scope, "target", "t") is validated
    cache.put_all(("other",), [])
    assert cache.get(scope, "target", "t") is None


@pytest.mark.parametrize(
    "bounds", [{"max_entries": 0}, {"max_source_ids": -1}, {"max_entries": 1.5}]
)
def test_invalid_bounds_refused(bounds):
    with pytest.raises(ValueError, match="cache bounds"):
        RollingVerificationCache(**bounds)


def test_journal_refuses_foreign_cache_objects(tmp_path):
    store, settings, *_ = chain(tmp_path, 1)
    with pytest.raises(ValueError, match="verification cache"):
        open_journal(tmp_path, store, settings, cache={})


def test_engine_requests_reuse_its_cache(tmp_path, monkeypatch):
    from test_rolling_engine import complete, select, setup_engine
    from test_rolling_selection import exchange

    engine, db, _ = setup_engine(tmp_path, monkeypatch)
    for tag in "ABCD":
        db.rows += exchange(tag, complete=False)[:1]
        select(engine, db, tag)
        db.rows += exchange(tag)[1:]
        complete(engine, db, tag)
    assert engine._published.sequence == 3
    db.rows += exchange("E", complete=False)[:1]
    assert select(engine, db, "E") is not None
    published = engine._published
    # Every generation artifact of the published chain, named by its id.
    chain_ids = {path.stem for path in Path(engine._home).rglob("generations/*.json")}
    assert published.generation_id in chain_ids and len(chain_ids) == published.sequence
    decodes = spy_decodes(monkeypatch)
    canonicalized = spy(monkeypatch, rolling, "_canonical_rows")
    assert select(engine, db, "E") is not None
    assert engine._published == published
    # A repeated request decodes no generation of the chain and canonicalizes no
    # archive row (selection still validates the current host rows themselves).
    assert not chain_ids & set(decodes)
    assert canonicalized["_canonical_rows"] <= 2
    assert engine.clone_for_agent()._verification is not engine._verification
