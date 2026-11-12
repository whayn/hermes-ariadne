"""Journal open validation cost (#19): bounded archive reads without skipped checks.

Controller-owned oracle: public journal behaviour only, no memo interface assumed.
"""

import json
import sqlite3
from collections import Counter

import pytest
from test_rolling_state import ready, setup, target

from hermes_ariadne import rolling, rolling_state


def chain(tmp_path, count, *, pending=False, **limits):
    """Publish ``count`` generations; optionally leave one ready successor pending."""
    store, settings = setup(tmp_path, **limits)
    views = []
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        parent = None
        for sequence in range(1, count + 1):
            frozen = target(store, settings, sequence, parent)
            ready(journal, frozen, f"summary {sequence}")
            parent = journal.publish_ready(
                frozen.job_id,
                rolling.generation_bytes(frozen, f"summary {sequence}", sequence=sequence),
            )
            views.append(parent)
        successor = None
        if pending:
            successor = target(store, settings, count + 1, parent)
            ready(journal, successor, "pending summary")
        paths = journal.path, journal.generations, journal.manifest_path
    return store, settings, views, successor, paths


def count_reads(monkeypatch, store):
    reads = Counter()
    original = store.get_event

    def counting(event_id, *, session_id):
        reads[event_id] += 1
        return original(event_id, session_id=session_id)

    monkeypatch.setattr(store, "get_event", counting)
    return reads


def referenced(views, successor):
    events = set()
    for view in views:
        events.update(view.event_ids)
    if successor is not None:
        events.update(successor.event_ids)
    return events


@pytest.mark.parametrize("count", [1, 4, 8])
def test_open_reads_each_archive_event_once(tmp_path, monkeypatch, count):
    store, settings, views, successor, _ = chain(tmp_path, count, pending=True)
    events = referenced(views, successor)
    reads = count_reads(monkeypatch, store)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        # Every referenced source is still read and hash-checked, but only once per open.
        assert set(reads) == events
        assert max(reads.values()) == 1
        assert sum(reads.values()) == len(events) == 2 * (count + 1)
        reads.clear()
        assert journal.load_view() == views[-1]
        assert not reads  # same open retains its insert-only source snapshot


def test_each_open_rereads_sources_no_cross_open_cache(tmp_path, monkeypatch):
    store, settings, views, _, _ = chain(tmp_path, 3)
    reads = count_reads(monkeypatch, store)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        for _ in range(2):
            reads.clear()
            journal.load_view()
            assert not reads  # operations within this open share validation
    reads.clear()
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
        assert set(reads) == set(views[-1].event_ids)


def test_shared_ancestor_validated_once_per_open(tmp_path, monkeypatch):
    store, settings, views, _, _ = chain(tmp_path, 5, pending=True)
    checked = Counter()
    original = rolling_state._generation_matches

    def counting(view, target, summary):
        checked[view.sequence] += 1
        return original(view, target, summary)

    # The journal cross-check runs once for every newly validated generation.
    monkeypatch.setattr(rolling_state, "_generation_matches", counting)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
        assert checked == Counter(range(1, 6))


# --- Mutation tests: the memoized path must still reject tampered state. ---


def test_ancestor_bytes_changed_after_first_validation_are_rejected(tmp_path, monkeypatch):
    store, settings, views, _, _ = chain(tmp_path, 4, pending=True)
    root = views[0]
    forged = json.loads(root.snapshot_bytes)
    forged["summary_text"] = "forged"
    forged["summary_sha256"] = rolling.sha256(b"forged")
    forged = rolling.encode_json(forged, settings.max_checkpoint_bytes)
    original = rolling_state.RollingJournal._parent_bytes
    loads = Counter()

    def swap_after_first(self, digest):
        loads[digest] += 1
        data = original(self, digest)
        return forged if digest == root.generation_id and loads[digest] > 1 else data

    monkeypatch.setattr(rolling_state.RollingJournal, "_parent_bytes", swap_after_first)
    # Either layer may refuse first (parent lineage, or the journal cross-check
    # when the forged bytes are read as a top-level parent); both fail closed.
    with pytest.raises(ValueError, match="lineage|unjournaled or mismatching generation"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    assert loads[root.generation_id] > 1, "mutation never reached a memoized ancestor"


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("observation", "generation source unavailable or changed"),
        ("payload", "archived event payload does not match its identity"),
    ],
)
def test_archive_tamper_of_shared_root_source_refuses_open(tmp_path, mutation, message):
    store, settings, views, _, paths = chain(tmp_path, 5, pending=True)
    event = views[0].event_ids[0]
    with sqlite3.connect(store.path) as conn:
        if mutation == "observation":
            conn.execute("DELETE FROM observations WHERE event_id=?", (event,))
        else:
            conn.execute(
                "UPDATE events SET payload_json=CAST(? AS BLOB) WHERE event_id=?",
                (b'{"message_uid":"0-user","role":"user","content":"forged"}', event),
            )
    with pytest.raises(ValueError, match=message):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    # Refusal never resets: charged accounting and published lineage survive.
    with sqlite3.connect(paths[0]) as conn:
        assert conn.execute("SELECT logical_reservations FROM campaign").fetchone()[0] == 6
        assert conn.execute("SELECT count(*) FROM jobs WHERE state='published'").fetchone()[0] == 5


def test_archive_change_between_opens_is_seen(tmp_path):
    store, settings, views, _, _ = chain(tmp_path, 4)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view() == views[-1]
        with sqlite3.connect(store.path) as conn:
            conn.execute("DELETE FROM observations WHERE event_id=?", (views[0].event_ids[0],))
        # Direct archive mutation is unsupported within an open snapshot.
        assert journal.load_view() == views[-1]
    with pytest.raises(ValueError, match="source"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)


def test_ancestor_journal_text_mismatch_refuses_open(tmp_path):
    store, settings, views, _, paths = chain(tmp_path, 5)
    with sqlite3.connect(paths[0]) as conn:
        # Self-consistent row (hash updated) so only the generation cross-check can catch it.
        conn.execute(
            "UPDATE jobs SET validated_text='forged',text_sha256=? WHERE job_id=?",
            (rolling.sha256(b"forged"), views[1].job_id),
        )
    with pytest.raises(ValueError, match="unjournaled or mismatching generation"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)


def test_interrupted_publication_artifact_with_memoized_parent_is_checked(tmp_path):
    store, settings, views, successor, paths = chain(tmp_path, 5, pending=True)
    # A leftover artifact whose parent is the (memoized) head but whose text the
    # journal never accepted must still be rejected, not trusted via its parent.
    data = rolling.generation_bytes(successor, "forged summary", sequence=6)
    artifact = paths[1] / (rolling.sha256(data) + ".json")
    artifact.write_bytes(data)
    artifact.chmod(0o600)
    with pytest.raises(ValueError, match="unjournaled or mismatching generation"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
    artifact.unlink()
    # The legitimate interrupted publication is still recovered on open.
    good = rolling.generation_bytes(successor, "pending summary", sequence=6)
    (paths[1] / (rolling.sha256(good) + ".json")).write_bytes(good)
    (paths[1] / (rolling.sha256(good) + ".json")).chmod(0o600)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.publish_ready(successor.job_id, good).sequence == 6
        assert journal.report()["logical_reservations"] == 6


@pytest.mark.parametrize(
    "sql,message",
    [
        (
            "UPDATE campaign SET logical_reservations=logical_reservations-1",
            "rolling accounting mismatch",
        ),
        (
            "UPDATE campaign SET cumulative_request_bytes=cumulative_request_bytes+1",
            "rolling accounting mismatch",
        ),
        ("UPDATE campaign SET next_sequence=next_sequence+1", "corrupt rolling sequence"),
    ],
)
def test_accounting_tamper_in_long_chain_refuses_open(tmp_path, sql, message):
    # Regression guard (also passes on the base commit): memoization must not weaken this.
    store, settings, _, _, paths = chain(tmp_path, 5, pending=True)
    with sqlite3.connect(paths[0]) as conn:
        conn.execute(sql)
    with pytest.raises(ValueError, match=message):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store)
