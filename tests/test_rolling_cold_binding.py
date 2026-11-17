"""Exact old-path verdicts and operation ownership for cold delta binding."""

import json
import shutil
from collections import Counter
from contextlib import contextmanager

import pytest
from test_rolling_cold_validation import delta_chain

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.store import EventStore


class ReferenceJournal(rolling_state.RollingJournal):
    """The two replaced paths from eeb5e0c; other validation is unchanged.

    Baseline startup _validate_records had no binding operation. Each explicit
    delta-parent lookup performed _view(_parent_bytes(id)) unconditionally.
    """

    @contextmanager
    def _operation(self, *, promote=True):
        if not promote:
            yield
        else:
            with super()._operation():
                yield

    def _parent_view(self, digest):
        return self._view(self._parent_bytes(digest))


def read_outcome(cls, home, settings):
    store = EventStore(home / "ariadne" / "events.sqlite3")
    try:
        with cls(home, "s", settings, store=store) as journal:
            view, pending = journal.load_view(), journal.pending_target()
            return (
                "ok",
                None if view is None else view.snapshot_bytes,
                None if pending is None else pending.to_bytes(settings),
                journal.report(),
                journal.charged_callback_ids(),
            )
    except Exception as exc:
        return type(exc), str(exc)


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "root-bytes",
        "wrong-parent-file",
        "extra-artifact",
        "missing-root",
        "manifest",
        "target-hash",
        "charge",
    ],
)
def test_cold_binding_matches_old_path_verdict(tmp_path, mutation):
    home = tmp_path / "fixture"
    store, settings, views, _, directory = delta_chain(home, 8)
    root = directory / "generations" / (views[0].generation_id + ".json")
    if mutation == "root-bytes":
        value = json.loads(root.read_bytes())
        value["summary_text"] = "changed"
        value["summary_sha256"] = rolling.sha256(b"changed")
        rolling_state._atomic(root, rolling.encode_json(value, settings.max_checkpoint_bytes))
    elif mutation == "wrong-parent-file":
        # A valid, journaled artifact at another valid artifact's filename.
        rolling_state._atomic(
            directory / "generations" / (views[-1].generation_id + ".json"), views[0].snapshot_bytes
        )
    elif mutation == "extra-artifact":
        rolling_state._atomic(directory / "generations" / ("0" * 64 + ".json"), b"{}")
    elif mutation == "missing-root":
        root.unlink()
    elif mutation == "manifest":
        value = json.loads((directory / "manifest.json").read_bytes())
        value["generation_id"] = "0" * 64
        rolling_state._atomic(directory / "manifest.json", rolling.encode_json(value, 8192))
    elif mutation in {"target-hash", "charge"}:
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(directory / "journal.sqlite3")) as conn:
            with conn:
                if mutation == "charge":
                    conn.execute("UPDATE campaign SET logical_reservations=0")
                else:
                    conn.execute("UPDATE targets SET payload_sha256=?", ("0" * 64,))
    # Recreate the same synthetic pathname so even filesystem exception text
    # compares exactly; restore the input after reference recovery may write it.
    candidate = tmp_path / "candidate"
    shutil.copytree(home, candidate)
    expected = read_outcome(ReferenceJournal, candidate, settings)
    shutil.rmtree(candidate)
    shutil.copytree(home, candidate)
    actual = read_outcome(rolling_state.RollingJournal, candidate, settings)
    assert actual == expected
    assert (actual[0] == "ok") == (mutation == "valid"), actual


def test_delta_parent_binding_is_once_per_operation_not_per_open(tmp_path, monkeypatch):
    store, settings, views, successor, _ = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        reads = Counter()
        original = journal._parent_bytes

        def counted(digest):
            reads[digest] += 1
            return original(digest)

        monkeypatch.setattr(journal, "_parent_bytes", counted)
        for _ in range(2):
            reads.clear()
            with journal._operation():
                assert journal._validate_target(successor) == successor
                assert reads == Counter(view.generation_id for view in views)
                reads.clear()
                assert journal._validate_target(successor) == successor
                assert not reads
            assert journal._memo.bound is None


def test_wrong_parent_name_does_not_bind_a_different_valid_view(tmp_path, monkeypatch):
    store, settings, views, successor, _ = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        original = journal._parent_bytes
        swapped = []

        def wrong_once(digest):
            if digest == views[-1].generation_id and not swapped:
                swapped.append(digest)
                return views[0].snapshot_bytes
            return original(digest)

        monkeypatch.setattr(journal, "_parent_bytes", wrong_once)
        with journal._operation():
            with pytest.raises(ValueError, match="parent prefix extension"):
                journal._validate_target(successor)
            assert views[-1].generation_id not in journal._memo.bound
            assert journal._validate_target(successor) == successor
            assert views[-1].generation_id in journal._memo.bound


def test_next_operation_rechecks_distant_ancestor_and_recovers_cleanly(tmp_path):
    store, settings, views, successor, directory = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with journal._operation():
            journal._validate_target(successor)
        root = directory / "generations" / (views[0].generation_id + ".json")
        value = json.loads(views[0].snapshot_bytes)
        value["summary_text"] = "changed"
        value["summary_sha256"] = rolling.sha256(b"changed")
        rolling_state._atomic(root, rolling.encode_json(value, settings.max_checkpoint_bytes))
        with pytest.raises(ValueError, match="lineage"):
            journal.load_view()
        assert journal._memo is None and not journal._busy
        rolling_state._atomic(root, views[0].snapshot_bytes)
        restored = journal.pending_target()
        assert tuple(restored.event_ids) == tuple(successor.event_ids)
        assert tuple(restored.message_uids) == tuple(successor.message_uids)


def test_binding_hit_still_checks_capture_and_keeps_caught_reentry_owned(tmp_path):
    store, settings, views, successor, _ = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with journal._operation():
            journal._validate_target(successor)
            bound = set(journal._memo.bound)
            with pytest.raises(ValueError, match="reentrant"):
                journal.load_view()
            assert journal._memo.bound == bound

            def revoke():
                raise RuntimeError("synthetic capture revoked")

            journal.guard = revoke
            with pytest.raises(RuntimeError, match="capture revoked"):
                journal._parent_view(views[-1].generation_id)
            journal.guard = None
            assert journal._memo.bound == bound


def test_failed_startup_audit_never_promotes_cache_facts(tmp_path):
    store, settings, views, _, directory = delta_chain(tmp_path, 8)
    cache = rolling.RollingVerificationCache()
    rolling_state._atomic(directory / "generations" / ("0" * 64 + ".json"), b"{}")
    assert cache.stats()["entries"] == 0
    with pytest.raises(ValueError, match="generation schema"):
        rolling_state.RollingJournal(tmp_path, "s", settings, store=store, cache=cache)
    assert cache.stats()["entries"] == 0


def test_delta_startup_audit_has_reentry_owner(tmp_path, monkeypatch):
    store, settings, *_ = delta_chain(tmp_path, 8)
    original = rolling_state.RollingJournal._validate_records
    checked = []

    def audit(self):
        with pytest.raises(ValueError, match="reentrant"):
            self.report()
        checked.append(self._memo.bound == set())
        return original(self)

    monkeypatch.setattr(rolling_state.RollingJournal, "_validate_records", audit)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store):
        pass
    assert checked == [True]


@pytest.mark.parametrize("field", ["store", "session_id", "policy", "settings_digest"])
def test_bound_parent_hit_refuses_a_different_memo_scope(tmp_path, monkeypatch, field):
    store, settings, views, successor, _ = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with journal._operation():
            journal._validate_target(successor)
            changed = "different" if field in {"session_id", "settings_digest"} else object()
            with monkeypatch.context() as patch:
                patch.setattr(journal._memo, field, changed)
                with pytest.raises(ValueError, match="scope mismatch"):
                    journal._parent_view(views[-1].generation_id)
            assert journal._parent_view(views[-1].generation_id) == views[-1]


@pytest.mark.parametrize("fault", ["unaccepted", "untied-prefix"])
def test_unaccepted_bound_parent_cannot_skip_the_reader(tmp_path, monkeypatch, fault):
    store, settings, views, successor, _ = delta_chain(tmp_path, 8)
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        with journal._operation():
            journal._validate_target(successor)
            if fault == "unaccepted":
                journal._memo.accepted.pop(views[-1].generation_id)
            else:
                from dataclasses import replace

                recorded = journal._memo.prefix_states[views[-1].generation_id]
                journal._memo.prefix_states[views[-1].generation_id] = (
                    replace(recorded[0]),
                    recorded[1],
                )
            reads = []
            original = journal._parent_bytes

            def counted(digest):
                reads.append(digest)
                return original(digest)

            monkeypatch.setattr(journal, "_parent_bytes", counted)
            assert journal._parent_view(views[-1].generation_id) == views[-1]
            assert views[-1].generation_id in reads
