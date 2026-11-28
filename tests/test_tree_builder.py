"""Profile tree builder: planning, reserved model calls, lanes, background worker."""

import contextvars
import hashlib
import os
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_ariadne import profile_tree as pt
from hermes_ariadne import tree_builder
from hermes_ariadne.store import EventStore, canonical_json
from hermes_ariadne.tree import Part
from hermes_ariadne.tree_builder import TASK, TreeBuilder, TreeSettings, estimate
from hermes_ariadne.tree_store import TreeStore

DATE = "2026-01-02T03:04:05Z"


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o755)
    os.chmod(home, 0o755)
    return home


class Log:
    """A fixture home: the exact archive plus the profile log that points into it."""

    def __init__(self, home):
        self.home = home
        self.store = TreeStore(home)
        self.archive = EventStore(home / "ariadne" / "events.sqlite3")
        self.sources = {}
        self._n = 0

    def add(self, text, *, role="user", session="s1", origin="live", extra=None):
        self._n += 1
        message = {"role": role, "content": text, "message_uid": f"uid-{self._n}", **(extra or {})}
        self.archive.ingest(
            session, [message], origin=origin if origin == "import" else "hermes_db"
        )
        event_id = hashlib.sha256(canonical_json(message)).hexdigest()
        item = {
            "event_id": event_id,
            "message_uid": message["message_uid"],
            "role": role,
            "date": DATE,
            "restricted": False,
        }
        (position,) = self.store.append(session, [item], origin=origin)
        entry = self.store.entry(position)
        self.sources[position] = pt.leaf_source(
            pt.LogEntry(**entry.__dict__),
            message,  # same projection the builder reads
        )
        return position


@pytest.fixture
def log(home):
    return Log(home)


def long(tag="x"):
    return tag * 700  # over node_bytes: needs a model call


class FakeLlm:
    """Scripted ``ctx.llm``: ``script(n, messages, kwargs)`` -> text or an exception."""

    def __init__(self, script=None, *, usage=None):
        self.script = script
        self.calls = []
        self.lock = threading.Lock()
        self.usage = usage or SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=0.0,
        )

    def complete(self, messages, **kwargs):
        with self.lock:
            self.calls.append((messages, kwargs))
            n = len(self.calls)
        out = f"summary {n}" if self.script is None else self.script(n, messages, kwargs)
        if isinstance(out, BaseException):
            raise out
        return SimpleNamespace(
            text=out, provider="p", model="m", agent_id=None, usage=self.usage, audit=None
        )

    def what(self):
        """The node each call was for, in call order: ``('leaf', i)`` or ``('merge', a, b)``."""
        found = []
        for messages, _ in self.calls:
            user = messages[1]["content"]
            leaf = re.search(r"compress message (\d+) into", user)
            merge = re.search(r"merge lines (\d+)\+(\d+) and", user)
            found.append(
                ("leaf", int(leaf[1])) if leaf else ("merge", int(merge[1]), int(merge[2]))
            )
        return found


def builder(home, llm, **settings):
    settings.setdefault("jobs", 4)
    return TreeBuilder(home, llm, TreeSettings(**settings))


def drain(b, limit=50):
    for _ in range(limit):
        if not b.step()["progress"]:
            return
    raise AssertionError("builder never went idle")


# -- settings -----------------------------------------------------------------------------


def test_settings_defaults_and_validation():
    s = TreeSettings()
    assert (s.high, s.seen_high, s.node_bytes, s.jobs) == (128000, 32000, 512, 4)
    assert (s.max_jobs, s.max_request_bytes, s.max_total_request_bytes) == (0, 65536, 0)
    assert (s.tail_bytes, s.tries, s.timeout) == (32768, 5, 300)
    assert 0 < s.max_tokens
    for name in ("high", "jobs", "node_bytes", "tries", "max_tokens", "timeout"):
        for bad in (0, -1, True, 1.5, "4", None):
            with pytest.raises(ValueError):
                TreeSettings(**{name: bad})
    with pytest.raises(ValueError):
        TreeSettings(max_request_bytes=10)


def test_lifetime_limits_accept_zero_and_reject_invalid_values():
    for name in ("max_jobs", "max_total_request_bytes"):
        assert getattr(TreeSettings(**{name: 0}), name) == 0
        for bad in (-1, True, 1.5, "1", None):
            with pytest.raises(ValueError):
                TreeSettings(**{name: bad})


def test_unlimited_reservation_reports_taken_not_caps(home, log):
    log.add(long())
    b = builder(home, FakeLlm())
    part = Part(0, 0)
    assert b._reserve(part, 10) == ("ok", 1)
    assert b._reserve(part, 10) == ("taken", 0)
    assert b.status()["caps_exhausted"] is False


def test_exit_wait_and_retry_defaults_and_validation():
    s = TreeSettings()
    assert s.exit_wait == 120 and (s.retry_base, s.retry_max) == (30, 3600)
    assert TreeSettings(exit_wait=0).exit_wait == 0  # 0 = never wait
    assert TreeSettings(exit_wait=2.5).exit_wait == 2.5  # int or float
    assert TreeSettings.from_config({"exit_wait": 5, "retry_base": 2}).retry_base == 2
    assert [s.retry_delay(n) for n in (1, 2, 3, 8, 50)] == [30, 60, 120, 3600, 3600]
    for bad in (-1, -0.5, True, "5", None, float("nan"), float("inf"), 10**6):
        with pytest.raises(ValueError):
            TreeSettings(exit_wait=bad)
    for name in ("retry_base", "retry_max"):
        for bad in (0, -1, True, 1.5, "1", None, 10**6):
            with pytest.raises(ValueError):
                TreeSettings(**{name: bad})


def test_settings_from_config_rejects_unknown_keys_and_bad_values():
    assert TreeSettings.from_config(None) == TreeSettings()
    assert TreeSettings.from_config({}) == TreeSettings()
    assert TreeSettings.from_config({"jobs": 2, "tries": 3}).jobs == 2
    with pytest.raises(ValueError, match="unknown"):
        TreeSettings.from_config({"jobz": 2})
    with pytest.raises(ValueError):
        TreeSettings.from_config({"jobs": True})
    with pytest.raises(ValueError):
        TreeSettings.from_config([("jobs", 2)])


def test_plugin_yaml_schema_matches_settings_defaults():
    from dataclasses import asdict
    from pathlib import Path

    text = (Path(tree_builder.__file__).parents[2] / "plugin.yaml").read_text()
    block = text.split("\n  tree:\n", 1)[1]  # the manifest has no YAML parser in CI
    defaults = block.split("    default:\n", 1)[1]
    found = {}
    for line in defaults.splitlines():
        match = re.fullmatch(r"      (\w+): (\d+)", line)
        if not match:
            break
        found[match[1]] = int(match[2])
    assert found == asdict(TreeSettings())


def test_register_summary_task_uses_a_plugin_owned_key():
    seen = []

    class Ctx:
        def register_auxiliary_task(self, key, **kwargs):
            seen.append((key, kwargs))

    tree_builder.register_summary_task(Ctx())
    ((key, kwargs),) = seen
    assert key == TASK == "ariadne_summary"
    assert kwargs["display_name"] == "Ariadne summaries"
    assert "chat model" in kwargs["description"] and "summar" in kwargs["description"]
    assert kwargs.get("defaults") is None and kwargs.get("inherit_from") is None


# -- estimate -----------------------------------------------------------------------------


def test_estimate_counts_calls_on_known_cases():
    assert estimate([]) == {
        "leaves": 0,
        "leaf_calls": 0,
        "internal_calls": 0,
        "calls": 0,
        "request_bytes": 0,
    }
    # four messages: two over 512 bytes; the full tree has 3 internal nodes (2 + 1)
    e = estimate([100, 600, 700, 50])
    assert (e["leaves"], e["leaf_calls"], e["internal_calls"], e["calls"]) == (4, 2, 3, 5)
    assert e["request_bytes"] > 0
    # exactly node_bytes still fits; one more does not
    assert estimate([512, 513])["leaf_calls"] == 1
    # three messages: one complete pair, so one internal node
    assert estimate([10, 10, 10])["internal_calls"] == 1
    assert estimate([10] * 8)["internal_calls"] == 7
    assert estimate([10, 10], node_bytes=8)["leaf_calls"] == 2


def test_estimate_for_an_appended_batch_adds_only_the_new_calls():
    whole = estimate([100, 600, 700, 50, 900])
    before = estimate([100, 600, 700])
    added = estimate([50, 900], existing=3)
    assert added["leaf_calls"] == 1
    assert added["internal_calls"] == whole["internal_calls"] - before["internal_calls"]
    assert added["calls"] == whole["calls"] - before["calls"]
    assert estimate([], existing=5)["calls"] == 0
    with pytest.raises(ValueError):
        estimate([1], existing=-1)
    with pytest.raises(TypeError):
        estimate([True])


# -- building -----------------------------------------------------------------------------


def test_fitting_leaf_is_stored_without_a_model_call(home, log):
    log.add("hello")
    log.add("world")
    llm = FakeLlm()
    b = builder(home, llm)
    report = b.step()
    nodes = log.store.nodes()
    assert nodes[Part(0, 0)] == pt.flat(log.sources[0])
    assert nodes[Part(0, 1)] == pt.flat(log.sources[1])
    assert report["fits"] == 2
    assert llm.what() == [("merge", 0, 1)]  # only the parent asks the model
    assert set(log.store.jobs()) == {Part(1, 0)}  # no reservation for a shortcut
    assert nodes[Part(1, 0)] == "summary 1" or log.store.nodes()[Part(1, 0)] == "summary 1"


def test_restricted_leaves_build_without_a_model_call(home, log):
    log.add(long(), extra={"data_use": ["no-train"]})  # message-level restriction
    # An entry restricted at admission is never even read from the archive.
    log.store.append(
        "s1",
        [
            {
                "event_id": "a" * 64,
                "message_uid": "u-r",
                "role": "assistant",
                "date": DATE,
                "restricted": True,
            }
        ],
        origin="live",
    )
    llm = FakeLlm()
    b = builder(home, llm)
    b.step()
    nodes = log.store.nodes()
    assert nodes[Part(0, 0)].endswith(pt.RESTRICTED) and nodes[Part(0, 1)].endswith(pt.RESTRICTED)
    assert llm.what() == [("merge", 0, 1)]  # the two placeholders are merged, never sent


def test_build_order_is_due_order_and_parents_follow_children(home, log):
    for i in range(4):
        log.add(long(str(i)))
    llm = FakeLlm()
    b = builder(home, llm, jobs=2)
    drain(b)
    order = llm.what()
    assert order == [
        ("leaf", 0),
        ("leaf", 1),
        ("merge", 0, 1),  # a merge before the leaf it ties with (due)
        ("leaf", 2),
        ("leaf", 3),
        ("merge", 2, 1),
        ("merge", 0, 2),
    ]
    assert set(log.store.nodes()) == {Part(0, i) for i in range(4)} | {
        Part(1, 0),
        Part(1, 1),
        Part(2, 0),
    }


def test_each_call_carries_task_purpose_and_limits(home, log):
    log.add(long())
    llm = FakeLlm()
    b = builder(home, llm, max_tokens=300, timeout=33)
    drain(b)
    ((messages, kwargs),) = llm.calls
    assert kwargs == {
        "max_tokens": 300,
        "timeout": 33,
        "purpose": "ariadne-tree-summary",
        "task": TASK,
    }
    assert [m["role"] for m in messages] == ["system", "user"]
    assert pt.COMPACTION_RULES in messages[0]["content"]
    assert log.sources[0] in messages[1]["content"]


def test_first_unbuilt_leaf_caps_the_context_of_every_job_in_a_pass(home, log):
    for i in range(4):
        log.add(long(str(i)))
    llm = FakeLlm()
    b = builder(home, llm, jobs=4)
    b.step()
    assert len(llm.calls) == 4
    for messages, _ in llm.calls:
        assert "<chat>\n</chat>" in messages[1]["content"]  # nothing before leaf 0 is built


def test_context_holds_the_seen_view_lines_up_to_the_cutoff(home, log):
    for i in range(3):
        log.add(long(str(i)))
    llm = FakeLlm(lambda n, m, k: f"line{n}")
    b = builder(home, llm, jobs=1)
    drain(b)
    chats = [re.search(r"<chat>\n(.*?)</chat>", m[1]["content"], re.S)[1] for m, _ in llm.calls]
    assert llm.what() == [("leaf", 0), ("leaf", 1), ("merge", 0, 1), ("leaf", 2)]
    assert chats[0] == ""
    assert chats[1] == "0+1|line1\n"
    assert chats[2] == "0+1|line1\n1+1|line2\n"  # the merge is shown the leaves it merges
    assert chats[3] == chats[2]  # leaf 2: seen is still under its high, nothing merged
    seen, revision = log.store.load_view("seen")
    assert seen is not None and revision >= 1


def test_over_length_line_is_asked_again_with_the_over_note(home, log):
    log.add(long())
    first = "a" * 600

    def script(n, messages, kwargs):
        return first if n == 1 else "short"

    llm = FakeLlm(script)
    b = builder(home, llm)
    drain(b)
    assert len(llm.calls) == 2
    one, two = (m[1]["content"] for m, _ in llm.calls)
    assert two == one + "\n" + pt.over(first)
    assert log.store.nodes()[Part(0, 0)] == "short"
    job = log.store.jobs()[Part(0, 0)]
    assert (job.state, job.attempts) == ("done", 2)
    assert log.store.counters()["logical_jobs"] == 2  # each ask is its own reservation


def test_attempts_are_capped_and_the_shortest_line_is_kept(home, log):
    log.add(long())
    lengths = {1: 700, 2: 520, 3: 650}
    llm = FakeLlm(lambda n, m, k: "b" * lengths[n])
    b = builder(home, llm, tries=3)
    drain(b)
    assert len(llm.calls) == 3
    node = log.store.nodes()[Part(0, 0)]
    assert node == pt.cut("b" * 520, 512) and len(node.encode()) <= 512
    assert log.store.jobs()[Part(0, 0)].state == "done"


def test_empty_answer_clips_the_source(home, log):
    log.add(long())
    log.add(long("y"))
    llm = FakeLlm(lambda n, m, k: "   \n ")
    b = builder(home, llm)
    drain(b)
    nodes = log.store.nodes()
    assert nodes[Part(0, 0)] == pt.cut(pt.flat(log.sources[0]), 512)
    expected = pt.cut(pt.flat(pt.merge_input(nodes[Part(0, 0)], nodes[Part(0, 1)])), 512)
    assert nodes[Part(1, 0)] == expected
    assert len(llm.calls) == 3  # a refusal is final: no second ask


def test_explicit_retry_rearms_an_uncertain_job(home, log):
    log.add(long())
    llm = FakeLlm(lambda n, m, k: RuntimeError("down") if n == 1 else "fine")
    b = builder(home, llm)
    drain(b)
    assert log.store.nodes() == {}
    assert b.retry() == [Part(0, 0)]
    drain(b)
    assert log.store.nodes()[Part(0, 0)] == "fine"
    job = log.store.jobs()[Part(0, 0)]
    assert (job.state, job.attempts) == ("done", 2)


def test_non_text_result_is_uncertain(home, log):
    log.add(long())

    class Weird(FakeLlm):
        def complete(self, messages, **kwargs):
            super().complete(messages, **kwargs)
            return SimpleNamespace(text=b"bytes", usage=self.usage)

    b = builder(home, Weird())
    drain(b)
    assert log.store.jobs()[Part(0, 0)].state == "uncertain"


def test_default_builder_continues_after_old_lifetime_caps(home, log):
    log.add(long())
    with log.store._connect(write=True) as conn:
        conn.execute("UPDATE counters SET value=100000 WHERE name='logical_jobs'")
        conn.execute("UPDATE counters SET value=? WHERE name='request_bytes'", (1 << 30,))
    llm = FakeLlm()
    b = builder(home, llm)
    drain(b)
    assert len(llm.calls) == 1
    assert b.status()["caps_exhausted"] is False
    counters = log.store.counters()
    assert counters["logical_jobs"] == 100001
    assert counters["request_bytes"] == (1 << 30) + len(tree_builder._request_json(llm.calls[0][0]))


def test_job_cap_stops_the_pass_and_is_reported(home, log):
    for i in range(4):
        log.add(long(str(i)))
    llm = FakeLlm()
    b = builder(home, llm, max_jobs=2)
    report = b.step()
    assert len(llm.calls) == 2 and report["caps_exhausted"] is True
    assert b.status()["caps_exhausted"] is True
    assert b.step()["progress"] is False and len(llm.calls) == 2
    assert log.store.counters()["logical_jobs"] == 2


def test_request_byte_cap_stops_before_the_first_call_and_leaves_no_reservation(home, log):
    for i in range(3):
        log.add(long(str(i)))
    llm = FakeLlm()
    b = builder(home, llm, jobs=1, max_total_request_bytes=5000)
    drain(b)
    size = len(tree_builder._request_json(llm.calls[0][0]))
    assert size <= 5000 < 2 * size  # premise: one request fits, two do not
    assert len(llm.calls) == 1
    assert b.status()["caps_exhausted"] is True
    assert log.store.counters()["request_bytes"] == size


def test_oversized_source_is_clipped_to_the_request_cap(home, log):
    log.add("z" * 100_000)
    llm = FakeLlm()
    b = builder(home, llm, max_request_bytes=8192)
    drain(b)
    ((messages, _),) = llm.calls
    size = len(tree_builder._request_json(messages))
    assert size <= 8192
    job = log.store.jobs()[Part(0, 0)]
    assert job.request_bytes == size  # the reserved bytes are the bytes sent
    assert "clipped" in messages[1]["content"]
    assert log.store.nodes()[Part(0, 0)] == "summary 1"


def test_source_with_many_escaped_characters_is_clipped_by_serialized_size(home, log):
    log.add('"' * 70_000)  # every quote doubles in the request JSON
    llm = FakeLlm()
    b = builder(home, llm)  # default settings
    drain(b)
    ((messages, _),) = llm.calls
    size = len(tree_builder._request_json(messages))
    assert size <= b.settings.max_request_bytes
    assert log.store.jobs()[Part(0, 0)].request_bytes == size
    assert "clipped" in messages[1]["content"]
    assert log.store.nodes()[Part(0, 0)] == "summary 1"


def test_missing_message_leaves_the_leaf_pending_and_creates_no_archive(home):
    store = TreeStore(home)
    store.append(
        "s1",
        [
            {
                "event_id": "b" * 64,
                "message_uid": "u",
                "role": "user",
                "date": DATE,
                "restricted": False,
            }
        ],
        origin="live",
    )
    llm = FakeLlm()
    b = builder(home, llm)
    report = b.step()
    assert report["progress"] is False and llm.calls == []
    assert b.status()["last_error"] == "source_unavailable"
    assert not (home / "ariadne" / "events.sqlite3").exists()
    assert store.nodes() == {}


def test_an_unreadable_message_is_retried_after_a_delay_not_every_pass(home, log):
    now = [1000.0]
    message = {"role": "user", "content": "late arrival", "message_uid": "u"}
    event_id = hashlib.sha256(canonical_json(message)).hexdigest()
    entry = {"event_id": event_id, "message_uid": "u", "role": "user", "date": DATE}
    log.store.append("s1", [{**entry, "restricted": False}], origin="live")
    b = TreeBuilder(home, FakeLlm(), TreeSettings(), clock=lambda: now[0])
    assert b.step()["progress"] is False  # the archive does not have it yet
    log.archive.ingest("s1", [message], origin="hermes_db")
    now[0] += 5
    assert b.step()["progress"] is False  # too soon: not read again on every pass
    now[0] += 60
    assert b.step()["progress"] is True
    assert log.store.nodes()[Part(0, 0)].endswith("user: late arrival")


def test_unavailable_prefix_does_not_stall_planning_of_later_messages(home, log):
    now = [1000.0]
    for n in range(97):  # more than the 8 * jobs + 64 scan allowance, none archived
        item = {
            "event_id": hashlib.sha256(f"missing-{n}".encode()).hexdigest(),
            "message_uid": f"missing-{n}",
            "role": "user",
            "date": DATE,
            "restricted": False,
        }
        log.store.append("s1", [item], origin="live")
    valid = log.add(long())
    assert valid == 97
    llm = FakeLlm()
    b = TreeBuilder(home, llm, TreeSettings(), clock=lambda: now[0])
    for _ in range(5):  # bounded: the valid message must not wait for the prefix to clear
        b.step()
        if llm.calls:
            break
    assert llm.what() == [("leaf", valid)]
    assert Part(0, valid) in log.store.nodes()


def test_two_builders_on_one_store_never_call_the_same_node_twice(home, log):
    for i in range(6):
        log.add(long(str(i)))

    def slow(n, messages, kwargs):
        time.sleep(0.05)  # widen the window in which both have planned the same jobs
        return f"s{n}"

    llm = FakeLlm(slow)
    one, two = builder(home, llm, jobs=3), builder(home, llm, jobs=3)
    start = threading.Barrier(2)

    def run(b):
        start.wait()
        drain(b)

    threads = [threading.Thread(target=run, args=(b,)) for b in (one, two)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    what = llm.what()
    assert len(what) == len(set(what)) == 6 + 4  # 6 leaves, 4 internal nodes, each once
    assert len(log.store.nodes()) == 10
    assert log.store.counters()["logical_jobs"] == 10


def test_restart_turns_dead_reservations_uncertain_and_keeps_live_ones(home, log):
    for i in range(4):
        log.add(long(str(i)))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    caps = {"max_jobs": 100, "max_total_bytes": 10**6}
    log.store.reserve(Part(0, 0), 10, f"{dead.pid}:gone", **caps)  # another process, now dead
    log.store.reserve(Part(0, 1), 10, f"{os.getpid()}:earlier-instance", **caps)  # same pid, gone
    log.store.reserve(Part(0, 2), 10, f"{os.getppid()}:elsewhere", **caps)  # a live process
    mine = builder(home, FakeLlm())
    log.store.reserve(Part(0, 3), 10, mine.owner, **caps)  # this instance, in flight
    llm = FakeLlm()
    other = builder(home, llm)
    other.recover()
    jobs = log.store.jobs()
    assert [jobs[Part(0, i)].state for i in range(4)] == [
        "uncertain",
        "uncertain",
        "reserved",
        "reserved",
    ]
    assert jobs[Part(0, 0)].error_type == "owner_lost"
    drain(other)
    # The lost ones are asked again (each a new reservation); in-flight ones are not.
    assert sorted(llm.what()[:2]) == [("leaf", 0), ("leaf", 1)]
    assert {("leaf", 2), ("leaf", 3)}.isdisjoint(llm.what())


def test_first_step_recovers_before_planning(home, log):
    log.add(long())
    log.store.reserve(Part(0, 0), 10, f"{os.getpid()}:old", max_jobs=10, max_total_bytes=10**6)
    llm = FakeLlm()
    drain(builder(home, llm))
    job = log.store.jobs()[Part(0, 0)]
    assert len(llm.calls) == 1 and (job.state, job.attempts) == ("done", 2)


def test_chat_view_is_advanced_and_persisted_with_cas(home, log):
    for i in range(3):
        log.add(f"short {i}")
    b = builder(home, FakeLlm())
    drain(b)
    view, revision = log.store.load_view("chat")
    assert view is not None and view.parts[-1].stop == 3 and revision >= 1
    assert view.high == 128000
    log.add("one more")
    drain(b)
    view2, revision2 = log.store.load_view("chat")
    assert view2.parts[-1].stop == 4 and revision2 > revision
    drain(b)
    assert log.store.load_view("chat")[1] == revision2  # unchanged view: nothing rewritten


def test_a_view_changed_by_another_writer_is_re_read_not_overwritten(home, log):
    log.add("one")
    b = builder(home, FakeLlm())
    real = log.store.save_view
    calls = []

    def racing(name, view, expected):
        calls.append((name, expected))
        if len(calls) == 1:  # someone else saves first
            real(name, view, expected)
        return real(name, view, expected)

    b._store.save_view = racing
    drain(b)
    view, revision = log.store.load_view("chat")
    assert view.parts[-1].stop == 1
    # the lost CAS was re-read, found equal to what the other writer stored, not repeated
    assert calls == [("chat", 0)] and revision == 1


# -- lanes --------------------------------------------------------------------------------


def test_imports_build_in_their_own_lane_live_first(home, log):
    for i in range(3):
        log.add(long(f"i{i}"), session="old", origin="import")
    live = log.add(long("L"), session="now", origin="live")
    llm = FakeLlm()
    b = builder(home, llm, jobs=1)
    b.step()
    assert llm.what() == [("leaf", live), ("leaf", 0)]  # one per lane, live first
    drain(b)
    assert Part(0, live) in log.store.nodes()


def test_import_backlog_does_not_delay_or_starve_a_live_node(home, log):
    for i in range(4):
        log.add(long(f"i{i}"), session="old", origin="import")
    live = log.add(long("L"), session="now", origin="live")
    llm = FakeLlm()
    b = builder(home, llm, jobs=2, max_jobs=1)  # one call allowed in total
    b.step()
    assert llm.what() == [("leaf", live)]  # the live node got the only call
    # an unbuilt import leaf before it neither blocks nor appears as a hole in its context
    assert "<chat>\n</chat>" in llm.calls[0][0][1]["content"]


# -- background worker --------------------------------------------------------------------


def wait_idle(b, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        thread = b._thread
        if thread is None or not thread.is_alive():
            return
        time.sleep(0.01)
    raise AssertionError("worker did not go idle")


def test_live_job_is_reserved_before_an_import_when_the_cap_fits_only_one(home, log):
    imported = log.add(long("i"), session="old", origin="import")
    live = log.add(long("L"), session="now", origin="live")
    llm = FakeLlm()
    b = builder(home, llm, jobs=1, max_jobs=1)  # one logical job in total
    gate = threading.Event()
    execute = b._execute

    def delayed(job, *args, **kwargs):
        if job.lane == "live":
            assert gate.wait(10)  # the live thread is slow to get going
        return execute(job, *args, **kwargs)

    b._execute = delayed
    b.start()
    try:
        wait_for(lambda: log.store.jobs())
        assert set(log.store.jobs()) == {Part(0, live)}  # reserved by the planner, in priority
    finally:
        gate.set()
    wait_idle(b)
    assert llm.what() == [("leaf", live)]
    assert Part(0, imported) not in log.store.jobs()
    b.stop()


def test_worker_exit_and_a_late_kick_cannot_lose_a_wake(home, log):
    log.add("short")  # stored without a call: the worker then goes idle
    llm = FakeLlm()
    b = builder(home, llm)
    late = []

    class Hooked:
        """A lock that runs a hook right after the worker releases it."""

        def __init__(self, real):
            self.real, self.hook = real, None

        def __enter__(self):
            self.real.acquire()
            return self

        def __exit__(self, *exc):
            self.real.release()
            hook, self.hook = self.hook, None
            if hook:
                hook()

    class Wake(threading.Event):
        def is_set(self):  # only the worker's idle check asks
            found = super().is_set()
            if not found and not late:
                late.append(True)

                def new_entries_arrive():
                    log.add(long("n"))
                    b.kick()  # lands after the idle check, before the worker is gone

                b._lock.hook = new_entries_arrive
            return found

    b._lock = Hooked(b._lock)
    b._wake = Wake()
    b.start()
    wait_for(lambda: Part(0, 1) in log.store.nodes(), timeout=3)  # the late entry is built
    wait_idle(b)
    assert late
    b.stop()


def test_start_kick_stop_never_block_and_the_work_gets_done(home, log):
    for i in range(4):
        log.add(long(str(i)))
    gate = threading.Event()

    def blocked(n, messages, kwargs):
        assert gate.wait(10)
        return f"s{n}"

    llm = FakeLlm(blocked)
    b = builder(home, llm)
    t0 = time.time()
    b.start()
    b.kick()
    b.kick()
    assert time.time() - t0 < 1  # the model is stuck, the caller is not
    deadline = time.time() + 5
    while b.status()["running"] < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert b.status()["running"] >= 1
    gate.set()
    wait_idle(b)
    assert len(log.store.nodes()) == 7
    assert b.status()["running"] == 0
    # new entries restart an idle worker
    log.add(long("n"))
    b.kick()
    wait_idle(b)
    assert Part(0, 4) in log.store.nodes()
    t1 = time.time()
    b.stop(timeout=5)
    assert time.time() - t1 < 5 and b._thread is None


def test_jobs_run_concurrently_up_to_the_setting(home, log):
    for i in range(3):
        log.add(long(str(i)))
    meet = threading.Barrier(3, timeout=10)

    def together(n, messages, kwargs):
        if "compress message" in messages[1]["content"]:
            meet.wait()  # all three leaves must be in flight at once
        return f"s{n}"

    llm = FakeLlm(together)
    b = builder(home, llm, jobs=3)
    b.start()
    wait_idle(b)
    assert all(Part(0, i) in log.store.nodes() for i in range(3))
    assert not meet.broken
    b.stop()


def test_worker_calls_run_in_the_callers_context(home, log):
    log.add(long())
    marker = contextvars.ContextVar("profile", default="none")
    marker.set("bound-profile")
    seen = []

    def spy(n, messages, kwargs):
        seen.append(marker.get())
        return "ok"

    b = builder(home, FakeLlm(spy))
    b.start()
    wait_idle(b)
    assert seen == ["bound-profile"]
    b.stop()


def test_worker_survives_an_llm_failure_and_waits_for_the_retry(home, log):
    log.add(long())
    b = builder(home, FakeLlm(lambda n, m, k: RuntimeError("x")), retry_base=3600)
    b.start()
    wait_for(lambda: log.store.jobs().get(Part(0, 0)) is not None)
    wait_for(lambda: log.store.jobs()[Part(0, 0)].state == "uncertain")
    assert b._thread is not None and b._thread.is_alive()  # waiting, not gone
    b.stop()
    assert b._thread is None


def test_status_reports_counts_and_unknown_usage(home, log):
    for i in range(2):
        log.add(long(str(i)))
    llm = FakeLlm()
    b = builder(home, llm, jobs=1)
    assert b.status()["count"] == 2 and b.status()["built"] == 0
    assert b.status()["pending_ready"] == 2
    drain(b)
    status = b.status()
    assert status["count"] == 2 and status["built"] == 3 and status["pending_ready"] == 0
    assert status["counters"]["logical_jobs"] == 3
    assert status["running"] == 0 and status["caps_exhausted"] is False
    assert status["last_error"] is None
    assert status["usage"] == {"known_calls": 0, "unknown_calls": 3, "total_tokens": 0}


def test_reported_usage_is_counted_only_when_present(home, log):
    log.add(long())
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.0,
    )
    b = builder(home, FakeLlm(usage=usage))
    drain(b)
    assert b.status()["usage"] == {"known_calls": 1, "unknown_calls": 0, "total_tokens": 15}


# -- lost owners: asked again at once -------------------------------------------------------


def dead_owner():
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    return f"{dead.pid}:gone"


CAPS = {"max_jobs": 100, "max_total_bytes": 10**6}


def test_start_asks_a_lost_job_again_and_builds_the_node(home, log):
    log.add(long())
    log.store.reserve(Part(0, 0), 10, dead_owner(), **CAPS)  # a process that exited mid-call
    llm = FakeLlm()
    b = builder(home, llm)
    b.start()
    wait_idle(b)
    assert log.store.nodes()[Part(0, 0)] == "summary 1"
    job = log.store.jobs()[Part(0, 0)]
    assert (job.state, job.attempts) == ("done", 2)
    assert log.store.counters()["logical_jobs"] == 2  # the re-ask is a charged reservation
    assert b.status()["retrying"] == 0
    b.stop()


def test_start_on_an_empty_home_creates_nothing(home):
    b = builder(home, FakeLlm())
    b.start()
    b.stop()
    assert not (home / "ariadne").exists()


# -- waiting at exit ----------------------------------------------------------------------


def wait_running(b, count=1, timeout=5):
    """``count`` jobs are in flight and their calls have been sent (reservation is past)."""

    def sent():
        status = b.status()
        return status["running"] >= count and status["calls"] >= count

    wait_for(sent, timeout)


def test_stop_waits_for_an_inflight_call_to_commit_and_schedules_nothing_new(home, log):
    log.add(long("a"))
    log.add(long("b"))
    gate = threading.Event()
    llm = FakeLlm(lambda n, m, k: gate.wait(10) and f"s{n}")
    b = builder(home, llm, jobs=1)
    b.start()
    wait_running(b)
    threading.Timer(0.3, gate.set).start()
    t0 = time.monotonic()
    b.stop(timeout=10)
    waited = time.monotonic() - t0
    assert 0.2 < waited < 5  # it waited for the call, not for the whole timeout
    assert log.store.nodes() == {Part(0, 0): "s1"}  # the in-flight call committed
    assert log.store.jobs()[Part(0, 0)].state == "done"
    assert b.status()["running"] == 0
    time.sleep(0.3)  # the finished call must not wake the worker into new work
    assert len(llm.calls) == 1 and Part(0, 1) not in log.store.jobs()


def test_stop_without_wait_returns_at_once_and_leaves_the_call_reserved(home, log):
    log.add(long())
    gate = threading.Event()
    b = builder(home, FakeLlm(lambda n, m, k: gate.wait(10) and "late"))
    b.start()
    wait_running(b)
    t0 = time.monotonic()
    b.stop(timeout=0)
    assert time.monotonic() - t0 < 1
    assert log.store.jobs()[Part(0, 0)].state == "reserved" and log.store.nodes() == {}
    gate.set()  # cleanup: the daemon thread finishes by itself
    wait_for(lambda: Part(0, 0) in log.store.nodes())


def test_stop_timeout_gives_up_on_a_slow_call_and_it_stays_reserved(home, log):
    log.add(long())
    gate = threading.Event()
    b = builder(home, FakeLlm(lambda n, m, k: gate.wait(10) and "late"))
    b.start()
    wait_running(b)
    t0 = time.monotonic()
    b.stop(timeout=0.3)
    assert 0.25 < time.monotonic() - t0 < 3
    job = log.store.jobs()[Part(0, 0)]
    assert job.state == "reserved" and job.owner == b.owner  # the next start reconciles it
    gate.set()
    wait_for(lambda: Part(0, 0) in log.store.nodes())


def test_stop_is_still_waiting_after_an_earlier_stop_without_wait(home, log):
    log.add(long())
    gate = threading.Event()
    b = builder(home, FakeLlm(lambda n, m, k: gate.wait(10) and "late"))
    b.start()
    wait_running(b)
    b.stop(timeout=0)  # a session end: stops scheduling, does not wait
    threading.Timer(0.3, gate.set).start()
    b.stop(timeout=10)  # the process exit that follows does wait
    assert log.store.nodes() == {Part(0, 0): "late"}


def test_stop_does_not_send_the_over_limit_re_ask(home, log):
    log.add(long())
    gate = threading.Event()
    llm = FakeLlm(lambda n, m, k: gate.wait(10) and "y" * 600)  # over node_bytes: asks again
    b = builder(home, llm)
    b.start()
    wait_running(b)
    threading.Timer(0.3, gate.set).start()
    b.stop(timeout=10)
    assert len(llm.calls) == 1  # no second ask once stopped
    job = log.store.jobs()[Part(0, 0)]
    assert job.state == "refused" and log.store.nodes() == {}  # reservable again later


def test_stop_during_the_re_ask_reservation_sends_no_second_call(home, log):
    log.add(long())
    llm = FakeLlm(lambda n, m, k: "y" * 600)  # over node_bytes: asks again
    b = builder(home, llm)
    reserve = b._reserve
    entered, release, count = threading.Event(), threading.Event(), []

    def slow(part, size):
        count.append(1)
        if len(count) == 2:  # the re-ask, past its stop check
            entered.set()
            release.wait(10)
        return reserve(part, size)

    b._reserve = slow
    b.start()
    assert entered.wait(10)
    b.stop(timeout=0)
    release.set()
    wait_for(lambda: b.status()["running"] == 0)
    time.sleep(0.2)
    assert len(llm.calls) == 1  # the reserved re-ask was released, not sent
    job = log.store.jobs()[Part(0, 0)]
    assert job.state == "refused" and log.store.nodes() == {}


def test_a_job_launched_after_stop_is_never_sent_or_reserved(home, log):
    log.add(long())
    llm = FakeLlm()
    b = builder(home, llm)
    plan = b._plan({"live": 1, "import": 1})
    assert len(plan.jobs) == 1
    b.stop(timeout=0)
    b._launch(plan.jobs[0])  # the worker was mid-plan when stop() came
    time.sleep(0.2)
    assert llm.calls == [] and log.store.jobs() == {} and b.status()["running"] == 0
    assert b.kick() is False


def wait_for(check, timeout=5):
    deadline = time.time() + timeout
    while not check() and time.time() < deadline:
        time.sleep(0.01)
    assert check()


def test_unavailable_prefix_does_not_stall_the_fit_sweep(home, log):
    now = [1000.0]
    n_missing = tree_builder.SWEEP_READS + 10
    for n in range(n_missing):  # more than the sweep's read budget, none archived
        item = {
            "event_id": hashlib.sha256(f"missing-{n}".encode()).hexdigest(),
            "message_uid": f"missing-{n}",
            "role": "user",
            "date": DATE,
            "restricted": False,
        }
        log.store.append("s1", [item], origin="live")
    short = log.add("fits without a call")
    b = TreeBuilder(home, FakeLlm(), TreeSettings(), clock=lambda: now[0])
    for _ in range(5):
        b.step()
        if Part(0, short) in log.store.nodes():
            break
    assert Part(0, short) in log.store.nodes()
