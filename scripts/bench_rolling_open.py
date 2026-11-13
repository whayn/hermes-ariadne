"""Benchmark: rolling journal open cost versus generations and source events (#22).

Synthetic and offline: no Hermes imports, model, network or live home. Run it
with the Ariadne tree under test first on ``PYTHONPATH`` and a scratch ``TMPDIR``:

    PYTHONPATH=src python scripts/bench_rolling_open.py build DIR --events 4096
    PYTHONPATH=src python scripts/bench_rolling_open.py measure DIR [--profile 25]

``build`` publishes ``--generations`` (default 64) generations through the
journal API (freeze, reserve, ready, publish). Their deltas total exactly
``--events`` canonical rows: native-shaped Codex exchanges of four rows (user,
assistant tool call with retained reasoning, tool result, final assistant), with
the exchanges spread as evenly as possible. Settings are the defaults except
``trigger_bytes=1`` and delta/request/checkpoint byte limits raised so large
deltas fit; ``max_prefix_events`` stays at its default 100,000, so
``--events 100000`` exercises the real cap.

``measure`` copies the fixture to a fresh temporary home (an open may write the
journal; ``--in-place`` opens DIR itself when disk space is short), then reports
one JSON line: the open (``RollingJournal(...)``, which
runs ``_reconcile``), ``load_view``, generation-artifact reads, and the bytes
produced by ``encode_json`` during the open. Fixtures are byte-identical across
Ariadne revisions that preserve durable identities, so one fixture can be
measured by several trees (set ``PYTHONPATH`` to each tree's ``src``). Wall
times are single runs on whatever host runs them; ``--profile`` adds cProfile
overhead and is reported separately from the untraced open.
"""

import argparse
import copy
import cProfile
import io
import json
import os
import pstats
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import hermes_ariadne
from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.prepared import ReasoningReplayPolicy
from hermes_ariadne.store import EventStore

SESSION = "s"
POLICY = ReasoningReplayPolicy("codex_backend", "gpt-test")
ROUTE = ("codex_backend", "gpt-test")
REQUEST = b'[{"content":"summarize","role":"user"}]'
ROWS_PER_EXCHANGE = 4


def settings():
    return rolling.RollingSettings(
        trigger_bytes=1,
        max_delta_exchanges=2**20,
        max_readable_delta_bytes=2**26,
        max_request_bytes=2**26,
        max_checkpoint_bytes=2**26,
    )


def exchange(tag):
    """One completed native-shaped Codex exchange (four canonical rows)."""

    def row(role, content, suffix, **extra):
        return {
            "session_id": SESSION,
            "message_uid": tag + suffix,
            "role": role,
            "content": content,
            "tool_calls": None,
            "tool_call_id": None,
            "tool_name": None,
            "effect_disposition": None,
            "reasoning": None,
            "reasoning_content": None,
            "reasoning_details": None,
            "codex_reasoning_items": None,
            "codex_message_items": None,
            "tool_call_uids": None,
            "tool_call_uid": None,
            "absorbed_message_uids": "[]",
            "display_metadata": None,
            "active": 1,
            **extra,
        }

    def reasoning(suffix, count):
        return json.dumps(
            [
                {
                    "type": "reasoning",
                    "id": "rs_" + tag + suffix + str(i),
                    "encrypted_content": "OPAQUE_" + tag + suffix + str(i),
                    "summary": [] if i % 2 == 0 else [{"type": "summary_text", "text": "note"}],
                    "_issuer_kind": ROUTE[0],
                    "_issuer_model": ROUTE[1],
                }
                for i in range(count)
            ]
        )

    text = " answer " + tag + " "
    return [
        row("user", " question " + tag + " ", "u"),
        row(
            "assistant",
            None,
            "o",
            tool_calls=[
                {
                    "id": "call_" + tag,
                    "call_id": "call_" + tag,
                    "response_item_id": "fc_native_" + tag,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"exact"}'},
                }
            ],
            tool_call_uids=json.dumps({"call_" + tag: "occ_" + tag}),
            codex_reasoning_items=reasoning("owner", 1),
            reasoning="",
        ),
        row(
            "tool",
            "result " + tag,
            "t",
            tool_call_id="call_" + tag,
            tool_name="lookup",
            effect_disposition="none",
            tool_call_uid="occ_" + tag,
        ),
        row(
            "assistant",
            text,
            "a",
            reasoning="readable companion",
            reasoning_content="",
            codex_reasoning_items=reasoning("answer", 4),
            codex_message_items=json.dumps(
                [
                    {
                        "type": "message",
                        "id": "msg_" + tag,
                        "role": "assistant",
                        "status": "completed",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ]
            ),
        ),
    ]


def build(directory, events, generations):
    if events % ROWS_PER_EXCHANGE or events < ROWS_PER_EXCHANGE * generations:
        raise SystemExit("--events must be a multiple of 4 and at least 4 per generation")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(f"{directory} is not empty")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = settings()
    store = EventStore(directory / "ariadne" / "events.sqlite3")
    base, extra = divmod(events // ROWS_PER_EXCHANGE, generations)
    started = time.perf_counter()
    rows, event_ids, parent = [], [], None
    with rolling_state.RollingJournal(
        directory, SESSION, config, store=store, policy=POLICY
    ) as journal:
        for sequence in range(1, generations + 1):
            count = base + (sequence <= extra)
            chunk = [r for j in range(count) for r in exchange(f"g{sequence}x{j}")]
            store.ingest(SESSION, chunk, origin="canonical")
            start = len(rows)
            rows += chunk
            event_ids += [
                rolling.sha256(rolling.encode_json(r, config.max_checkpoint_bytes)) for r in chunk
            ]
            delta = rolling.readable_delta(copy.deepcopy(chunk), policy=POLICY, settings=config)
            # Built field by field so construction stays O(delta); freeze_target
            # then validates it against the archive exactly like any other target.
            frozen = rolling.Target(
                session_id=SESSION,
                settings_digest=config.digest,
                parent_generation=None if parent is None else parent.generation_id,
                parent_summary_sha256=None if parent is None else parent.summary_sha256,
                event_ids=tuple(event_ids),
                message_uids=tuple(r["message_uid"] for r in rows),
                delta_start=start,
                delta_stop=len(rows),
                readable_input=rolling._summary_input(
                    None if parent is None else parent.summary_text, delta, config
                ),
                request_bytes=REQUEST,
                route_labels=ROUTE,
            )
            journal.freeze_target(frozen)
            if not journal.reserve(frozen.job_id, REQUEST, lambda: None, lambda: None):
                raise RuntimeError("reservation refused")
            summary = f"summary {sequence}"
            journal.record_ready(frozen.job_id, summary, {"provider": "t", "model": "t"}, None)
            parent = journal.publish_ready(
                frozen.job_id,
                rolling.generation_bytes(frozen, summary, sequence=sequence, settings=config),
            )
    return {
        "command": "build",
        "events": len(rows),
        "generations": generations,
        "head": parent.generation_id,
        "build_s": round(time.perf_counter() - started, 2),
    }


def retained_mib(cache):
    """Deep size of the objects the cache's entries keep alive (each object once)."""
    seen, size = set(), 0
    stack = [known for known, _ in cache._entries.values()]
    while stack:
        item = stack.pop()
        if id(item) in seen or item is None or isinstance(item, (type, int, bool)):
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            stack.extend(item)
        elif hasattr(item, "__dict__"):
            size += sys.getsizeof(vars(item))
            stack.extend(vars(item).values())
    return round(size / 2**20, 1)


def measure(directory, profile, in_place=False, cache=False):
    config = settings()
    # --cache: one RollingVerificationCache shared by every open below, as an
    # engine holds one. The first open is cold (it populates the cache); a
    # second, warm open is then timed separately.
    extra = {"cache": rolling.RollingVerificationCache()} if cache else {}
    with tempfile.TemporaryDirectory(prefix="bench-rolling-") as scratch:
        home = Path(directory) if in_place else Path(scratch) / "home"
        if not in_place:
            shutil.copytree(directory, home)
        store = EventStore(home / "ariadne" / "events.sqlite3")
        reads = {"count": 0, "bytes": 0}
        encoded = {"calls": 0, "bytes": 0}
        read, encode = rolling_state._read, rolling.encode_json

        def counting_read(path, limit):
            data = read(path, limit)
            if path.parent.name == "generations":
                reads["count"] += 1
                reads["bytes"] += len(data)
            return data

        def counting_encode(value, limit):
            data = encode(value, limit)
            encoded["calls"] += 1
            encoded["bytes"] += len(data)
            return data

        rolling_state._read = counting_read
        rolling.encode_json = rolling_state.encode_json = counting_encode
        try:
            started = time.perf_counter()
            journal = rolling_state.RollingJournal(
                home, SESSION, config, store=store, policy=POLICY, **extra
            )
            open_s = time.perf_counter() - started
            open_reads, open_encoded = dict(reads), dict(encoded)
            try:
                started = time.perf_counter()
                view = journal.load_view()
                load_s = time.perf_counter() - started
            finally:
                journal.close()
            if cache:
                reads.update(count=0, bytes=0)
                encoded.update(calls=0, bytes=0)
                started = time.perf_counter()
                journal = rolling_state.RollingJournal(
                    home, SESSION, config, store=store, policy=POLICY, **extra
                )
                warm_open_s = time.perf_counter() - started
                warm_reads, warm_encoded = dict(reads), dict(encoded)
                try:
                    started = time.perf_counter()
                    assert journal.load_view() == view
                    warm_load_s = time.perf_counter() - started
                finally:
                    journal.close()
        finally:
            rolling_state._read, rolling.encode_json = read, encode
            rolling_state.encode_json = encode
        result = {
            "command": "measure",
            "source": str(Path(hermes_ariadne.__file__).resolve().parent),
            "python": sys.version.split()[0],
            "events": len(view.event_ids),
            "generations": view.sequence,
            "open_s": round(open_s, 3),
            "open_artifact_reads": open_reads["count"],
            "open_artifact_mib": round(open_reads["bytes"] / 2**20, 2),
            "open_encode_json_calls": open_encoded["calls"],
            "open_encode_json_mib": round(open_encoded["bytes"] / 2**20, 2),
            "load_view_s": round(load_s, 3),
            "maxrss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        }
        if cache:
            result.update(
                warm_open_s=round(warm_open_s, 3),
                warm_open_artifact_reads=warm_reads["count"],
                warm_open_encode_json_mib=round(warm_encoded["bytes"] / 2**20, 2),
                warm_load_view_s=round(warm_load_s, 3),
                cache=extra["cache"].stats(),
                cache_retained_mib=retained_mib(extra["cache"]),
            )
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
            rolling_state.RollingJournal(
                home, SESSION, config, store=store, policy=POLICY, **extra
            ).close()
            profiler.disable()
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
            stats.print_stats("hermes_ariadne", profile)
            result["profile"] = stream.getvalue()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="publish a synthetic fixture into DIR")
    build_parser.add_argument("directory")
    build_parser.add_argument("--events", type=int, required=True)
    build_parser.add_argument("--generations", type=int, default=64)
    measure_parser = commands.add_parser("measure", help="time one journal open of DIR")
    measure_parser.add_argument("directory")
    measure_parser.add_argument("--profile", type=int, default=0, metavar="LINES")
    measure_parser.add_argument(
        "--in-place",
        action="store_true",
        help="open DIR itself instead of a copy (large fixtures; the open may write it)",
    )
    measure_parser.add_argument(
        "--cache",
        action="store_true",
        help="share a RollingVerificationCache across opens and also time a warm open",
    )
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.directory, args.events, args.generations)
    else:
        result = measure(args.directory, args.profile, args.in_place, args.cache)
    profile = result.pop("profile", None)
    print(json.dumps(result), flush=True)
    if profile:
        print(profile, file=sys.stderr)


if __name__ == "__main__":
    os.umask(0o077)
    main()
