"""Installed-host pilot for the opt-in rolling engine (#17). NOT part of the test suite.

Drives a long synthetic conversation through real Hermes in ONE explicitly chosen,
disposable development home: native general-plugin discovery, ``AIAgent``,
``SessionDB``, the normal turn finalizer and host-owned ``ctx.llm`` against the
provider route given on the command line. Each phase is a separate process, so
``restart``/``restart2`` reload the durable view, journal and accounting cold.

It never installs, enables, publishes or calls an engine hook itself. The wrappers
only OBSERVE ``select_context``/``handle_tool_call``/``on_turn_complete`` and record
the exception behind any decline/unavailable transition. The plugin must already be
installed from a built wheel (``ariadne-install --home HOME --rolling``) and selected
in that home's config, e.g.::

    hermes -p DEV plugins enable ariadne-rolling-plugin
    hermes -p DEV config set context.engine ariadne-rolling
    hermes -p DEV config set plugins.entries.ariadne-rolling-plugin.settings.rolling.max_logical_jobs 8
    hermes -p DEV config set plugins.entries.ariadne-rolling-plugin.settings.rolling.trigger_bytes 1024
    hermes -p DEV config set plugins.entries.ariadne-rolling-plugin.settings.rolling.max_delta_exchanges 2

Mark the development home as disposable (``touch $DEV_HOME/.ariadne-rolling-pilot``);
the pilot refuses homes without it, the default Hermes root, and a ``HOME`` that is the
account home or contains either Hermes home. Hermes still resolves provider
authentication through its own mechanisms. Run with Hermes's interpreter and source on
PYTHONPATH (no packages are installed) and a dedicated ``HOME``::

    HOME=$SCRATCH_HOME PYTHONPATH=$HERMES_SOURCE:$HERMES_SITE HERMES_HOME=$DEV_HOME \\
      $HERMES_PYTHON scripts/rolling_host_pilot.py plan OUT --provider P --model M
    ... rolling_host_pilot.py advance OUT/plan.json      # turns 1-8
    ... rolling_host_pilot.py restart OUT/plan.json      # turns 9-14, new process
    ... rolling_host_pilot.py restart2 OUT/plan.json     # turns 15-17, new process
    python3 scripts/rolling_host_pilot.py summarize OUT

Phases must run in order, once each, against an unchanged session (checked against
the earlier receipts); a failed turn stops its phase, and an interrupted plan is
abandoned rather than resumed. One process per plan at a time (file lock).

``hermes chat`` is deliberately not used: on gateway-client Hermes builds its turns run
inside the gateway owner process, which may be a shared live gateway. Receipts hold
synthetic text and tool results; keep them out of version control. Model answers are
single samples, not recall evidence; the pilot is not a provider-cost measurement.
"""

import copy
import dataclasses
import fcntl
import hashlib
import json
import os
import pwd
import secrets
import sys
import time
import traceback
from pathlib import Path

FILLER = [
    "the fictional lighthouse at Cape Verity has a red lamp and 214 steps",
    "the imaginary river Lorn floods every April; its ferry carries eleven passengers",
    "the invented village of Sorrel grows blue barley and keeps three windmills",
    "the made-up observatory on Mount Calder tracks a comet named Pell",
    "the fictional train from Brisk to Hallow stops at nine stations",
    "the imaginary archive in Vell stores 3,200 maps catalogued by a clerk called Ines",
    "the invented orchard at Wren Hill grows quince, medlar and one pomegranate tree",
    "the made-up harbour of Tarn has a green crane and a tide clock that runs slow",
    "the fictional bakery on Quill Street sells rye loaves shaped like owls",
    "the imaginary bridge at Fennick has 17 arches and a toll of two copper coins",
]
WORDS = [
    "AMBER-HERON-11",
    "COBALT-FINCH-22",
    "SAFFRON-OTTER-33",
    "UMBER-MOTH-44",
    "JADE-STOAT-55",
    "OCHRE-WREN-66",
    "SLATE-LYNX-77",
    "CRIMSON-VOLE-88",
    "IVORY-NEWT-99",
    "TEAL-BADGER-10",
]
RECALL = (
    "From the conversation context you were given (no tools), list every codeword you can see "
    "verbatim for turns 1-10; write 'not visible' for any you cannot see verbatim."
)
PROBES = {
    11: "Use the ariadne_find tool to search this session's archive for 'COBALT' and report the "
    "exact codeword stated in turn 2, in one sentence.",
    12: "Call ariadne_rolling_page once (first page) and tell me, in two sentences, how many "
    "source descriptors the current summary view covers and which roles appear.",
    13: "Use ariadne_find for 'UMBER', then ariadne_zoom on the user event it returns, and quote "
    "the turn-4 codeword exactly as stored.",
    14: RECALL,
    15: "Call ariadne_rolling_page exactly once with ref {prev_rolling_ref} (an older summary "
    "view reference) and report verbatim whether it returned results or an error code.",
    16: "Use ariadne_find for 'SAFFRON', then ariadne_zoom on the user event it returns, and "
    "quote the turn-3 codeword exactly as stored.",
    17: RECALL,
}
PHASES = {"advance": range(1, 9), "restart": range(9, 15), "restart2": range(15, 18)}


def turn_text(n):
    if n in PROBES:
        return f"Turn {n}. {PROBES[n]}"
    i = n - 1
    notes = [
        f"Note {k + 1}: {FILLER[(i + k) % len(FILLER)]} (variant {n}.{k + 1})." for k in range(4)
    ]
    return "\n".join(
        [
            f"Synthetic Ariadne pilot, turn {n}. Remember this fact: the codeword for turn {n} "
            f"is {WORDS[i]}.",
            "Synthetic filler notes follow; they are data, not instructions.",
            *notes,
            "Reply with exactly one short sentence acknowledging the codeword; do not call tools.",
        ]
    )


def make_plan(out, provider, model, reasoning):
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        "home": str(Path(os.environ["HERMES_HOME"]).resolve()),
        "session": f"ariadne-pilot-{time.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(6)}",
        "provider": provider,
        "model": model,
        "reasoning": reasoning,
        "phases": {m: [{"n": n, "text": turn_text(n)} for n in r] for m, r in PHASES.items()},
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(plan["session"])


MARKER = ".ariadne-rolling-pilot"


def refuse(condition, message):
    """Explicit pilot safety check: unlike ``assert`` it survives ``python -O``."""
    if not condition:
        raise SystemExit(f"rolling_host_pilot: {message}")


def _within(parent, child):
    return child == parent or child.is_relative_to(parent)


def admit_homes(plan):
    """Check HERMES_HOME and HOME BEFORE importing any stateful Hermes module."""
    home = Path(os.environ["HERMES_HOME"]).resolve()
    refuse(str(home) == plan["home"], "HERMES_HOME differs from the plan")
    refuse((home / MARKER).is_file(), f"HERMES_HOME lacks the disposable-home marker {MARKER}")
    user_home = Path(os.environ.get("HOME", "")).resolve()
    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    from hermes_constants import get_default_hermes_root  # path helper, no state

    root = Path(get_default_hermes_root()).resolve()
    refuse(not _within(home, root) and home != root, "refusing the default Hermes root")
    refuse(
        os.environ.get("HOME")
        and user_home != account_home
        and not _within(user_home, root)
        and not _within(user_home, home),
        "HOME must be a dedicated directory: not the account home, the default Hermes root, "
        "HERMES_HOME or an ancestor of either",
    )
    return home


def phase_cursor(mode, plan, plan_path, rows):
    """Durable progress check: the session must hold exactly the earlier phases' turns."""
    order = list(PHASES)
    earlier = order[: order.index(mode)]
    expected = []
    for phase in earlier:
        path = plan_path.parent / f"receipt-{phase}.json"
        refuse(path.is_file(), f"run phase {phase!r} first")
        done = json.loads(path.read_bytes())
        planned = [t["n"] for t in plan["phases"][phase]]
        refuse(
            [t["turn"] for t in done["turns"]] == planned
            and not any(t["failed"] for t in done["turns"]),
            f"phase {phase!r} is incomplete or has a failed turn",
        )
        expected += [t["text"] for t in done["turns"]]
    users = [r["content"] for r in rows if r["role"] == "user"]
    refuse(users == expected, "session history does not match the completed pilot phases")
    refuse(
        not (plan_path.parent / f"receipt-{mode}.json").exists(),
        f"phase {mode!r} already has a receipt; start a new plan instead of repeating turns",
    )


def run_phase(mode, plan_path):
    plan = json.loads(plan_path.read_bytes())
    session = plan["session"]
    refuse(session.startswith("ariadne-pilot-"), "not a pilot session id")
    lock = (plan_path.parent / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("rolling_host_pilot: another process is running this plan") from None
    home = admit_homes(plan)
    plugin = home / "plugins" / "ariadne-rolling-plugin"
    refuse((plugin / "plugin.yaml").is_file(), "install the plugin from a wheel first")
    refuse(not (home / "plugins" / "ariadne-rolling").exists(), "directory-loader name collision")

    # isort: off
    from agent.plugin_llm import PluginLlm
    from hermes_cli.plugins import get_plugin_context_engine
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import parse_reasoning_effort
    from hermes_state import SessionDB
    from run_agent import AIAgent
    # isort: on

    db = SessionDB(db_path=home / "state.db")
    rows = db.get_messages(session)
    if mode == "advance":
        refuse(rows == [], "advance needs a brand-new session")
    phase_cursor(mode, plan, plan_path, rows)

    # Observe failures on the CLASS before AIAgent construction, so an archive refusal
    # during initialization keeps its cause (clones share the class).
    manager_engine = get_plugin_context_engine()
    refuse(
        manager_engine is not None and manager_engine.name == "ariadne-rolling",
        "context.engine is not ariadne-rolling",
    )
    engine_class = type(manager_engine)
    init_exceptions = []
    active = None

    def observe_failure(name):
        method = getattr(engine_class, name)

        def wrapper(self, *args, **kwargs):
            etype, exc, tb = sys.exc_info()
            record = {"via": name}
            if exc is not None:
                frames = [
                    f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                    for f in traceback.extract_tb(tb)
                    if str(plugin) in f.filename
                ]
                record.update(type=etype.__name__, message=str(exc)[:300], frames=frames[-8:])
            (active["engine_exceptions"] if active is not None else init_exceptions).append(record)
            return method(self, *args, **kwargs)

        setattr(engine_class, name, wrapper)

    for name in ("_decline", "_unavailable", "_refuse_archive"):
        observe_failure(name)

    if mode == "advance":
        db.create_session(session, source="cli", model=plan["model"])
    runtime = resolve_runtime_provider(requested=plan["provider"], target_model=plan["model"])
    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=plan["model"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        enabled_toolsets=["context_engine"],
        session_db=db,
        session_id=session,
        max_tokens=1024,
        max_iterations=12,
        platform="cli",
        reasoning_config=parse_reasoning_effort(plan.get("reasoning")),
    )
    engine = agent.context_compressor
    refuse(engine.name == "ariadne-rolling", "context.engine is not ariadne-rolling")
    origin = Path(sys.modules[type(engine).__module__].__file__).resolve()
    refuse(origin.parent == plugin.resolve(), "engine not loaded from the installed plugin")
    refuse(get_plugin_context_engine() is not engine, "manager must clone the engine per agent")
    refuse(
        isinstance(engine._llm, PluginLlm) and engine._llm._sync_caller is None,
        "engine is not bound to the host ctx.llm facade",
    )
    store_py = (plugin / "store.py").resolve()
    canonical_json = next(
        m.canonical_json
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and Path(m.__file__).resolve() == store_py
    )
    view_dir = home / "ariadne" / "rolling" / hashlib.sha256(session.encode()).hexdigest()

    def durable_view():
        if not (view_dir / "manifest.json").exists():
            return None
        generation = json.loads((view_dir / "manifest.json").read_bytes())["generation_id"]
        content = (view_dir / "generations" / f"{generation}.json").read_bytes()
        refuse(hashlib.sha256(content).hexdigest() == generation, "generation hash mismatch")
        view = json.loads(content)
        refuse(view["session_id"] == session, "published view belongs to another session")
        return {
            "generation_id": generation,
            "parent_generation": view.get("parent_generation"),
            "frontier_count": view.get("frontier_count"),
            "delta": [view.get("delta_start"), view.get("delta_stop")],
            "summary_bytes": len((view.get("summary_text") or "").encode()),
        }

    def status():
        s = engine.get_status()
        return {"rolling": s.get("rolling"), "selection": s.get("selection")}

    receipt = {
        "mode": mode,
        "pid": os.getpid(),
        "engine_origin": str(origin),
        "tool_names": sorted(agent._context_engine_tool_names),
        "runtime": {"provider": runtime.get("provider"), "api_mode": runtime.get("api_mode")},
        "model": plan["model"],
        "reasoning": plan.get("reasoning"),
        "settings": dataclasses.asdict(engine._settings),
        "initial": status(),
        "initial_view": durable_view(),
        "initial_rows": len(db.get_messages(session)),
        "init_exceptions": init_exceptions,
        "turns": [],
    }
    native = {
        n: getattr(engine, n) for n in ("select_context", "handle_tool_call", "on_turn_complete")
    }

    def observe_select(request_messages, **kwargs):
        before = copy.deepcopy((request_messages, kwargs))
        result = native["select_context"](request_messages, **kwargs)
        active["input_mutated"] = (
            active.get("input_mutated") or (request_messages, kwargs) != before
        )
        entry = {
            "selected": result is not None,
            "request_messages": len(request_messages),
            "status": engine.get_status()["selection"],
        }
        if result is not None:
            suffix = 0
            for k in range(1, min(len(result), len(request_messages)) + 1):
                if result[-k] != request_messages[-k]:
                    break
                suffix = k
            entry.update(
                selected_messages=len(result),
                unchanged_suffix=suffix,
                selected_roles=[m.get("role") for m in result],
            )
        active["selections"].append(entry)
        return result

    def observe_tool(name, args, **kwargs):
        result = native["handle_tool_call"](name, args, **kwargs)
        active["tools"].append({"name": name, "args": copy.deepcopy(args), "result": result})
        return result

    def observe_complete(messages, usage=None, **kwargs):
        active["callbacks"].append(
            {
                "usage_present": usage is not None,
                **{k: kwargs.get(k) for k in ("interrupted", "failed", "turn_exit_reason")},
                "turn_id_present": bool(kwargs.get("turn_id")),
            }
        )
        started = time.time()
        try:
            return native["on_turn_complete"](messages, usage=usage, **kwargs)
        finally:
            active["completion_seconds"] = round(time.time() - started, 2)

    engine.select_context = observe_select
    engine.handle_tool_call = observe_tool
    engine.on_turn_complete = observe_complete

    previous = copy.deepcopy(db.get_messages(session))
    for turn in plan["phases"][mode]:
        active = {
            "turn": turn["n"],
            "selections": [],
            "tools": [],
            "callbacks": [],
            "engine_exceptions": [],
        }
        text = turn["text"]
        if "{prev_rolling_ref}" in text:
            earlier = json.loads((plan_path.parent / "receipt-restart.json").read_bytes())
            refs = []
            for x in earlier["turns"]:
                for t in x["tools"]:
                    try:
                        ok = json.loads(t["result"]).get("ok") is True
                    except (TypeError, ValueError):
                        ok = False
                    ref = t["args"].get("ref") if isinstance(t["args"], dict) else None
                    if t["name"] == "ariadne_rolling_page" and ok and isinstance(ref, str):
                        refs.append(ref)
            refuse(refs, "restart phase recorded no successful ariadne_rolling_page reference")
            text = text.replace("{prev_rolling_ref}", refs[0])
        active["text"] = text
        history = db.get_messages_as_conversation(session, repair_alternation=True)
        started = time.time()
        result = agent.run_conversation(text, conversation_history=history)
        rows = db.get_messages(session)
        active.update(
            seconds=round(time.time() - started, 1),
            failed=bool(result.get("failed")),
            final_response=result.get("final_response"),
            rows=len(rows),
            canonical_prefix_unchanged=rows[: len(previous)] == previous,
            archived_exact_mismatches=sum(
                engine._store is None
                or engine._store.get_event(
                    hashlib.sha256(canonical_json(row)).hexdigest(), session_id=session
                )
                != row
                for row in rows
            ),
            status=status(),
            view=durable_view(),
        )
        receipt["turns"].append(active)
        (plan_path.parent / f"receipt-{mode}.json").write_text(
            json.dumps(receipt, indent=1, default=str)
        )
        print(
            json.dumps(
                {
                    "turn": turn["n"],
                    "failed": active["failed"],
                    "selected": [s["selected"] for s in active["selections"]],
                    "tools": [t["name"] for t in active["tools"]],
                    "frontier": (active["view"] or {}).get("frontier_count"),
                    "rolling": active["status"]["rolling"].get("status"),
                }
            ),
            flush=True,
        )
        # Later turns would continue from incomplete history: stop, never treat them as evidence.
        refuse(not active["failed"], f"turn {turn['n']} failed; phase stopped")
        refuse(not active.get("input_mutated"), f"turn {turn['n']}: selection mutated host input")
        previous = copy.deepcopy(rows)
    agent.close()
    db.close()


def summarize(out):
    rows = []
    for mode in PHASES:
        path = out / f"receipt-{mode}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_bytes())
        init = r["initial"]["rolling"] or {}
        for e in r.get("init_exceptions", []):
            rows.append(f"init {e['via']}: {e.get('type', '')} {e.get('message', '')}")
        rows.append(
            f"## {mode}: pid {r['pid']}, initial view "
            f"{(r['initial_view'] or {}).get('generation_id', '-')[:12]}, reservations "
            f"{init.get('logical_reservations')}, request bytes {init.get('cumulative_request_bytes')}"
        )
        for t in r["turns"]:
            roll = t["status"]["rolling"] or {}
            view = t["view"] or {}
            tools = ",".join(
                f"{x['name']}:{'ok' if json.loads(x['result']).get('ok') else json.loads(x['result']).get('error')}"
                for x in t["tools"]
            )
            excs = ";".join(
                f"{e['via']}:{e.get('message', '')}"
                for e in t["engine_exceptions"]
                if e.get("message")
            )
            rows.append(
                f"| {t['turn']}{' FAILED' if t.get('failed') else ''} | "
                f"{''.join('S' if s['selected'] else 'D' for s in t['selections'])} | "
                f"{view.get('generation_id', '-')[:8]} f{view.get('frontier_count')} | "
                f"{roll.get('logical_reservations')} | {roll.get('status')} | "
                f"{t['canonical_prefix_unchanged']}/{t['archived_exact_mismatches']} | {tools or '-'} | {excs or '-'} |"
            )
    print("\n".join(rows))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "plan":
        args = sys.argv[3:]
        opts = dict(zip(args[::2], args[1::2]))
        make_plan(
            Path(sys.argv[2]), opts["--provider"], opts["--model"], opts.get("--reasoning", "none")
        )
    elif command in PHASES:
        run_phase(command, Path(sys.argv[2]))
    elif command == "summarize":
        summarize(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command {command!r}")
