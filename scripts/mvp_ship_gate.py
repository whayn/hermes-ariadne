"""Ariadne MVP ship gate (#41 phase 5). NOT part of the test suite.

Drives real Hermes (``AIAgent``, ``SessionDB``, the profile's configured provider)
in ONE explicitly chosen disposable profile where the ``ariadne`` plugin is already
installed with ``hermes plugins install`` and selected as memory provider and
context engine. Each phase is a separate process. The driver only observes engine
and provider calls (timing, outcome, tool names); it never calls a hook itself.

Phases, in order::

    import   – ask the agent to import a synthetic export with the ariadne:import skill
    long     – an 8-turn conversation that should trigger rolling summaries; last turn
               asks about the first turn
    restart  – a new process continues the long conversation for one turn
    newchat  – a brand-new chat asks about imported and earlier-chat facts, then asks
               the agent to save a built-in memory entry (mirror check)

Usage (Hermes interpreter, Hermes source on PYTHONPATH)::

    HERMES_HOME=$DEV_PROFILE GATE_RUN=r2 python scripts/mvp_ship_gate.py PHASE OUT_DIR

Refuses homes without ``.ariadne-mvp-gate`` and the default Hermes root. Outputs are
JSON receipts with synthetic text; keep them out of version control.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

SESSIONS = {
    "import": "gate-import",
    "long": "gate-long",
    "newchat": "gate-newchat",
    "probe": "gate-probe",
}

EXPORT = [
    {
        "id": "conv-ebike",
        "title": "E-bike tyres",
        "create_time": 1714560000,
        "current_node": "n4",
        "mapping": {
            "n1": {"parent": None, "message": None, "children": ["n2"]},
            "n2": {
                "parent": "n1",
                "children": ["n3", "n3b"],
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1714560001,
                    "content": {
                        "content_type": "text",
                        "parts": [
                            "My e-bike is a Nakamura Crossover XV with a 12x148 rear "
                            "thru-axle. Which inner tubes do I need?"
                        ],
                    },
                },
            },
            "n3b": {
                "parent": "n2",
                "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1714560002,
                    "content": {"content_type": "text", "parts": ["(abandoned draft answer)"]},
                },
            },
            "n3": {
                "parent": "n2",
                "children": ["n4"],
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1714560003,
                    "content": {
                        "content_type": "text",
                        "parts": ["You need 27.5x2.10 tubes, ETRTO 54-584."],
                    },
                },
            },
            "n4": {
                "parent": "n3",
                "children": [],
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1714560004,
                    "content": {
                        "content_type": "text",
                        "parts": ["Thanks, I'll buy two from the shop on rue Gallieni."],
                    },
                },
            },
        },
    },
    {
        "id": "conv-lyon",
        "title": "Lyon trip",
        "create_time": 1717000000,
        "current_node": "m3",
        "mapping": {
            "m1": {"parent": None, "message": None, "children": ["m2"]},
            "m2": {
                "parent": "m1",
                "children": ["m3"],
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1717000001,
                    "content": {
                        "content_type": "text",
                        "parts": [
                            "Plan a weekend in Lyon for 12 March. I'm vegetarian. "
                            "Booking reference HERON-7."
                        ],
                    },
                },
            },
            "m3": {
                "parent": "m2",
                "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "create_time": 1717000002,
                    "content": {
                        "content_type": "text",
                        "parts": ["Saturday: Vieux Lyon and a bouchon with veggie options."],
                    },
                },
            },
        },
    },
]

FILLER = (
    " Some context so this turn is long enough to matter for summarization: we are "
    "planning a small home-automation project with a Raspberry Pi, a few Zigbee "
    "sensors, and a dashboard; please keep answers to two or three sentences."
)
LONG_TURNS = [
    "For this whole conversation, our project codeword is AMBER-LANTERN-42." + FILLER,
    "The living-room sensor is named 'otter' and the kitchen one 'heron'." + FILLER,
    "Suggest a polling interval for temperature sensors." + FILLER,
    "Should the dashboard run on the Pi or on a separate laptop?" + FILLER,
    "How should I name MQTT topics for the two sensors?" + FILLER,
    "What backup strategy makes sense for the Pi's SD card?" + FILLER,
    "Give one tip for reducing Zigbee interference." + FILLER,
    "Without guessing: what was the project codeword I gave at the very start? "
    "If it is no longer in your context, use your Ariadne tools to check the originals.",
]
RESTART_TURN = (
    "What names did I give the living-room and kitchen sensors? Check the originals "
    "with your Ariadne tools if they are not in your context."
)
NEWCHAT_TURNS = [
    "Two questions. 1) What e-bike do I own and which tubes does it need? 2) In an "
    "earlier conversation I gave a project codeword; what was it? Use your Ariadne "
    "memory and conversation tools; do not guess.",
    "Please save to your built-in memory (the `memory` tool, user profile) that I "
    "prefer metric units.",
]


def refuse(ok, why):
    if not ok:
        raise SystemExit(f"refused: {why}")


def tool_names(messages, start):
    names = []
    for message in messages[start:]:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name == "tool_call":
                try:
                    inner = json.loads(fn.get("arguments") or "{}").get("calls") or []
                    name = "tool_call:" + ",".join(c.get("name", "?") for c in inner)
                except Exception:
                    pass
            names.append(name)
    return names


def main():
    phase, out = sys.argv[1], Path(sys.argv[2])
    home = Path(os.environ["HERMES_HOME"]).resolve()
    refuse((home / ".ariadne-mvp-gate").is_file(), "home is not marked as a gate profile")
    refuse(home != Path("/opt/data").resolve() and "profiles" in home.parts, "not a profile")
    out.mkdir(parents=True, exist_ok=True)

    # Load this profile's .env exactly as the hermes CLI does (provider keys live there).
    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv()

    # isort: off
    from hermes_cli.config import load_config
    from hermes_cli.plugins import get_plugin_context_engine
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import parse_reasoning_effort
    from hermes_state import SessionDB
    from run_agent import AIAgent
    # isort: on

    cfg = load_config()
    model = cfg["model"]["default"]
    provider = cfg["model"]["provider"]
    plugin = home / "plugins" / "ariadne"
    events = []

    engine_class = type(get_plugin_context_engine())
    refuse(get_plugin_context_engine().name == "ariadne", "context engine is not ariadne")

    def observe(name):
        method = getattr(engine_class, name)

        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            outcome = "ok"
            if name == "select_context" and os.environ.get("GATE_DUMP_KEYS"):
                request = args[0] if args else kwargs.get("request_messages") or []
                events.append(
                    {
                        "request_keys": [
                            [m.get("role"), sorted(m)] for m in request if isinstance(m, dict)
                        ],
                        "history_keys": [
                            [m.get("role"), sorted(m)]
                            for m in kwargs.get("conversation_messages") or []
                            if isinstance(m, dict)
                        ],
                    }
                )
            try:
                result = method(self, *args, **kwargs)
                if name == "select_context":
                    outcome = "selected" if result is not None else "declined"
                return result
            except Exception as exc:
                outcome = f"raised:{type(exc).__name__}"
                raise
            finally:
                events.append(
                    {
                        "hook": name,
                        "ms": round((time.perf_counter() - start) * 1000, 1),
                        "outcome": outcome,
                        **({"tool": args[0]} if name == "handle_tool_call" and args else {}),
                    }
                )

        setattr(engine_class, name, wrapper)

    for name in ("select_context", "on_turn_complete", "handle_tool_call"):
        observe(name)

    def observe_failure(name):
        method = getattr(engine_class, name)

        def wrapper(self, *args, **kwargs):
            etype, exc, tb = sys.exc_info()
            record = {"failure": name}
            if exc is not None:
                frames = [
                    f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                    for f in traceback.extract_tb(tb)
                    if str(plugin) in f.filename
                ]
                record.update(type=etype.__name__, message=str(exc)[:300], frames=frames[-6:])
            events.append(record)
            return method(self, *args, **kwargs)

        setattr(engine_class, name, wrapper)

    for name in ("_decline", "_unavailable", "_refuse_archive"):
        if hasattr(engine_class, name):
            observe_failure(name)

    db = SessionDB(db_path=home / "state.db")
    runtime = resolve_runtime_provider(requested=provider, target_model=model)
    run = os.environ.get("GATE_RUN", "r1")
    session = SESSIONS["long" if phase == "restart" else phase] + "-" + run
    if phase != "restart":
        refuse(not db.get_messages(session), f"{phase} needs a fresh session")
        db.create_session(session, source="cli", model=model)
    toolsets = ["memory", "context_engine"] + (["file", "skills"] if phase == "import" else [])
    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=model,
        quiet_mode=True,
        skip_context_files=True,
        skip_background_review=True,
        enabled_toolsets=toolsets,
        session_db=db,
        session_id=session,
        max_tokens=2048,
        max_iterations=40 if phase == "import" else 12,
        platform="cli",
        reasoning_config=parse_reasoning_effort(cfg.get("agent", {}).get("reasoning_effort")),
    )
    engine = agent.context_compressor
    refuse(engine.name == "ariadne", "agent did not select the ariadne engine")
    origin = Path(sys.modules[type(engine).__module__].__file__).resolve()
    refuse(plugin.resolve() in origin.parents, "engine not loaded from the installed plugin")

    if phase == "import":
        export = home / "workspace" / "synthetic-chat-export.json"
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(json.dumps(EXPORT, indent=1))
        prompts = [
            f"Please import my chat export at {export} into Ariadne memory. Load the "
            "ariadne:import skill first and follow it."
        ]
    elif phase == "long":
        prompts = LONG_TURNS
    elif phase == "restart":
        prompts = [RESTART_TURN]
    elif phase == "probe":
        prompts = ["Say hi in five words."]
    else:
        prompts = NEWCHAT_TURNS

    history = db.get_messages_as_conversation(session, repair_alternation=True) or []
    turns = []
    for index, prompt in enumerate(prompts, 1):
        del events[:]
        start = time.perf_counter()
        error = None
        try:
            result = agent.run_conversation(prompt, conversation_history=history)
        except Exception:
            result, error = {}, traceback.format_exc()[-1500:]
        elapsed = round(time.perf_counter() - start, 2)
        messages = result.get("messages") or []
        status = engine.get_status()
        turns.append(
            {
                "turn": index,
                "seconds": elapsed,
                "failed": bool(result.get("failed")) or error is not None,
                "error": error,
                "api_calls": result.get("api_calls"),
                "tools": tool_names(messages, len(history)),
                "answer": (result.get("final_response") or "")[:1500],
                "hooks": list(events),
                "status": json.loads(json.dumps(status, default=str)),
                "selection": {k: status.get("selection", {}).get(k) for k in ("status", "reason")},
                "rolling": {
                    k: status.get("rolling", {}).get(k)
                    for k in (
                        "status",
                        "published_sequence",
                        "frontier_count",
                        "logical_reservations",
                        "blocked_reason",
                    )
                },
            }
        )
        history = messages or db.get_messages_as_conversation(session, repair_alternation=True)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "turn": index,
                    "seconds": elapsed,
                    "tools": turns[-1]["tools"],
                    "selection": turns[-1]["selection"],
                    "rolling": turns[-1]["rolling"],
                }
            ),
            flush=True,
        )
        if turns[-1]["failed"]:
            break
    agent.close()

    from importlib import import_module

    notes_mod = import_module(type(engine).__module__.rsplit(".", 1)[0] + ".notes")
    store = notes_mod.NoteStore(home)
    receipt = {
        "phase": phase,
        "model": model,
        "provider": provider,
        "plugin_commit": (plugin / ".git" / "HEAD").read_text().strip()
        if (plugin / ".git" / "HEAD").is_file()
        else None,
        "turns": turns,
        "notes": store.list(limit=100)["notes"],
        "conversations": store.conversations(limit=50)["conversations"],
    }
    (out / f"{phase}.json").write_text(json.dumps(receipt, indent=1, ensure_ascii=False))
    db.close()


if __name__ == "__main__":
    main()
