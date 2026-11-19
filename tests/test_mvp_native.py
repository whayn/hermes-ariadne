"""Installed-host proof: real `hermes plugins install` + the host's own loaders.

Synthetic and loopback-free: no model is called. Establishes discovery,
enablement, slot selection and tool dispatch from the installed copy only.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_PYTHON = os.environ.get("HERMES_PYTHON")
HERMES_SOURCE = os.environ.get("HERMES_SOURCE")

pytestmark = pytest.mark.skipif(
    not (HERMES_SOURCE and HERMES_PYTHON), reason="set HERMES_SOURCE and HERMES_PYTHON"
)

CHILD = r"""
import json, os, sys
from pathlib import Path
home = Path(os.environ["HERMES_HOME"])
from hermes_cli.config import load_config
from hermes_cli.plugins import discover_plugins
from agent.agent_init import _select_context_engine
from plugins.context_engine import discover_context_engines
from plugins.memory import discover_memory_providers, load_memory_provider
cfg = load_config()
discover_plugins()
engine = _select_context_engine(cfg)
provider = load_memory_provider("ariadne")
provider.initialize("native-s1", hermes_home=str(home), agent_context="primary")
added = json.loads(provider.handle_tool_call("ariadne_notes", {"action": "add", "text": "native note"}))
provider.on_memory_write("add", "user", "mirrored entry", {})
recall = provider.prefetch("tell me the native note")
from hermes_cli.plugins import get_plugin_manager
skill = get_plugin_manager().find_plugin_skill("ariadne:import")
origin = sys.modules[type(provider).__module__].__file__
print(json.dumps({
    "engine": getattr(engine, "name", None),
    "engine_has_llm": callable(getattr(getattr(engine, "_llm", None), "complete", None)),
    "engine_origin": sys.modules[type(engine).__mro__[1].__module__].__file__,
    "provider": provider.name,
    "provider_origin": origin,
    "listed_engines": [n for n, *_ in discover_context_engines()],
    "listed_providers": [n for n, *_ in discover_memory_providers()],
    "added": added["ok"],
    "recall": recall,
    "notes_db": (home / "ariadne" / "notes.sqlite3").is_file(),
    "skill": str(skill) if skill else None,
}))
"""


def _snapshot(dest):
    files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for rel in files:
        if (ROOT / rel).is_file():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
    git = [
        "git",
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "init.defaultBranch=main",
    ]
    subprocess.run(git + ["init", "-q"], cwd=dest, check=True)
    subprocess.run(git + ["add", "-A"], cwd=dest, check=True)
    subprocess.run(git + ["commit", "-qm", "snapshot"], cwd=dest, check=True)


def _env(home):
    # Only the host source: plugin code must come from the installed copy.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(HOME=str(home), HERMES_HOME=str(home), NO_COLOR="1", PYTHONPATH=HERMES_SOURCE)
    return env


def _run(args, home, stderr_path):
    with open(stderr_path, "w") as err:
        proc = subprocess.run(
            args,
            env=_env(home),
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            timeout=600,
            cwd=home,
        )
    returncode = proc.returncode
    assert returncode == 0, f"{args[1:3]} failed; see {stderr_path}"
    return proc.stdout


def _install(tmp_path):
    repo, home = tmp_path / "repo", tmp_path / "home"
    repo.mkdir()
    home.mkdir(mode=0o700)
    _snapshot(repo)
    hermes = [HERMES_PYTHON, "-m", "hermes_cli.main"]
    _run(
        [*hermes, "plugins", "install", f"file://{repo}", "--enable", "--yes-deps"],
        home,
        tmp_path / "install.err",
    )
    for key, value in (
        ("memory.provider", "ariadne"),
        ("context.engine", "ariadne"),
        ("model.context_length", "128000"),
        ("auxiliary.title_generation.enabled", "false"),
    ):
        _run([*hermes, "config", "set", key, value], home, tmp_path / "config.err")
    _run([*hermes, "plugins", "enable", "ariadne"], home, tmp_path / "enable.err")
    return home


@pytest.fixture(scope="module")
def installed_home(tmp_path_factory):
    return _install(tmp_path_factory.mktemp("mvp-native"))


def test_install_enable_select_and_use(installed_home, tmp_path):
    home = installed_home
    out = _run([HERMES_PYTHON, "-c", CHILD], home, tmp_path / "child.err")
    result = json.loads(out.strip().splitlines()[-1])

    installed = str(home / "plugins" / "ariadne")
    assert result["engine"] == "ariadne" and result["engine_has_llm"]
    assert result["provider"] == "ariadne"
    assert result["provider_origin"].startswith(installed)
    assert result["engine_origin"].startswith(installed)
    assert "ariadne" in result["listed_engines"] and "ariadne" in result["listed_providers"]
    assert result["added"] and result["notes_db"]
    assert "native note" in result["recall"] and "mirrored entry" not in result["recall"]
    assert result["skill"] and result["skill"].startswith(installed)


SESSION_A, SESSION_B = "mvp-chat-a", "mvp-chat-b"
PROMPT_A = "NOTE-IT: my e-bike is a Nakamura Crossover XV, remember that."
PROMPT_B = "ASK-BIKE: which tyres fit my bike?"
NOTE = "User's e-bike: Nakamura Crossover XV"


def _texts(messages):
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.extend(p.get("text", "") for p in content if isinstance(p, dict))
    return "\n".join(out)


class _Provider(BaseHTTPRequestHandler):
    requests, errors, observed = [], [], {}

    def log_message(self, *args):
        pass

    def _call(self, body, name, args):
        names = [t["function"]["name"] for t in body.get("tools") or []]
        if name in names:
            self.observed.setdefault("direct", set()).add(name)
            return name, args
        assert "tool_call" in names, f"{name} neither exposed nor deferred: {names}"
        self.observed.setdefault("deferred", set()).add(name)
        return "tool_call", {"calls": [{"name": name, "arguments": args}]}

    def _answer(self, body):
        messages = body["messages"]
        assert all("message_uid" not in m for m in messages)
        last = messages[-1]
        prompt = next((m.get("content") for m in reversed(messages) if m["role"] == "user"), "")
        prompt = prompt if isinstance(prompt, str) else _texts([{"content": prompt}])
        # Checked first: chat B's recalled index quotes chat A's NOTE-IT preview.
        if "ASK-BIKE" in prompt:
            if last["role"] == "tool":
                assert PROMPT_A in last["content"], last["content"][:500]
                self.observed["read_exact_original"] = True
                return {"role": "assistant", "content": "answered"}, "stop"
            text = _texts(messages)
            self.observed["recall_note"] = NOTE in text
            self.observed["recall_index"] = f"[{SESSION_A}]" in text
            name, args = self._call(
                body, "ariadne_conversations", {"action": "read", "id": SESSION_A}
            )
        elif "NOTE-IT" in prompt:
            if last["role"] == "tool":
                assert '"ok": true' in last["content"], last["content"]
                return {"role": "assistant", "content": "noted"}, "stop"
            name, args = self._call(body, "ariadne_notes", {"action": "add", "text": NOTE})
        else:
            self.observed.setdefault("other", 0)
            self.observed["other"] += 1
            return {"role": "assistant", "content": "ok"}, "stop"
        call = {
            "id": "call_" + name,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"

    def do_POST(self):
        streaming = False
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.requests.append(body)
            message, finish = self._answer(body)
            answer = {
                "id": "offline",
                "object": "chat.completion",
                "created": 1,
                "model": "ariadne/test",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            }
            streaming = bool(body.get("stream"))
            if streaming:
                delta = dict(message)
                if delta.get("tool_calls"):
                    delta["tool_calls"] = [
                        {**tc, "index": i} for i, tc in enumerate(delta["tool_calls"])
                    ]
                chunks = [
                    {
                        **answer,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    },
                    {
                        **answer,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    },
                ]
                payload = (
                    "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
                ).encode()
            else:
                payload = json.dumps(answer).encode()
            self.send_response(200)
        except Exception as exc:
            self.errors.append(repr(exc))
            streaming = False
            payload = json.dumps({"error": str(exc)}).encode()
            self.send_response(400)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_new_chat_gets_notes_index_and_reads_previous_chat(installed_home, tmp_path):
    home = installed_home
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    worker = Path(__file__).parent / "fixtures" / "mvp_chat_worker.py"
    env = _env(home)
    env.update(HERMES_MANAGED_DIR=tempfile.mkdtemp(prefix="managed-", dir=tmp_path), TZ="UTC")
    reports = []
    try:
        for session, prompt in ((SESSION_A, PROMPT_A), (SESSION_B, PROMPT_B)):
            err = tmp_path / f"{session}.err"
            with open(err, "w") as stderr:
                proc = subprocess.run(
                    [HERMES_PYTHON, str(worker), session, url, prompt],
                    env=env,
                    cwd=home,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    timeout=300,
                )
            returncode = proc.returncode
            assert returncode == 0, f"worker {session} failed; see {err}"
            reports.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not _Provider.errors, _Provider.errors
    a, b = reports
    assert a["engine"] == b["engine"] == "ariadne" and a["pid"] != b["pid"]
    assert (a["final_response"], b["final_response"]) == ("noted", "answered")
    seen = _Provider.observed
    assert seen.get("recall_note") and seen.get("recall_index"), seen
    assert seen.get("read_exact_original"), seen
    print(
        json.dumps(
            {
                "exposure": {k: sorted(v) for k, v in seen.items() if isinstance(v, set)},
                "requests": len(_Provider.requests),
                "other": seen.get("other", 0),
            }
        )
    )
    assert sys.prefix
