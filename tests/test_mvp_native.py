"""Installed-host proof: real `hermes plugins install` + the host's own loaders.

Synthetic and loopback-free: no model is called. Establishes discovery,
enablement, slot selection and tool dispatch from the installed copy only.
"""

import json
import os
import shutil
import subprocess
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


def test_install_enable_select_and_use(tmp_path):
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
    _run([*hermes, "plugins", "enable", "ariadne"], home, tmp_path / "enable.err")
    _run([*hermes, "config", "set", "memory.provider", "ariadne"], home, tmp_path / "m.err")
    _run([*hermes, "config", "set", "context.engine", "ariadne"], home, tmp_path / "c.err")
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
