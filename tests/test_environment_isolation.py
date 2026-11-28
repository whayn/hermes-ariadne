"""Synthetic environment tripwires; no host startup, credentials, or network."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# config.py's protected location names, plus actual backend/identity selectors in
# config_backend.py, hermes_constants.py and secret_sources/registry.py.
OVERRIDES = (
    "HERMES_CONFIG",
    "HERMES_ENV",
    "HERMES_CONFIG_PATH",
    "HERMES_ENV_PATH",
    "HERMES_PROFILE",
    "HERMES_ACTIVE_PROFILE",
    "HERMES_PROFILE_NAME",
    "HERMES_CONFIG_BACKEND",
    "HERMES_REAL_HOME",
)


@pytest.mark.parametrize("explicit_host", [False, True])
def test_collection_and_each_test_replace_caller_configuration(tmp_path, explicit_host):
    caller = tmp_path / "synthetic-caller"
    managed = caller / "managed"
    managed.mkdir(parents=True)
    for directory in (caller, caller / ".hermes", managed):
        directory.mkdir(exist_ok=True)
        (directory / "config.yaml").write_text("model: CALLER_CONFIG_SENTINEL\n")
        (directory / ".env").write_text("ARIADNE_SENTINEL=CALLER_DOTENV_SENTINEL\n")
    before = {path: path.read_bytes() for path in caller.rglob("*") if path.is_file()}
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copyfile(REPO / "tests" / "conftest.py", suite / "conftest.py")
    # The import-time assertion runs during collection, before PYTEST_CURRENT_TEST
    # can suppress native /etc/hermes fallback. Every test also gets a new scope.
    (suite / "test_scopes.py").write_text(
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"OVERRIDES = {OVERRIDES!r}\n"
        f"CALLER = Path({str(caller)!r})\n"
        "def check():\n"
        "    home = Path(os.environ['HOME'])\n"
        "    hermes = Path(os.environ['HERMES_HOME'])\n"
        "    managed = Path(os.environ['HERMES_MANAGED_DIR'])\n"
        "    assert not home.is_relative_to(CALLER), 'inherited HOME'\n"
        "    assert not hermes.is_relative_to(CALLER), 'inherited HERMES_HOME'\n"
        "    assert not managed.is_relative_to(CALLER), 'inherited managed directory'\n"
        "    assert managed.is_dir() and not list(managed.iterdir())\n"
        "    assert not any(key in os.environ for key in OVERRIDES)\n"
        "    assert 'ARIADNE_SENTINEL' not in os.environ\n"
        "    assert 'agent' not in sys.modules\n"
        f"    assert all((name in sys.modules) == {explicit_host!r} for name in ('openai', 'httpx'))\n"
        "    return str(home), str(hermes), str(managed)\n"
        "COLLECTION = check()\n"
        "SEEN = []\n"
        "def assert_fresh(current, earlier, boundary):\n"
        "    for key, value, prior in zip(('HOME', 'HERMES_HOME', 'HERMES_MANAGED_DIR'), current, earlier):\n"
        "        assert value != prior, f'{key} reused from {boundary}'\n"
        "def test_01_scope():\n"
        "    current = check()\n"
        "    assert_fresh(current, COLLECTION, 'collection')\n"
        "    SEEN.append(current)\n"
        "    Path(current[2], '.env').write_text('ARIADNE_SENTINEL=PRIOR_TEST_SENTINEL')\n"
        "    os.environ['HERMES_CONFIG_PATH'] = str(CALLER / 'config.yaml')\n"
        "def test_02_fresh_scope():\n"
        "    current = check()\n"
        "    assert_fresh(current, COLLECTION, 'collection')\n"
        "    assert_fresh(current, SEEN[0], 'preceding test')\n"
        "    code = \"import os; assert not any('SENTINEL' in v for v in os.environ.values())\"\n"
        "    subprocess.run([sys.executable, '-I', '-S', '-c', code], check=True)\n"
    )
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    env.update({key: str(caller / "config.yaml") for key in OVERRIDES})
    env.update(
        HOME=str(caller),
        HERMES_HOME=str(caller / ".hermes"),
        HERMES_MANAGED_DIR=str(managed),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    if explicit_host:
        # These are import tripwires, NOT SDK/transport evidence. They assert the
        # environment at the very first dependency import without starting Hermes.
        dependencies = suite / "synthetic-dependencies"
        dependencies.mkdir()
        import_check = (
            "import os\nfrom pathlib import Path\n"
            f"assert not any(key in os.environ for key in {OVERRIDES!r})\n"
            f"assert not Path(os.environ['HOME']).is_relative_to(Path({str(caller)!r}))\n"
            f"assert not Path(os.environ['HERMES_HOME']).is_relative_to(Path({str(caller)!r}))\n"
            "managed = Path(os.environ['HERMES_MANAGED_DIR'])\n"
            f"assert not managed.is_relative_to(Path({str(caller)!r}))\n"
            "assert managed.is_dir() and not list(managed.iterdir())\n"
        )
        for name in ("httpx", "openai"):
            (dependencies / f"{name}.py").write_text(import_check)
        env.update(HERMES_SOURCE=str(dependencies), PYTHONPATH=str(dependencies))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(suite)],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert {path: path.read_bytes() for path in before} == before
