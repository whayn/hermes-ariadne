"""Synthetic environment tripwires; no host startup, credentials, or network."""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "test_restart.py",
    "test_live_transport.py",
    "test_codex_transport.py",
    "test_native_compactor.py",
    "test_compaction_transport.py",
)
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


@pytest.fixture
def native_worker_guard(tmp_path):
    """Compile ONLY the real guard: no worker imports, audit hook, or startup."""
    worker = REPO / "tests" / "fixtures" / "codex_prepared_worker.py"
    tree = ast.parse(worker.read_text())
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "credential_read_guard"
    ]
    assert len(guards) == 1
    home = tmp_path / "synthetic-home" / "isolated-home"
    managed = tmp_path / "synthetic-managed"
    home.mkdir(parents=True)
    managed.mkdir()
    namespace: dict[str, Any] = dict(os=os, Path=Path, home=home, managed=managed)
    exec(compile(ast.Module(body=guards, type_ignores=[]), str(worker), "exec"), namespace)
    return namespace["credential_read_guard"], home.parent, managed


@pytest.mark.parametrize("root_index", [0, 1], ids=["home", "managed"])
@pytest.mark.parametrize("escape", ["symlink", "parent-traversal"])
def test_native_worker_guard_refuses_path_escape(tmp_path, native_worker_guard, root_index, escape):
    guard, *roots = native_worker_guard
    root = roots[root_index]
    outside = tmp_path / "synthetic-outside"
    outside.mkdir()
    target = outside / "auth.json"
    target.write_text("SYNTHETIC_AUTH_SENTINEL")
    if escape == "symlink":
        (root / "escape").symlink_to(outside, target_is_directory=True)
        path = root / "escape" / target.name
    else:
        path = root / ".." / outside.name / target.name
    with pytest.raises(AssertionError, match="shared credential/config read forbidden"):
        guard("open", (path, "r", 0))


@pytest.mark.parametrize("root_index", [0, 1], ids=["home", "managed"])
@pytest.mark.parametrize(
    ("alias_name", "target_name"),
    [("harmless.txt", "auth.json"), ("auth.json", "harmless.txt")],
)
def test_native_worker_guard_refuses_filename_alias_escape(
    tmp_path, native_worker_guard, root_index, alias_name, target_name
):
    guard, *roots = native_worker_guard
    outside = tmp_path / "synthetic-outside"
    outside.mkdir()
    target = outside / target_name
    target.write_text("SYNTHETIC_OUTSIDE_SENTINEL")
    alias = roots[root_index] / alias_name
    alias.symlink_to(target)
    with pytest.raises(AssertionError, match="shared credential/config read forbidden"):
        guard("open", (alias, "r", 0))


@pytest.mark.parametrize("root_index", [0, 1], ids=["home", "managed"])
@pytest.mark.parametrize(
    ("alias_name", "target_name"),
    [("harmless.txt", "auth.json"), ("auth.json", "harmless.txt")],
)
def test_native_worker_guard_allows_alias_to_synthetic_target(
    tmp_path, native_worker_guard, root_index, alias_name, target_name
):
    guard, *roots = native_worker_guard
    target = roots[root_index] / target_name
    target.write_text("SYNTHETIC_ALLOWED_SENTINEL")
    alias = tmp_path / alias_name
    alias.symlink_to(target)
    guard("open", (alias, "r", 0))
    assert alias.read_text() == "SYNTHETIC_ALLOWED_SENTINEL"


@pytest.mark.parametrize("root_index", [0, 1], ids=["home", "managed"])
@pytest.mark.parametrize(
    "name", ["auth.json", ".env", ".op.env", "config.yaml", "credentials.json"]
)
def test_native_worker_guard_allows_synthetic_files(native_worker_guard, root_index, name):
    guard, *roots = native_worker_guard
    path = roots[root_index] / name
    path.write_text("SYNTHETIC_ALLOWED_SENTINEL")
    guard("open", (path, "r", 0))
    assert path.read_text() == "SYNTHETIC_ALLOWED_SENTINEL"


@pytest.mark.parametrize("root_index", [0, 1], ids=["home", "managed"])
def test_native_worker_guard_resolves_allowed_roots(tmp_path, native_worker_guard, root_index):
    guard, *roots = native_worker_guard
    root = roots[root_index]
    alias = tmp_path / "synthetic-root-alias"
    alias.symlink_to(root, target_is_directory=True)
    if root_index == 0:
        guard.__globals__["home"] = alias / "isolated-home"
    else:
        guard.__globals__["managed"] = alias
    path = root / "config.yaml"
    path.write_text("SYNTHETIC_ALLOWED_SENTINEL")
    guard("open", (path, "r", 0))
    assert path.read_text() == "SYNTHETIC_ALLOWED_SENTINEL"


@pytest.mark.parametrize(
    "name", ["auth.json", ".env", ".op.env", "config.yaml", "credentials.json"]
)
def test_native_worker_guard_refuses_existing_outside(tmp_path, native_worker_guard, name):
    guard, *_ = native_worker_guard
    path = tmp_path / name
    path.write_text("SYNTHETIC_OUTSIDE_SENTINEL")
    with pytest.raises(AssertionError, match="shared credential/config read forbidden"):
        guard("open", (path, "r", 0))


@pytest.mark.parametrize(
    "name", ["auth.json", ".env", ".op.env", "config.yaml", "credentials.json"]
)
def test_native_worker_guard_preserves_missing_layer_enoent(tmp_path, native_worker_guard, name):
    guard, *_ = native_worker_guard
    path = tmp_path / name
    guard("open", (path, "r", 0))
    with pytest.raises(FileNotFoundError):
        path.read_text()


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


@pytest.mark.parametrize("name", LAUNCHERS)
def test_each_native_launcher_selects_empty_managed_scope(tmp_path, monkeypatch, name):
    caller = tmp_path / "synthetic-caller"
    caller.mkdir()
    (caller / ".env").write_text("ARIADNE_SENTINEL=CALLER_DOTENV_SENTINEL\n")
    (caller / "config.yaml").write_text("model: CALLER_CONFIG_SENTINEL\n")
    for key in OVERRIDES:
        monkeypatch.setenv(key, str(caller / "config.yaml"))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(caller))
    monkeypatch.setenv("HOME", str(caller))
    monkeypatch.setenv("HERMES_HOME", str(caller))
    monkeypatch.setenv("HERMES_SOURCE", str(tmp_path / "synthetic-host"))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "synthetic-parent")

    # Extract only the exact env-building statements, NOT AIAgent/the server.
    # This avoids importing the test modules or accidentally executing a native
    # worker while the parallel codec moves. Dependency locals are synthetic.
    tree = ast.parse((REPO / "tests" / name).read_text())
    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(stmt, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "env" for target in stmt.targets)
            for stmt in node.body
        )
    ]
    assert len(scopes) == 1, "update the launcher inventory if its construction changes"
    statements: list[ast.stmt] = [
        stmt
        for stmt in scopes[0].body
        if (
            isinstance(stmt, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "env" for target in stmt.targets)
        )
        or (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "env"
            and stmt.value.func.attr == "update"
        )
    ]
    assert len(statements) == 2
    launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert launches
    assert all(
        any(
            kw.arg == "env" and isinstance(kw.value, ast.Name) and kw.value.id == "env"
            for kw in call.keywords
        )
        for call in launches
    )
    # The bounded extractor must account for EVERY env reference, not silently
    # skip later writes (including nested/conditional writes or alias escapes).
    # Only the initial binding, extracted update/get calls and launch keywords
    # are understood; a new shape requires reviewing the oracle before execution.
    allowed_env: set[ast.AST] = {
        target
        for stmt in statements
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name) and target.id == "env"
    }
    for stmt in statements:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "env"
                and (
                    node.func.attr == "get"
                    or (
                        isinstance(stmt, ast.Expr)
                        and node is stmt.value
                        and node.func.attr == "update"
                    )
                )
            ):
                allowed_env.add(node.func.value)
    allowed_env.update(kw.value for call in launches for kw in call.keywords if kw.arg == "env")
    unknown = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "env" and node not in allowed_env
    ]
    assert not unknown, f"{name}: unrecognized env use at lines {unknown}"
    home = tmp_path / "isolated-home"
    home.mkdir()
    namespace: dict[str, Any] = dict(
        os=os,
        Path=Path,
        tempfile=tempfile,
        tmp_path=tmp_path,
        root=tmp_path,
        home=home,
        source=os.environ["HERMES_SOURCE"],
        host=os.environ["HERMES_SOURCE"],
        __file__=str(REPO / "tests" / name),
    )
    code = compile(ast.Module(body=statements, type_ignores=[]), name, "exec")
    observed = []
    for _ in range(2):
        exec(code, namespace)
        env = namespace["env"]
        assert "HERMES_MANAGED_DIR" in env, f"{name} leaves system managed scope selectable"
        managed = Path(env["HERMES_MANAGED_DIR"])
        assert managed.is_relative_to(tmp_path) and managed != caller
        assert managed.is_dir() and not list(managed.iterdir())
        assert managed not in observed, "each env construction must get a fresh managed scope"
        observed.append(managed)
        assert env["HOME"] == str(tmp_path)
        assert env["HERMES_HOME"] == str(
            tmp_path if name == "test_compaction_transport.py" else home
        )
        # The CLI compaction test intentionally seeds four synthetic overrides to
        # prove profile selection clears them before imports; do not erase that test.
        intentional = (
            {"HERMES_CONFIG", "HERMES_ENV", "HERMES_PROFILE", "HERMES_ACTIVE_PROFILE"}
            if name == "test_compaction_transport.py"
            else set()
        )
        assert not any(key in env for key in set(OVERRIDES) - intentional)
        assert "PYTEST_CURRENT_TEST" not in env
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                """
import os, sys
from pathlib import Path
# Fail before any fallback file read; no reliance on pytest's managed-dir bypass.
assert 'PYTEST_CURRENT_TEST' not in os.environ
managed = Path(os.environ['HERMES_MANAGED_DIR'])
assert managed.is_dir() and not list(managed.iterdir())
for directory in (Path(os.environ['HOME']), Path(os.environ['HERMES_HOME']), managed):
    for filename in ('.env', 'config.yaml'):
        path = directory / filename
        assert not path.exists() or 'SENTINEL' not in path.read_text()
assert not any(name in sys.modules for name in ('agent', 'hermes_cli', 'openai', 'httpx'))
print('synthetic child scope clean')
""",
            ],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "synthetic child scope clean"


@pytest.mark.parametrize("name", LAUNCHERS)
@pytest.mark.parametrize("conditional", [False, True], ids=["direct", "conditional"])
def test_launcher_oracle_rejects_late_env_mutation(tmp_path, monkeypatch, name, conditional):
    source = (REPO / "tests" / name).read_text()
    updates = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "env"
        and node.value.func.attr == "update"
    ]
    assert len(updates) == 1
    update = updates[0]
    assert update.end_lineno is not None
    indent = " " * update.col_offset
    poison = "env['HERMES_MANAGED_DIR'] = os.environ['HOME']\n"
    mutation = f"{indent}if True:\n{indent}    {poison}" if conditional else f"{indent}{poison}"
    lines = source.splitlines(keepends=True)
    lines.insert(update.end_lineno, mutation)
    disposable = tmp_path / "mutant-repo"
    (disposable / "tests").mkdir(parents=True)
    (disposable / "tests" / name).write_text("".join(lines))
    monkeypatch.setitem(globals(), "REPO", disposable)
    probe = tmp_path / "probe"
    probe.mkdir()
    with pytest.raises(AssertionError, match="unrecognized env use"):
        test_each_native_launcher_selects_empty_managed_scope(probe, monkeypatch, name)


@pytest.mark.parametrize(
    "bindings",
    [("HOME",), ("HERMES_HOME",), ("HOME", "HERMES_HOME")],
    ids=["HOME", "HERMES_HOME", "both"],
)
@pytest.mark.parametrize("reuse", [False, True], ids=["missing-binding", "reused-binding"])
def test_scope_oracle_rejects_unrenewed_home_bindings(tmp_path, monkeypatch, bindings, reuse):
    source = (REPO / "tests" / "conftest.py").read_text()
    for key in bindings:
        original = f'    monkeypatch.setenv("{key}", ' + (
            "str(tmp_path))\n" if key == "HOME" else 'str(tmp_path / "home"))\n'
        )
        assert source.count(original) == 1
        replacement = (
            f'    monkeypatch.setenv("{key}", str(_TEST_ROOT / "reused-{key}"))\n' if reuse else ""
        )
        source = source.replace(original, replacement)
    disposable = tmp_path / "mutant-repo"
    (disposable / "tests").mkdir(parents=True)
    (disposable / "tests" / "conftest.py").write_text(source)
    monkeypatch.setitem(globals(), "REPO", disposable)
    boundary = "preceding test" if reuse else "collection"
    with pytest.raises(AssertionError, match=f"{bindings[0]} reused from {boundary}"):
        test_collection_and_each_test_replace_caller_configuration(tmp_path / "probe", False)
