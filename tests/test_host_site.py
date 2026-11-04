"""Read-only host dependency bootstrap regressions (no host installation needed)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def clean_env():
    return {key: os.environ[key] for key in ("PATH", "TMPDIR", "SYSTEMROOT") if key in os.environ}


def copy_launcher(tmp_path, name):
    root = tmp_path / "project with spaces"
    script = root / "scripts" / name
    script.parent.mkdir(parents=True)
    shutil.copyfile(REPO / "scripts" / name, script)
    helper = REPO / "src" / "hermes_ariadne" / "_host_site.py"
    if helper.exists():
        target = root / "src" / "hermes_ariadne" / helper.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(helper, target)
    return root, script


def test_test_runner_preserves_caller_paths_after_selected_host_site(tmp_path):
    root, script = copy_launcher(tmp_path, "test.sh")
    host = tmp_path / "host venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(host)], check=True)
    host_python = host / "bin" / "python"
    site = subprocess.check_output(
        [str(host_python), "-I", "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        text=True,
    ).strip()
    selected = Path(site)
    (selected / "selected_dependency.py").write_text("ORIGIN = 'selected'\n")
    caller = tmp_path / "caller dependencies"
    caller.mkdir()
    (caller / "caller_dependency.py").write_text("ORIGIN = 'caller'\n")
    (caller / "selected_dependency.py").write_text("ORIGIN = 'wrong caller'\n")
    # The runner really imports this tiny pytest stand-in with a real interpreter.
    # No host/provider or pytest dependency is needed for this bootstrap-only probe.
    (caller / "pytest.py").write_text(
        "import os, selected_dependency, caller_dependency\n"
        "assert selected_dependency.ORIGIN == 'selected'\n"
        "assert caller_dependency.ORIGIN == 'caller'\n"
        f"assert os.environ['PYTHONPATH'] == {site + os.pathsep + str(caller)!r}\n"
        "print('dependency order preserved')\n"
    )
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    env = clean_env()
    env.update(HERMES_SOURCE=str(tmp_path), HERMES_PYTHON=str(host_python), PYTHONPATH=str(caller))
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "dependency order preserved" in result.stdout


@pytest.mark.parametrize(
    "initial_path",
    [None, "", os.pathsep.join(["selected-sdk", "caller dependencies", "selected-sdk"])],
)
@pytest.mark.parametrize("collection_mutation", [False, True])
def test_test_fixtures_preserve_initial_pythonpath_between_tests(
    tmp_path, initial_path, collection_mutation
):
    # Exercise the real autouse fixture in a separate, Hermes-independent pytest run.
    shutil.copyfile(REPO / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_worker_paths.py").write_text(
        "import os, subprocess, sys\n"
        f"INITIAL_PATH = {initial_path!r}\n"
        "# Host imports can rewrite the process environment during collection too.\n"
        f"if {collection_mutation!r}:\n"
        "    os.environ['PYTHONPATH'] = 'host-managed-sdk-at-collection'\n"
        "def test_01_host_mutates_pythonpath():\n"
        "    assert os.environ.get('PYTHONPATH') == INITIAL_PATH\n"
        "    os.environ['PYTHONPATH'] = 'host-managed-sdk-during-test'\n"
        "def test_02_next_worker_inherits_runner_path():\n"
        "    assert os.environ.get('PYTHONPATH') == INITIAL_PATH\n"
        "    observed = subprocess.check_output([sys.executable, '-I', '-c',\n"
        "        \"import os; print(repr(os.environ.get('PYTHONPATH')))\"], text=True)\n"
        "    assert observed.strip() == repr(INITIAL_PATH)\n"
    )
    env = clean_env()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if initial_path is not None:
        env["PYTHONPATH"] = initial_path
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path / "test_worker_paths.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


@pytest.mark.parametrize("version", [None, "3.13"])
def test_site_disabled_discovery_uses_real_venv_without_executing_startup(tmp_path, version):
    interpreter = sys.executable if version is None else shutil.which(f"python{version}")
    if interpreter is None:
        pytest.skip(f"Python {version} is not already installed; never download in unit tests")
    host = tmp_path / "host venv with spaces"
    subprocess.run([interpreter, "-I", "-m", "venv", "--without-pip", str(host)], check=True)
    python = host / "bin" / "python"
    expected = subprocess.check_output(
        [str(python), "-I", "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        text=True,
    ).strip()
    old = subprocess.check_output(
        [str(python), "-I", "-S", "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        text=True,
    ).strip()
    if version == "3.13":
        assert old != expected, "This interpreter must reproduce the pre-3.14 bootstrap bug"
    marker = tmp_path / "startup code executed"
    payload = f"from pathlib import Path; Path({str(marker)!r}).touch()\n"
    (Path(expected) / "sitecustomize.py").write_text(payload)
    (Path(expected) / "bootstrap.pth").write_text("import pathlib; " + payload.split("; ")[1])
    caller = tmp_path / "inherited dependencies"
    caller.mkdir()
    (caller / "sitecustomize.py").write_text(payload)
    env = clean_env()
    env.update(PYTHONPATH=str(caller))
    # Running a private module by filename must not execute its package root.
    package = tmp_path / "private helper source" / "hermes_ariadne"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(payload)
    helper = package / "_host_site.py"
    shutil.copyfile(REPO / "src" / "hermes_ariadne" / helper.name, helper)
    result = subprocess.run(
        [str(python), "-I", "-S", str(helper)],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
    assert not marker.exists()


@pytest.mark.parametrize("name", ["test.sh", "compactor-demo.sh"])
def test_launchers_discover_real_venv_without_site_startup(tmp_path, name):
    interpreter = shutil.which("python3.13")
    if interpreter is None:
        pytest.skip("Python 3.13 not installed; no downloads")
    root, script = copy_launcher(tmp_path, name)
    host = tmp_path / "selected host venv"
    subprocess.run([interpreter, "-I", "-m", "venv", "--without-pip", str(host)], check=True)
    python = host / "bin" / "python"
    site = subprocess.check_output(
        [str(python), "-I", "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        text=True,
    ).strip()
    marker = tmp_path / "host startup executed"
    (Path(site) / "unsafe.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
    )
    (Path(site) / "selected_dependency.py").write_text("ORIGIN = 'selected'\n")
    worker = root / ".venv" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.symlink_to(sys.executable)
    caller = tmp_path / "caller dependencies"
    caller.mkdir()
    (caller / "caller_dependency.py").write_text("ORIGIN = 'caller'\n")
    code = (
        "import selected_dependency, caller_dependency\n"
        "assert selected_dependency.ORIGIN == 'selected'\n"
        "assert caller_dependency.ORIGIN == 'caller'\n"
        "print('selected host dependencies available')\n"
    )
    if name == "test.sh":
        (caller / "pytest.py").write_text(code)
    else:
        package = root / "src" / "hermes_ariadne"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
        (package / "compactor_demo.py").write_text(code)
        (caller / "sitecustomize.py").write_text(
            f"from pathlib import Path; Path({str(marker)!r}).touch()\n"
        )
    env = clean_env()
    env.update(HERMES_SOURCE=str(tmp_path), HERMES_PYTHON=str(python), PYTHONPATH=str(caller))
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert not marker.exists(), result.stderr
    assert result.returncode == 0, result.stderr
    assert "selected host dependencies available" in result.stdout


def test_non_venv_executable_uses_bare_python_fallback(tmp_path):
    bare = Path(sys.executable).resolve()
    expected = subprocess.check_output(
        [str(bare), "-I", "-S", "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        text=True,
    ).strip()
    prefix = tmp_path / "not a venv"
    python = prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(bare)
    helper = REPO / "src" / "hermes_ariadne" / "_host_site.py"
    for executable in (bare, python):
        result = subprocess.run(
            [str(executable), "-I", "-S", str(helper)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected


def test_venv_directory_symlink_keeps_lexical_selected_path(tmp_path):
    host = tmp_path / "real host"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(host)], check=True)
    alias = tmp_path / "selected alias"
    alias.symlink_to(host, target_is_directory=True)
    python = alias / "bin" / "python"
    helper = REPO / "src" / "hermes_ariadne" / "_host_site.py"
    result = subprocess.run([str(python), "-I", "-S", str(helper)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith(str(alias) + os.sep)
    assert not result.stdout.strip().startswith(str(host) + os.sep)


@pytest.mark.parametrize("name", ["test.sh", "compactor-demo.sh"])
def test_launcher_propagates_discovery_failure_without_starting_worker(tmp_path, name):
    root, script = copy_launcher(tmp_path, name)
    host_python = tmp_path / "failing host python"
    host_python.write_text("#!/bin/sh\nexit 73\n")
    host_python.chmod(0o700)
    marker = tmp_path / "worker executed"
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    python.chmod(0o700)
    env = clean_env()
    env.update(HERMES_SOURCE=str(tmp_path), HERMES_PYTHON=str(host_python))
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 73, result.stderr
    assert not marker.exists()
