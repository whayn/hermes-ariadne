"""Development-profile selection needs only stdlib + pytest, never host/auth imports."""

import builtins
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_ariadne import compactor_demo


@pytest.fixture
def no_native_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "native_compactor" or name.split(".")[0] in {
            "agent",
            "hermes_cli",
            "hermes_constants",
            "openai",
            "httpx",
        }:
            raise AssertionError(f"host/native import before profile selection: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_build_requires_explicit_home_even_with_inherited_profile(
    tmp_path, monkeypatch, no_native_imports
):
    inherited = tmp_path / "profiles" / "inherited-dev"
    inherited.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(inherited))
    destination = tmp_path / "case"
    with pytest.raises(ValueError, match="explicit.*hermes.home"):
        compactor_demo.build(destination, allow_model_calls=True)
    assert not destination.exists()


@pytest.mark.parametrize(
    "choice",
    [
        "empty",
        "missing",
        "file",
        "default-root",
        "default-alias",
        "ordinary-directory",
        "ordinary-profile",
        "profiles-alias",
        "missing-config",
        "config-alias",
    ],
)
def test_invalid_home_refuses_before_native_import_or_output(
    choice, tmp_path, monkeypatch, no_native_imports
):
    root = tmp_path / "default-root"
    root.mkdir()
    (root / "config.yaml").write_text("# synthetic default config\n")
    profiles = root / "profiles"
    profiles.mkdir()
    selected = profiles / "candidate-dev"
    if choice == "empty":
        selected = ""
    elif choice == "missing":
        pass
    elif choice == "file":
        selected.write_text("not a directory")
    elif choice == "default-root":
        selected = root
    elif choice == "default-alias":
        selected.symlink_to(root, target_is_directory=True)
    elif choice == "ordinary-directory":
        selected = tmp_path / "ordinary-dev"
        selected.mkdir()
        (selected / "config.yaml").write_text("# synthetic\n")
    elif choice == "ordinary-profile":
        selected = profiles / "personal"
        selected.mkdir()
        (selected / "config.yaml").write_text("# synthetic\n")
    elif choice == "profiles-alias":
        outside = tmp_path / "other"
        outside.mkdir()
        profiles.rmdir()
        profiles.symlink_to(outside, target_is_directory=True)
        selected.mkdir()
        (selected / "config.yaml").write_text("# synthetic\n")
    elif choice == "missing-config":
        selected.mkdir()
    elif choice == "config-alias":
        selected.mkdir()
        (selected / "config.yaml").symlink_to(root / "config.yaml")
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("HERMES_HOME", str(root))
    before = dict(os.environ)
    destination = tmp_path / "case"
    with pytest.raises(ValueError, match="development profile"):
        compactor_demo.build(destination, hermes_home=selected, allow_model_calls=True)
    assert not destination.exists()
    assert dict(os.environ) == before


@pytest.mark.parametrize("as_string", [False, True])
def test_explicit_development_home_binds_environment_before_adapter(
    tmp_path, monkeypatch, no_native_imports, as_string
):
    selected = tmp_path / "profiles" / "ariadne-dev"
    selected.mkdir(parents=True)
    config = selected / "config.yaml"
    config.write_text("# synthetic profile config; never rewritten\n")
    before = config.read_bytes()
    overrides = ("HERMES_CONFIG", "HERMES_ENV", "HERMES_PROFILE", "HERMES_ACTIVE_PROFILE")
    for key in ("HOME", "HERMES_HOME", *overrides):
        monkeypatch.setenv(key, str(tmp_path / "inherited-default"))

    class SelectionObserved(Exception):
        pass

    def adapter(*args):
        assert Path(os.environ["HOME"]) == selected.resolve()
        assert Path(os.environ["HERMES_HOME"]) == selected.resolve()
        assert not any(key in os.environ for key in overrides)
        raise SelectionObserved

    monkeypatch.setattr(compactor_demo, "_adapter", adapter)
    with pytest.raises(SelectionObserved):
        compactor_demo.build(
            tmp_path / "case",
            hermes_home=str(selected) if as_string else selected,
            allow_model_calls=True,
        )
    assert config.read_bytes() == before
    assert sorted(path.name for path in selected.iterdir()) == ["config.yaml"]


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_build_validates_profile_and_consent_before_native_import(
    tmp_path, monkeypatch, no_native_imports, explicit
):
    argv = ["compactor-demo", str(tmp_path / "case"), "--phase", "build"]
    if explicit:
        selected = tmp_path / "profiles" / "cli-dev"
        selected.mkdir(parents=True)
        (selected / "config.yaml").write_text("# synthetic\n")
        argv += ["--hermes-home", str(selected)]
        expected = "allow-model-calls"
    else:
        argv += ["--allow-model-calls"]
        expected = "explicit.*hermes.home"
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match=expected):
        compactor_demo.main()
    assert not (tmp_path / "case").exists()


def test_launcher_defers_host_imports_and_preserves_read_only_dependency_path(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    # Exercise the actual launcher with a disposable stdlib interpreter link;
    # this pure test does not require a host installation or repo-local venv.
    root = tmp_path / "project"
    script = root / "scripts" / "compactor-demo.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes((repo / "scripts" / "compactor-demo.sh").read_bytes())
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    imports = tmp_path / "inherited-dependencies"
    imports.mkdir()
    marker = tmp_path / "premature-import"
    (imports / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    host = tmp_path / "host-source"
    host.mkdir()
    (host / "agent.py").write_text("raise AssertionError('premature native import')\n")
    host_python = tmp_path / "host-python"
    host_python.write_text(
        '#!/bin/sh\n[ "$1" = -I ] && [ "$2" = -S ] || exit 91\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    host_python.chmod(0o700)
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    env.update(
        HOME=str(tmp_path),
        HERMES_HOME=str(tmp_path),
        HERMES_SOURCE=str(host),
        HERMES_PYTHON=str(host_python),
        PYTHONPATH=os.pathsep.join([str(repo / "src"), str(imports)]),
    )
    result = subprocess.run(
        ["bash", str(script), str(tmp_path / "case"), "--phase", "build", "--allow-model-calls"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert not marker.exists(), "Python startup imported code before selection"
    assert result.returncode != 0
    assert "explicit --hermes-home" in result.stderr
    assert not (tmp_path / "case").exists()
