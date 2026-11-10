from pathlib import Path

import pytest

from hermes_ariadne.install import install


def test_installer_is_self_contained_and_does_not_modify_configuration(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("context:\n  engine: compressor\n", encoding="utf-8")
    before = config.read_bytes()
    target = install(home)
    assert target == home / "plugins" / "ariadne"
    assert (target / "plugin.yaml").exists()
    assert (target / "store.py").exists()
    assert "register_context_engine" in (target / "__init__.py").read_text()
    assert config.read_bytes() == before
    assert not (home / "ariadne").exists()
    with pytest.raises(FileExistsError):
        install(home)
    assert config.read_bytes() == before
    assert not list(target.glob("__pycache__/*"))


@pytest.mark.parametrize("destination", ["outside", "dangling", "inside"])
@pytest.mark.parametrize("prepared", [False, True])
def test_installer_rejects_plugins_symlink_without_outside_writes(tmp_path, destination, prepared):
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = outside
    if destination == "dangling":
        linked = outside / "missing"
    elif destination == "inside":
        linked = home / "actual-plugins"
        linked.mkdir()
    plugins = home / "plugins"
    plugins.symlink_to(linked, target_is_directory=True)
    before = set(home.iterdir())

    with pytest.raises(FileExistsError):
        install(home, prepared=prepared)

    assert plugins.is_symlink()
    assert plugins.readlink() == linked
    assert list(outside.iterdir()) == []
    assert set(home.iterdir()) == before
    if destination == "inside":
        assert list(linked.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "directory", "symlink-file", "symlink-dir", "dangling"])
@pytest.mark.parametrize("prepared", [False, True])
def test_installer_preserves_every_preexisting_final_destination(tmp_path, kind, prepared):
    home = tmp_path / "home"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    target = plugins / ("ariadne-prepared" if prepared else "ariadne")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"do not replace")
    linked = {
        "symlink-file": sentinel,
        "symlink-dir": outside,
        "dangling": outside / "missing",
    }.get(kind)
    if kind == "file":
        target.write_bytes(b"existing plugin")
    elif kind == "directory":
        target.mkdir()
        (target / "sentinel").write_bytes(b"existing plugin")
    else:
        assert linked is not None
        target.symlink_to(linked, target_is_directory=kind == "symlink-dir")
    before = target.lstat()

    with pytest.raises(FileExistsError):
        install(home, prepared=prepared)

    assert target.lstat() == before
    assert list(plugins.iterdir()) == [target]
    assert list(outside.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"do not replace"
    if kind == "file":
        assert target.read_bytes() == b"existing plugin"
    elif kind == "directory":
        assert list(target.iterdir()) == [target / "sentinel"]
        assert (target / "sentinel").read_bytes() == b"existing plugin"
    else:
        assert target.is_symlink()
        assert target.readlink() == linked


def test_installer_canonicalizes_explicit_home_symlink(tmp_path):
    canonical_home = tmp_path / "canonical-home"
    canonical_home.mkdir()
    home = tmp_path / "home"
    home.symlink_to(canonical_home, target_is_directory=True)

    target = install(home)

    assert target == canonical_home / "plugins" / "ariadne"
    assert target.resolve().is_relative_to(canonical_home)
    assert home.is_symlink()
    assert {path.name for path in target.iterdir()} == {
        "__init__.py",
        "engine.py",
        "store.py",
        "plugin.yaml",
    }


def test_installed_directory_loads_through_real_hermes_discovery(tmp_path, monkeypatch):
    from agent.agent_init import _select_context_engine
    from plugins.context_engine import load_context_engine

    home = tmp_path / "home"
    install(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert load_context_engine("ariadne").name == "ariadne"
    assert _select_context_engine({"context": {"engine": "ariadne"}}).name == "ariadne"
    assert _select_context_engine({"context": {"engine": "compressor"}}) is None
    # Merely discovering/selecting the plugin must not create the archive.
    assert not (home / "ariadne").exists()
    assert Path(home / "plugins" / "ariadne" / "engine.py").is_file()


def test_prepared_install_is_explicit_separate_and_leaves_passive_unchanged(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("context:\n  engine: compressor\n")
    before = config.read_bytes()
    passive = install(home)
    passive_bytes = {p.name: p.read_bytes() for p in passive.iterdir()}
    target = install(home, prepared=True)
    assert target == home / "plugins" / "ariadne-prepared"
    assert "PreparedAriadneEngine" in (target / "__init__.py").read_text()
    assert "ariadne-prepared" in (target / "plugin.yaml").read_text()
    assert {p.name for p in target.iterdir()} == {
        "__init__.py",
        "plugin.yaml",
        "engine.py",
        "store.py",
        "live_engine.py",
        "prepared.py",
        "retrieval.py",
        "kernel.py",
        "tree.py",
        "sources.py",
        "simulation.py",
    }
    assert {p.name: p.read_bytes() for p in passive.iterdir()} == passive_bytes
    assert config.read_bytes() == before
    assert not (home / "ariadne").exists()
    with pytest.raises(FileExistsError):
        install(home, prepared=True)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_installer_rejects_ambiguous_prepared_mode_before_writes(tmp_path, value):
    home = tmp_path / "no-writes"
    with pytest.raises(ValueError, match="explicit boolean"):
        install(home, prepared=value)
    assert not home.exists()


def test_installer_cli_requires_explicit_prepared_flag(tmp_path, monkeypatch, capsys):
    import importlib
    import sys

    module = importlib.import_module("hermes_ariadne.install")
    calls = []

    def observed(home, *, prepared=False):
        calls.append((home, prepared))
        return home / "plugins" / "ariadne-prepared"

    monkeypatch.setattr(module, "install", observed)
    monkeypatch.setattr(sys, "argv", ["ariadne-install", "--home", str(tmp_path), "--prepared"])
    module.main()
    assert calls == [(tmp_path.resolve(), True)]
    assert "context.engine=ariadne-prepared" in capsys.readouterr().out


ROLLING_FILES = {
    "__init__.py": "rolling_entry.py",
    "plugin.yaml": "rolling_plugin.yaml",
    **{
        name: name
        for name in (
            "engine.py",
            "store.py",
            "live_engine.py",
            "prepared.py",
            "retrieval.py",
            "kernel.py",
            "tree.py",
            "sources.py",
            "simulation.py",
            "receipts.py",
            "rolling.py",
            "rolling_engine.py",
            "rolling_state.py",
            "rolling_selection.py",
            "rolling_retrieval.py",
        )
    },
}


def test_rolling_installer_copies_exact_runtime_without_activation(tmp_path):
    import ast

    import hermes_ariadne.install as module

    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_bytes(b"context:\n  engine: compressor\n")
    passive = install(home)
    prepared = install(home, prepared=True)
    old = {p: p.read_bytes() for folder in (passive, prepared) for p in folder.iterdir()}
    target = install(home, rolling=True)
    assert target == home / "plugins" / "ariadne-rolling-plugin"
    assert not (home / "plugins" / "ariadne-rolling").exists()
    assert {p.name for p in target.iterdir()} == set(ROLLING_FILES)
    source = Path(module.__file__).parent
    for name, original in ROLLING_FILES.items():
        assert (target / name).read_bytes() == (source / original).read_bytes()
        if name.endswith(".py"):
            for node in ast.walk(ast.parse((target / name).read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                    assert node.module.split(".")[0] + ".py" in ROLLING_FILES
    assert config.read_bytes() == b"context:\n  engine: compressor\n"
    assert all(p.read_bytes() == value for p, value in old.items())
    assert not (home / "ariadne").exists()
    assert target.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("value", [1, "yes", None, [], {}])
def test_rolling_mode_requires_explicit_boolean_before_writes(tmp_path, value):
    with pytest.raises(ValueError, match="explicit boolean"):
        install(tmp_path / "absent", rolling=value)
    assert not (tmp_path / "absent").exists()


def test_rolling_and_prepared_are_mutually_exclusive_before_writes(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        install(tmp_path / "absent", rolling=True, prepared=True)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling", "plugins-symlink"])
def test_rolling_install_refuses_existing_targets_and_symlink_parents(tmp_path, kind):
    home = tmp_path / "home"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    target = plugins / "ariadne-rolling-plugin"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"untouched")
    if kind == "plugins-symlink":
        plugins.rmdir()
        plugins.symlink_to(outside, target_is_directory=True)
    elif kind == "file":
        target.write_bytes(b"untouched")
    elif kind == "directory":
        target.mkdir()
        (target / "sentinel").write_bytes(b"untouched")
    else:
        target.symlink_to(sentinel if kind == "symlink" else outside / "absent")
    protected = plugins if kind == "plugins-symlink" else target
    before = protected.lstat()
    with pytest.raises(FileExistsError):
        install(home, rolling=True)
    assert protected.lstat() == before
    assert list(outside.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"untouched"


@pytest.mark.parametrize("mode", [{}, {"prepared": True}, {"rolling": True}])
def test_install_cleans_partial_copy_on_failure(tmp_path, monkeypatch, mode):
    import hermes_ariadne.install as module

    original = module.shutil.copyfile
    calls = []

    def broken(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("synthetic copy failure")
        return original(source, destination)

    monkeypatch.setattr(module.shutil, "copyfile", broken)
    with pytest.raises(OSError, match="synthetic"):
        install(tmp_path / "home", **mode)
    assert list((tmp_path / "home" / "plugins").iterdir()) == []


def test_rolling_cli_is_explicit_and_prints_separate_enablement(tmp_path, monkeypatch, capsys):
    import sys

    import hermes_ariadne.install as module

    monkeypatch.setattr(sys, "argv", ["ariadne-install", "--home", str(tmp_path), "--rolling"])
    module.main()
    output = capsys.readouterr().out
    assert "context.engine=ariadne-rolling" in output
    assert "plugins enable ariadne-rolling-plugin" in output
    assert "plugins.isolation=in_process" in output
    assert "configuration unchanged" in output
    assert (tmp_path / "plugins" / "ariadne-rolling-plugin" / "plugin.yaml").exists()
    assert not (tmp_path / "config.yaml").exists()


def test_cli_refuses_both_modes_before_writes(tmp_path, monkeypatch):
    import sys

    import hermes_ariadne.install as module

    monkeypatch.setattr(
        sys,
        "argv",
        ["ariadne-install", "--home", str(tmp_path / "absent"), "--prepared", "--rolling"],
    )
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not (tmp_path / "absent").exists()


def test_package_data_includes_all_opt_in_manifests():
    import tomllib

    project = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    assert metadata["tool"]["setuptools"]["package-data"]["hermes_ariadne"] == [
        "plugin.yaml",
        "prepared_plugin.yaml",
        "rolling_plugin.yaml",
    ]
