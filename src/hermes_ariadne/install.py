"""Explicit directory installation; never edit config or an existing plugin."""

import argparse
import shutil
from pathlib import Path

# Copy only runtime files, not caches, tests, or the installer's environment.
_RUNTIME_FILES = ("__init__.py", "engine.py", "store.py", "plugin.yaml")
_PREPARED_FILES = {
    "__init__.py": "prepared_entry.py",
    "plugin.yaml": "prepared_plugin.yaml",
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
        )
    },
}


def install(home, *, prepared=False):
    """Copy beneath canonical home; reject pre-existing destination symlinks.

    The explicit home itself may be a symlink. Existing final files, directories,
    and symlinks (including dangling ones) raise FileExistsError unchanged.
    Concurrent malicious directory replacement is outside this installer's scope.
    """
    if type(prepared) is not bool:
        raise ValueError("prepared must be an explicit boolean")
    home = Path(home).expanduser().resolve()
    target = home / "plugins" / ("ariadne-prepared" if prepared else "ariadne")
    if target.parent.is_symlink():
        raise FileExistsError(f"Refusing symlink destination component: {target.parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700)  # Refuse ANY pre-existing directory or symlink.
    try:
        files = _PREPARED_FILES if prepared else {name: name for name in _RUNTIME_FILES}
        for target_name, source_name in files.items():
            shutil.copyfile(Path(__file__).parent / source_name, target / target_name)
    except BaseException:
        shutil.rmtree(target)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home",
        required=True,
        type=Path,
        help="Explicit destination HERMES_HOME (use a disposable development home)",
    )
    parser.add_argument(
        "--prepared",
        action="store_true",
        help="Install the separate opt-in ariadne-prepared development engine",
    )
    args = parser.parse_args()
    target = install(args.home.expanduser().resolve(), prepared=args.prepared)
    print(
        f"Installed {target}; configuration unchanged. "
        f"Select context.engine={target.name} explicitly."
    )


if __name__ == "__main__":
    main()
