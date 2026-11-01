"""Explicit directory installation; never edit config or an existing plugin."""

import argparse
import shutil
from pathlib import Path

# Copy only runtime files, not caches, tests, or the installer's environment.
_RUNTIME_FILES = ("__init__.py", "engine.py", "store.py", "plugin.yaml")


def install(home):
    """Copy beneath canonical home; reject pre-existing destination symlinks.

    The explicit home itself may be a symlink. Existing final files, directories,
    and symlinks (including dangling ones) raise FileExistsError unchanged.
    Concurrent malicious directory replacement is outside this installer's scope.
    """
    home = Path(home).expanduser().resolve()
    target = home / "plugins" / "ariadne"
    if target.parent.is_symlink():
        raise FileExistsError(f"Refusing symlink destination component: {target.parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700)  # Refuse ANY pre-existing directory or symlink.
    try:
        for name in _RUNTIME_FILES:
            shutil.copyfile(Path(__file__).parent / name, target / name)
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
    args = parser.parse_args()
    target = install(args.home.expanduser().resolve())
    print(f"Installed {target}; configuration unchanged. Select context.engine=ariadne explicitly.")


if __name__ == "__main__":
    main()
