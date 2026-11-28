"""Admission of Ariadne-owned storage paths for the MVP plugin (stdlib only)."""

import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageRefused(ValueError):
    """Ariadne's private storage failed admission (links, permissions, foreign files)."""


def tighten_private(path, *, regular):
    """Admit one Ariadne-owned path, restoring private permissions if they drifted.

    Refuses symlinks, non-regular/hardlinked files and paths owned by someone else;
    a too-open mode on our own file or directory is reset to 0600/0700 with a
    warning instead of refusing. Missing paths are accepted (nothing to admit).
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise StorageRefused("ariadne paths must not be symlinks")
    if regular and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise StorageRefused("ariadne store must be a regular single-link file")
    if not regular and not stat.S_ISDIR(info.st_mode):
        raise StorageRefused("ariadne directory is not a directory")
    if info.st_mode & 0o077:
        if info.st_uid != os.geteuid():
            raise StorageRefused("ariadne path is owned by another user")
        os.chmod(path, 0o600 if regular else 0o700)
        logger.warning("ariadne: reset too-open permissions on its private storage")


def mkdir_private(path):
    """Create the missing directories of ``path`` (and itself) with mode 0700."""
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)


def archive_path(home, *, store=None):
    """Metadata-only admission, before EventStore can open or create SQLite files.

    The Hermes home itself only has to be free of symlinks, since Hermes creates 0755
    profile homes; Ariadne's own directory and files must still be private and are
    reset to 0700/0600 if their mode drifted, rather than refused.
    """
    root = Path(home).absolute()
    archive = root / "ariadne" / "events.sqlite3"
    for item in (root, *root.parents):
        if item.is_symlink():
            raise ValueError("ariadne home must not traverse symlinks")
    tighten_private(archive.parent, regular=False)
    for suffix in ("", "-wal", "-shm", "-journal"):
        tighten_private(Path(str(archive) + suffix), regular=True)
    if store is not None and (store.path.absolute() != archive or not archive.is_file()):
        raise ValueError("rolling requires the initialized explicit-home archive")
    return archive
