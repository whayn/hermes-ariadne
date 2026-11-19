"""Admission of Ariadne-owned storage paths for the MVP plugin (stdlib only)."""

import logging
import os
import stat

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
