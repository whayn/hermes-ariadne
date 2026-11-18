"""Describe a selected source byte range; never print its contents or follow links."""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--start-bytes", type=int, required=True)
    parser.add_argument("--length-bytes", type=int, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    try:
        path = args.path
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("explicit non-symlink absolute source required")
        if not 0 <= args.start_bytes or not 1 <= args.length_bytes <= 1048576:
            raise ValueError("source range must be bounded")
        if not args.label or len(args.label.encode("utf-8")) > 160:
            raise ValueError("short source label required")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or args.start_bytes + args.length_bytes > info.st_size
            ):
                raise ValueError("regular single-link source and existing range required")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                stream.seek(args.start_bytes)
                data = stream.read(args.length_bytes)
            if len(data) != args.length_bytes:
                raise ValueError("source range changed while reading")
        finally:
            os.close(fd)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "start_bytes": args.start_bytes,
                    "length_bytes": args.length_bytes,
                    "label": args.label,
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
                ensure_ascii=True,
            )
        )
    except (OSError, ValueError):
        # Do not echo source data, paths or OS diagnostics.
        parser.exit(1, "source range refused\n")


if __name__ == "__main__":
    main()
