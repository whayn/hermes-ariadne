"""Concurrent-append worker for tests/test_tree_store.py (no pytest import).

Run as ``python tree_store_writer.py <src> <home> <name> <count> <start_file>``: waits for
``start_file`` so both writers begin together, then appends ``count`` single entries.
"""

import hashlib
import sys
import time
from pathlib import Path


def main():
    src, home, name, count, start = sys.argv[1:6]
    sys.path.insert(0, src)
    from hermes_ariadne.tree_store import TreeStore

    deadline = time.monotonic() + 60
    while not Path(start).exists():
        if time.monotonic() > deadline:
            raise SystemExit("start signal timed out")
        time.sleep(0.01)
    store = TreeStore(home)
    for i in range(int(count)):
        text = f"{name}-{i}"
        store.append(
            f"{name}-{i % 3}",
            [
                {
                    "event_id": hashlib.sha256(text.encode()).hexdigest(),
                    "message_uid": "uid-" + text,
                    "role": "user",
                }
            ],
            origin="live",
        )


if __name__ == "__main__":
    main()
