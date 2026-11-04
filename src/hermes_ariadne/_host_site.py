"""Print a selected interpreter's purelib without importing site or host code.

Execute this file directly with ``python -I -S``. Before Python 3.14, -S leaves
sys.prefix at the base installation even when sys.executable names a venv.
Only the lexical executable location and presence of pyvenv.cfg are needed;
never resolve its symlink to the base interpreter or execute/read venv config.
"""

import sys
import sysconfig
from pathlib import Path


def host_site() -> str:
    """Return this interpreter's venv purelib, or its ordinary installation path."""
    prefix = Path(sys.executable).parent.parent
    if (prefix / "pyvenv.cfg").is_file():
        return sysconfig.get_path(
            "purelib", scheme="venv", vars={"base": str(prefix), "platbase": str(prefix)}
        )
    return sysconfig.get_path("purelib")


if __name__ == "__main__":
    print(host_site())
