"""The repository root is the Hermes plugin, so its pyproject must install nothing.

Hermes builds every enabled plugin that declares packages into one shared
environment for all profiles. A buildable member keeps its declared name, so a
second profile installing Ariadne would fail with duplicate workspace members,
and the package would be installed into Hermes's managed environment.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_pyproject_declares_nothing_for_hermes_to_install():
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = document["project"]
    assert "build-system" not in document
    assert not project.get("dependencies")
    assert not any(project.get("optional-dependencies", {}).values())
    assert not any(document.get("dependency-groups", {}).values())
    uv = document.get("tool", {}).get("uv", {})
    assert uv.get("package") is not True and not uv.get("dev-dependencies")
