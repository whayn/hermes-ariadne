"""Pure generic-plugin contract; installed real-host coverage lives separately."""

import importlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path


def test_profile_bound_registration_preparation_and_user_approval(tmp_path, monkeypatch):
    assert importlib.util.find_spec("hermes_ariadne.library_plugin") is not None, (
        "library plugin missing"
    )
    plugin = importlib.import_module("hermes_ariadne.library_plugin")
    first = tmp_path / "first"
    first.mkdir(mode=0o700)
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    home = [first]
    monkeypatch.setitem(
        sys.modules, "hermes_constants", types.SimpleNamespace(get_hermes_home=lambda: home[0])
    )

    class Context:
        def __init__(self):
            self.tools, self.commands, self.skills = {}, {}, {}

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

        def register_command(self, name, handler, **kwargs):
            self.commands[name] = handler

        def register_skill(self, name, path, **kwargs):
            assert path.is_file()
            self.skills[name] = path

    ctx = Context()
    plugin.register(ctx)
    assert not (first / "ariadne-library").exists()
    assert len(ctx.tools) == 4 and "ariadne_memory_approve" not in ctx.tools
    assert "ariadne-import" in ctx.skills
    home[0] = other  # No later ambient-home lookup can retarget tools.
    raw = b"synthetic portable source"
    path = tmp_path / "input.txt"
    path.write_bytes(raw)
    import hashlib

    response = json.loads(
        ctx.tools["ariadne_memory_prepare"]["handler"](
            {
                "sources": [
                    {
                        "path": str(path),
                        "label": "Test",
                        "start_bytes": 0,
                        "length_bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                ],
                "notes": [
                    {
                        "title": "A",
                        "text": "portable source",
                        "kind": "quotation",
                        "tags": [],
                        "sources": [0],
                    }
                ],
            }
        )
    )
    assert response["ok"]
    ref = response["result"]["proposal_ref"]
    assert not (other / "ariadne-library").exists()
    approved = json.loads(ctx.commands["ariadne-approve"](ref))
    assert approved["ok"] and approved["result"]["inserted"] == 1
    found = json.loads(ctx.tools["ariadne_memory_search"]["handler"]({"query": "portable"}))
    assert found["ok"] and len(found["result"]["results"]) == 1
    for name, entry in ctx.tools.items():
        assert entry["schema"]["name"] == name
        assert not (
            {"home", "profile", "destination"} & set(entry["schema"]["parameters"]["properties"])
        )
        bad = entry["handler"]({"home": str(other)})
        assert len(bad.encode()) <= plugin.MAX_RESPONSE_BYTES and not json.loads(bad)["ok"]


def test_bundled_source_range_helper_captures_bytes_not_characters(tmp_path):
    root = Path(__file__).resolve().parents[1]
    helper = root / "src/hermes_ariadne/skills/ariadne-import/scripts/source_range.py"
    assert helper.is_file(), "bundled generic byte-range helper missing"
    path = tmp_path / "source.txt"
    path.write_bytes("α café\nNO".encode())
    proc = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(path),
            "--start-bytes",
            "3",
            "--length-bytes",
            "6",
            "--label",
            "Selected",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    descriptor = json.loads(proc.stdout)
    import hashlib

    assert descriptor["sha256"] == hashlib.sha256("café\n".encode()).hexdigest()
    assert descriptor["start_bytes"] == 3 and descriptor["length_bytes"] == 6
    assert "café" not in proc.stdout


def test_response_bound_includes_escaped_metadata_and_can_page_smaller(tmp_path):
    import hashlib

    from hermes_ariadne.library import Library
    from hermes_ariadne.library_plugin import MAX_RESPONSE_BYTES, TOOLS, _handler

    path = tmp_path / "source.txt"
    path.write_bytes(b"DO_NOT_EXPORT_RAW")
    store = Library(tmp_path)
    sources = [
        {
            "path": str(path),
            "label": "Private source",
            "start_bytes": 0,
            "length_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]
    notes = [
        {
            "title": "\x01" * 160,
            "text": f"needle{i}" + "\x01" * 200,
            "kind": "summary",
            "tags": ["\x01" * 48] * 8,
            "sources": [0],
        }
        for i in range(5)
    ]
    proposal = store.prepare(sources=sources, notes=notes)
    store.approve(proposal["proposal_ref"])
    handler = _handler(store.search, TOOLS["ariadne_memory_search"][1])
    large = handler({"query": "needle"})
    assert len(large.encode()) <= MAX_RESPONSE_BYTES
    assert json.loads(large)["error"] == "response_exceeds_limit"
    small = handler({"query": "needle", "limit": 1})
    assert json.loads(small)["ok"] and len(small.encode()) <= MAX_RESPONSE_BYTES
    assert "DO_NOT_EXPORT_RAW" not in small


def test_tool_errors_do_not_echo_source_paths_or_notes(tmp_path):
    from hermes_ariadne.library import Library
    from hermes_ariadne.library_plugin import TOOLS, _handler

    handler = _handler(Library(tmp_path).prepare, TOOLS["ariadne_memory_prepare"][1])
    result = handler({"sources": [{"path": "SECRET_PRIVATE_PATH"}], "notes": ["SECRET_NOTE"]})
    assert not json.loads(result)["ok"]
    assert "SECRET" not in result
