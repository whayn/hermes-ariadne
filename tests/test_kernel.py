"""Archive-backed kernel tests use only synthetic messages and isolated files."""

import hashlib

import pytest

from hermes_ariadne.store import EventStore, canonical_json


def event_id(message):
    return hashlib.sha256(canonical_json(message)).hexdigest()


def test_exact_event_lookup_is_session_scoped_and_preserves_versions(tmp_path):
    archive = EventStore(tmp_path / "archive.sqlite3")
    original = {"message_uid": "same", "role": "user", "content": "code λ\n  keep spaces  \ud800"}
    edited = {**original, "content": "edited later"}
    archive.ingest("chosen", [original], origin="fixture")
    archive.ingest("chosen", [edited], origin="fixture")
    assert archive.get_event(event_id(original), session_id="chosen") == original
    assert archive.get_event(event_id(edited), session_id="chosen") == edited
    assert archive.get_event(event_id(original), session_id="foreign") is None
    returned = archive.get_event(event_id(original), session_id="chosen")
    returned["content"] = "local edit"
    assert archive.get_event(event_id(original), session_id="chosen") == original
    with pytest.raises(ValueError, match="event"):
        archive.get_event("not-an-event", session_id="chosen")


def test_projection_requires_explicit_scoped_versions_and_pages_exact_bytes(tmp_path):
    import base64
    import json

    from hermes_ariadne.sources import Projection

    archive = EventStore(tmp_path / "archive.sqlite3")
    original = {"message_uid": "same", "role": "user", "content": "λ\n  spaces  \ud83d\ude00\ud800"}
    edited = {**original, "content": "later revision"}
    archive.ingest("chosen", [original, edited], origin="fixture")
    ids = [event_id(original)]
    projection = Projection(archive, "chosen", ids)
    assert projection.event_ids == tuple(ids)
    assert projection.message(0) == original
    data = bytearray()
    offset = 0
    while True:
        page = projection.page(0, offset=offset, limit=7)
        assert page["event_id"] == ids[0]
        data.extend(base64.b64decode(page["data_base64"], validate=True))
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert bytes(data) == canonical_json(original)
    assert json.loads(data.decode("utf-8", "surrogatepass")) == original
    assert (
        Projection(archive, "chosen", [event_id(edited)]).projection_id != projection.projection_id
    )
    with pytest.raises(ValueError, match="revision|identity"):
        Projection(archive, "chosen", [event_id(original), event_id(edited)])
    with pytest.raises(ValueError, match="scope|missing"):
        Projection(archive, "foreign", ids)
    with pytest.raises(ValueError, match="duplicate"):
        Projection(archive, "chosen", ids + ids)
    with pytest.raises(ValueError, match="offset|limit"):
        projection.page(0, offset=True, limit=1)
    with pytest.raises(ValueError, match="offset|range"):
        projection.page(0, offset=10000, limit=7)


def kernel_fixture(tmp_path, count=32):
    from hermes_ariadne.sources import Projection

    archive = EventStore(tmp_path / "archive.sqlite3")
    messages = [
        {"message_uid": f"m{i}", "role": "user", "content": f"topic {i}: " + "λ" * 150}
        for i in range(count)
    ]
    archive.ingest("chosen", messages, origin="fixture")
    return Projection(archive, "chosen", [event_id(m) for m in messages])


def test_kernel_builds_persists_and_zooms_without_future_context(tmp_path):
    from hermes_ariadne.kernel import MemoryKernel

    projection = kernel_fixture(tmp_path)
    path = tmp_path / "derived.json"
    kernel = MemoryKernel(projection, path, node_bytes=64, chat_high=256, compactor_high=512)
    requests = []

    def compact(request):
        requests.append(request)
        expected_cutoff = request.part.start if request.part.level == 0 else request.part.stop
        assert request.context_end <= expected_cutoff
        for part in request.context_parts:
            assert part.stop <= request.context_end
        if request.part.level == 0:
            assert request.inputs == (kernel.source_input(request.part.index),)
        else:
            assert len(request.inputs) == 2
        return f"topic range {request.part.start}+{request.part.count}"

    result = kernel.build(compact)
    assert result["built"] == 63
    assert result["pending"] == 0
    assert kernel.view()["usable"]
    assert kernel.view()["summary_bytes"] <= 256
    first = kernel.view()["parts"][0]["ref"]
    zoomed = kernel.zoom(first)
    assert zoomed["kind"] == "node"
    assert len(zoomed["children"]) == 2
    while zoomed["kind"] != "source":
        first = zoomed["children"][0]["ref"]
        zoomed = kernel.zoom(first, limit=19)
    assert zoomed["page"]["event_id"] == projection.event_ids[0]
    reloaded = MemoryKernel(projection, path, node_bytes=64, chat_high=256, compactor_high=512)
    assert reloaded.view() == kernel.view()
    assert reloaded.zoom(first, limit=19) == zoomed
    assert reloaded.build(lambda _: pytest.fail("ready summaries must be reused"))["built"] == 0
    assert requests


@pytest.mark.parametrize("ready", [False, True], ids=["pending", "ready"])
@pytest.mark.parametrize("present", [False, True], ids=["absent-path", "existing-path"])
def test_kernel_from_snapshot_never_reads_or_initializes_a_state_file(
    tmp_path, monkeypatch, ready, present
):
    import copy
    import json
    from pathlib import Path

    from hermes_ariadne.kernel import MemoryKernel

    projection = kernel_fixture(tmp_path, 2)
    source = MemoryKernel(projection, tmp_path / "source.json", node_bytes=64)
    if ready:
        source.build(lambda _: "captured summary")
    snapshot = json.loads(source.path.read_bytes())
    original = copy.deepcopy(snapshot)
    target = tmp_path / "snapshot.json"
    if present:
        target.write_bytes(b"not the captured snapshot")

    def forbidden(*args, **kwargs):
        pytest.fail("snapshot construction must not probe/read/create/save state")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(MemoryKernel, "_load", forbidden)
        patch.setattr(MemoryKernel, "_save", forbidden)
        loaded = MemoryKernel.from_snapshot(
            projection, target, snapshot=snapshot, **source._settings()
        )
    assert loaded.path == target and loaded.projection is projection
    assert loaded.view() == source.view()
    assert loaded._settings() == source._settings()
    assert snapshot == original
    if ready:
        assert loaded.prefix_view(2) == source.prefix_view(2)
    else:
        with pytest.raises(ValueError, match="pending"):
            loaded.prefix_view(2)
    if present:
        assert target.read_bytes() == b"not the captured snapshot"
    else:
        assert not target.exists()


@pytest.mark.parametrize("mutation", ["shape", "projection", "settings", "node", "view"])
def test_kernel_from_snapshot_rejects_invalid_state_without_blank_fallback(
    tmp_path, monkeypatch, mutation
):
    import json

    from hermes_ariadne.kernel import MemoryKernel

    projection = kernel_fixture(tmp_path, 2)
    source = MemoryKernel(projection, tmp_path / "source.json", node_bytes=64)
    source.build(lambda _: "captured summary")
    snapshot = json.loads(source.path.read_bytes())
    if mutation == "shape":
        snapshot = None
    elif mutation == "projection":
        snapshot["projection"]["id"] = "0" * 64
    elif mutation == "settings":
        snapshot["settings"]["summary_version"] = "foreign"
    elif mutation == "node":
        snapshot["nodes"][0]["text"] = "x" * 65
    elif mutation == "view":
        snapshot["chat"]["size"] += 1
    target = tmp_path / "must-not-create" / "snapshot.json"

    def forbidden(*args, **kwargs):
        pytest.fail("invalid captured state must not create a blank checkpoint")

    monkeypatch.setattr(MemoryKernel, "_save", forbidden)
    with pytest.raises(ValueError):
        MemoryKernel.from_snapshot(projection, target, snapshot=snapshot, **source._settings())
    assert not target.parent.exists()


def test_kernel_pending_oversized_and_foreign_nodes_are_explicit(tmp_path):
    from hermes_ariadne.kernel import MemoryKernel

    projection = kernel_fixture(tmp_path, 2)
    kernel = MemoryKernel(projection, tmp_path / "derived.json", node_bytes=64, chat_high=64)
    result = kernel.build(lambda _: None)
    assert result["pending"] == 3
    assert not kernel.view()["usable"]
    assert kernel.view()["pending"]
    assert not kernel.zoom(kernel.reference(kernel.parts[0]))["ready"]
    with pytest.raises(ValueError, match="byte|summary"):
        kernel.build(lambda _: "λ" * 33)
    with pytest.raises(ValueError, match="projection|reference"):
        kernel.zoom("0" * 64 + ":0+1")
    with pytest.raises(ValueError, match="reference|range"):
        kernel.zoom(projection.projection_id + ":0+3")
    kernel.build(lambda request: f"range {request.part.start}+{request.part.count}")
    assert kernel.view()["usable"]


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("alias", [False, True], ids=["canonical", "resolved-alias"])
def test_kernel_rejects_absent_archive_companion_paths(tmp_path, suffix, alias):
    from hermes_ariadne.kernel import MemoryKernel

    projection = kernel_fixture(tmp_path, 1)
    archive = projection._archive
    original = projection.message(0)
    companions = [
        archive.path.with_name(archive.path.name + s) for s in ("-wal", "-shm", "-journal")
    ]
    assert all(not path.exists() for path in companions)
    parent = archive.path.parent
    if alias:
        parent = tmp_path / "archive-alias"
        parent.symlink_to(archive.path.parent, target_is_directory=True)
    path = parent / (archive.path.name + suffix)
    before = set(archive.path.parent.iterdir())

    with pytest.raises(ValueError, match="archive|separate"):
        MemoryKernel(projection, path)
    with pytest.raises(ValueError, match="archive|separate"):
        MemoryKernel.from_snapshot(projection, path, snapshot={})

    assert set(archive.path.parent.iterdir()) == before
    assert all(not companion.exists() for companion in companions)
    assert archive.get_event(projection.event_ids[0], session_id="chosen") == original


@pytest.mark.parametrize("view", ["chat", "seen"])
def test_kernel_rejects_pending_internal_view_without_changing_checkpoint(tmp_path, view):
    import json

    from hermes_ariadne.kernel import MemoryKernel

    projection = kernel_fixture(tmp_path, 2)
    path = tmp_path / "derived.json"
    kernel = MemoryKernel(projection, path, node_bytes=64)
    # Pending leaves are legal, including across a checkpoint reload.
    assert MemoryKernel(projection, path, node_bytes=64).view() == kernel.view()
    result = kernel.build(lambda request: "leaf" if request.part.level == 0 else None)
    assert result["pending"] == 1
    assert MemoryKernel(projection, path, node_bytes=64).view() == kernel.view()
    data = json.loads(path.read_bytes())
    assert data[view]["parts"] == [[0, 0], [0, 1]]
    assert all(node["level"] == 0 for node in data["nodes"])
    data[view]["parts"] = [[1, 0]]
    data[view]["size"] = 0
    path.write_bytes(canonical_json(data))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="missing|pending|internal"):
        MemoryKernel(projection, path, node_bytes=64)

    assert path.read_bytes() == before


def test_kernel_refuses_mismatched_or_foreign_derived_files(tmp_path):
    from hermes_ariadne.kernel import MemoryKernel
    from hermes_ariadne.sources import Projection

    projection = kernel_fixture(tmp_path, 2)
    path = tmp_path / "derived.json"
    MemoryKernel(projection, path, node_bytes=64)
    different = Projection(projection._archive, "chosen", projection.event_ids[:1])
    with pytest.raises(ValueError, match="projection|settings"):
        MemoryKernel(different, path, node_bytes=64)
    with pytest.raises(ValueError, match="settings"):
        MemoryKernel(projection, path, node_bytes=65)
    foreign = tmp_path / "foreign.json"
    foreign.write_text('{"unrelated": true}')
    before = foreign.read_bytes()
    with pytest.raises(ValueError, match="derived|format"):
        MemoryKernel(projection, foreign)
    assert foreign.read_bytes() == before
    with pytest.raises(ValueError, match="archive|separate"):
        MemoryKernel(projection, projection._archive.path)
