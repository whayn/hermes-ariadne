"""Two-process offline demo using synthetic data only; no Hermes imports/models."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

from .kernel import MemoryKernel
from .simulation import SourceMessage, simulate_request
from .sources import Projection
from .store import EventStore, canonical_json
from .tree import Part

SETTINGS = {
    "node_bytes": 96,
    "chat_high": 384,
    "compactor_high": 1536,
    "summary_version": "synthetic-topic-markers-v1",
}


def synthetic_compactor(request):
    """Fixture navigation labels, NOT an evaluation of an AI's summary quality."""
    combined = " ".join(request.inputs)
    labels = [word for word in ("NEEDLE", "garden", "project", "deadline") if word in combined]
    return "Topics: " + ", ".join(labels or ["conversation"])


def build(directory):
    # A new directory is required so the demo cannot overwrite unrelated data.
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    code = "def NEEDLE(value):\n    # λ and emoji 🧭; keep whitespace\n    return value + 1  \n"
    code += "    # original indentation and trailing spaces  \n" * 120
    messages = []
    for index in range(65):
        topic = ("garden", "project", "deadline")[index % 3]
        content = code if index == 17 else f"{topic} discussion {index}. " + ("λ details " * 40)
        messages.append(
            {
                "message_uid": f"synthetic-{index}",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": content,
            }
        )
    archive = EventStore(directory / "archive" / "events.sqlite3")
    archive.ingest("synthetic-life-chat", messages, origin="synthetic-demo")
    ids = [hashlib.sha256(canonical_json(message)).hexdigest() for message in messages]
    projection = Projection(archive, "synthetic-life-chat", ids)
    kernel = MemoryKernel(projection, directory / "derived.json", **SETTINGS)
    result = kernel.build(synthetic_compactor)
    retained_from = len(messages) - 3
    slices = []
    for item in kernel.prefix_view(retained_from):
        count = item["stop"] - item["start"]
        part = Part(count.bit_length() - 1, item["start"] // count)
        slices.append(
            {
                "node_id": kernel.reference(part),
                "event_ids": ids[item["start"] : item["stop"]],
                "text": item["text"],
            }
        )
    simulation = simulate_request(
        [{"role": "system", "content": "Synthetic offline fixture."}],
        [SourceMessage(identity, message) for identity, message in zip(ids, messages, strict=True)],
        retain_from=retained_from,
        summaries=slices,
        budget_units=20000,
    )
    exact_suffix = simulation.messages is not None and simulation.messages[-3:] == messages[-3:]
    expectation = {
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "event_id": ids[17],
    }
    # Hash and identity only: the recall phase cannot recover the code from this.
    expectation_path = directory / "expectation.json"
    fd = os.open(expectation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(expectation))
    return {
        "phase": "build",
        "pid": os.getpid(),
        **result,
        "projection_id": projection.projection_id,
        "source_messages": len(messages),
        "view_usable": kernel.view()["usable"],
        "summary_bytes": kernel.view()["summary_bytes"],
        "simulation_status": simulation.status,
        "simulation_ascii_json_bytes": simulation.used_units,
        "retained_suffix_exact": exact_suffix,
        "expected_code_sha256": expectation["code_sha256"],
    }


def _archive_count(archive):
    with archive._connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]


def recall(directory):
    # No fixture construction, ingestion, summary calls or regenerated source.
    state = json.loads((directory / "derived.json").read_bytes().decode("utf-8", "surrogatepass"))
    expectation = json.loads((directory / "expectation.json").read_bytes())
    archive = EventStore(directory / "archive" / "events.sqlite3")
    before = _archive_count(archive)
    projection = Projection(
        archive, state["projection"]["session_id"], state["projection"]["event_ids"]
    )
    kernel = MemoryKernel(projection, directory / "derived.json", **SETTINGS)
    candidates = kernel.view()["parts"]
    navigation = []
    while True:
        matches = [item for item in candidates if item["ready"] and "NEEDLE" in item["text"]]
        if len(matches) != 1:
            raise RuntimeError("synthetic marker is not uniquely navigable")
        reference = matches[0]["ref"]
        navigation.append(reference)
        zoomed = kernel.zoom(reference, limit=127)
        if zoomed["kind"] == "source":
            break
        candidates = zoomed["children"]
    page = zoomed["page"]
    recovered = bytearray()
    pages = 0
    while True:
        recovered.extend(base64.b64decode(page["data_base64"], validate=True))
        pages += 1
        if page["next_offset"] is None:
            break
        page = kernel.zoom(reference, offset=page["next_offset"], limit=127)["page"]
    message = json.loads(recovered.decode("utf-8", "surrogatepass"))
    digest = hashlib.sha256(message["content"].encode("utf-8")).hexdigest()
    exact = (
        digest == expectation["code_sha256"]
        and zoomed["page"]["event_id"] == expectation["event_id"]
        and hashlib.sha256(recovered).hexdigest() == expectation["event_id"]
    )
    if not exact:
        raise RuntimeError("exact original code was not recovered")
    return {
        "phase": "recall",
        "pid": os.getpid(),
        "navigation": navigation,
        "pages": pages,
        "recovered_event_id": zoomed["page"]["event_id"],
        "recovered_sha256": digest,
        "exact_code_recovered": exact,
        "archive_events_before": before,
        "archive_events_after": _archive_count(archive),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new synthetic demo directory")
    parser.add_argument("--phase", choices=("build", "recall"), required=True)
    args = parser.parse_args()
    result = build(args.directory) if args.phase == "build" else recall(args.directory)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
