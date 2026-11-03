"""Opt-in, synthetic native-compactor demo; never activates the live engine."""

import argparse
import base64
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from .compaction import CompactionPolicy, DurableCompactor
from .kernel import MemoryKernel
from .sources import Projection
from .store import EventStore, canonical_json

KERNEL_SETTINGS = {
    "node_bytes": 320,
    "chat_high": 640,
    "compactor_high": 2048,
    "summary_version": "native-demo-v1",
}
DEMO_FORMAT = "ariadne-native-demo-v1"


def _write_private(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(data))
        stream.flush()
        os.fsync(stream.fileno())


def _adapter(task, provider, model, allow_unbounded):
    from .native_compactor import NativeCompactorAdapter

    kwargs = {
        "task": task,
        "max_request_bytes": 8000,
        "max_output_tokens": None if allow_unbounded else 512,
        "allow_unbounded_output": allow_unbounded,
        "timeout": 30,
        "isolated_worker": True,
    }
    if provider is None and model is None:
        return NativeCompactorAdapter.from_hermes(**kwargs)
    if not provider or not model:
        raise ValueError("an explicit route needs both provider and model")
    # Temporary host-owned task routing; do NOT edit active profile configuration.
    from agent.auxiliary_task_config import session_auxiliary_scope

    with session_auxiliary_scope({task: {"provider": provider, "model": model}}):
        return NativeCompactorAdapter.from_hermes(**kwargs)


def build(
    directory,
    *,
    allow_model_calls=False,
    allow_unbounded=False,
    task="compression",
    provider=None,
    model=None,
):
    if not allow_model_calls:
        raise ValueError("model dispatch requires explicit --allow-model-calls")
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    code = "def NEEDLE(value):\n    # λ; retain indentation and trailing whitespace\n    return value + 1  \n"
    code += "    # archived code comment; exact whitespace matters  \n" * 15
    garden = "Separate garden topic: basil needs water; project deadline remains undecided. " * 16
    messages = [
        {"message_uid": "native-code", "role": "user", "content": code},
        {"message_uid": "native-garden", "role": "assistant", "content": garden},
    ]
    archive = EventStore(directory / "archive" / "events.sqlite3")
    archive.ingest("synthetic-native", messages, origin="synthetic-native-demo")
    ids = [hashlib.sha256(canonical_json(message)).hexdigest() for message in messages]
    projection = Projection(archive, "synthetic-native", ids)
    kernel = MemoryKernel(projection, directory / "kernel.json", **KERNEL_SETTINGS)
    adapter = _adapter(task, provider, model, allow_unbounded)
    policy = CompactionPolicy(context_limit_tokens=8192)
    limits = {
        "max_attempts": 3,
        "max_request_bytes": 8000,
        "max_total_request_bytes": 3 * 8000,
        "max_attempts_per_job": 2,
        "max_jobs": 16,
    }
    manifest = {
        "format": DEMO_FORMAT,
        "session_id": projection.session_id,
        "event_ids": ids,
        "kernel_settings": KERNEL_SETTINGS,
        "policy": asdict(policy),
        "limits": limits,
        "route": adapter.fingerprint,
        "model": adapter.model,
        "output_tokens": adapter.output_tokens,
        "expected_event_id": ids[0],
        "expected_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    }
    _write_private(directory / "manifest.json", manifest)
    with DurableCompactor(
        kernel, adapter, directory / "jobs.sqlite", policy=policy, limits=limits
    ) as runner:
        result = runner.build()
        return {
            "phase": "build",
            "pid": os.getpid(),
            **result,
            "route": adapter.fingerprint,
            "output_token_ceiling": adapter.output_tokens,
            "ledger": runner.status()["ledger"],
            "blocked": runner.status()["blocked"],
            "summaries": kernel.view()["parts"],
        }


class _ReceiptOnlyAdapter:
    """Read existing receipts without any host imports, authentication or network."""

    def __init__(self, manifest):
        route = manifest["route"]
        if (
            route.get("transport") == "native_codex_responses"
            and route.get("completion_validation") != "codex-terminal-v1"
        ):
            raise ValueError("Codex receipt validation version is unsupported")
        self.fingerprint = route
        self.model = manifest["model"]
        self.output_tokens = manifest["output_tokens"]
        self.calls = 0

    def dispatch(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("receipt-only rebuild must never dispatch a model")

    def estimate_tokens(self, messages):
        raise RuntimeError("cached receipts must not require a new request estimate")


def rebuild(directory):
    manifest = json.loads(
        (directory / "manifest.json").read_bytes().decode("utf-8", "surrogatepass")
    )
    if manifest.get("format") != DEMO_FORMAT or manifest["session_id"] != "synthetic-native":
        raise ValueError("not an owned synthetic compactor demo")
    archive = EventStore(directory / "archive" / "events.sqlite3")
    projection = Projection(archive, manifest["session_id"], manifest["event_ids"])
    # A new state path simulates a lost checkpoint without deleting any file.
    kernel = MemoryKernel(
        projection, directory / "rebuild-kernel.json", **manifest["kernel_settings"]
    )
    initial_ready = len(kernel._nodes)
    adapter = _ReceiptOnlyAdapter(manifest)
    with DurableCompactor(
        kernel,
        adapter,
        directory / "jobs.sqlite",
        policy=CompactionPolicy(**manifest["policy"]),
        limits=manifest["limits"],
    ) as runner:
        result = runner.build()
        if result["pending"]:
            return {
                "phase": "rebuild",
                "pid": os.getpid(),
                **result,
                "model_dispatches": adapter.calls,
                "exact_source_recovered": False,
            }
        candidates = kernel.view()["parts"]
        navigation = []
        while True:
            matches = [
                item for item in candidates if item["ready"] and "needle" in item["text"].lower()
            ]
            if not matches:
                raise RuntimeError("generated summaries did not retain the navigation identifier")
            chosen = min(matches, key=lambda item: item["start"])
            navigation.append(chosen["ref"])
            zoomed = kernel.zoom(chosen["ref"], limit=97)
            if zoomed["kind"] == "source":
                break
            candidates = zoomed["children"]
        data = bytearray()
        page = zoomed["page"]
        pages = 0
        while True:
            data.extend(base64.b64decode(page["data_base64"], validate=True))
            pages += 1
            if page["next_offset"] is None:
                break
            page = kernel.zoom(chosen["ref"], offset=page["next_offset"], limit=97)["page"]
        message = json.loads(data.decode("utf-8", "surrogatepass"))
        recovered_hash = hashlib.sha256(message["content"].encode()).hexdigest()
        exact = (
            recovered_hash == manifest["expected_code_sha256"]
            and hashlib.sha256(data).hexdigest() == manifest["expected_event_id"]
        )
        if not exact:
            raise RuntimeError("retrieved source does not equal the archived original")
        return {
            "phase": "rebuild",
            "pid": os.getpid(),
            **result,
            "initial_ready_nodes": initial_ready,
            "model_dispatches": adapter.calls,
            "navigation": navigation,
            "pages": pages,
            "exact_source_recovered": exact,
            "recovered_sha256": recovered_hash,
            "ledger": runner.status()["ledger"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="NEW synthetic demo directory for build")
    parser.add_argument("--phase", choices=("build", "rebuild"), required=True)
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--allow-unbounded-codex-output", action="store_true")
    parser.add_argument("--task", default="compression")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    args = parser.parse_args()
    result = (
        build(
            args.directory,
            allow_model_calls=args.allow_model_calls,
            allow_unbounded=args.allow_unbounded_codex_output,
            task=args.task,
            provider=args.provider,
            model=args.model,
        )
        if args.phase == "build"
        else rebuild(args.directory)
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    if result.get("pending"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
