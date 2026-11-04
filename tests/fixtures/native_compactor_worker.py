"""Dedicated sequential native-factory worker; never imports a live Hermes home."""

import ipaddress
import json
import os
import socket
import sys
from pathlib import Path

assert Path(os.environ["HERMES_HOME"]).name == "isolated-home"
assert Path(os.environ["HOME"]) == Path(os.environ["HERMES_HOME"]).parent
assert not any("proxy" in key.lower() for key in os.environ)

# A second independent guard makes any unexpected discovery/auth network fail offline.
_connect = socket.socket.connect
_getaddrinfo = socket.getaddrinfo


def loopback_getaddrinfo(host, *args, **kwargs):
    if host == "localhost":
        host = "127.0.0.1"
    if not ipaddress.ip_address(host).is_loopback:
        raise AssertionError("non-loopback DNS forbidden")
    return _getaddrinfo(host, *args, **kwargs)


def loopback_connect(sock, address):
    if not isinstance(address, tuple) or not ipaddress.ip_address(address[0]).is_loopback:
        raise AssertionError("non-loopback connection forbidden")
    return _connect(sock, address)


socket.socket.connect = loopback_connect
socket.getaddrinfo = loopback_getaddrinfo

from agent import auxiliary_client as native  # noqa: E402
from agent.model_metadata import estimate_request_tokens_rough  # noqa: E402

payload = json.loads(sys.argv[1])
messages = payload["messages"] or [
    {"role": "system", "content": "Summarize only the synthetic messages."},
    {"role": "user", "content": "Synthetic note café: a sample, not a private corpus."},
]
reservations = []
report = {"ok": False, "reservations": reservations}
transport = hooks = leaf = None
observed_responses = []


def record_response(response):
    observed_responses.append((response, response.stream, dict(response.__dict__)))


try:
    from hermes_ariadne.native_compactor import NativeCompactorAdapter

    if payload["action"] == "unsupported":
        native.get_text_auxiliary_client = lambda *a, **kw: (object(), "ariadne/test")
    adapter = NativeCompactorAdapter.from_hermes(**{"isolated_worker": True, **payload["options"]})
    leaf = adapter._sdk
    transport = leaf._client
    transport.event_hooks["response"].append(record_response)
    if payload["action"] == "raw-decoder-unavailable":
        import builtins

        original_import = builtins.__import__

        def without_zstd(name, *args, **kwargs):
            if name == "compression":
                raise ImportError("simulated pre-3.14 stdlib")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = without_zstd
    if payload["action"].startswith("zstd-"):
        from agent.codex_headers import CODEX_AUX_BASE_URL, codex_request_compression_hooks

        compression_hooks = codex_request_compression_hooks(CODEX_AUX_BASE_URL)
        assert compression_hooks is not None
        transport.event_hooks["request"].extend(compression_hooks["request"])
        report["actual_compression_hook"] = True
    if payload["action"] == "wire-mutation":

        def mutate(request):
            request.url = request.url.copy_with(path="/v1/embeddings")

        transport.event_hooks["request"].append(mutate)
    if payload["action"] == "no-send":
        from types import SimpleNamespace

        adapter._client.chat.completions.create = lambda **kw: SimpleNamespace(choices=[])
    if payload["action"] == "retry-config":
        leaf.max_retries = 7
    if payload["action"] == "redirect-config":
        transport.follow_redirects = True
    if payload["action"] in {"wire-model-mutation", "cap-mutation"}:
        original_create = adapter._client.chat.completions.create

        def change_kwargs(**kwargs):
            if payload["action"] == "wire-model-mutation":
                kwargs["model"] = "not-the-frozen-model"
            else:
                kwargs.pop("max_tokens", None)
                kwargs.pop("max_completion_tokens", None)
            return original_create(**kwargs)

        adapter._client.chat.completions.create = change_kwargs
    hooks = {key: list(value) for key, value in transport.event_hooks.items()}
    if payload["action"] == "transport-replaced":
        leaf._client = object()

    class LedgerBudgetError(Exception):
        pass

    def reserve(data):
        if payload["action"] == "zstd-budget-refusal":
            raise LedgerBudgetError("synthetic-budget-refusal")
        reservations.append(data)
        if payload["action"] == "budget-refusal":
            raise LedgerBudgetError("synthetic-budget-refusal")
        if payload["action"] == "concurrent":
            try:
                adapter.dispatch(messages, before_send=reservations.append)
            except Exception as exc:
                report["concurrent_outcome"] = exc.outcome

    if payload["action"] in {"durable", "durable-terminal"}:
        import hashlib

        from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor
        from hermes_ariadne.kernel import MemoryKernel
        from hermes_ariadne.sources import Projection
        from hermes_ariadne.store import EventStore, canonical_json

        root = Path(os.environ["HOME"])
        archive = EventStore(root / "source.sqlite")
        message = {"message_uid": "refusal-case", "role": "user", "content": "synthetic " * 100}
        archive.ingest("synthetic", [message], origin="test")
        projection = Projection(
            archive, "synthetic", [hashlib.sha256(canonical_json(message)).hexdigest()]
        )
        kernel = MemoryKernel(
            projection,
            root / "kernel.json",
            node_bytes=96,
            chat_high=256,
            compactor_high=512,
            summary_version="native-test-v1",
        )
        limits = {
            "max_attempts": 1,
            "max_request_bytes": 8000,
            "max_total_request_bytes": 8000,
            "max_attempts_per_job": 1,
            "max_jobs": 8,
        }
        if payload["action"] == "durable-terminal":
            # Leave retry budget available: uncertainty, not exhaustion, must block restart.
            limits.update(max_attempts=3, max_attempts_per_job=3, max_total_request_bytes=24000)
        with DurableCompactor(
            kernel,
            adapter,
            root / "jobs.sqlite",
            policy=CompactionPolicy(context_limit_tokens=8192),
            limits=limits,
        ) as runner:
            report.update(
                build=runner.build(),
                ledger=runner.ledger.report(),
                job=runner.ledger.inspect("node:0+1"),
                receipt=runner.ledger.get_receipt("node:0+1"),
                usable=kernel.view()["usable"],
                published_nodes=len(kernel._nodes),
            )
        completion = None
    else:
        completion = adapter.dispatch(messages, before_send=reserve)
    report.update(
        ok=True,
        completion=completion,
        fingerprint=adapter.fingerprint,
        estimate=adapter.estimate_tokens(messages),
        native_estimate=estimate_request_tokens_rough(messages),
        hooks_clean=transport.event_hooks == hooks,
        same_transport=leaf._client is transport,
        retries=leaf.max_retries,
    )
except Exception as exc:
    report.update(
        error_type=type(exc).__name__,
        outcome=getattr(exc, "outcome", None),
        reason=getattr(exc, "reason", str(exc)),
        status_code=getattr(exc, "status_code", None),
        cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
    )
if transport is not None:
    report.update(
        hooks_clean=transport.event_hooks == hooks,
        same_transport=leaf._client is transport,
        retries=leaf.max_retries,
    )
report["responses_closed"] = all(response.is_closed for response, _, _ in observed_responses)
report["response_streams_unchanged"] = all(
    response.stream is stream for response, stream, _ in observed_responses
)
report["response_observers_clean"] = all(
    response.__dict__.get("iter_bytes") is previous.get("iter_bytes")
    for response, _, previous in observed_responses
)
print(json.dumps(report))
