"""Synthetic compaction planning and durable kernel integration."""

import hashlib
import json

import pytest

from hermes_ariadne.kernel import SummaryRequest
from hermes_ariadne.tree import Part


def request(text, *, limit=96):
    return SummaryRequest(Part(0, 2), (text,), (Part(1, 0),), ("0+2|past project",), 2, limit)


def test_request_fits_reserves_requested_output_cap():
    from hermes_ariadne.compaction import CompactionPolicy, request_fits

    assert not request_fits(
        [{"role": "user", "content": "source"}],
        policy=CompactionPolicy(context_limit_tokens=1024, output_reserve_tokens=1),
        model="fixture",
        output_tokens=512,
        estimate_tokens=lambda messages: 1023,
    )


@pytest.mark.parametrize(
    ("reserve", "cap", "estimate", "fits"),
    [
        (1, 512, 512, True),  # Requested cap fits exactly, not just the smaller reserve.
        (1, 512, 513, False),
        (768, 512, 256, True),  # A larger configured reserve must still be honored.
        (768, 512, 257, False),
        (1, None, 1023, True),  # Explicitly uncapped Codex retains the rough reserve.
        (1, None, 1024, False),
        (512, None, 512, True),
        (512, None, 513, False),
    ],
)
def test_request_fits_effective_reserve_boundaries(reserve, cap, estimate, fits):
    from hermes_ariadne.compaction import CompactionPolicy, request_fits

    assert (
        request_fits(
            [{"role": "user", "content": "source"}],
            policy=CompactionPolicy(context_limit_tokens=1024, output_reserve_tokens=reserve),
            model="fixture",
            output_tokens=cap,
            estimate_tokens=lambda messages: estimate,
        )
        is fits
    )


@pytest.mark.parametrize("cap", [True, False, 0, -1, 1.5, "512"])
def test_request_fits_rejects_invalid_output_caps(cap):
    from hermes_ariadne.compaction import CompactionPolicy, request_fits

    with pytest.raises(ValueError, match="output token cap"):
        request_fits(
            [{"role": "user", "content": "source"}],
            policy=CompactionPolicy(context_limit_tokens=1024),
            model="fixture",
            output_tokens=cap,
            estimate_tokens=lambda messages: 1,
        )


def test_plan_covers_whole_escaped_unicode_input_with_bounded_requests():
    from hermes_ariadne.compaction import (
        CompactionPolicy,
        make_plan,
        messages_for,
        serialize_inputs,
    )

    req = request('long λ😀 source \\"\n' * 400)
    policy = CompactionPolicy(
        context_limit_tokens=8192, max_request_bytes=2400, request_margin_bytes=128, max_chunks=64
    )
    plan = make_plan(req, policy=policy, model="fixture", output_tokens=128)
    source = serialize_inputs(req.inputs)
    assert len(plan["spans"]) > 1
    assert "source" not in plan  # The canonical archive remains the source, not this plan.
    pieces = []
    end = 0
    for span in plan["spans"]:
        assert span["start"] == end
        text = source[span["start"] : span["stop"]]
        assert hashlib.sha256(text.encode("ascii")).hexdigest() == span["sha256"]
        pieces.append(text)
        end = span["stop"]
        messages = messages_for(plan, text, kind="chunk", span=span)
        payload = {"model": "fixture", "messages": messages, "max_tokens": 128}
        assert len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()) <= 2272
    assert end == len(source)
    assert "".join(pieces) == source
    assert json.loads(source) == list(req.inputs)
    assert plan["context_end"] == 2


def test_plan_rejects_future_context_and_chunk_limit_without_clipping():
    from hermes_ariadne.compaction import CompactionIncomplete, CompactionPolicy, make_plan

    req = request("a" * 20000)
    policy = CompactionPolicy(
        context_limit_tokens=8192, max_request_bytes=1800, request_margin_bytes=128, max_chunks=1
    )
    with pytest.raises(CompactionIncomplete, match="chunk|fit"):
        make_plan(req, policy=policy, model="fixture", output_tokens=128)
    future = SummaryRequest(Part(0, 2), ("data",), (Part(0, 3),), ("3+1|FUTURE",), 4, 96)
    with pytest.raises(ValueError, match="future|cutoff"):
        make_plan(future, policy=policy, model="fixture", output_tokens=128)


class ScriptedAdapter:
    """Explicit test fixture; real native HTTP transport is tested separately."""

    model = "scripted"
    output_tokens = 128
    fingerprint = {"provider": "fixture", "model": "scripted", "transport": "fixture-v1"}

    def __init__(self, replies=None):
        self.calls = 0
        self.replies = list(replies or [])

    def estimate_tokens(self, messages):
        return len(json.dumps(messages)) // 3

    def dispatch(self, messages, *, before_send):
        payload = json.dumps(
            {"model": self.model, "messages": messages, "max_tokens": self.output_tokens},
            separators=(",", ":"),
        ).encode()
        before_send(
            {
                "request_sha256": hashlib.sha256(payload).hexdigest(),
                "request_bytes": len(payload),
                "output_tokens": self.output_tokens,
            }
        )
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else "NEEDLE code and garden project"
        if isinstance(reply, Exception):
            raise reply
        return {
            "text": reply,
            "finish_reason": "stop",
            "usage": None,
            "provider": "fixture",
            "model": self.model,
            "refusal": False,
        }


def kernel_and_limits(tmp_path, count=2):
    from hermes_ariadne.kernel import MemoryKernel
    from hermes_ariadne.sources import Projection
    from hermes_ariadne.store import EventStore, canonical_json

    archive = EventStore(tmp_path / "source.sqlite")
    messages = [
        {"message_uid": f"m{i}", "role": "user", "content": "NEEDLE code λ\n" * 80}
        for i in range(count)
    ]
    archive.ingest("synthetic", messages, origin="test")
    ids = [hashlib.sha256(canonical_json(message)).hexdigest() for message in messages]
    projection = Projection(archive, "synthetic", ids)
    kernel = MemoryKernel(
        projection,
        tmp_path / "kernel.json",
        node_bytes=96,
        chat_high=256,
        compactor_high=512,
        summary_version="native-test-v1",
    )
    limits = {
        "max_attempts": 20,
        "max_request_bytes": 8000,
        "max_total_request_bytes": 160000,
        "max_attempts_per_job": 2,
        "max_jobs": 128,
    }
    return kernel, limits


def test_output_cap_forces_token_bounded_chunks_before_reservation(tmp_path):
    from hermes_ariadne.compaction import (
        CompactionPolicy,
        DurableCompactor,
        make_plan,
        messages_for,
        serialize_inputs,
    )

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    req = SummaryRequest(Part(0, 0), (kernel.source_input(0),), (), (), 0, kernel.node_bytes)
    source = serialize_inputs(req.inputs)
    whole_plan = make_plan(
        req, policy=CompactionPolicy(context_limit_tokens=8192), model="scripted", output_tokens=512
    )
    whole_messages = messages_for(whole_plan, source, kind="chunk", span=whole_plan["spans"][0])
    estimate = ScriptedAdapter().estimate_tokens(whole_messages)
    policy = CompactionPolicy(context_limit_tokens=estimate + 1, output_reserve_tokens=1)

    class CappedAdapter(ScriptedAdapter):
        output_tokens = 512

        def dispatch(self, messages, *, before_send):
            # Check BEFORE the fixture invokes the durable reservation callback.
            assert (
                self.estimate_tokens(messages) + self.output_tokens <= policy.context_limit_tokens
            )
            return super().dispatch(messages, before_send=before_send)

    adapter = CappedAdapter()
    with DurableCompactor(
        kernel, adapter, tmp_path / "capped-chunks.sqlite", policy=policy, limits=limits
    ) as runner:
        assert runner.build()["pending"] == 0
        plan = runner.ledger.get_plan("node:0+1")
        assert plan is not None
        assert len(plan["spans"]) > 1
        assert "".join(source[s["start"] : s["stop"]] for s in plan["spans"]) == source
        assert adapter.calls > len(plan["spans"])  # Bounded reductions also dispatched.
        assert runner.ledger.report()["attempts"] == adapter.calls


def test_output_cap_blocks_unplannable_request_without_durable_reservation(tmp_path):
    from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor

    class NoRoomAdapter(ScriptedAdapter):
        output_tokens = 512

        def estimate_tokens(self, messages):
            return 1023

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    adapter = NoRoomAdapter()
    policy = CompactionPolicy(context_limit_tokens=1024, output_reserve_tokens=1)
    with DurableCompactor(
        kernel, adapter, tmp_path / "no-room.sqlite", policy=policy, limits=limits
    ) as runner:
        assert runner.build()["pending"] == 1
        assert adapter.calls == 0
        assert runner.ledger.get_plan("node:0+1") is None
        report = runner.ledger.report()
        assert report["attempts"] == 0
        assert report["reserved_request_bytes"] == 0
        assert "cannot fit" in runner.status()["blocked"]["node:0+1"]


def test_receipts_rebuild_lost_kernel_checkpoint_without_new_dispatches(tmp_path):
    from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor

    kernel, limits = kernel_and_limits(tmp_path)
    policy = CompactionPolicy(context_limit_tokens=8192)
    adapter = ScriptedAdapter()
    ledger = tmp_path / "jobs.sqlite"
    with DurableCompactor(kernel, adapter, ledger, policy=policy, limits=limits) as runner:
        assert runner.build()["pending"] == 0
        assert adapter.calls == 3
        expected = kernel.view()
    kernel.path.unlink()  # Lose derived state, never the canonical source.
    kernel, _ = kernel_and_limits(tmp_path)
    resumed = ScriptedAdapter()
    with DurableCompactor(kernel, resumed, ledger, policy=policy, limits=limits) as runner:
        assert runner.build()["pending"] == 0
        assert resumed.calls == 0
        assert kernel.view() == expected


def test_long_source_chunks_reduce_and_resume_without_repaying(tmp_path):
    from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor, serialize_inputs

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    limits.update(max_request_bytes=1800, max_attempts=100)
    policy = CompactionPolicy(
        context_limit_tokens=8192, max_request_bytes=1800, request_margin_bytes=128
    )
    adapter = ScriptedAdapter()
    path = tmp_path / "long-jobs.sqlite"
    with DurableCompactor(kernel, adapter, path, policy=policy, limits=limits) as runner:
        result = runner.build()
        assert result["pending"] == 0
        plan = runner.ledger.get_plan("node:0+1")
        source = serialize_inputs((kernel.source_input(0),))
        assert len(plan["spans"]) > 1
        assert "".join(source[s["start"] : s["stop"]] for s in plan["spans"]) == source
        assert adapter.calls > len(
            plan["spans"]
        )  # Actual reduction jobs, not clipping/aliasing input.
    kernel.path.unlink()
    kernel, _ = kernel_and_limits(tmp_path, count=1)
    again = ScriptedAdapter()
    with DurableCompactor(kernel, again, path, policy=policy, limits=limits) as runner:
        assert runner.build()["pending"] == 0
        assert again.calls == 0


def test_durable_job_budget_blocks_without_publishing_partial_summary(tmp_path):
    from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    limits.update(max_request_bytes=1800, max_jobs=1)
    policy = CompactionPolicy(
        context_limit_tokens=8192, max_request_bytes=1800, request_margin_bytes=128
    )
    adapter = ScriptedAdapter()
    with DurableCompactor(
        kernel, adapter, tmp_path / "limited.sqlite", policy=policy, limits=limits
    ) as runner:
        assert runner.build()["pending"] == 1
        assert adapter.calls == 0
        assert runner.status()["blocked"]
        assert not kernel.view()["usable"]


def test_uncertain_model_attempt_is_not_automatically_retried_after_restart(tmp_path):
    from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor

    class LostResponse(RuntimeError):
        outcome = "uncertain"
        reason = "synthetic_disconnect"

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    policy = CompactionPolicy(context_limit_tokens=8192)
    ledger = tmp_path / "uncertain.sqlite"
    adapter = ScriptedAdapter([LostResponse()])
    with DurableCompactor(kernel, adapter, ledger, policy=policy, limits=limits) as runner:
        assert runner.build()["pending"] == 1
        assert adapter.calls == 1
        assert runner.ledger.inspect("node:0+1")["state"].lower() == "uncertain"
    recovered = ScriptedAdapter()
    with DurableCompactor(kernel, recovered, ledger, policy=policy, limits=limits) as runner:
        assert runner.build()["pending"] == 1
        assert recovered.calls == 0
        assert runner.status()["blocked"]


def test_invalid_outputs_exhaust_durable_cap_and_missing_usage_stays_unknown(tmp_path):
    from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    policy = CompactionPolicy(context_limit_tokens=8192)
    ledger = tmp_path / "failed.sqlite"
    bad = ScriptedAdapter(["λ" * 100, ""])
    with DurableCompactor(kernel, bad, ledger, policy=policy, limits=limits) as runner:
        assert runner.build()["pending"] == 1
        assert bad.calls == 2
        assert runner.ledger.get_receipt("node:0+1") is None
    good = ScriptedAdapter()
    with DurableCompactor(kernel, good, ledger, policy=policy, limits=limits) as runner:
        assert runner.build()["pending"] == 1
        assert good.calls == 0
    # Explicitly creating a DIFFERENT campaign is not an automatic budget reset.
    with DurableCompactor(
        kernel, good, tmp_path / "authorized-new-campaign.sqlite", policy=policy, limits=limits
    ) as runner:
        assert runner.build()["pending"] == 0
        assert runner.ledger.get_receipt("node:0+1")["usage"] is None


def test_unverified_existing_summaries_and_namespace_changes_are_rejected(tmp_path):
    from hermes_ariadne.compaction import CompactionPolicy, DurableCompactor
    from hermes_ariadne.receipts import LedgerError

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    kernel.build(lambda _: "unproven manual summary")
    with pytest.raises(ValueError, match="unverified"):
        DurableCompactor(
            kernel,
            ScriptedAdapter(),
            tmp_path / "unverified.sqlite",
            policy=CompactionPolicy(context_limit_tokens=8192),
            limits=limits,
        )
    kernel.path.unlink()
    kernel, _ = kernel_and_limits(tmp_path, count=1)
    ledger = tmp_path / "versioned.sqlite"
    with DurableCompactor(
        kernel,
        ScriptedAdapter(),
        ledger,
        policy=CompactionPolicy(context_limit_tokens=8192),
        limits=limits,
    ):
        pass
    with pytest.raises(LedgerError, match="namespace|match|metadata"):
        DurableCompactor(
            kernel,
            ScriptedAdapter(),
            ledger,
            policy=CompactionPolicy(context_limit_tokens=16384),
            limits=limits,
        )


def test_prompt_content_is_part_of_durable_namespace_even_without_version_change(
    tmp_path, monkeypatch
):
    import hermes_ariadne.compaction as compaction
    from hermes_ariadne.receipts import LedgerError

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    path = tmp_path / "prompt-version.sqlite"
    policy = compaction.CompactionPolicy(context_limit_tokens=8192)
    with compaction.DurableCompactor(kernel, ScriptedAdapter(), path, policy=policy, limits=limits):
        pass
    monkeypatch.setattr(compaction, "INSTRUCTIONS", compaction.INSTRUCTIONS + " changed semantics")
    with pytest.raises(LedgerError, match="namespace|match|metadata"):
        compaction.DurableCompactor(kernel, ScriptedAdapter(), path, policy=policy, limits=limits)


@pytest.mark.parametrize("version", [None, "codex-terminal-v0"])
def test_receipt_only_adapter_rejects_legacy_codex_validation(version):
    from hermes_ariadne.compactor_demo import _ReceiptOnlyAdapter

    route = {"transport": "native_codex_responses"}
    if version is not None:
        route["completion_validation"] = version
    manifest = {"route": route, "model": "fixture", "output_tokens": None}
    with pytest.raises(ValueError, match="Codex.*validation"):
        _ReceiptOnlyAdapter(manifest)


def test_receipt_only_adapter_accepts_current_codex_validation_without_host_imports(monkeypatch):
    import builtins

    from hermes_ariadne.compactor_demo import _ReceiptOnlyAdapter

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"agent", "openai", "httpx"}:
            raise AssertionError("receipt-only adapter attempted host import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest = {
        "route": {
            "transport": "native_codex_responses",
            "completion_validation": "codex-terminal-v1",
        },
        "model": "fixture",
        "output_tokens": None,
    }
    adapter = _ReceiptOnlyAdapter(manifest)
    assert adapter.fingerprint == manifest["route"]
    assert adapter.calls == 0


def test_prompt_mutation_inside_campaign_is_rejected_before_dispatch(tmp_path, monkeypatch):
    import hermes_ariadne.compaction as compaction

    kernel, limits = kernel_and_limits(tmp_path, count=1)
    adapter = ScriptedAdapter()
    with compaction.DurableCompactor(
        kernel,
        adapter,
        tmp_path / "frozen.sqlite",
        policy=compaction.CompactionPolicy(context_limit_tokens=8192),
        limits=limits,
    ) as runner:
        monkeypatch.setattr(compaction, "INSTRUCTIONS", compaction.INSTRUCTIONS + " mutation")
        with pytest.raises(ValueError, match="prompt.*changed"):
            runner.build()
        assert adapter.calls == 0
