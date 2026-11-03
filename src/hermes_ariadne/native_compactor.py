"""Offline, single-worker compatibility bridge through Hermes' native factory.

This is deliberately NOT the live plugin LLM API or a general provider layer.
Only the inspected synchronous OpenAI SDK and native Codex wrapper are supported.
No call_llm recovery ladder, provider construction, credential handling or rotation.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any, Callable

# Private SDK _client access is source-backed only for these exercised versions.
_SUPPORTED_SDK = "2.24.0"
_SUPPORTED_HTTPX = "0.28.1"
_DISPATCH_LOCK = threading.Lock()


class NativeDispatchError(RuntimeError):
    """A safe classification; the exception message never contains provider data."""

    def __init__(self, reason: str, *, outcome: str = "not_dispatched", status_code=None):
        super().__init__(reason)
        self.reason = reason
        self.outcome = outcome
        self.status_code = status_code


class NativeCompactorAdapter:
    """One frozen native route, reserved at the serialized HTTP request boundary.

    Inspected host seams: auxiliary_client.get_text_auxiliary_client (5506),
    CodexAuxiliaryClient (1665), its single-stream create (1555-1621), and
    model_metadata.estimate_request_tokens_rough (2581). Line numbers describe
    the locally inspected source, not a historical-release compatibility claim;
    fingerprint.host_fingerprint records its actual auxiliary_client.py hash.
    """

    _client: Any
    _sdk: Any
    _http: Any
    _codex: bool
    _model: str
    _output_tokens: int | None
    _max_request_bytes: int
    _timeout: float
    _endpoint: Any
    _wire_model: str | None
    _cap_field: str
    _provider: str
    _fingerprint: dict[str, Any]

    @classmethod
    def from_hermes(
        cls,
        *,
        task="compression",
        main_runtime=None,
        max_request_bytes=8000,
        max_output_tokens=512,
        timeout=30,
        allow_unbounded_output=False,
        isolated_worker=False,
    ) -> NativeCompactorAdapter:
        if isolated_worker is not True:
            raise NativeDispatchError("dedicated isolated worker required")
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise NativeDispatchError("invalid request byte limit")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise NativeDispatchError("invalid timeout")
        if not math.isfinite(timeout) or timeout <= 0:
            raise NativeDispatchError("invalid timeout")
        import httpx
        import openai
        from agent import auxiliary_client as native

        if openai.__version__ != _SUPPORTED_SDK or httpx.__version__ != _SUPPORTED_HTTPX:
            raise NativeDispatchError("unverified SDK transport version")
        try:
            client, model = native.get_text_auxiliary_client(task, main_runtime=main_runtime)
        except Exception as exc:
            raise NativeDispatchError("native factory unavailable") from exc
        codex = type(client) is native.CodexAuxiliaryClient
        sdk = client._real_client if codex else client
        if type(sdk) is not openai.OpenAI:
            raise NativeDispatchError("unsupported native client")
        if not isinstance(model, str) or not model:
            raise NativeDispatchError("native factory returned no model")
        if not isinstance(getattr(sdk, "_client", None), httpx.Client):
            raise NativeDispatchError("unsupported SDK HTTP transport")
        if codex:
            if allow_unbounded_output is not True or max_output_tokens is not None:
                raise NativeDispatchError(
                    "Codex requires explicit unbounded output consent and no cap"
                )
        elif type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise NativeDispatchError("Chat requires a positive output cap")
        self = cls()
        self._client, self._sdk, self._codex = client, sdk, codex
        self._http = sdk._client
        self._model, self._output_tokens = model, max_output_tokens
        self._max_request_bytes, self._timeout = max_request_bytes, timeout
        base = sdk.base_url
        if base.scheme not in {"http", "https"} or not base.host or base.username or base.password:
            raise NativeDispatchError("unsupported native origin")
        self._endpoint = base.join("responses" if codex else "chat/completions")
        self._wire_model = model
        if codex:
            from agent.codex_responses_adapter import _wire_model_identity

            self._wire_model = _wire_model_identity(model)
        # No auth lookup for cap selection: the host's model predicate plus frozen host.
        from utils import model_forces_max_completion_tokens

        self._cap_field = (
            "max_completion_tokens"
            if model_forces_max_completion_tokens(model)
            or base.host in {"api.openai.com", "api.githubcopilot.com"}
            or base.host.endswith(".githubcopilot.com")
            else "max_tokens"
        )
        provider = getattr(client, "_hermes_aux_effective_provider", "")
        if provider not in {"openai", "openai-codex", "openrouter", "nous", "copilot", "actual"}:
            provider = (
                "openai-codex"
                if codex and base.host == "chatgpt.com"
                else "openai"
                if base.host == "api.openai.com"
                else "custom"
            )
        self._provider = provider
        self._fingerprint = {
            "provider": provider,
            "model": model,
            "transport": "native_codex_responses" if codex else "openai_chat_completions",
            "host_fingerprint": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "route_fingerprint": hashlib.sha256(str(self._endpoint).encode()).hexdigest(),
            "sdk_version": openai.__version__,
            "httpx_version": httpx.__version__,
            "estimator": "hermes_native_rough",
            "output_tokens": max_output_tokens,
            "cap_field": None if codex else self._cap_field,
        }
        if codex:
            self._fingerprint["completion_validation"] = "codex-terminal-v1"
        return self

    @property
    def model(self) -> str:
        return self._model

    @property
    def output_tokens(self) -> int | None:
        return self._output_tokens

    @property
    def fingerprint(self) -> dict[str, Any]:
        return dict(self._fingerprint)

    def estimate_tokens(self, messages) -> int:
        """Hermes' rough full-prompt estimate, NOT an exact provider token count."""
        from agent.model_metadata import estimate_request_tokens_rough

        return estimate_request_tokens_rough(messages)

    def dispatch(self, messages, *, before_send: Callable[[dict], Any]) -> dict:
        import openai

        if not callable(before_send):
            raise NativeDispatchError("reservation callback required")
        sdk, http = self._sdk, self._http
        if sdk._client is not http or http.is_closed:
            raise NativeDispatchError("native HTTP transport changed or closed")
        hooks = http.event_hooks["request"]
        if not _DISPATCH_LOCK.acquire(blocking=False):
            raise NativeDispatchError("concurrent dispatch forbidden")
        attempts, reserved, callback_error = 0, False, None
        owner = threading.get_ident()
        old_retries = sdk.max_retries
        metadata = {
            "usage": None,
            "request_id": None,
            "refusal": False,
            "terminal": None,
            "tool_call": False,
        }
        response_hooks = http.event_hooks["response"]
        observed_responses = []
        missing = object()

        def observe_response(response):
            metadata["request_id"] = response.headers.get("x-request-id")
            if not self._codex or not response.is_success:
                return
            from openai._streaming import SSEDecoder

            media_type = response.headers.get("content-type")
            if media_type is not None and media_type.split(";", 1)[0] != "text/event-stream":
                raise ValueError("unsupported Codex response encoding")
            original = response.iter_bytes
            previous = response.__dict__.get("iter_bytes", missing)

            def metadata_bytes(chunk_size=None):
                # Observe HTTPX-decoded bytes from this response instance only.
                # SDK SSEDecoder and Hermes STILL own all text/finish handling.
                decoder = SSEDecoder()
                pending_bytes = 0

                def bounded_fragments():
                    nonlocal pending_bytes
                    for chunk in original(chunk_size=chunk_size):
                        # Match the pinned SDK's inner fragment sequence exactly,
                        # but check the limit BEFORE its _iter_chunks buffers it.
                        for fragment in chunk.splitlines(keepends=True):
                            pending_bytes += len(fragment)
                            if pending_bytes > 1024 * 1024:
                                raise ValueError("Codex metadata frame exceeds observation limit")
                            yield fragment

                def observe_frame(frame):
                    # Observe ALL SDK events in a frame before native can stop.
                    for raw_line in frame.splitlines():
                        event = decoder.decode(raw_line.decode("utf-8"))
                        if event is None or event.data == "[DONE]":
                            continue
                        data = json.loads(event.data)
                        kind = data.get("type")
                        items = [data.get("item") or data.get("part") or {}]
                        if kind in {
                            "response.completed",
                            "response.incomplete",
                            "response.failed",
                        }:
                            if metadata["terminal"] is not None:
                                raise ValueError("duplicate Codex terminal response")
                            terminal = data.get("response")
                            if (
                                not isinstance(terminal, dict)
                                or not isinstance(terminal.get("id"), str)
                                or not terminal["id"].strip()
                                or terminal.get("status") != kind.removeprefix("response.")
                                or "output" not in terminal
                                or (
                                    terminal.get("output") is not None
                                    and (
                                        not isinstance(terminal["output"], list)
                                        or not all(isinstance(i, dict) for i in terminal["output"])
                                    )
                                )
                            ):
                                raise ValueError("invalid Codex terminal metadata")
                            # SDK Response.id/status/error/incomplete_details are the
                            # completion proof. Codex output=None is source-backed in
                            # auxiliary_client.create; text remains native-owned.
                            if kind == "response.completed" and (
                                terminal.get("error") is not None
                                or terminal.get("incomplete_details") is not None
                            ):
                                raise ValueError("contradictory Codex completion metadata")
                            metadata["terminal"] = kind
                            metadata["usage"] = terminal.get("usage")
                            items.extend(terminal.get("output") or [])
                        if kind in {"response.refusal.delta", "response.refusal.done"}:
                            metadata["refusal"] = True
                        for item in items:
                            if isinstance(item, dict) and str(item.get("type", "")).endswith(
                                "_call"
                            ):
                                metadata["tool_call"] = True
                            parts = item.get("content", []) if isinstance(item, dict) else []
                            if isinstance(item, dict) and item.get("type") == "refusal":
                                metadata["refusal"] = True
                            if any(
                                isinstance(p, dict) and p.get("type") == "refusal" for p in parts
                            ):
                                metadata["refusal"] = True

                frames = decoder._iter_chunks(bounded_fragments())
                for frame in frames:
                    pending_bytes = 0
                    observe_frame(frame)
                    if metadata["terminal"] is not None:
                        # Native stops at its first terminal. Validate the remaining
                        # decoded stream first, without buffering it or assembling
                        # Responses, so later conflicting terminals cannot hide.
                        for tail in frames:
                            pending_bytes = 0
                            observe_frame(tail)
                    yield frame

            observed_responses.append((response, previous, metadata_bytes))
            response.iter_bytes = metadata_bytes

        def request_hook(request):
            nonlocal attempts, reserved, callback_error
            attempts += 1
            if attempts != 1 or threading.get_ident() != owner:
                raise NativeDispatchError("second or concurrent physical send forbidden")
            endpoint = self._endpoint
            if request.method != "POST" or (
                request.url.scheme,
                request.url.host,
                request.url.port,
                request.url.path,
            ) != (endpoint.scheme, endpoint.host, endpoint.port, endpoint.path):
                raise NativeDispatchError("native inference route changed")
            body = request.content
            if len(body) > self._max_request_bytes:
                raise NativeDispatchError("serialized request exceeds byte limit")
            coding = request.headers.get("content-encoding")
            decoded = body
            if coding is not None:
                if coding.strip().lower() != "zstd":
                    raise NativeDispatchError("unsupported native request encoding")
                try:
                    from compression import zstd
                except ImportError as exc:
                    raise NativeDispatchError("native zstd decoder unavailable") from exc
                # Single ordinary frame only; no skippable/concatenated frames.
                # Bound output AND the decoder window before allocating either.
                if not body.startswith(b"\x28\xb5\x2f\xfd"):
                    raise NativeDispatchError("invalid native zstd frame")
                info = zstd.get_frame_info(body)
                if (
                    info.decompressed_size is not None
                    and info.decompressed_size > self._max_request_bytes
                ):
                    raise NativeDispatchError("decoded request exceeds byte limit")
                decoder = zstd.ZstdDecompressor(
                    options={
                        zstd.DecompressionParameter.window_log_max: max(
                            10, (self._max_request_bytes - 1).bit_length()
                        )
                    }
                )
                decoded = decoder.decompress(body, max_length=self._max_request_bytes + 1)
                if len(decoded) > self._max_request_bytes:
                    raise NativeDispatchError("decoded request exceeds byte limit")
                if not decoder.eof or decoder.unused_data:
                    raise NativeDispatchError("incomplete or trailing native zstd frame")
            wire = json.loads(decoded)
            if wire.get("model") != self._wire_model:
                raise NativeDispatchError("native wire model changed")
            if not self._codex and wire.get(self._cap_field) != self.output_tokens:
                raise NativeDispatchError("native output cap changed")
            if self._codex and any(
                k in wire for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            ):
                raise NativeDispatchError("unexpected Codex output cap")
            try:
                before_send(
                    {
                        "request_sha256": hashlib.sha256(body).hexdigest(),
                        "request_bytes": len(body),
                        "output_tokens": self.output_tokens,
                    }
                )
            except Exception as exc:
                callback_error = exc
                raise
            reserved = True

        try:
            sdk.max_retries = 0
            hooks.append(request_hook)
            response_hooks.append(observe_response)
            kwargs = {"model": self.model, "messages": messages, "timeout": self._timeout}
            if not self._codex:
                kwargs[self._cap_field] = self.output_tokens
            response = self._client.chat.completions.create(**kwargs)
            if not reserved:
                raise NativeDispatchError("native client did not reserve a physical request")
            if self._codex and metadata["terminal"] is None:
                raise ValueError("Codex stream has no terminal response")
            choices = getattr(response, "choices", None) or []
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None)
            text = getattr(message, "content", None)
            finish_reason = getattr(choice, "finish_reason", None)
            if (
                self._codex
                and finish_reason == "stop"
                and (metadata["terminal"] != "response.completed" or metadata["tool_call"])
            ):
                raise ValueError("Codex terminal does not prove a successful text completion")
            if text is not None and not isinstance(text, str):
                raise ValueError("invalid completion text type")
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise ValueError("invalid completion finish reason type")
            if getattr(message, "tool_calls", None) and finish_reason != "tool_calls":
                raise ValueError("inconsistent tool completion")
            usage = metadata["usage"] if self._codex else getattr(response, "usage", None)
            if usage is not None:
                usage = (
                    usage.model_dump(mode="json", exclude_unset=True)
                    if hasattr(usage, "model_dump")
                    else (usage if isinstance(usage, dict) else vars(usage))
                )
                usage = json.loads(json.dumps(usage, allow_nan=False))
            return {
                "text": text,
                "finish_reason": finish_reason,
                "usage": usage,
                "provider": self._provider,
                "model": self.model,
                "request_id": metadata["request_id"] or getattr(response, "_request_id", None),
                "refusal": metadata["refusal"] or bool(getattr(message, "refusal", None)),
            }
        except Exception as exc:
            outcome = "uncertain" if reserved else "not_dispatched"
            status = exc.status_code if isinstance(exc, openai.APIStatusError) else None
            if status is not None:
                outcome = "failed"
            reason = (
                "reservation refused"
                if callback_error is not None
                else (
                    "native HTTP failure"
                    if status is not None
                    else "native dispatch blocked or interrupted"
                )
            )
            raise NativeDispatchError(reason, outcome=outcome, status_code=status) from (
                callback_error if callback_error is not None else exc
            )
        finally:
            for response, previous, observer in observed_responses:
                if response.__dict__.get("iter_bytes") is observer:
                    if previous is missing:
                        del response.iter_bytes
                    else:
                        response.iter_bytes = previous
            if request_hook in hooks:
                hooks.remove(request_hook)
            if observe_response in response_hooks:
                response_hooks.remove(observe_response)
            sdk.max_retries = old_retries
            _DISPATCH_LOCK.release()
