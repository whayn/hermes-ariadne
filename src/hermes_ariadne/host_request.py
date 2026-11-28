"""Hermes request correspondence for the tree engine (issue #70, phase 1).

Proves that the host's assembled request, its conversation history and the canonical
SessionDB rows correspond one-to-one (host prefix, ``message_uid`` identity, exact
payloads and tool occurrences, the ``<memory-context>`` recall block, admitted
reasoning carriers) before the engine replaces any of them.

Verbatim copies, unchanged, of the helpers the tree engine used from the development
engines: ``prepared`` (identity, comparison, carrier admission), ``rolling``
(``POLICY_ID``, ``completed_exchange_starts``, ``sha256``), ``rolling_selection``
(``_admit_messages``, ``incoming_anchor``) and ``simulation`` (``_tool_sequence``,
``ascii_json_units``). The originals are deleted with those engines (#70, phase 2).
"""

import hashlib
import json
import re
from dataclasses import dataclass

from .retrieval import _require_supported_data_use
from .store import REASONING_CARRIERS, canonical_json


def ascii_json_units(messages):
    """Canonical ASCII-JSON byte count, NOT a provider token estimate."""
    return len(
        json.dumps(
            messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    )


def _tool_sequence(messages):
    pending = set()
    previous_role = None
    for message in messages:
        role = message["role"]
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("orphan, duplicate or mismatched tool result")
            pending.remove(call_id)
        else:
            if pending:
                raise ValueError("pending tool results cannot be skipped")
            if previous_role == role:
                raise ValueError("retained user/assistant messages must alternate")
            calls = message.get("tool_calls")
            if calls is not None:
                if role != "assistant" or not isinstance(calls, list):
                    raise ValueError("tool calls must be an assistant list")
                for call in calls:
                    call_id = call.get("id") if isinstance(call, dict) else None
                    if not isinstance(call_id, str) or not call_id or call_id in pending:
                        raise ValueError(
                            "parallel tool call identities must be nonempty and unique"
                        )
                    pending.add(call_id)
        previous_role = role
    if pending:
        raise ValueError("missing tool results at the provider-request boundary")


POLICY_ID = "aged-codex-retirement-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def completed_exchange_starts(payloads, *, allow_current=False):
    """Require a final assistant before EVERY next user boundary.

    Tool-result completeness alone is not completed-response coverage. Inputs
    must already have passed the shared payload and tool occurrence validators.
    """
    starts = [i for i, message in enumerate(payloads) if message["role"] == "user"]
    if not starts or starts[0] != 0:
        raise ValueError("history must start with a user exchange")
    ends = starts[1:] + ([] if allow_current else [len(payloads)])
    for end in ends:
        final = payloads[end - 1]
        if final["role"] != "assistant" or final.get("tool_calls"):
            raise ValueError("completed exchange requires a final assistant")
    return tuple(starts)


_MAX_CANONICAL_CODEX_JSON_BYTES = 1_048_576


@dataclass(frozen=True)
class ReasoningReplayPolicy:
    """Trusted host label binding, not issuer authentication or provider-fit proof."""

    issuer_kind: str
    issuer_model: str


# Provider reasoning/replay state that only an explicit policy may admit. Keys only:
# a refusal names the key so the rolling engine can report it, never its value.
_REASONING_CARRIERS = REASONING_CARRIERS


# Rolling only (#29): other providers' reasoning, admitted as OPAQUE values. Never compared,
# read, summarized, logged or shown; it leaves requests with its retired exchange. Codex
# items keep their audited codec; bedrock_content_blocks (and any other key) stays refused.
_OPAQUE_CARRIERS = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "thinking",
        "anthropic_content_blocks",
        "_anthropic_content_blocks",
        "extra_content",
        "thought_signature",
    }
)


_MAX_CARRIER_MESSAGE_BYTES = 1_048_576


_MAX_CARRIER_REQUEST_BYTES = 8 * 1_048_576


_MUST_ECHO_PAD = " "  # Hermes pads reasoning_content on must-echo routes (DeepSeek/Kimi/MiMo).


_ABSENT = (None, "", False, [], {})  # The provider codec's legacy empty-as-absent values.


def _carrier_error(message, key):
    """A plain ``ValueError`` (same type and text for every engine) naming the carrier key."""
    error = ValueError(message)
    error.carrier = key
    return error


def _tag_policy_refusal(error, key):
    """Name the carrier on the policy refusal itself; any other error is left untouched.

    Re-raised unchanged by the caller, so its exact type (e.g. a UnicodeError from label
    validation) and message stay what every engine already sees.
    """
    if type(error) is ValueError and str(error) == "unsupported reasoning replay policy":
        error.carrier = key


def _require_reasoning_policy(policy):
    if (
        type(policy) is not ReasoningReplayPolicy
        or type(policy.issuer_kind) is not str
        or policy.issuer_kind != "codex_backend"
        or type(policy.issuer_model) is not str
        or not policy.issuer_model
        or policy.issuer_model[0].isspace()
        or policy.issuer_model[-1].isspace()
    ):
        raise ValueError("unsupported reasoning replay policy")
    # Policy labels have no independent length limit (including reasoning-free
    # selections), but validating one must not allocate its entire UTF-8 copy.
    for start in range(0, len(policy.issuer_model), 4096):
        policy.issuer_model[start : start + 4096].encode("utf-8")


def _json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate prepared JSON key")
            result[key] = item
        return result

    def invalid(_):
        raise ValueError("nonfinite prepared JSON")

    decoded = json.loads(
        value.decode("utf-8", "surrogatepass"), object_pairs_hook=pairs, parse_constant=invalid
    )
    _safe_json(decoded)  # Also reject finite-JSON numbers overflowing Python floats.
    return decoded


# Deliberately narrow, audited text/chat-completions mapping. Unknown carriers
# decline instead of silently assuming they are persistence-only fields.
_WIRE = {"role", "content", "tool_calls", "tool_call_id", "name"}


_METADATA = {
    "data_use",  # Only an explicitly empty list survives evidence admission.
    "message_uid",
    "timestamp",
    "display_kind",
    "display_metadata",
    "_row_id",
    # Hermes stamps restored dicts as durable; its chat transport strips this
    # persistence-only marker (hermes_state_messages._rows_to_conversation).
    "_db_persisted",
    "_submit_row_session_id",
    "_db_row_snapshot",
    "_canonical_row",
    "_restored_from_store",
    "id",
    "session_id",
    "token_count",
    "finish_reason",
    "tool_name",
    "effect_disposition",
    "platform_message_id",
    "observed",
    "compacted",
    "active",
    "display_order",
}


_CARRIERS = {
    "api_content",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "_compressed_summary",
    "_anthropic_content_blocks",
    "thinking",
    "_merged_turn_prefix",
}


_IDENTITIES = {
    "_absorbed_message_uids",
    "absorbed_message_uids",
    "_tool_call_uids",
    "tool_call_uids",
    "_tool_call_uid",
    "tool_call_uid",
}


def _safe_json(value):
    ascii_json_units(value)  # Reject cycles and non-finite numbers before traversal.
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("non-string JSON key")
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
        elif item is not None and type(item) not in {str, bool, int, float}:
            raise ValueError("non-JSON value")


def _ordinary_codex_message(message, *, wire=False, durable=False):
    """Validate only one reasoning-free, completed assistant output-text item.

    codex_responses_adapter._OutputScan._message stores this shape and
    _replay_message_items replays its text/status/id/phase. Deliberately accept
    only clean ASCII msg_* ids (narrower than the host's msg prefix rule), so
    no provider id/phase/status repair is needed or hidden by comparison.
    """
    items = message["codex_message_items"]
    if durable and type(items) is str:
        # SessionDB._reasoning_json_text persists this column as JSON TEXT;
        # _rows_to_conversation decodes it. Decode ONLY canonical active rows
        # for comparison, never arbitrary restored/request substitutions.
        # UTF-8 cannot be shorter than its input: reject before allocating bytes.
        if len(items) > _MAX_CANONICAL_CODEX_JSON_BYTES:
            raise ValueError("canonical Codex JSON exceeds its admission bound")
        encoded = items.encode("utf-8", "surrogatepass")
        if len(encoded) > _MAX_CANONICAL_CODEX_JSON_BYTES:
            raise ValueError("canonical Codex JSON exceeds its admission bound")
        items = _json(encoded)
    if (
        message["role"] != "assistant"
        or any(
            message.get(key) is not None
            for key in ("tool_calls", "tool_call_id", "tool_name", "effect_disposition")
        )
        or type(items) is not list
        or len(items) != 1
    ):
        raise ValueError("unsupported ordinary Codex message")
    item = items[0]
    required = {"type", "role", "status", "content"}
    if (
        type(item) is not dict
        or not required <= item.keys()
        or item.keys() - required - {"id", "phase"}
        or item["type"] != "message"
        or item["role"] != "assistant"
        or item["status"] != "completed"
        or ("phase" in item and item["phase"] != "final_answer")
    ):
        raise ValueError("unsupported Codex message item")
    if "id" in item and (
        type(item["id"]) is not str or re.fullmatch(r"msg_[A-Za-z0-9_-]{1,60}", item["id"]) is None
    ):
        raise ValueError("unsupported Codex message identity")
    parts = item["content"]
    if type(parts) is not list or len(parts) != 1:
        raise ValueError("unsupported Codex message content")
    part = parts[0]
    if (
        type(part) is not dict
        or set(part) != {"type", "text"}
        or part["type"] != "output_text"
        or type(part["text"]) is not str
        or not part["text"]
    ):
        raise ValueError("contradictory Codex message text")
    if part["text"] != message["content"] and not (
        (wire or message.get("_db_persisted") is True)
        and part["text"].strip() == message["content"]
    ):
        raise ValueError("contradictory Codex message text")
    return items


def _reasoning_list_bytes(items):
    """Exact compact UTF-8 JSON size for the finite encrypted carrier shape.

    Resource admission only: this grants no route, role, source or occurrence
    permission. Check builtin lengths before key sets or string serialization;
    a cumulative lower bound caps traversal even for many empty summaries.
    """
    if type(items) is not list or not items:
        raise ValueError("unsupported encrypted reasoning envelope")

    def bounded(size):
        if size > _MAX_CANONICAL_CODEX_JSON_BYTES:
            raise ValueError("Codex reasoning list exceeds its admission bound")
        return size

    # Each empty six-field item costs 114 bytes, plus commas and list brackets.
    # Admit the structural minimum BEFORE traversing items or allocating key sets.
    size = bounded(1 + 115 * len(items))
    for item in items:
        if (
            type(item) is not dict
            or len(item) != 6
            or set(item)
            != {
                "type",
                "encrypted_content",
                "id",
                "summary",
                "_issuer_kind",
                "_issuer_model",
            }
        ):
            raise ValueError("unsupported encrypted reasoning fields")
        if type(item["_issuer_kind"]) is str and item["_issuer_kind"] != "codex_backend":
            # Another issuer's Responses reasoning: same refusal, named for the rolling engine.
            raise _carrier_error("unsupported encrypted reasoning item", "codex_reasoning_items")
        if (
            item["type"] != "reasoning"
            or type(item["encrypted_content"]) is not str
            or not item["encrypted_content"]
            or type(item["id"]) is not str
            or len(item["id"]) > 64
            or re.fullmatch(r"rs_[A-Za-z0-9_-]{1,61}", item["id"]) is None
            or item["id"].startswith("rs_tmp_")
            or type(item["_issuer_kind"]) is not str
            or item["_issuer_kind"] != "codex_backend"
            or type(item["_issuer_model"]) is not str
            or type(item["summary"]) is not list
        ):
            raise ValueError("unsupported encrypted reasoning item")
        # Bound every item's raw payload and summary structure before encoding
        # ANY strings, including when only a later item exceeds the shared budget.
        parts = item["summary"]
        count = len(parts)
        size = bounded(size + 33 * count + max(count - 1, 0))
        for key in ("encrypted_content", "id", "_issuer_model"):
            size = bounded(size + len(item[key]))
        for part in parts:
            if (
                type(part) is not dict
                or len(part) != 2
                or set(part) != {"type", "text"}
                or part["type"] != "summary_text"
                or type(part["text"]) is not str
            ):
                raise ValueError("unsupported reasoning summary")
            size = bounded(size + len(part["text"]))
        if not item["encrypted_content"].isascii():
            raise ValueError("unsupported encrypted reasoning item")

    def payload(value):
        nonlocal size
        # iterencode alone may produce a whole giant escaped string. Encode
        # <=4096 characters instead (<=24578 UTF-8 JSON bytes with quotes).
        # Escaping and UTF-8 sizes are additive across scalar Unicode slices;
        # strict UTF-8 also rejects surrogate summary/model text.
        for start in range(0, len(value), 4096):
            chunk = value[start : start + 4096]
            encoded = json.dumps(chunk, ensure_ascii=False).encode("utf-8")
            size = bounded(size + len(encoded) - 2 - len(chunk))

    for item in items:
        for key in ("encrypted_content", "id", "_issuer_model"):
            payload(item[key])
        for part in item["summary"]:
            payload(part["text"])
    return size


def _reasoning_resource_preflight(message):
    """Bound only this carrier before generic public-path safety serialization."""
    items = message.get("codex_reasoning_items")
    if items in (None, "", False, [], {}):
        return  # Preserve legacy falsey markers, not serialized JSON absence.
    if type(items) is str:
        if len(items) > _MAX_CANONICAL_CODEX_JSON_BYTES:
            raise ValueError("canonical Codex JSON exceeds its admission bound")
        size = 0
        for start in range(0, len(items), 4096):
            size += len(items[start : start + 4096].encode("utf-8"))
            if size > _MAX_CANONICAL_CODEX_JSON_BYTES:
                raise ValueError("canonical Codex JSON exceeds its admission bound")
        # Only canonical active rows may decode later. Generic re-escaping of
        # this bounded text is now bounded too; do not normalize or cache it.
    else:
        _reasoning_list_bytes(items)


def _encrypted_codex_reasoning(message, policy, *, durable=False):
    """Validate an opaque replay envelope, without interpreting its ciphertext."""
    try:
        _require_reasoning_policy(policy)
    except ValueError as error:
        _tag_policy_refusal(error, "codex_reasoning_items")
        raise
    items = message["codex_reasoning_items"]
    if durable and type(items) is str:
        _reasoning_resource_preflight(message)
        items = _json(items.encode("utf-8"))
    _reasoning_list_bytes(items)
    if message["role"] != "assistant":
        raise ValueError("unsupported encrypted reasoning envelope")
    if any(item["_issuer_model"] != policy.issuer_model for item in items):
        raise _carrier_error("unsupported encrypted reasoning item", "codex_reasoning_items")
    return items


def _opaque(message, carriers):
    """Whether the rolling-only opaque admission applies: never to a Codex reasoning row."""
    return carriers and message.get("codex_reasoning_items") in _ABSENT


def _carrier_absent(value):
    """Type-aware absence: None or an EMPTY string/list/object. Unlike the legacy codec's
    ``_ABSENT`` equality, False/0/0.0 are present (and then refused as non-JSON)."""
    return value is None or (type(value) in (str, list, dict) and not value)


def _carrier_values(message):
    """``(key, value)`` of every present non-empty opaque carrier, tool-call ones included."""
    found = [
        (key, message[key])
        for key in sorted(_OPAQUE_CARRIERS & message.keys())
        if not _carrier_absent(message[key])
    ]
    calls = message.get("tool_calls")
    for call in calls if type(calls) is list else ():
        if type(call) is not dict:
            continue
        if not _carrier_absent(call.get("extra_content")):
            found.append(("extra_content", call["extra_content"]))
        function = call.get("function")
        if type(function) is dict and not _carrier_absent(function.get("thought_signature")):
            found.append(("thought_signature", function["thought_signature"]))
    return found


class _OverBound(ValueError):
    """Internal: a bounded measurement passed its limit."""


def _bounded_json_units(value, limit):
    """``ascii_json_units(value)`` without serializing past ``limit``.

    Same byte unit (compact, sorted, ASCII-escaped JSON), computed by an iterative walk
    that raises ``_OverBound`` as soon as the running total exceeds ``limit``: a string
    is only escaped once its raw length (a lower bound of its escaped length) fits, and a
    container's framing is counted before its items are queued, so cycles and huge
    fan-out stop at the bound too. Non-JSON values raise ``ValueError``.
    """
    size, stack = 0, [value]
    while stack:
        item = stack.pop()
        if type(item) is str:
            if size + len(item) + 2 > limit:
                raise _OverBound()
            size += len(json.dumps(item, ensure_ascii=True))
        elif type(item) is dict:
            size += 1 + 2 * len(item) if item else 2  # Braces, colons and commas.
            if size > limit:
                raise _OverBound()
            for key, member in item.items():
                if type(key) is not str:
                    raise ValueError("non-string JSON key")
                stack.extend((key, member))
        elif type(item) is list:
            size += 1 + len(item) if item else 2  # Brackets and commas.
            if size > limit:
                raise _OverBound()
            stack.extend(item)
        elif item is None or type(item) is bool:
            size += 4 if item is None or item is True else 5
        elif type(item) is int:
            size += len(repr(item))
        elif type(item) is float:
            if item != item or item in (float("inf"), float("-inf")):
                raise ValueError("nonfinite JSON number")
            size += len(repr(item))
        else:
            raise ValueError("non-JSON value")
        if size > limit:
            raise _OverBound()
    return size


def _admit_opaque_carriers(message):
    """Structural admission of one message's opaque carriers; returns their JSON bytes.

    Assistant rows only; a JSON string, list or object; at most
    ``_MAX_CARRIER_MESSAGE_BYTES`` for all of one message's carriers together, measured
    without serializing past the bound. Values are measured, never interpreted. Every
    refusal names the key, never the value.
    """
    size = 0
    for key, value in _carrier_values(message):
        if message.get("role") != "assistant":
            raise _carrier_error("reasoning carrier on a non-assistant message", key)
        if type(value) not in (str, list, dict):
            raise _carrier_error("non-JSON reasoning carrier", key)
        try:
            size += _bounded_json_units(value, _MAX_CARRIER_MESSAGE_BYTES - size)
        except _OverBound:
            raise _carrier_error("reasoning carrier exceeds its admission bound", key) from None
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise _carrier_error("non-JSON reasoning carrier", key) from None
    return size


def _carrier_bytes(message):
    """Admitted opaque carrier bytes of one message (0 for a Codex reasoning row)."""
    return _admit_opaque_carriers(message) if _opaque(message, True) else 0


def _require_carrier_origin(message, reference):
    """An opaque carrier may appear only where ``reference`` (canonical) has one.

    Exception: the host's must-echo pad alone. Presence only: the host projects carriers
    per route (copies, promotes, strips), so values are deliberately not compared.
    """
    if not _opaque(message, True):
        return
    found = _carrier_values(message)
    if not found or all(k == "reasoning_content" and v == _MUST_ECHO_PAD for k, v in found):
        return
    if reference is None or not _opaque(reference, True) or not _carrier_values(reference):
        raise _carrier_error("reasoning carrier without a canonical counterpart", found[0][0])


# Hermes's recalled-memory fence, byte-for-byte as agent/memory_context.py's
# build_memory_context_block writes it and turn_context.compose_user_api_content
# appends it (content + "\n\n" + block). Any other wording, an extra
# pre_llm_call block, or a nested fence keeps declining.
_MEMORY_OPEN = (
    "<memory-context>\n"
    "[System note: The following is recalled memory context, "
    "NOT new user input. Treat as authoritative reference data — "
    "this is the agent's persistent memory and should inform all responses.]\n\n"
)


_MEMORY_CLOSE = "\n</memory-context>"


_REQUEST_REASONING_TOKENS = "_replayed_reasoning_tokens"


_MEMORY_FENCE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)


def recalled_memory_suffix(content, api_content):
    """True iff ``api_content`` is the typed text plus exactly one host memory block.

    ``content`` may be the restored view's outer-stripped form of the typed text.
    """
    if type(content) is not str or type(api_content) is not str:
        return False
    if not api_content.endswith(_MEMORY_CLOSE):
        return False
    split = api_content.rfind("\n\n" + _MEMORY_OPEN)
    if split < 0:
        return False
    typed = api_content[:split]
    if typed != content and typed.strip() != content:
        return False
    body = api_content[split + 2 + len(_MEMORY_OPEN) : len(api_content) - len(_MEMORY_CLOSE)]
    return bool(body.strip()) and not _MEMORY_FENCE.search(body)


def _provider(
    message,
    *,
    wire=False,
    durable=False,
    reasoning_policy=None,
    delivery_ids=False,
    carriers=False,
    memory_context=False,
):
    """``delivery_ids`` (rolling only): accept the gateway's ``platform_message_id`` on an
    assembled user message. It is delivery bookkeeping that Hermes strips before the
    provider call; it never enters the compared payload.

    ``carriers`` (rolling only, #29): admit ``_OPAQUE_CARRIERS`` structurally on rows
    without Codex reasoning and leave them out of the compared payload.

    ``memory_context`` (rolling only, #41): a user row whose ``api_content`` is its
    ``content`` plus one host recalled-memory block is compared as that sent text
    (the assembled view already carries it as ``content``). Readable summaries keep
    using the canonical ``content`` only."""
    if type(message) is not dict:
        raise ValueError("message must be a mapping")
    opaque = _opaque(message, carriers)
    # At select_context this is assembled, not final HTTP: Hermes strips the
    # born-durable marker only in the later chat-completions transport.
    allowed = (
        _WIRE
        | {"_db_persisted", "tool_name", "effect_disposition", "codex_message_items"}
        | ({"codex_reasoning_items"} if reasoning_policy is not None else set())
        | ({"platform_message_id"} if delivery_ids else set())
        | ({_REQUEST_REASONING_TOKENS} if carriers else set())
        if wire
        else _WIRE | _METADATA | _CARRIERS | _IDENTITIES
    ) | (_OPAQUE_CARRIERS if opaque else set())
    if "_db_persisted" in message and message["_db_persisted"] is not True:
        raise ValueError("invalid persistence marker")
    if set(message) - allowed:
        carriers = sorted((set(message) - allowed) & _REASONING_CARRIERS)
        if carriers:
            raise _carrier_error("unknown message shape", carriers[0])
        raise ValueError("unknown message shape")
    if wire and _REQUEST_REASONING_TOKENS in message:
        # Hermes's request-only reasoning-size estimate for an assistant replay row;
        # assemble_api_request pops it before the provider call. Never compared.
        count = message[_REQUEST_REASONING_TOKENS]
        if message.get("role") != "assistant" or type(count) is not int or count <= 0:
            raise ValueError("invalid request reasoning estimate")
    if wire and delivery_ids and "platform_message_id" in message:
        delivery = message["platform_message_id"]
        if (
            message.get("role") != "user"
            or type(delivery) not in (str, int)
            or isinstance(delivery, bool)
        ):
            raise ValueError("invalid delivery identifier")
    role, content = message["role"], message["content"]
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise ValueError("unsupported role")
    if not isinstance(content, str) and not (
        role == "assistant" and content is None and message.get("tool_calls")
    ):
        raise ValueError("only text and assistant tool-call messages are supported")
    if opaque:
        _admit_opaque_carriers(message)
    reasoning_items = None
    if message.get("codex_reasoning_items") not in (None, "", False, [], {}):
        reasoning_items = _encrypted_codex_reasoning(message, reasoning_policy, durable=durable)
    codex_items = None
    sent = None
    for key in sorted(_CARRIERS & message.keys()):  # Deterministic: names the reported carrier.
        value = message[key]
        if reasoning_items is not None and key in {
            "codex_reasoning_items",
            "reasoning",
            "reasoning_content",
        }:
            continue  # Companions are checked independently before this projection.
        if key == "api_content" and value == content:
            continue  # An identical sidecar cannot change this supported wire mapping.
        if (
            key == "api_content"
            and memory_context
            and role == "user"
            and recalled_memory_suffix(content, value)
        ):
            sent = value
            continue
        if opaque and key in _OPAQUE_CARRIERS:
            continue  # Admitted above; opaque, so never part of the compared payload.
        if value not in (None, "", False, [], {}):
            if key == "codex_message_items":
                codex_items = _ordinary_codex_message(message, wire=wire, durable=durable)
                continue
            if key in _REASONING_CARRIERS:
                raise _carrier_error("unsupported replay carrier", key)
            raise ValueError("unsupported replay carrier")
    for key in ("_absorbed_message_uids", "absorbed_message_uids"):
        if message.get(key) not in (None, [], "[]", ""):
            raise ValueError("absorbed identity is ambiguous")
    if "active" in message and message["active"] not in (1, True):
        raise ValueError("inactive source")
    result = {
        key: value
        for key, value in message.items()
        if key in _WIRE and (key in {"role", "content"} or value is not None)
    }
    if sent is not None:
        result["content"] = sent
    if codex_items is not None:
        # A real replay payload, never a redundant text sidecar: compare every
        # validated field across active, restored and assembled views unchanged.
        result["codex_message_items"] = codex_items
    if reasoning_items is not None:
        if codex_items is None and not message.get("tool_calls"):
            raise ValueError("reasoning requires a completed answer or tool-call envelope")
        result["codex_reasoning_items"] = reasoning_items
    if "tool_calls" in result and role != "assistant":
        raise ValueError("cross-role tool calls")
    if "tool_call_id" in result and role != "tool":
        raise ValueError("cross-role tool result")
    return result


def _comparison(
    message,
    *,
    wire=False,
    durable=False,
    reasoning_policy=None,
    delivery_ids=False,
    carriers=False,
    memory_context=False,
):
    """Audited pretransport comparison only; never returned as request/source.

    _assistant_tool_call_dict adds these two Responses sidecars, then
    reasoning_params strips them before the selection hook off Codex. Only
    deterministic sidecars remain a projection, but an independently assigned
    native item ID is comparison identity and must agree across all views.
    """
    result = _provider(
        message,
        wire=wire,
        durable=durable,
        reasoning_policy=reasoning_policy,
        delivery_ids=delivery_ids,
        carriers=carriers,
        memory_context=memory_context,
    )
    opaque = _opaque(message, carriers)
    # make_tool_result_message writes name + tool_name; SessionDB restores only
    # tool_name. Reconcile this alias ONLY for tool results, never speakers.
    for key in ("tool_name", "effect_disposition"):
        value = message.get(key)
        if value is not None:
            if message["role"] != "tool" or not isinstance(value, str) or not value:
                raise ValueError("invalid tool metadata")
            result[key] = value
    if message["role"] == "tool" and "name" in message:
        name = message["name"]
        if (
            not isinstance(name, str)
            or not name
            or ("tool_name" in result and result["tool_name"] != name)
        ):
            raise ValueError("contradictory tool name")
        result["tool_name"] = name
        result.pop("name", None)
    if "tool_calls" in result:
        calls = result["tool_calls"]
        if type(calls) is not list or not calls:
            raise ValueError("tool calls must be a nonempty list")
        projected = []
        native_item_ids = set()  # Response-local; deterministic sidecars are not native identity.
        for call in calls:
            if type(call) is not dict or set(call) - {
                "id",
                "type",
                "function",
                "call_id",
                "response_item_id",
                *(("extra_content",) if opaque else ()),
            }:
                if type(call) is dict and "extra_content" in call:
                    raise _carrier_error("unsupported tool call", "extra_content")
                raise ValueError("unsupported tool call")
            native_item = False
            if {"call_id", "response_item_id"} & call.keys():
                ident = call["id"]
                if not isinstance(ident, str) or not ident or ident != ident.strip():
                    raise ValueError("invalid sidecar identity")
                if "call_id" in call and call["call_id"] != ident:
                    raise ValueError("contradictory call identity")
                # codex_responses_adapter._derive_responses_function_call_id,
                # restricted to an absent native response_item_id.
                sanitized = re.sub(r"[^A-Za-z0-9_-]", "", ident)
                if ident.startswith("fc_"):
                    expected = ident
                elif ident.startswith("call_") and len(ident) > len("call_"):
                    expected = "fc_" + ident[len("call_") :]
                elif sanitized.startswith("fc_"):
                    expected = sanitized
                elif sanitized.startswith("call_") and len(sanitized) > len("call_"):
                    expected = "fc_" + sanitized[len("call_") :]
                elif sanitized:
                    expected = "fc_" + sanitized[:48]
                else:
                    expected = "fc_" + hashlib.sha1(ident.encode("utf-8")).hexdigest()[:24]
                if "response_item_id" in call and call["response_item_id"] != expected:
                    # _assistant_tool_call_dict preserves the provider's item ID
                    # independently of call_id. _replay_tool_call_items may omit
                    # it on HTTP, but the selection hook still carries it.
                    item_id = call["response_item_id"]
                    if (
                        "call_id" not in call
                        or type(item_id) is not str
                        or re.fullmatch(r"fc_[A-Za-z0-9_-]{1,61}", item_id) is None
                    ):
                        raise ValueError("unsupported native response item")
                    if item_id in native_item_ids:
                        raise ValueError("duplicate native response item")
                    native_item_ids.add(item_id)
                    native_item = True
            call = {
                k: v
                for k, v in call.items()
                if k != "call_id"
                and (k != "response_item_id" or native_item)
                and not (opaque and k == "extra_content")
            }
            if opaque and type(call["function"]) is dict:
                # Admitted by _provider; opaque, so not part of the compared payload.
                call["function"] = {
                    k: v for k, v in call["function"].items() if k != "thought_signature"
                }
            projected.append(call)
        result["tool_calls"] = projected
    return result


def _uids(messages):
    values = [item.get("message_uid") for item in messages]
    if any(not isinstance(uid, str) or not uid for uid in values) or len(set(values)) != len(
        values
    ):
        raise ValueError("missing or duplicate logical identity")
    return values


def _leading(messages):
    split = 0
    while split < len(messages) and messages[split].get("role") in {"system", "developer"}:
        split += 1
    return split


def _identity_value(message, name):
    values = []
    for key in (name, "_" + name):
        value = message.get(key)
        if value is None or value == "":
            continue
        if name == "tool_call_uids" and isinstance(value, str):
            value = _json(value.encode("utf-8"))
        values.append(value)
    if len(values) == 2 and canonical_json(values[0]) != canonical_json(values[1]):
        raise ValueError("conflicting occurrence metadata")
    return values[0] if values else None


def _occurrences(
    messages, *, durable=False, reasoning_policies=None, carriers=False, memory_context=False
):
    _tool_sequence(messages)
    pending = {}
    seen = set()
    identities = []
    for index, message in enumerate(messages):
        calls = _comparison(
            message,
            durable=durable,
            reasoning_policy=reasoning_policies[index] if reasoning_policies else None,
            carriers=carriers,
            memory_context=memory_context,
        ).get("tool_calls")
        mapping = _identity_value(message, "tool_call_uids")
        result_uid = _identity_value(message, "tool_call_uid")
        if mapping is not None and (message["role"] != "assistant" or not calls):
            raise ValueError("cross-role occurrence mapping")
        if result_uid is not None and (
            message["role"] != "tool" or not isinstance(result_uid, str)
        ):
            raise ValueError("invalid result occurrence")
        if calls is not None:
            if type(calls) is not list or not calls:
                raise ValueError("tool calls must be a nonempty list")
            if mapping is not None and (
                type(mapping) is not dict or set(mapping) != {c["id"] for c in calls}
            ):
                raise ValueError("occurrences must match all parallel calls")
            for call in calls:
                if (
                    type(call) is not dict
                    or not {"id", "type", "function"} <= call.keys()
                    or call.keys() - {"id", "type", "function", "response_item_id"}
                    or call["type"] != "function"
                ):
                    raise ValueError("unsupported tool call")
                function = call["function"]
                if (
                    type(function) is not dict
                    or set(function) != {"name", "arguments"}
                    or not isinstance(function["name"], str)
                    or not function["name"]
                    or not isinstance(function["arguments"], str)
                ):
                    if type(function) is dict and "thought_signature" in function:
                        raise _carrier_error("unsupported function payload", "thought_signature")
                    raise ValueError("unsupported function payload")
                uid = mapping[call["id"]] if mapping is not None else None
                if uid is not None:
                    if not isinstance(uid, str) or not uid or uid in seen:
                        raise ValueError("occurrence identities must be unique")
                    seen.add(uid)
                pending[call["id"]] = uid
        if message["role"] == "tool":
            expected = pending.pop(message["tool_call_id"])
            if result_uid != expected:
                raise ValueError("tool result occurrence mismatch")
        identities.append((mapping, result_uid))
    return identities


def _reasoning_permissions(history, active, request, count, policy, *, retirement_policy=None):
    """Bind companions; aged permission is a separate explicit representation policy.

    The default remains retained-only, including for the existing prepared engines.
    """
    if retirement_policy not in (None, "aged-codex-retirement-v1"):
        raise ValueError("unsupported retirement policy")
    permissions = [None] * len(history)
    if policy is None:
        return permissions  # Preserve the legacy reasoning-free comparison path.
    _require_reasoning_policy(policy)
    views = (active, history) if request is None else (active, history, request)
    for index in range(max(map(len, views))):
        rows = [rows[index] for rows in views if index < len(rows)]
        if not any(m.get("codex_reasoning_items") not in (None, "", False, [], {}) for m in rows):
            continue
        lower = 0 if retirement_policy == "aged-codex-retirement-v1" else count
        if not lower <= index < min(map(len, views)):
            raise ValueError("encrypted reasoning requires a retained active source")
        source, restored = active[index], history[index]
        assembled = None if request is None else request[index]
        for key in ("reasoning", "reasoning_content"):
            value = source[key]  # Both canonical SQL columns are required.
            if value is not None and type(value) is not str:
                raise ValueError("invalid canonical reasoning companion")
            if type(value) is str:
                value.encode("utf-8")
            if assembled is not None and key in assembled:
                raise _carrier_error("readable reasoning in assembled request", key)
            if value is None:
                if restored.get(key) is not None:
                    raise ValueError("NULL reasoning companion mismatch")
            elif key not in restored:
                if not (
                    key == "reasoning" and value == "" and restored.get("_db_persisted") is True
                ):
                    raise ValueError("missing reasoning companion")
            elif type(restored[key]) is not str or restored[key] != value:
                raise ValueError("reasoning companion mismatch")
        permissions[index] = policy
    return permissions


def _comparison_payloads(
    messages,
    policies,
    *,
    wire=False,
    durable=False,
    delivery_ids=False,
    carriers=False,
    memory_context=False,
):
    """Use a separate request-local reasoning ID set for each compared view."""
    payloads = []
    reasoning_ids = set()
    for index, message in enumerate(messages):
        payload = _comparison(
            message,
            wire=wire,
            durable=durable,
            reasoning_policy=policies[index],
            delivery_ids=delivery_ids,
            carriers=carriers,
            memory_context=memory_context,
        )
        items = payload.get("codex_reasoning_items")
        if items:
            for item in items:
                ident = item["id"]
                if ident in reasoning_ids:
                    raise ValueError("duplicate retained reasoning identity")
                reasoning_ids.add(ident)
        payloads.append(payload)
    return payloads


def _exact_payloads(
    history,
    active,
    policies,
    *,
    assembled=None,
    delivery_ids=False,
    carriers=False,
    memory_context=False,
):
    """Shared audited canonical/restored comparison; assembled is real or absent.

    Callers establish UID coverage separately. A callback has no assembled view;
    omitting it proves only the actual canonical/restored correspondence.
    ``carriers`` (rolling only, #29): opaque carriers are excluded from the equality;
    instead every restored/assembled carrier needs a canonical one at its position
    (or is the must-echo pad), and the assembled request's carriers share one bound.
    """
    raw_payload = _comparison_payloads(
        history, policies, carriers=carriers, memory_context=memory_context
    )
    if any(m["role"] not in {"user", "assistant", "tool"} for m in raw_payload):
        raise ValueError("host role inside history")
    active_payload = _comparison_payloads(
        active, policies, durable=True, carriers=carriers, memory_context=memory_context
    )
    for index, replay in enumerate(raw_payload):
        items = replay.get("codex_message_items")
        if items:
            text = items[0]["content"][0]["text"]
            # Only restored outer content may strip, never the canonical replay.
            if index >= len(active_payload) or active_payload[index]["content"] != text:
                raise ValueError("Codex replay requires exact active source")
    for source, replay, message in zip(active_payload, raw_payload, history):
        if canonical_json(source) == canonical_json(replay):
            continue
        if (
            message.get("_db_persisted") is True
            and source["role"] in {"user", "assistant"}
            and isinstance(source["content"], str)
            and canonical_json({**source, "content": source["content"].strip()})
            == canonical_json(replay)
        ):
            continue
        raise ValueError("active/conversation payload mismatch")
    if assembled is not None and canonical_json(
        _comparison_payloads(
            assembled, policies, wire=True, delivery_ids=delivery_ids, carriers=carriers
        )
    ) != canonical_json(raw_payload):
        raise ValueError("request must be an exact one-to-one mapping")
    if carriers:
        # Past this point every view is aligned one-to-one with history.
        for index, message in enumerate(history[: len(active)]):
            _require_carrier_origin(message, active[index])
        total = 0
        for index, message in enumerate(assembled or ()):
            # In-flight rows of the current turn are not durable yet: the host's own
            # conversation message is then the origin of what it assembled.
            _require_carrier_origin(
                message, active[index] if index < len(active) else history[index]
            )
            total += _carrier_bytes(message)
            if total > _MAX_CARRIER_REQUEST_BYTES:
                raise _carrier_error(
                    "reasoning carriers exceed the request admission bound",
                    _carrier_values(message)[0][0],
                )
    return raw_payload


def _exact_occurrences(history, active, policies, *, carriers=False, memory_context=False):
    occurrences = _occurrences(
        history, reasoning_policies=policies, carriers=carriers, memory_context=memory_context
    )
    if canonical_json(
        _occurrences(
            active,
            durable=True,
            reasoning_policies=policies,
            carriers=carriers,
            memory_context=memory_context,
        )
    ) != canonical_json(occurrences[: len(active)]):
        raise ValueError("active/conversation occurrence mismatch")
    return occurrences


def _admit_messages(*groups):
    for messages in groups:
        if type(messages) is not list or not messages or any(type(m) is not dict for m in messages):
            raise ValueError("nonempty message lists required")
        for message in messages:
            _require_supported_data_use(message)
            _reasoning_resource_preflight(message)
            if _opaque(message, True):  # Name the carrier before generic serialization.
                _admit_opaque_carriers(message)
        _safe_json(messages)


def incoming_anchor(history, incoming):
    """UID plus exact validated payload of the latest USER, not last message.

    This is an immutable pin key, not a mutable pin/adoption implementation.
    The caller first validates the whole history's correspondence.
    """
    _admit_messages(history, [incoming])
    uids = _uids(history)
    starts = [i for i, row in enumerate(history) if row["role"] == "user"]
    if not starts or incoming.get("role") != "user":
        raise ValueError("incoming user anchor required")
    current = starts[-1]
    payload = canonical_json(_comparison(history[current], memory_context=True))
    if (
        incoming.get("message_uid") != uids[current]
        or canonical_json(_comparison(incoming, memory_context=True)) != payload
    ):
        raise ValueError("incoming must anchor the latest user by UID and payload")
    return uids[current], sha256(payload)
