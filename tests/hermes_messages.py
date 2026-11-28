"""Synthetic Hermes SessionDB rows, restored histories and assembled requests.

``exchange(tag)`` is one user turn with a tool call and Codex reasoning carriers;
``restored`` is what Hermes rebuilds from those rows, ``assembled`` the request it
sends (host prefix of two messages). Synthetic data only.
"""

import copy
import json


def exchange(tag, *, complete=True):
    def row(role, content, suffix, **extra):
        return {
            "session_id": "s",
            "message_uid": tag + suffix,
            "role": role,
            "content": content,
            "tool_calls": None,
            "tool_call_id": None,
            "tool_name": None,
            "effect_disposition": None,
            "reasoning": None,
            "reasoning_content": None,
            "reasoning_details": None,
            "codex_reasoning_items": None,
            "codex_message_items": None,
            "tool_call_uids": None,
            "tool_call_uid": None,
            "absorbed_message_uids": "[]",
            "display_metadata": None,
            "active": 1,
            **extra,
        }

    def reasoning(suffix, count):
        return json.dumps(
            [
                {
                    "type": "reasoning",
                    "id": "rs_" + tag + suffix + str(i),
                    "encrypted_content": "OPAQUE_" + tag + suffix + str(i),
                    "summary": []
                    if i % 2 == 0
                    else [{"type": "summary_text", "text": "companion"}],
                    "_issuer_kind": "codex_backend",
                    "_issuer_model": "gpt-test",
                }
                for i in range(count)
            ]
        )

    result = [
        row("user", " question " + tag + " ", "u"),
        row(
            "assistant",
            None,
            "o",
            tool_calls=[
                {
                    "id": "call_" + tag,
                    "call_id": "call_" + tag,
                    "response_item_id": "fc_native_" + tag,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"exact"}'},
                }
            ],
            tool_call_uids=json.dumps({"call_" + tag: "occ_" + tag}),
            codex_reasoning_items=reasoning("owner", 1),
            reasoning="",
        ),
        row(
            "tool",
            "result " + tag,
            "t",
            tool_call_id="call_" + tag,
            tool_name="lookup",
            effect_disposition="none",
            tool_call_uid="occ_" + tag,
        ),
    ]
    if complete:
        text = " answer " + tag + " "
        result.append(
            row(
                "assistant",
                text,
                "a",
                reasoning="readable companion",
                reasoning_content="",
                codex_reasoning_items=reasoning("answer", 4),
                codex_message_items=json.dumps(
                    [
                        {
                            "type": "message",
                            "id": "msg_" + tag,
                            "role": "assistant",
                            "status": "completed",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    ]
                ),
            )
        )
    return result


def restored(source):
    result = []
    for row in source:
        item = {k: copy.deepcopy(row[k]) for k in ("role", "content", "message_uid")}
        item["_db_persisted"] = True
        if item["role"] in {"user", "assistant"} and isinstance(item["content"], str):
            item["content"] = item["content"].strip()
        for key in ("codex_reasoning_items", "codex_message_items", "tool_call_uids"):
            if row[key] is not None:
                item[key] = json.loads(row[key])
        for key in (
            "tool_calls",
            "tool_call_id",
            "tool_call_uid",
            "tool_name",
            "effect_disposition",
            "reasoning",
            "reasoning_content",
        ):
            if row[key] is not None and not (key == "reasoning" and row[key] == ""):
                item[key] = copy.deepcopy(row[key])
        result.append(item)
    return result


def assembled(history):
    return [
        {"role": "system", "content": "host"},
        {"role": "developer", "content": "host rules"},
    ] + [
        {
            k: copy.deepcopy(v)
            for k, v in row.items()
            if k
            in {
                "role",
                "content",
                "tool_calls",
                "tool_call_id",
                "tool_name",
                "effect_disposition",
                "codex_reasoning_items",
                "codex_message_items",
                "_db_persisted",
            }
        }
        for row in history
    ]
