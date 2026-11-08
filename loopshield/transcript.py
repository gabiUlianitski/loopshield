"""Convert chat transcripts from any framework into :class:`AgentStep` trajectories."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from loopshield.mitigator import estimate_tokens, generate_nudge
from loopshield.models import AgentStep, LoopVerdict

REPLY_TOOL = "reply"

_ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "chat": "assistant",
    "model": "assistant",
}


def content_text(content: Any) -> str:
    """Read plain text or a list of content parts as one string."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            content_text(part.get("text", part.get("content", "")))
            if isinstance(part, Mapping)
            else content_text(part)
            for part in content
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def argument_dict(arguments: Any) -> dict[str, Any]:
    """Return tool arguments as a dict, parsing a JSON string when needed."""

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return {"raw": arguments}
        arguments = parsed
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    return {"value": arguments}


def describe_call(name: str, arguments: Mapping[str, Any]) -> str:
    """Describe a tool call as text for steps that carry no written thought."""

    return f"{name} {json.dumps(dict(arguments), sort_keys=True, default=str)}"


def coerce_messages(raw: Iterable[Any]) -> list[dict[str, Any]]:
    """Return OpenAI-style message dicts for dicts, role tuples, or message objects.

    LangChain-style objects (``type``, ``content``, ``tool_calls``) are read by
    attribute, so no framework import is needed.
    """

    return [coerce_message(item) for item in raw]


def coerce_message(item: Any) -> dict[str, Any]:
    """Return one message as an OpenAI-style dict. The input is not modified."""

    if isinstance(item, Mapping):
        data = dict(item)
        data["role"] = _role(data.get("role") or data.get("type"))
        _carry_marker(data, item.get("additional_kwargs"))
        return data
    if isinstance(item, str):
        return {"role": "user", "content": item}
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return {"role": _role(item[0]), "content": item[1]}

    data = {
        "role": _role(getattr(item, "role", None) or getattr(item, "type", None)),
        "content": getattr(item, "content", ""),
    }
    for key in ("name", "tool_call_id", "status"):
        value = getattr(item, key, None)
        if isinstance(value, str):
            data[key] = value
    calls = getattr(item, "tool_calls", None)
    if isinstance(calls, list) and calls:
        data["tool_calls"] = [_openai_call(call) for call in calls]
    _carry_marker(data, getattr(item, "additional_kwargs", None))
    return data


def messages_to_steps(messages: Sequence[Mapping[str, Any]]) -> list[AgentStep]:
    """Turn assistant turns into steps: one per tool call, or one text reply.

    A text-only reply is a ``reply`` step whose arguments carry its position,
    so replies never form a tool cycle on their own.
    """

    steps: list[AgentStep] = []
    for index, message in enumerate(messages):
        if str(message.get("role", "")).casefold() != "assistant":
            continue
        thought = content_text(message.get("content")).strip()
        calls = tool_calls(message)
        tokens_used = estimate_tokens([message])
        if calls:
            for call in calls:
                name, arguments = parse_call(call)
                steps.append(
                    AgentStep(
                        thought=thought or describe_call(name, arguments),
                        tool_name=name,
                        tool_args=arguments,
                        tokens_used=tokens_used,
                    )
                )
        elif thought:
            steps.append(
                AgentStep(
                    thought=thought,
                    tool_name=REPLY_TOOL,
                    tool_args={"index": index},
                    tokens_used=tokens_used,
                )
            )
    return steps


def tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return OpenAI ``tool_calls`` and Anthropic ``tool_use`` blocks of a message."""

    calls: list[Mapping[str, Any]] = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        calls.extend(call for call in raw_calls if isinstance(call, Mapping))
    content = message.get("content")
    if isinstance(content, list):
        calls.extend(
            part for part in content if isinstance(part, Mapping) and part.get("type") == "tool_use"
        )
    return calls


def parse_call(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the tool name and argument dict of one call in any supported shape."""

    function = call.get("function")
    payload = function if isinstance(function, Mapping) else call
    name = str(payload.get("name") or "").strip() or "tool"
    arguments = payload.get("arguments", payload.get("input", payload.get("args", {})))
    return name, argument_dict(arguments)


def nudge_for(verdict: LoopVerdict, steps: Sequence[AgentStep]) -> str:
    """Return the intervention text for a flagged trajectory."""

    tool = steps[-1].tool_name if steps else "tool"
    if tool != REPLY_TOOL:
        return str(generate_nudge(verdict, tool)["content"])
    if verdict.intervention_prompt:
        return str(verdict.intervention_prompt)
    return "LoopShield detected a loop. Choose a different strategy immediately."


def is_intervention(message: Mapping[str, Any]) -> bool:
    """Return True for a message that already carries a LoopShield nudge."""

    if message.get("loopshield_intervention") is True:
        return True
    text = content_text(message.get("content"))
    return text.startswith("LoopShield detected") or (
        "is temporarily locked" in text and "different strategy" in text
    )


def _role(value: Any) -> str:
    role = str(value or "").strip().casefold()
    return _ROLE_ALIASES.get(role, role)


def _carry_marker(data: dict[str, Any], extras: Any) -> None:
    if isinstance(extras, Mapping) and extras.get("loopshield") is True:
        data["loopshield_intervention"] = True


def _openai_call(call: Any) -> Any:
    if not isinstance(call, Mapping) or isinstance(call.get("function"), Mapping):
        return call
    arguments = call.get("args", call.get("arguments", call.get("input", {})))
    return {
        "id": call.get("id"),
        "type": "function",
        "function": {
            "name": call.get("name"),
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, default=str),
        },
    }
