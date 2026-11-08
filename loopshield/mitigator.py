"""Self-healing nudges and context pruning for looping agents."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from loopshield.graph_analyzer import tool_signature
from loopshield.models import AgentStep, LoopVerdict

_LOOP_LABELS: dict[str, str] = {
    "graph_cycle": "repeated tool cycle",
    "semantic": "repeated line of thought",
    "both": "repeated tool cycle and a repeated line of thought",
}

_ERROR_TEXT = re.compile(r"\b(error|exception|traceback|failed|failure|timeout|timed out)\b", re.I)
_TURN_STARTS = {"user", "assistant"}
_TOOL_ROLES = {"tool", "function", "tool_result"}


def generate_nudge(verdict: LoopVerdict, blocked_tool: str) -> dict[str, str]:
    """Return a system message that locks ``blocked_tool`` and demands a new strategy."""

    tool = blocked_tool.strip()
    if not tool:
        raise ValueError("blocked_tool must name the locked tool")
    label = _LOOP_LABELS.get(verdict.loop_type or "", "detected loop")
    content = (
        f"The tool `{tool}` is temporarily locked due to a {label}. "
        f"You must not call `{tool}` again. "
        "Choose a different strategy immediately, using only the evidence you already have, "
        "or give the final answer."
    )
    return {"role": "system", "content": content}


def prune_context(messages: list[dict[str, Any]], max_tokens_estimate: int) -> list[dict[str, Any]]:
    """Drop repeated tool errors from the middle of a chat transcript.

    The original leading system prompt and the latest two turns are kept.
    When :func:`estimate_tokens` is above ``max_tokens_estimate``, repetitive
    tool failures between those anchors collapse into one system summary.
    OpenAI ``tool_calls`` and Anthropic ``tool_use`` blocks that produced
    those failures are removed with them, so a call is never left without its
    result. The input list is not mutated.
    """

    if max_tokens_estimate < 0:
        raise ValueError("max_tokens_estimate must be zero or greater")
    if any(not isinstance(message, dict) for message in messages):
        raise TypeError("messages must be a list of dictionaries")

    copied = [dict(message) for message in messages]
    if estimate_tokens(copied) <= max_tokens_estimate:
        return copied

    system, body = _split_system_prompt(copied)
    turns = _split_turns(body)
    if len(turns) <= 2:
        return system + _flatten(turns)

    middle_turns, tail_turns = turns[:-2], turns[-2:]
    middle = _flatten(middle_turns)
    tail = _flatten(tail_turns)
    collapsed_middle, summary = _collapse_repetitive_tool_errors(middle, _collected_result_ids(tail))
    collapsed_messages = system + collapsed_middle + tail
    collapsed = _legalize_tool_exchanges(
        collapsed_messages if summary is None else _place_summary(collapsed_messages, summary)
    )
    summarized = _legalize_tool_exchanges(system + [_summarize_middle(middle)] + tail)
    anchors = _legalize_tool_exchanges(system + tail)
    for candidate in (collapsed, summarized, anchors):
        if estimate_tokens(candidate) <= max_tokens_estimate:
            return candidate
    return anchors


def estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimate the token cost of OpenAI-style messages.

    An explicit ``token_count``, ``tokens``, or ``usage.total_tokens`` value
    is used when a message carries one. Otherwise each message is estimated
    from its character count at about four characters per token.
    """

    return sum(_estimate_message(message) for message in messages)


class Mitigator:
    """Author loop nudges and keep agent transcripts from growing without bound."""

    def generate_nudge(self, verdict: LoopVerdict, blocked_tool: str) -> dict[str, str]:
        """Return the system message from :func:`generate_nudge`."""

        return generate_nudge(verdict, blocked_tool)

    def prune_context(
        self,
        messages: list[dict[str, Any]],
        max_tokens_estimate: int,
    ) -> list[dict[str, Any]]:
        """Return the pruned transcript from :func:`prune_context`."""

        return prune_context(messages, max_tokens_estimate)

    def prune(self, steps: Sequence[AgentStep]) -> list[AgentStep]:
        """Return the trajectory with earlier exact tool repeats removed.

        A repeat is the same tool name and normalized arguments. The original
        sequence is not mutated.
        """

        last_index: dict[str, int] = {}
        for index, step in enumerate(steps):
            last_index[tool_signature(step.tool_name, step.tool_args)] = index
        return [
            step
            for index, step in enumerate(steps)
            if last_index[tool_signature(step.tool_name, step.tool_args)] == index
        ]


def _estimate_message(message: Mapping[str, Any]) -> int:
    explicit = _explicit_token_count(message)
    if explicit is not None:
        return explicit
    return _characters_to_tokens(_message_characters(message))


def _explicit_token_count(message: Mapping[str, Any]) -> int | None:
    for key in ("token_count", "tokens"):
        value = message.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    usage = message.get("usage")
    if isinstance(usage, Mapping):
        total = usage.get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
    return None


def _characters_to_tokens(characters: int) -> int:
    if characters <= 0:
        return 0
    return max(1, (characters + 3) // 4)


def _message_characters(message: Mapping[str, Any]) -> int:
    chunks = [_role(message), _content_text(message.get("content"))]
    for key in ("name", "tool_name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str):
            chunks.append(value)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        chunks.extend(_tool_call_text(call) for call in calls)
    return sum(len(chunk) for chunk in chunks)


def _content_text(content: Any) -> str:
    """Read plain text or OpenAI content parts."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_text(part) if not isinstance(part, Mapping) else _part_text(part) for part in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        return _part_text(content)
    return str(content)


def _part_text(part: Mapping[str, Any]) -> str:
    for key in ("text", "content", "refusal"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _tool_call_text(call: Any) -> str:
    if not isinstance(call, Mapping):
        return str(call)
    function = call.get("function")
    payload = function if isinstance(function, Mapping) else call
    name = payload.get("name") if isinstance(payload, Mapping) else ""
    arguments = payload.get("arguments") if isinstance(payload, Mapping) else ""
    return f"{name if isinstance(name, str) else ''}{arguments if isinstance(arguments, str) else ''}"


def _collapse_repetitive_tool_errors(
    messages: Sequence[dict[str, Any]],
    preserved_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    """Replace repeated tool failures with a summary slot and strip their calls.

    The returned summary is placed by the caller across the whole transcript,
    so it can move past a result that survived in a later turn. ``preserved_ids``
    are those surviving results: their assistant calls stay.
    """

    preserved = preserved_ids or set()
    names = _tool_call_names(messages)
    failures = _error_results(messages, names)
    counts: dict[str, int] = {}
    for _index, _block, _call_id, tool, _reason in failures:
        counts[tool] = counts.get(tool, 0) + 1
    repetitive = {tool for tool, count in counts.items() if count >= 2}
    if not repetitive:
        return [dict(message) for message in messages], None

    dropped = [failure for failure in failures if failure[3] in repetitive]
    dropped_whole = {failure[0] for failure in dropped if failure[1] is None}
    dropped_blocks: dict[int, set[int]] = {}
    samples: dict[str, list[str]] = {}
    for index, block_index, _call_id, tool, reason in dropped:
        if block_index is not None:
            dropped_blocks.setdefault(index, set()).add(block_index)
        bucket = samples.setdefault(tool, [])
        if reason and reason not in bucket:
            bucket.append(reason)
    still_answered = _remaining_result_ids(messages, dropped_whole, dropped_blocks) | preserved
    strip_ids = {
        call_id
        for _index, _block, call_id, _tool, _reason in dropped
        if call_id and call_id not in still_answered
    }

    slot = {"role": "system", "content": "", "_loopshield_summary_slot": True}
    rebuilt: list[dict[str, Any]] = []
    placed = False
    for index, message in enumerate(messages):
        if index in dropped_whole:
            if not placed:
                rebuilt.append(slot)
                placed = True
            continue
        updated = dict(message)
        blocks = dropped_blocks.get(index)
        if blocks and isinstance(message.get("content"), list):
            if not placed:
                rebuilt.append(slot)
                placed = True
            updated["content"] = [
                part for block_index, part in enumerate(message["content"]) if block_index not in blocks
            ]
            if _role(updated) == "user" and _message_is_blank(updated):
                continue
        rebuilt.append(updated)
    if not placed:
        rebuilt.append(slot)
    return _strip_call_ids_backward(rebuilt, strip_ids), _repetitive_summary(counts, samples)


def _repetitive_summary(counts: Mapping[str, int], samples: Mapping[str, list[str]]) -> dict[str, str]:
    parts: list[str] = []
    total = 0
    for tool, reasons in samples.items():
        total += counts[tool]
        shown = "; ".join(reasons[:2])
        label = f"{tool} x{counts[tool]}"
        parts.append(f"{label} ({shown})" if shown else label)
    listed = "; ".join(parts)
    if len(listed) > 240:
        listed = listed[:237] + "..."
    noun = "message" if total == 1 else "messages"
    return {
        "role": "system",
        "content": (
            f"Compressed {total} repetitive tool-error {noun}: {listed}. "
            "Do not repeat those failed calls."
        ),
    }


def _split_system_prompt(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = 0
    while index < len(messages) and _role(messages[index]) == "system":
        index += 1
    return messages[:index], messages[index:]


def _split_turns(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if _role(message) in _TURN_STARTS and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def _flatten(turns: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [message for turn in turns for message in turn]


def _summarize_middle(messages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    snippets: list[str] = []
    for message in messages:
        role = _role(message) or "message"
        text = " ".join(_content_text(message.get("content")).split())
        if len(text) > 80:
            text = text[:77] + "..."
        snippets.append(f"{role}: {text}" if text else role)
    body = " | ".join(snippets)
    if len(body) > 300:
        body = body[:297] + "..."
    return {"role": "system", "content": f"Compressed {len(messages)} earlier messages: {body}"}


def _is_tool_error(message: Mapping[str, Any]) -> bool:
    if _role(message) not in _TOOL_ROLES:
        return False
    if message.get("is_error") is True:
        return True
    if str(message.get("status", "")).casefold() in {"error", "failed", "failure"}:
        return True
    text = _content_text(message.get("content"))
    return _ERROR_TEXT.search(text) is not None


def _tool_call_names(messages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map a tool call id to its name, for OpenAI calls and Anthropic tool_use blocks."""

    names: dict[str, str] = {}
    for message in messages:
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                _remember_call_name(names, call)
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "tool_use":
                    _remember_call_name(names, part)
    return names


def _remember_call_name(names: dict[str, str], call: Any) -> None:
    if not isinstance(call, Mapping):
        return
    function = call.get("function")
    payload = function if isinstance(function, Mapping) else call
    call_id = call.get("id")
    name = payload.get("name") if isinstance(payload, Mapping) else None
    if isinstance(call_id, str) and call_id and isinstance(name, str) and name.strip():
        names[call_id] = name.strip()


def _error_results(
    messages: Sequence[Mapping[str, Any]],
    names: Mapping[str, str],
) -> list[tuple[int, int | None, str | None, str, str]]:
    """Return tool failures as ``(message, block or None, call id, tool, reason)``.

    A whole OpenAI ``tool`` message has no block index. An Anthropic
    ``tool_result`` block does, so a user message that also carries text can
    lose the failed result and keep the text.
    """

    found: list[tuple[int, int | None, str | None, str, str]] = []
    for index, message in enumerate(messages):
        if _role(message) in _TOOL_ROLES:
            if _is_tool_error(message):
                found.append(
                    (
                        index,
                        None,
                        _text_id(message.get("tool_call_id")) or _text_id(message.get("tool_use_id")),
                        _message_tool_name(message, names),
                        _short_reason(message),
                    )
                )
            continue
        content = message.get("content")
        if _role(message) != "user" or not isinstance(content, list):
            continue
        for block_index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "tool_result" or not _part_is_error(part):
                continue
            call_id = _text_id(part.get("tool_use_id")) or _text_id(part.get("tool_call_id"))
            explicit = part.get("name")
            tool = explicit.strip() if isinstance(explicit, str) and explicit.strip() else names.get(call_id or "", "tool")
            found.append((index, block_index, call_id, tool, _clip_text(_content_text(part.get("content")))))
    return found


def _part_is_error(part: Mapping[str, Any]) -> bool:
    if part.get("is_error") is True:
        return True
    if str(part.get("status", "")).casefold() in {"error", "failed", "failure"}:
        return True
    return _ERROR_TEXT.search(_content_text(part.get("content"))) is not None


def _text_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _remaining_result_ids(
    messages: Sequence[Mapping[str, Any]],
    dropped_whole: set[int],
    dropped_blocks: Mapping[int, set[int]],
) -> set[str]:
    found: set[str] = set()
    for index, message in enumerate(messages):
        if index in dropped_whole:
            continue
        blocks = dropped_blocks.get(index)
        if not blocks or not isinstance(message.get("content"), list):
            found.update(_result_ids(message))
            continue
        trimmed = dict(message)
        trimmed["content"] = [
            part for block_index, part in enumerate(message["content"]) if block_index not in blocks
        ]
        found.update(_result_ids(trimmed))
    return found


def _collected_result_ids(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    found: set[str] = set()
    for message in messages:
        found.update(_result_ids(message))
    return found


def _result_ids(message: Mapping[str, Any]) -> list[str]:
    """Return ids of OpenAI tool results and Anthropic tool_result blocks."""

    found: list[str] = []
    if _role(message) in _TOOL_ROLES:
        call_id = _text_id(message.get("tool_call_id")) or _text_id(message.get("tool_use_id"))
        if call_id:
            found.append(call_id)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "tool_result":
                continue
            call_id = _text_id(part.get("tool_use_id")) or _text_id(part.get("tool_call_id"))
            if call_id:
                found.append(call_id)
    return found


def _call_ids(message: Mapping[str, Any]) -> list[str]:
    """Return ids of OpenAI tool_calls and Anthropic tool_use blocks."""

    if _role(message) != "assistant":
        return []
    found: list[str] = []
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, Mapping):
                call_id = _text_id(call.get("id"))
                if call_id:
                    found.append(call_id)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "tool_use":
                call_id = _text_id(part.get("id"))
                if call_id:
                    found.append(call_id)
    return found


def _strip_call_ids_backward(messages: Sequence[dict[str, Any]], strip_ids: set[str]) -> list[dict[str, Any]]:
    """Drop ``strip_ids`` from assistant calls, walking from the latest message backward."""

    if not strip_ids:
        return [dict(message) for message in messages]
    kept: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("_loopshield_summary_slot") is True:
            kept.append(message)
            continue
        if _role(message) == "assistant":
            message = _without_call_ids(message, strip_ids)
            if _assistant_is_empty(message):
                continue
        kept.append(dict(message))
    kept.reverse()
    return kept


def _without_call_ids(message: Mapping[str, Any], strip_ids: set[str]) -> dict[str, Any]:
    updated = dict(message)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        kept_calls = [
            dict(call)
            for call in calls
            if not (isinstance(call, Mapping) and _text_id(call.get("id")) in strip_ids)
        ]
        if kept_calls:
            updated["tool_calls"] = kept_calls
        else:
            updated.pop("tool_calls", None)
    content = message.get("content")
    if isinstance(content, list):
        updated["content"] = [
            part
            for part in content
            if not (
                isinstance(part, Mapping)
                and part.get("type") == "tool_use"
                and _text_id(part.get("id")) in strip_ids
            )
        ]
    return updated


def _assistant_is_empty(message: Mapping[str, Any]) -> bool:
    if _role(message) != "assistant" or _call_ids(message):
        return False
    return _message_is_blank(message)


def _message_is_blank(message: Mapping[str, Any]) -> bool:
    if _call_ids(message) or _result_ids(message):
        return False
    return not _content_text(message.get("content")).strip()


def _place_summary(messages: Sequence[dict[str, Any]], summary: dict[str, str]) -> list[dict[str, Any]]:
    """Insert the summary where it does not split a tool call from its results."""

    slot = next(
        (index for index, message in enumerate(messages) if message.get("_loopshield_summary_slot") is True),
        len(messages),
    )
    body = [dict(message) for message in messages if message.get("_loopshield_summary_slot") is not True]
    preferred = sum(1 for message in messages[:slot] if message.get("_loopshield_summary_slot") is not True)
    while preferred < len(body) and _insertion_splits_exchange(body, preferred):
        preferred += 1
    return [*body[:preferred], summary, *body[preferred:]]


def _insertion_splits_exchange(messages: Sequence[Mapping[str, Any]], index: int) -> bool:
    pending: list[str] = []
    for message in messages[:index]:
        if _role(message) == "assistant":
            pending.extend(_call_ids(message))
        for result_id in _result_ids(message):
            if result_id in pending:
                pending.remove(result_id)
    if not pending:
        return False
    later: list[str] = []
    for message in messages[index:]:
        later.extend(_result_ids(message))
    return any(call_id in later for call_id in pending)


def _legalize_tool_exchanges(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop calls with no result and id-bearing results whose call is gone."""

    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in messages:
        if _role(message) == "assistant":
            call_ids.update(_call_ids(message))
        result_ids.update(_result_ids(message))
    stripped = _strip_call_ids_backward(messages, call_ids - result_ids)
    return _drop_result_ids(stripped, result_ids - call_ids)


def _drop_result_ids(messages: Sequence[dict[str, Any]], drop_ids: set[str]) -> list[dict[str, Any]]:
    if not drop_ids:
        return [dict(message) for message in messages]
    kept: list[dict[str, Any]] = []
    for message in messages:
        updated = dict(message)
        if _role(updated) in _TOOL_ROLES:
            call_id = _text_id(updated.get("tool_call_id")) or _text_id(updated.get("tool_use_id"))
            if call_id in drop_ids:
                continue
        content = updated.get("content")
        if isinstance(content, list):
            updated["content"] = [
                part
                for part in content
                if not (
                    isinstance(part, Mapping)
                    and part.get("type") == "tool_result"
                    and (
                        _text_id(part.get("tool_use_id")) in drop_ids
                        or _text_id(part.get("tool_call_id")) in drop_ids
                    )
                )
            ]
        if _role(updated) in {"user", "assistant"} and _message_is_blank(updated):
            continue
        kept.append(updated)
    return kept


def _message_tool_name(message: Mapping[str, Any], names: Mapping[str, str]) -> str:
    for key in ("name", "tool_name", "tool"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    call_id = message.get("tool_call_id")
    if isinstance(call_id, str) and call_id in names:
        return names[call_id]
    return "tool"


def _short_reason(message: Mapping[str, Any]) -> str:
    return _clip_text(_content_text(message.get("content")))


def _clip_text(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > 60:
        return collapsed[:57] + "..."
    return collapsed


def _role(message: Mapping[str, Any]) -> str:
    return str(message.get("role", "")).casefold()
