"""Native LangChain, CrewAI, and LangGraph integrations for LoopShield.

Every framework import is optional. Message objects are read by attribute, so
the integrations run, and can be tested, against any class with the same shape.
Use them through :class:`~loopshield.shield.LoopShield`::

    shield = LoopShield()
    agent = create_agent(model, tools, middleware=[shield.langchain_middleware()])

    @shield.crewai_agent()
    class GuardedAgent(Agent): ...

    graph.add_node("agent", shield.langgraph_node(agent_node))

None of the hooks records history on the shield: each call judges the
trajectory it was given, so one shield can serve many concurrent runs.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from loopshield.mitigator import estimate_tokens, prune_context
from loopshield.models import AgentStep
from loopshield.transcript import (
    argument_dict,
    coerce_messages,
    content_text,
    describe_call,
    is_intervention,
    messages_to_steps,
    nudge_for,
    tool_calls,
)

if TYPE_CHECKING:
    from loopshield.shield import LoopShield


def _optional(module: str, attribute: str) -> Any | None:
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError):
        return None


AgentMiddleware: Any | None = _optional("langchain.agents.middleware", "AgentMiddleware")
wrap_model_call: Any | None = _optional("langchain.agents.middleware", "wrap_model_call")
wrap_tool_call: Any | None = _optional("langchain.agents.middleware", "wrap_tool_call")
_AIMessage: Any | None = _optional("langchain_core.messages", "AIMessage")
_SystemMessage: Any | None = _optional("langchain_core.messages", "SystemMessage")
_ToolMessage: Any | None = _optional("langchain_core.messages", "ToolMessage")
_LangGraphCommand: Any | None = _optional("langgraph.types", "Command")

LANGCHAIN_AVAILABLE = AgentMiddleware is not None
LANGGRAPH_AVAILABLE = _LangGraphCommand is not None

_INDEX_KEY = "_loopshield_index"
_MARKER = {"loopshield": True}


class IntegrationConfig(BaseModel):
    """Options shared by the framework integrations.

    ``max_tokens`` is the budget above which ``prune_context`` runs. It falls
    back to the shield's ``max_tokens_estimate``. ``on_loop`` picks what the
    LangChain model hook does for a flagged trajectory: ``short_circuit``
    answers with the nudge and never calls the model, ``inject`` appends the
    nudge as a system message and lets the model run. ``retry_node`` is the
    LangGraph node to re-enter after a nudge is injected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int | None = Field(default=None, ge=1)
    on_loop: Literal["short_circuit", "inject"] = "short_circuit"
    retry_node: str | None = None


@dataclass(frozen=True)
class FallbackCommand:
    """Stand-in for ``langgraph.types.Command`` when LangGraph is not installed."""

    update: Any = None
    goto: Any = None
    resume: Any = None
    graph: Any = None


def resolve_config(config: IntegrationConfig | Mapping[str, Any] | None) -> IntegrationConfig:
    """Return ``config`` as an :class:`IntegrationConfig`."""

    if config is None:
        return IntegrationConfig()
    if isinstance(config, IntegrationConfig):
        return config
    return IntegrationConfig.model_validate(dict(config))


# --------------------------------------------------------------------------
# Shared screening
# --------------------------------------------------------------------------


def _loop_nudge(shield: LoopShield, messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the nudge for a flagged transcript, or None.

    A transcript that already ends in a nudge is let through, which is what
    gives the model its turn to change strategy.
    """

    if not messages or is_intervention(messages[-1]):
        return None
    steps = messages_to_steps(messages)
    if not steps:
        return None
    verdict = shield.evaluate(steps)
    return nudge_for(verdict, steps) if verdict.is_loop else None


def _build_message(
    kind: Literal["system", "ai", "tool"],
    content: str,
    *,
    intervention: bool = True,
    **fields: Any,
) -> Any:
    """Build a LangChain message when ``langchain_core`` is present, else a dict."""

    extras = dict(_MARKER) if intervention else {}
    classes = {"system": _SystemMessage, "ai": _AIMessage, "tool": _ToolMessage}
    message_class = classes[kind]
    if message_class is not None:
        if extras:
            fields["additional_kwargs"] = extras
        return message_class(content=content, **fields)
    roles = {"system": "system", "ai": "assistant", "tool": "tool"}
    message: dict[str, Any] = {"role": roles[kind], "content": content, **fields}
    if extras:
        message["additional_kwargs"] = extras
    return message


def _prune_messages(
    raw: Sequence[Any],
    messages: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[Any]:
    """Run ``prune_context`` and map the result back onto the original objects."""

    tagged = [{**message, _INDEX_KEY: index} for index, message in enumerate(messages)]
    pruned = prune_context(tagged, limit)
    restored: list[Any] = []
    for message in pruned:
        index = message.get(_INDEX_KEY)
        if not isinstance(index, int):
            restored.append(
                _build_message("system", content_text(message.get("content")), intervention=False)
            )
            continue
        if _same_payload(messages[index], message):
            restored.append(raw[index])
            continue
        restored.append(_project_pruned(raw[index], message))
    return restored


def _same_payload(source: Mapping[str, Any], pruned: Mapping[str, Any]) -> bool:
    return all(source.get(key) == pruned.get(key) for key in ("role", "content", "tool_calls", "name", "tool_call_id"))


def _project_pruned(raw: Any, pruned: Mapping[str, Any]) -> Any:
    """Copy ``raw`` with the calls and content ``prune_context`` kept."""

    if isinstance(raw, Mapping):
        return {key: value for key, value in pruned.items() if key != _INDEX_KEY}
    clone = copy.copy(raw)
    if hasattr(clone, "content") and "content" in pruned:
        clone.content = pruned.get("content")
    if hasattr(clone, "tool_calls"):
        calls = pruned.get("tool_calls")
        clone.tool_calls = list(calls) if isinstance(calls, list) else []
    return clone


# --------------------------------------------------------------------------
# LangChain
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelOutcome:
    request: Any
    reply: Any | None = None


def _with_messages(request: Any, messages: list[Any]) -> Any:
    override = getattr(request, "override", None)
    if callable(override):
        return override(messages=messages)
    clone = copy.copy(request)
    clone.messages = messages
    return clone


def _screen_model_request(shield: LoopShield, config: IntegrationConfig, request: Any) -> _ModelOutcome:
    raw = list(getattr(request, "messages", None) or [])
    messages = coerce_messages(raw)
    nudge = _loop_nudge(shield, messages)
    if nudge is not None and config.on_loop == "short_circuit":
        return _ModelOutcome(request, _build_message("ai", nudge))

    outgoing = raw
    if nudge is not None:
        outgoing = [*raw, _build_message("system", nudge)]
        messages = coerce_messages(outgoing)
    limit = config.max_tokens if config.max_tokens is not None else shield.max_tokens_estimate
    if limit is not None and estimate_tokens(messages) > limit:
        outgoing = _prune_messages(outgoing, messages, limit)
    if outgoing is raw:
        return _ModelOutcome(request)
    return _ModelOutcome(_with_messages(request, outgoing))


def _call_parts(call: Any) -> tuple[str, dict[str, Any], str | None]:
    if isinstance(call, Mapping):
        name, arguments, call_id = call.get("name"), call.get("args"), call.get("id")
    else:
        name, arguments, call_id = (getattr(call, key, None) for key in ("name", "args", "id"))
    return (
        str(name or "").strip() or "tool",
        argument_dict(arguments),
        str(call_id) if call_id is not None else None,
    )


def _state_messages(state: Any) -> list[Any]:
    messages = state.get("messages") if isinstance(state, Mapping) else getattr(state, "messages", None)
    return list(messages) if isinstance(messages, (list, tuple)) else []


def _before_pending_call(
    messages: Sequence[Mapping[str, Any]],
    call_id: str | None,
) -> tuple[list[Mapping[str, Any]], str]:
    """Split a transcript at the assistant message that issued ``call_id``."""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role", "")) != "assistant":
            continue
        issued = tool_calls(message)
        if not issued:
            continue
        ids = {str(call.get("id")) for call in issued if call.get("id") is not None}
        if call_id is None or call_id in ids:
            return list(messages[:index]), content_text(message.get("content")).strip()
    return list(messages), ""


def _screen_tool_call(shield: LoopShield, request: Any) -> Any | None:
    raw_call = getattr(request, "tool_call", None)
    name, arguments, call_id = _call_parts(raw_call)
    messages = coerce_messages(_state_messages(getattr(request, "state", None)))
    prior, thought = _before_pending_call(messages, call_id)
    pending = AgentStep(
        thought=thought or describe_call(name, arguments),
        tool_name=name,
        tool_args=arguments,
        tokens_used=0,
    )
    steps = [*messages_to_steps(prior), pending]
    verdict = shield.evaluate(steps)
    if not verdict.is_loop:
        return None
    fields: dict[str, Any] = {"name": name, "status": "error"}
    if call_id is not None:
        fields["tool_call_id"] = call_id
    return _build_message("tool", nudge_for(verdict, steps), **fields)


_MiddlewareBase: Any = AgentMiddleware if AgentMiddleware is not None else object


class LoopShieldMiddleware(_MiddlewareBase):
    """LangChain ``AgentMiddleware`` that guards the model and tool calls of an agent.

    ``wrap_tool_call`` replaces a looping tool call with an error result that
    carries the nudge, so the tool never runs and the model sees why.
    ``wrap_model_call`` prunes an over-budget transcript and, for a loop that
    reaches the model, answers with the nudge (or injects it) before the LLM
    is called. Handlers are invoked unchanged for every clean call.
    """

    def __init__(self, shield: LoopShield, config: IntegrationConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.shield = shield
        self.config = resolve_config(config)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        outcome = _screen_model_request(self.shield, self.config, request)
        if outcome.reply is not None:
            return outcome.reply
        return handler(outcome.request)

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        outcome = _screen_model_request(self.shield, self.config, request)
        if outcome.reply is not None:
            return outcome.reply
        return await handler(outcome.request)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        blocked = _screen_tool_call(self.shield, request)
        if blocked is not None:
            return blocked
        return handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        blocked = _screen_tool_call(self.shield, request)
        if blocked is not None:
            return blocked
        return await handler(request)


def langchain_hooks(
    shield: LoopShield,
    config: IntegrationConfig | Mapping[str, Any] | None = None,
) -> list[Any]:
    """Return the guards as ``wrap_model_call`` / ``wrap_tool_call`` decorated middleware.

    Falls back to the single :class:`LoopShieldMiddleware` when LangChain's
    decorators cannot be imported.
    """

    middleware = LoopShieldMiddleware(shield, config)
    if wrap_model_call is None or wrap_tool_call is None:
        return [middleware]

    def guard_model(request: Any, handler: Callable[[Any], Any]) -> Any:
        return middleware.wrap_model_call(request, handler)

    def guard_tool(request: Any, handler: Callable[[Any], Any]) -> Any:
        return middleware.wrap_tool_call(request, handler)

    return [wrap_model_call(guard_model), wrap_tool_call(guard_tool)]


# --------------------------------------------------------------------------
# CrewAI
# --------------------------------------------------------------------------

_CREW_ACTION = re.compile(
    r"Action:\s*(?P<name>[^\n]+?)\s*\n\s*Action Input:\s*(?P<input>.*?)(?=\n\s*(?:Observation|Thought|Action):|\Z)",
    re.DOTALL,
)
_NAME_KEYS = ("tool_name", "name", "tool")
_ARGUMENT_KEYS = ("tool_args", "args", "input", "arguments")


def _field(item: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        value = item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)
        if value is not None:
            return value
    return None


def _crew_tool_steps(agent: Any, trace: str) -> list[AgentStep]:
    """Rebuild the tool-use trajectory from the agent's results, else its string trace."""

    uses: list[tuple[str, dict[str, Any]]] = []
    results = getattr(agent, "tools_results", None)
    if isinstance(results, (list, tuple)) and results:
        for item in results:
            name = _field(item, _NAME_KEYS)
            if name is not None and str(name).strip():
                uses.append((str(name).strip(), argument_dict(_field(item, _ARGUMENT_KEYS))))
    else:
        for match in _CREW_ACTION.finditer(trace):
            uses.append((match.group("name").strip(), argument_dict(match.group("input").strip())))
    return [
        AgentStep(thought=describe_call(name, arguments), tool_name=name, tool_args=arguments, tokens_used=0)
        for name, arguments in uses
    ]


def _crew_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, bool]:
    if "context" in kwargs:
        return kwargs["context"], True
    if args:
        return args[0], False
    return None, True


def _nudged_call(
    shield: LoopShield,
    agent: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    context, by_keyword = _crew_context(args, kwargs)
    trace = context if isinstance(context, str) else ""
    steps = _crew_tool_steps(agent, trace)
    if not steps:
        return args, kwargs
    verdict = shield.evaluate(steps)
    if not verdict.is_loop:
        return args, kwargs
    nudge = nudge_for(verdict, steps)
    if trace.startswith(nudge):
        return args, kwargs
    updated = f"{nudge}\n\n{trace}" if trace else nudge
    if by_keyword:
        return args, {**kwargs, "context": updated}
    return (updated, *args[1:]), kwargs


def _wrap_execute_task(shield: LoopShield, execute: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(execute):

        @wraps(execute)
        async def async_wrapper(agent: Any, task: Any, *args: Any, **kwargs: Any) -> Any:
            call_args, call_kwargs = _nudged_call(shield, agent, args, kwargs)
            return await execute(agent, task, *call_args, **call_kwargs)

        return async_wrapper

    @wraps(execute)
    def wrapper(agent: Any, task: Any, *args: Any, **kwargs: Any) -> Any:
        call_args, call_kwargs = _nudged_call(shield, agent, args, kwargs)
        return execute(agent, task, *call_args, **call_kwargs)

    return wrapper


def crewai_agent(shield: LoopShield) -> Callable[[Any], Any]:
    """Return a decorator for ``execute_task`` or for a CrewAI ``Agent`` class.

    Before the wrapped step runs, the agent's tool results (or the ReAct
    ``Action`` / ``Action Input`` lines in the string context) are rebuilt
    into a tool trajectory. A loop prepends the nudge to the ``context``
    string, which is the execution trace the next step reads.
    """

    def decorate(target: Any) -> Any:
        if inspect.isclass(target):
            target.execute_task = _wrap_execute_task(shield, target.execute_task)
            return target
        return _wrap_execute_task(shield, target)

    return decorate


# --------------------------------------------------------------------------
# LangGraph
# --------------------------------------------------------------------------


def _graph_update(
    shield: LoopShield,
    config: IntegrationConfig,
    state: Any,
) -> Any | None:
    nudge = _loop_nudge(shield, coerce_messages(_state_messages(state)))
    if nudge is None:
        return None
    update = {"messages": [_build_message("system", nudge)]}
    if config.retry_node is None:
        return update
    command = _LangGraphCommand if _LangGraphCommand is not None else FallbackCommand
    return command(update=update, goto=config.retry_node)


def langgraph_node(
    shield: LoopShield,
    node: Callable[..., Any] | None = None,
    config: IntegrationConfig | Mapping[str, Any] | None = None,
) -> Any:
    """Wrap a LangGraph node, or return a decorator when ``node`` is omitted.

    The wrapper reads ``state["messages"]`` and, when the trajectory is a loop
    that has not been nudged yet, returns the system nudge as a state update
    without running the node. With ``retry_node`` set it returns
    ``Command(update=..., goto=retry_node)`` so the graph re-enters that node
    with the nudge in context. Otherwise the node runs exactly as written.
    The update needs a ``messages`` channel with an appending reducer such as
    ``add_messages``.
    """

    resolved = resolve_config(config)

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
                intervention = _graph_update(shield, resolved, state)
                if intervention is not None:
                    return intervention
                return await function(state, *args, **kwargs)

            return async_wrapper

        @wraps(function)
        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            intervention = _graph_update(shield, resolved, state)
            if intervention is not None:
                return intervention
            return function(state, *args, **kwargs)

        return wrapper

    return decorate if node is None else decorate(node)
