"""Structural tests for the LangChain, CrewAI, and LangGraph integrations.

The mock classes below mirror the shapes the real frameworks hand to the
hooks, so the tests run with or without those packages installed.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from loopshield import LoopShield, frameworks
from loopshield.frameworks import IntegrationConfig, LoopShieldMiddleware


# --- LangChain-shaped mocks -------------------------------------------------


@dataclass
class Message:
    type: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    messages: list[Any]
    model: str = "mock-model"

    def override(self, **changes: Any) -> ModelRequest:
        return replace(self, **changes)


@dataclass(frozen=True)
class ToolCallRequest:
    tool_call: dict[str, Any]
    state: dict[str, Any]


def _call(call_id: str, name: str = "read_file", path: str = "config.txt") -> dict[str, Any]:
    return {"name": name, "args": {"path": path}, "id": call_id, "type": "tool_call"}


def _ai(call_id: str, name: str = "read_file", path: str = "config.txt") -> Message:
    return Message("ai", tool_calls=[_call(call_id, name, path)])


def _result(call_id: str, text: str = "FileNotFoundError: config.txt") -> Message:
    return Message("tool", text, tool_call_id=call_id, name="read_file")


def _failed_reads(count: int) -> list[Message]:
    messages: list[Message] = [Message("human", "Load the service configuration.")]
    for number in range(1, count + 1):
        messages.extend([_ai(str(number)), _result(str(number))])
    return messages


def _text(message: Any) -> str:
    return str(message["content"] if isinstance(message, dict) else message.content)


def _marked(message: Any) -> bool:
    extras = message["additional_kwargs"] if isinstance(message, dict) else message.additional_kwargs
    return bool(extras.get("loopshield"))


class RecordingHandler:
    def __init__(self, response: Any = "native-response") -> None:
        self.requests: list[Any] = []
        self.response = response

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        return self.response

    async def acall(self, request: Any) -> Any:
        return self(request)


# --- LangChain --------------------------------------------------------------


def test_langchain_model_hook_short_circuits_a_loop_before_the_llm() -> None:
    shield = LoopShield()
    middleware = shield.langchain_middleware()
    handler = RecordingHandler()

    reply = middleware.wrap_model_call(ModelRequest(_failed_reads(3)), handler)

    assert handler.requests == []
    assert "`read_file`" in _text(reply)
    assert "different strategy" in _text(reply)
    assert _marked(reply)
    assert shield.history == ()


def test_langchain_model_hook_leaves_a_clean_call_untouched() -> None:
    middleware = LoopShield().langchain_middleware()
    handler = RecordingHandler()
    request = ModelRequest(_failed_reads(1))

    response = middleware.wrap_model_call(request, handler)

    assert response == "native-response"
    assert handler.requests == [request]
    assert handler.requests[0] is request


def test_langchain_model_hook_lets_the_model_answer_after_a_nudge() -> None:
    middleware = LoopShield().langchain_middleware()
    handler = RecordingHandler()
    nudge = _failed_reads(3)
    nudge[-1] = Message(
        "tool",
        "The tool `read_file` is temporarily locked. Choose a different strategy.",
        tool_call_id="3",
        additional_kwargs={"loopshield": True},
    )

    response = middleware.wrap_model_call(ModelRequest(nudge), handler)

    assert response == "native-response"
    assert len(handler.requests) == 1


def test_langchain_model_hook_injects_the_nudge_when_configured() -> None:
    middleware = LoopShield().langchain_middleware({"on_loop": "inject"})
    handler = RecordingHandler()
    original = _failed_reads(3)

    response = middleware.wrap_model_call(ModelRequest(original), handler)

    assert response == "native-response"
    forwarded = handler.requests[0].messages
    assert forwarded[: len(original)] == original
    assert len(forwarded) == len(original) + 1
    assert "`read_file`" in _text(forwarded[-1])


def test_langchain_model_hook_prunes_an_over_budget_transcript() -> None:
    middleware = LoopShield().langchain_middleware(IntegrationConfig(max_tokens=120))
    handler = RecordingHandler()
    error = "Error: timeout. " * 40
    messages = [
        Message("human", "Find the incident."),
        Message("ai", "Searching.", tool_calls=[_call("a", "search", "q1")]),
        Message("tool", error, tool_call_id="a", name="search"),
        Message("ai", "Searching again.", tool_calls=[_call("b", "search", "q2")]),
        Message("tool", error, tool_call_id="b", name="search"),
        Message("human", "What caused it?"),
        Message("ai", "Still working on it."),
    ]

    response = middleware.wrap_model_call(ModelRequest(messages), handler)

    assert response == "native-response"
    forwarded = handler.requests[0].messages
    assert len(forwarded) < len(messages)
    assert all(error not in _text(message) for message in forwarded)
    assert forwarded[-1] is messages[-1]
    assert forwarded[-2] is messages[-2]
    forwarded_call_ids = []
    for message in forwarded:
        calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        forwarded_call_ids.extend(call.get("id") for call in calls or [] if isinstance(call, dict))
    assert forwarded_call_ids == []


def test_langchain_tool_hook_blocks_the_third_identical_call() -> None:
    shield = LoopShield()
    middleware = shield.langchain_middleware()
    handler = RecordingHandler()
    state = {"messages": [*_failed_reads(2), _ai("3")]}

    blocked = middleware.wrap_tool_call(ToolCallRequest(_call("3"), state), handler)

    assert handler.requests == []
    assert "`read_file`" in _text(blocked)
    assert "temporarily locked" in _text(blocked)
    assert blocked["tool_call_id"] == "3" if isinstance(blocked, dict) else blocked.tool_call_id == "3"
    assert _marked(blocked)
    assert shield.history == ()


def test_langchain_tool_hook_runs_normal_and_changed_calls_natively() -> None:
    middleware = LoopShield().langchain_middleware()
    handler = RecordingHandler("PORT=8080")

    second = ToolCallRequest(_call("2"), {"messages": [*_failed_reads(1), _ai("2")]})
    assert middleware.wrap_tool_call(second, handler) == "PORT=8080"

    switched = ToolCallRequest(
        _call("4", "read_readme", "README.md"),
        {"messages": [*_failed_reads(3), _ai("4", "read_readme", "README.md")]},
    )
    assert middleware.wrap_tool_call(switched, handler) == "PORT=8080"
    assert handler.requests == [second, switched]


def test_langchain_hooks_work_through_async_handlers() -> None:
    middleware = LoopShield().langchain_middleware()
    handler = RecordingHandler()

    async def run() -> tuple[Any, Any, Any]:
        reply = await middleware.awrap_model_call(ModelRequest(_failed_reads(3)), handler.acall)
        clean = await middleware.awrap_model_call(ModelRequest(_failed_reads(1)), handler.acall)
        blocked = await middleware.awrap_tool_call(
            ToolCallRequest(_call("3"), {"messages": [*_failed_reads(2), _ai("3")]}),
            handler.acall,
        )
        return reply, clean, blocked

    reply, clean, blocked = asyncio.run(run())

    assert "`read_file`" in _text(reply)
    assert clean == "native-response"
    assert "`read_file`" in _text(blocked)
    assert len(handler.requests) == 1


def test_langchain_middleware_exposes_the_native_hook_names() -> None:
    middleware = LoopShield().langchain_middleware()

    assert isinstance(middleware, LoopShieldMiddleware)
    for hook in ("wrap_model_call", "awrap_model_call", "wrap_tool_call", "awrap_tool_call"):
        assert callable(getattr(middleware, hook))


def test_langchain_hooks_register_through_the_decorators(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[tuple[str, Any]] = []

    def fake_model_decorator(function: Any) -> tuple[str, Any]:
        registered.append(("model", function))
        return ("model", function)

    def fake_tool_decorator(function: Any) -> tuple[str, Any]:
        registered.append(("tool", function))
        return ("tool", function)

    monkeypatch.setattr(frameworks, "wrap_model_call", fake_model_decorator)
    monkeypatch.setattr(frameworks, "wrap_tool_call", fake_tool_decorator)

    hooks = frameworks.langchain_hooks(LoopShield())
    handler = RecordingHandler()

    assert [kind for kind, _ in hooks] == ["model", "tool"]
    reply = registered[0][1](ModelRequest(_failed_reads(3)), handler)
    assert "`read_file`" in _text(reply)
    assert registered[1][1](ToolCallRequest(_call("1"), {"messages": []}), handler) == "native-response"


def test_langchain_hooks_fall_back_to_the_middleware_without_decorators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(frameworks, "wrap_model_call", None)
    monkeypatch.setattr(frameworks, "wrap_tool_call", None)

    hooks = frameworks.langchain_hooks(LoopShield())

    assert len(hooks) == 1
    assert isinstance(hooks[0], LoopShieldMiddleware)


# --- CrewAI-shaped mocks ----------------------------------------------------


class MockCrewAgent:
    """Mirrors ``crewai.Agent``: ``execute_task(task, context, tools) -> str``."""

    def __init__(self, tools_results: list[dict[str, Any]] | None = None) -> None:
        self.tools_results = tools_results or []
        self.contexts: list[str | None] = []

    def execute_task(self, task: str, context: str | None = None, tools: list[str] | None = None) -> str:
        self.contexts.append(context)
        return f"done:{task}"


def _tool_results(count: int) -> list[dict[str, Any]]:
    return [
        {"tool_name": "read_file", "tool_args": {"path": "config.txt"}, "result": "FileNotFoundError"}
        for _ in range(count)
    ]


def _react_trace(count: int) -> str:
    block = 'Action: read_file\nAction Input: {"path": "config.txt"}\nObservation: FileNotFoundError\n'
    return "Thought: retry\n" + "Thought: retry\n".join([block] * count)


def test_crewai_decorator_nudges_the_next_step_when_tools_loop() -> None:
    shield = LoopShield()

    @shield.crewai_agent()
    class GuardedAgent(MockCrewAgent):
        pass

    agent = GuardedAgent(_tool_results(3))

    result = agent.execute_task("load config", context="Earlier output.")

    assert result == "done:load config"
    context = agent.contexts[0]
    assert context is not None
    assert "`read_file`" in context
    assert "different strategy" in context
    assert context.endswith("Earlier output.")
    assert shield.history == ()


def test_crewai_decorator_reads_the_string_trace_when_no_tool_results_exist() -> None:
    shield = LoopShield()
    agent_class = shield.crewai_agent()(type("TraceAgent", (MockCrewAgent,), {}))
    agent = agent_class()

    agent.execute_task("load config", context=_react_trace(3))
    agent.execute_task("load config", _react_trace(3))

    for context in agent.contexts:
        assert context is not None
        assert context.startswith("The tool `read_file` is temporarily locked")


def test_crewai_decorator_passes_clean_runs_through_unchanged() -> None:
    shield = LoopShield()
    agent_class = shield.crewai_agent()(type("CleanAgent", (MockCrewAgent,), {}))
    agent = agent_class(_tool_results(2))
    trace = "Earlier output."

    result = agent.execute_task("load config", context=trace, tools=["read_file"])

    assert result == "done:load config"
    assert agent.contexts[0] is trace


def test_crewai_decorator_does_not_stack_nudges() -> None:
    shield = LoopShield()
    agent_class = shield.crewai_agent()(type("RepeatAgent", (MockCrewAgent,), {}))
    agent = agent_class(_tool_results(3))

    agent.execute_task("load config")
    first = agent.contexts[0]
    agent.execute_task("load config", context=first)

    assert agent.contexts[1] == first


def test_crewai_decorator_wraps_a_bare_method_and_keeps_its_signature() -> None:
    shield = LoopShield()
    original = MockCrewAgent.execute_task

    wrapped = shield.crewai_agent()(original)

    assert wrapped.__name__ == "execute_task"
    assert inspect.signature(wrapped) == inspect.signature(original)
    agent = MockCrewAgent(_tool_results(3))
    assert wrapped(agent, "load config") == "done:load config"
    assert agent.contexts[0] is not None
    assert "`read_file`" in agent.contexts[0]


def test_crewai_decorator_supports_async_execute_task() -> None:
    shield = LoopShield()
    seen: list[str | None] = []

    class AsyncAgent:
        tools_results = _tool_results(3)

        async def execute_task(self, task: str, context: str | None = None) -> str:
            seen.append(context)
            return task

    guarded = shield.crewai_agent()(AsyncAgent)

    assert asyncio.run(guarded().execute_task("go")) == "go"
    assert seen[0] is not None
    assert "`read_file`" in seen[0]


# --- LangGraph-shaped mocks -------------------------------------------------


def _agent_node(state: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [Message("ai", "Calling the model.")]}


def test_langgraph_node_injects_the_nudge_instead_of_running_a_looping_node() -> None:
    shield = LoopShield()
    calls: list[dict[str, Any]] = []

    @shield.langgraph_node()
    def agent(state: dict[str, Any]) -> dict[str, Any]:
        calls.append(state)
        return _agent_node(state)

    update = agent({"messages": _failed_reads(3)})

    assert calls == []
    assert list(update) == ["messages"]
    assert len(update["messages"]) == 1
    assert "`read_file`" in _text(update["messages"][0])
    assert _marked(update["messages"][0])
    assert agent.__name__ == "agent"
    assert shield.history == ()


def test_langgraph_node_runs_native_logic_for_a_clean_state() -> None:
    shield = LoopShield()
    seen: list[dict[str, Any]] = []

    def agent(state: dict[str, Any], extra: str = "") -> dict[str, Any]:
        seen.append(state)
        return {"messages": [Message("ai", f"native{extra}")]}

    wrapped = shield.langgraph_node(agent)
    state = {"messages": _failed_reads(1)}

    update = wrapped(state, extra="!")

    assert seen == [state]
    assert seen[0] is state
    assert _text(update["messages"][0]) == "native!"


def test_langgraph_node_runs_after_the_nudge_is_in_the_state() -> None:
    shield = LoopShield()
    wrapped = shield.langgraph_node(_agent_node)
    first = wrapped({"messages": _failed_reads(3)})

    second = wrapped({"messages": [*_failed_reads(3), *first["messages"]]})

    assert _text(second["messages"][0]) == "Calling the model."


def test_langgraph_node_returns_a_command_to_retry_the_node() -> None:
    shield = LoopShield()
    wrapped = shield.langgraph_node(_agent_node, config={"retry_node": "agent"})

    command = wrapped({"messages": _failed_reads(3)})

    assert command.goto == "agent"
    assert "`read_file`" in _text(command.update["messages"][0])
    if not frameworks.LANGGRAPH_AVAILABLE:
        assert isinstance(command, frameworks.FallbackCommand)


def test_langgraph_node_flags_a_semantic_loop_without_tool_calls() -> None:
    shield = LoopShield()
    wrapped = shield.langgraph_node(_agent_node)
    plan = "I will retry the same failing plan."
    messages: list[Any] = [Message("human", "Fix the build.")]
    for _ in range(3):
        messages.extend([Message("ai", plan), Message("human", "Continue.")])

    update = wrapped({"messages": messages})

    assert "repeated line of thought" in _text(update["messages"][0])


def test_langgraph_node_accepts_dict_messages_and_async_nodes() -> None:
    shield = LoopShield()

    @shield.langgraph_node()
    async def agent(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": "native"}]}

    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "config.txt"}'}}
        ],
    }
    failure = {"role": "tool", "tool_call_id": "x", "content": "FileNotFoundError"}
    loop_state = {"messages": [{"role": "user", "content": "go"}, call, failure, call, failure, call, failure]}

    assert "`read_file`" in _text(asyncio.run(agent(loop_state))["messages"][0])
    assert asyncio.run(agent({"messages": [{"role": "user", "content": "go"}]}))["messages"][0]["content"] == "native"


def test_integration_config_rejects_unknown_options() -> None:
    with pytest.raises(ValueError):
        LoopShield().langchain_middleware({"on_loop": "shout"})
    with pytest.raises(ValueError):
        IntegrationConfig.model_validate({"unknown": 1})
