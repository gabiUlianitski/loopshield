"""Simulated agent trajectories for tool cycles and semantic repeats."""

from __future__ import annotations

import networkx as nx
import pytest

from loopshield import (
    AgentStep,
    GraphAnalyzer,
    LoopShield,
    LoopVerdict,
    Mitigator,
    SemanticAnalyzer,
    generate_nudge,
    prune_context,
)
from loopshield.graph_analyzer import tool_signature
from loopshield.mitigator import estimate_tokens
from loopshield.semantic_analyzer import cosine_similarity, tfidf_embed


def _step(
    thought: str,
    tool_name: str = "search",
    tool_args: dict[str, object] | None = None,
    tokens_used: int = 10,
) -> AgentStep:
    return AgentStep(
        thought=thought,
        tool_name=tool_name,
        tool_args={} if tool_args is None else tool_args,
        tokens_used=tokens_used,
    )


def _search(thought: str, query: str) -> AgentStep:
    return _step(thought, tool_name="search", tool_args={"query": query})


def _open(thought: str, url: str) -> AgentStep:
    return _step(thought, tool_name="open", tool_args={"url": url})


def test_normalized_tool_args_ignore_key_order_and_string_formatting() -> None:
    left = tool_signature("search", {"b": 1, "a": " Paris "})
    right = tool_signature("search", {"a": "paris", "b": 1})

    assert left == right


def test_progressing_research_agent_is_not_a_loop() -> None:
    steps = [
        _search("Find the outage report", "outage report"),
        _open("Open the incident timeline", "https://status.example/incidents/41"),
        _step("The outage lasted 12 minutes.", tool_name="finish", tool_args={"status": "done"}),
    ]

    verdict, healed = LoopShield()(steps)

    assert verdict.is_loop is False
    assert verdict.loop_type is None
    assert verdict.intervention_prompt is None
    assert healed == steps


def test_same_tool_cycle_must_repeat_more_than_max_cycles() -> None:
    analyzer = GraphAnalyzer(max_cycles=1)
    one_lap = [
        _search("Search for the report", "outage"),
        _open("Open the first result", "https://status.example/incidents/41"),
        _search("Search for the report again", "outage"),
    ]
    second_lap = [
        *one_lap,
        _open("Open the same result", "https://status.example/incidents/41"),
        _search("Search for the report a third time", "outage"),
    ]

    graph = analyzer.build_graph(second_lap)

    assert isinstance(graph, nx.DiGraph)
    assert not nx.is_directed_acyclic_graph(graph)
    assert analyzer.repeat_count(one_lap) == 1
    assert analyzer.detect(one_lap) is False
    assert analyzer.repeat_count(second_lap) == 2
    assert analyzer.detect(second_lap) is True


def test_identical_tool_call_is_a_self_loop_after_it_repeats() -> None:
    analyzer = GraphAnalyzer(max_cycles=1)
    twice = [
        _search("Look up the weather in Paris", "weather in paris"),
        _search("Look up the weather in Paris", "weather in paris"),
    ]
    three_times = [*twice, _search("Look up the weather in Paris again", "weather in paris")]

    graph = analyzer.build_graph(three_times)

    assert graph.has_edge(tool_signature("search", {"query": "weather in paris"}), tool_signature("search", {"query": "weather in paris"}))
    assert analyzer.detect(twice) is False
    assert analyzer.detect(three_times) is True


def test_zero_max_cycles_flags_the_first_completed_cycle() -> None:
    steps = [
        _search("Look up the weather in Paris", "weather in paris"),
        _search("Look up the weather in Paris", "weather in paris"),
    ]

    assert GraphAnalyzer(max_cycles=0).detect(steps) is True


def test_cosine_similarity_uses_numpy_for_identical_and_orthogonal_vectors() -> None:
    identical = cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    orthogonal = cosine_similarity([1.0, 0.0], [0.0, 1.0])

    assert identical == pytest.approx(1.0)
    assert orthogonal == pytest.approx(0.0)


def test_tfidf_fallback_flags_consecutive_repeated_thoughts() -> None:
    thought = "Look up the weather in Paris"
    steps = [
        _search(thought, "paris weather"),
        _search(thought, "weather paris"),
        _search(thought, "paris forecast"),
    ]
    vectors = tfidf_embed([step.thought for step in steps])
    analyzer = SemanticAnalyzer()

    assert cosine_similarity(vectors[0], vectors[-1]) == pytest.approx(1.0)
    assert analyzer.similarity_streak(steps[:2]) == 1
    assert analyzer.detect(steps[:2]) is False
    assert analyzer.similarity_streak(steps) == 2
    assert analyzer.detect(steps) is True


def test_injected_embedder_flags_rephrased_thoughts() -> None:
    thoughts = {
        "Look up the weather in Paris": [1.0, 0.0, 0.0],
        "Please check the weather in Paris": [0.99, 0.01, 0.0],
        "Check the Paris weather one more time": [0.98, 0.02, 0.0],
    }
    steps = [
        _step(thought, tool_args={"q": str(index)})
        for index, thought in enumerate(thoughts)
    ]

    verdict = LoopShield(
        embed_fn=lambda text: thoughts[text],
        max_semantic_repeats=2,
    ).inspect(steps)

    assert verdict.is_loop is True
    assert verdict.loop_type == "semantic"


def test_similarity_must_exceed_the_threshold() -> None:
    steps = [
        _step("alpha", tool_args={"q": "1"}),
        _step("alpha again", tool_args={"q": "2"}),
    ]

    def embed(_text: str) -> list[float]:
        return [1.0, 0.0]

    at_threshold = SemanticAnalyzer(embed, similarity_threshold=1.0, max_semantic_repeats=1)
    above_threshold = SemanticAnalyzer(embed, similarity_threshold=0.99, max_semantic_repeats=1)

    assert at_threshold.detect(steps) is False
    assert above_threshold.detect(steps) is True


def test_unrelated_thought_breaks_the_consecutive_streak() -> None:
    steps = [
        _search("Look up the weather in Paris", "paris weather"),
        _open("Open the transit map", "https://example.test/metro"),
        _search("Please check the weather in Paris again", "weather paris"),
    ]

    assert SemanticAnalyzer(max_semantic_repeats=1).detect(steps) is False


def test_custom_embed_fn_is_used_for_semantic_detection() -> None:
    seen: list[str] = []

    def embed(text: str) -> list[float]:
        seen.append(text)
        if "paris" in text.casefold():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    steps = [
        _search("Find hotels in Paris", "hotels 1"),
        _search("Find more hotels in Paris", "hotels 2"),
        _open("Open the transit map", "https://example.test/metro"),
    ]
    analyzer = SemanticAnalyzer(embed, max_semantic_repeats=1)

    assert analyzer.detect(steps[:2]) is True
    assert analyzer.detect(steps) is False
    assert seen
    assert all("paris" in text.casefold() or "transit" in text.casefold() for text in seen)


def test_shield_reports_graph_and_semantic_loops_with_an_intervention() -> None:
    graph_steps = [
        _search("Search for the report", "outage"),
        _open("Open the report", "https://status.example/incidents/41"),
        _search("Search for the report again", "outage"),
        _open("Open the report again", "https://status.example/incidents/41"),
        _search("Search for the report a third time", "outage"),
    ]
    semantic_steps = [
        _step("Look up the weather in Paris", tool_args={"q": "1"}),
        _step("Look up the weather in Paris", tool_args={"q": "2"}),
        _step("Look up the weather in Paris", tool_args={"q": "3"}),
    ]

    graph_verdict = LoopShield().inspect(graph_steps)
    semantic_verdict = LoopShield(max_semantic_repeats=2).inspect(semantic_steps)

    assert graph_verdict.is_loop is True
    assert graph_verdict.loop_type == "graph_cycle"
    assert graph_verdict.intervention_prompt is not None
    assert "tool cycle" in graph_verdict.intervention_prompt
    assert semantic_verdict.loop_type == "semantic"
    assert semantic_verdict.intervention_prompt is not None


def test_protect_keeps_the_latest_copy_of_a_repeated_tool_call() -> None:
    steps = [
        _search("Look up the weather in Paris", "weather in paris"),
        _search("Look up the weather in Paris", "weather in paris"),
        _search("Look up the weather in Paris again", "weather in paris"),
    ]

    verdict, healed = LoopShield(max_cycles=0, max_semantic_repeats=1).protect(steps)

    assert verdict.is_loop is True
    assert healed == [steps[-1]]
    assert Mitigator().prune(steps) == [steps[-1]]


def test_embedding_dimension_mismatch_is_rejected() -> None:
    steps = [
        _step("alpha", tool_args={"q": "1"}),
        _step("beta", tool_args={"q": "2"}),
    ]

    def embed(text: str) -> list[float]:
        return [1.0, 0.0] if text == "alpha" else [1.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="dimensions"):
        SemanticAnalyzer(embed, max_semantic_repeats=1).detect(steps)


def test_models_reject_negative_token_counts() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _step("thought", tokens_used=-1)


def _transcript() -> tuple[list[dict[str, str]], str]:
    error = "Error: timeout. " * 80
    messages = [
        {"role": "system", "content": "You are a careful agent."},
        {"role": "user", "content": "Find the incident."},
        {"role": "assistant", "content": "Searching the status page."},
        {"role": "tool", "name": "search", "content": error},
        {"role": "assistant", "content": "Retrying the same search."},
        {"role": "tool", "name": "search", "content": error},
        {"role": "user", "content": "What caused it?"},
        {"role": "assistant", "content": "A bad deploy."},
    ]
    return messages, error


def test_generate_nudge_locks_the_blocked_tool() -> None:
    verdict = LoopVerdict(is_loop=True, loop_type="graph_cycle", intervention_prompt=None)

    message = generate_nudge(verdict, " search ")

    assert message["role"] == "system"
    assert "`search`" in message["content"]
    assert "temporarily locked" in message["content"]
    assert "different strategy" in message["content"]
    assert Mitigator().generate_nudge(verdict, "search") == message


def test_prune_context_keeps_the_system_prompt_and_last_two_turns() -> None:
    messages, error = _transcript()

    unchanged = prune_context(messages, estimate_tokens(messages))
    pruned = prune_context(messages, estimate_tokens(messages) - 1)
    anchors = prune_context(messages, 1)

    assert [message["content"] for message in unchanged] == [message["content"] for message in messages]
    assert pruned[0]["content"] == messages[0]["content"]
    assert pruned[-2]["content"] == "What caused it?"
    assert pruned[-1]["content"] == "A bad deploy."
    assert any("Find the incident." == message["content"] for message in pruned)
    assert all(error not in message["content"] for message in pruned)
    assert any("tool-error" in message["content"] for message in pruned)
    assert estimate_tokens(pruned) < estimate_tokens(messages)
    assert [message["content"] for message in anchors] == [
        "You are a careful agent.",
        "What caused it?",
        "A bad deploy.",
    ]
    assert messages[3]["content"] == error


def test_prune_context_collapses_repeated_openai_tool_errors_once() -> None:
    timeout = "Error: timeout. " * 40
    tail_error = "Error: disk full. " * 20
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are a careful agent."},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Searching."}],
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": timeout},
        {"role": "tool", "tool_call_id": "call-1", "content": timeout},
        {"role": "tool", "name": "open", "content": "Error: missing index"},
        {"role": "user", "content": "What caused it?"},
        {"role": "assistant", "content": "A bad deploy."},
        {"role": "tool", "name": "read_file", "content": tail_error},
    ]

    pruned = prune_context(messages, estimate_tokens(messages) - 1)
    summaries = [message["content"] for message in pruned if "tool-error" in message["content"]]

    assert len(summaries) == 1
    assert "search x2" in summaries[0]
    assert any(message["content"] == "Error: missing index" for message in pruned)
    assert all(timeout not in message["content"] for message in pruned)
    assert pruned[0] == {"role": "system", "content": "You are a careful agent."}
    assert pruned[-3]["content"] == "What caused it?"
    assert pruned[-2]["content"] == "A bad deploy."
    assert pruned[-1]["content"] == tail_error
    assert messages[2]["content"] == timeout


def _assert_tool_exchanges_are_paired(messages: list[dict[str, object]]) -> None:
    """Fail when a tool call has no result, or a result arrives without its call."""

    pending: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        calls = _message_call_ids(message) if role == "assistant" else []
        results = _message_result_ids(message)
        if calls and pending:
            raise AssertionError(f"tool calls at message {index} start before {pending} were answered")
        if calls and message.get("tool_calls") == []:
            raise AssertionError(f"message {index} has an empty tool_calls list")
        pending.extend(calls)
        for result_id in results:
            if result_id not in pending:
                raise AssertionError(f"{result_id} at message {index} has no matching tool call")
            pending.remove(result_id)
        if pending and not calls and not results and role in {"system", "user", "assistant"}:
            raise AssertionError(f"{role} message at {index} splits pending tool calls {pending}")
    if pending:
        raise AssertionError(f"unanswered tool calls: {pending}")


def _message_call_ids(message: dict[str, object]) -> list[str]:
    found: list[str] = []
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                found.append(call["id"])
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use" and isinstance(part.get("id"), str):
                found.append(part["id"])
    return found


def _message_result_ids(message: dict[str, object]) -> list[str]:
    found: list[str] = []
    if isinstance(message.get("tool_call_id"), str):
        found.append(message["tool_call_id"])
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_result"
                and isinstance(part.get("tool_use_id"), str)
            ):
                found.append(part["tool_use_id"])
    return found


def test_prune_context_strips_openai_calls_that_lost_their_results() -> None:
    timeout = "Error: timeout. " * 30
    kept_call = {
        "id": "call-3",
        "type": "function",
        "function": {"name": "read_readme", "arguments": '{"path": "README.md"}'},
    }
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Be careful."},
        {"role": "user", "content": "Audit the files."},
        {
            "role": "assistant",
            "content": "Reading the config.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": timeout},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                },
                kept_call,
            ],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": timeout},
        {"role": "tool", "tool_call_id": "call-3", "content": "PORT=8080"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-4",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-4", "content": timeout},
        {"role": "user", "content": "What is the port?"},
        {"role": "assistant", "content": "Checking."},
    ]
    original_calls = messages[2]["tool_calls"]

    pruned = prune_context(messages, estimate_tokens(messages) - 1)

    _assert_tool_exchanges_are_paired(pruned)
    summaries = [message for message in pruned if "tool-error" in str(message["content"])]
    assert len(summaries) == 1
    assert summaries[0]["role"] == "system"
    assert "read_file x3" in summaries[0]["content"]
    assert all(timeout not in str(message.get("content")) for message in pruned)
    assert pruned[-2]["content"] == "What is the port?"
    assert pruned[-1]["content"] == "Checking."
    readme = next(message for message in pruned if message.get("tool_calls"))
    assert readme["tool_calls"] == [kept_call]
    assert pruned[pruned.index(readme) + 1]["tool_call_id"] == "call-3"
    assert messages[2]["tool_calls"] is original_calls
    assert original_calls[0]["id"] == "call-1"


def test_prune_context_strips_anthropic_tool_use_that_lost_its_result() -> None:
    timeout = "Error: timeout. " * 30
    kept_use = {"type": "tool_use", "id": "toolu_3", "name": "read_readme", "input": {"path": "README.md"}}
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "Audit the files."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading the config."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": timeout, "is_error": True}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {"path": "a.txt"}},
                kept_use,
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": timeout, "is_error": True},
                {"type": "text", "text": "The readme is separate."},
                {"type": "tool_result", "tool_use_id": "toolu_3", "content": "PORT=8080"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_4", "name": "read_file", "input": {"path": "a.txt"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_4", "content": timeout, "is_error": True}
            ],
        },
        {"role": "user", "content": "What is the port?"},
        {"role": "assistant", "content": "Checking."},
    ]
    original_content = messages[1]["content"]

    pruned = prune_context(messages, estimate_tokens(messages) - 1)

    _assert_tool_exchanges_are_paired(pruned)
    summaries = [message for message in pruned if "tool-error" in str(message["content"])]
    assert len(summaries) == 1
    assert summaries[0]["role"] == "system"
    assert "read_file x3" in summaries[0]["content"]
    assert all(timeout not in str(message.get("content")) for message in pruned)
    assert pruned[-2]["content"] == "What is the port?"
    assert pruned[-1]["content"] == "Checking."
    readme = next(
        message
        for message in pruned
        if isinstance(message.get("content"), list)
        and any(isinstance(part, dict) and part.get("id") == "toolu_3" for part in message["content"])
    )
    assert pruned[pruned.index(readme) + 1]["content"][-1]["tool_use_id"] == "toolu_3"
    assert "The readme is separate." in str(pruned[pruned.index(readme) + 1]["content"])
    assert not any(
        isinstance(message.get("content"), list) and message["content"] == []
        for message in pruned
    )
    assert messages[1]["content"] is original_content
    assert original_content[1]["id"] == "toolu_1"


def test_prune_context_drops_a_result_whose_call_was_summarized_away() -> None:
    blob = "context " * 200
    messages: list[dict[str, object]] = [
        {"role": "user", "content": blob},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}},
            ],
        },
        {"role": "user", "content": blob},
        {"role": "assistant", "content": "Noted."},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "Error: missing",
                    "is_error": True,
                }
            ],
        },
        {"role": "assistant", "content": "Final."},
    ]

    pruned = prune_context(messages, estimate_tokens(messages) - 1)

    _assert_tool_exchanges_are_paired(pruned)
    assert pruned[-1]["content"] == "Final."
    assert not any(_message_call_ids(message) or _message_result_ids(message) for message in pruned)


def test_token_estimate_prefers_explicit_counts_over_characters() -> None:
    huge = {"role": "user", "content": "x" * 400, "token_count": 7}
    parts = {"role": "assistant", "content": [{"type": "text", "text": "abcd"}]}

    assert estimate_tokens([huge]) == 7
    assert estimate_tokens([parts]) == estimate_tokens([{"role": "assistant", "content": "abcd"}])
    assert estimate_tokens([{"role": "user", "content": "a" * 40}]) > estimate_tokens(
        [{"role": "user", "content": "a"}]
    )


def test_step_locks_the_tool_that_closed_the_loop() -> None:
    shield = LoopShield(max_cycles=0)
    first = shield.step(_search("Look up the weather in Paris", "weather in paris"))
    second = shield.step(_search("Look up the weather in Paris again", "weather in paris"))

    assert first.is_loop is False
    assert second.is_loop is True
    assert second.loop_type == "graph_cycle"
    assert second.intervention_prompt == generate_nudge(second, "search")["content"]
    assert [step.tool_name for step in shield.history] == ["search", "search"]


def test_guard_skips_the_tool_call_when_a_loop_closes() -> None:
    shield = LoopShield(max_cycles=0)
    ran: list[str] = []

    @shield.guard
    def execute(agent_step: AgentStep) -> str:
        ran.append(agent_step.tool_name)
        return "ok"

    assert execute(_search("Look up the weather", "weather in paris")) == "ok"
    blocked = execute(_search("Look up the weather again", "weather in paris"))

    assert isinstance(blocked, LoopVerdict)
    assert blocked.is_loop is True
    assert blocked.intervention_prompt is not None
    assert "search" in blocked.intervention_prompt
    assert ran == ["search"]


def test_guard_refuses_a_looping_batch_before_the_loop_runs() -> None:
    shield = LoopShield(max_cycles=0)
    executed = False

    @shield.guard
    def run(steps: list[AgentStep]) -> int:
        nonlocal executed
        executed = True
        return len(steps)

    result = run([
        _search("Look up the weather", "weather in paris"),
        _search("Look up the weather again", "weather in paris"),
    ])

    assert isinstance(result, LoopVerdict)
    assert result.is_loop is True
    assert executed is False


def test_guard_prunes_returned_messages_to_the_token_budget() -> None:
    messages, error = _transcript()
    shield = LoopShield(max_tokens_estimate=estimate_tokens(messages) - 1)

    @shield.guard
    def finish() -> list[dict[str, str]]:
        return messages

    pruned = finish()

    assert isinstance(pruned, list)
    assert all(error not in message["content"] for message in pruned)
    assert pruned[-1]["content"] == "A bad deploy."


def _call(tool_name: str, thought: str, **tool_args: object) -> AgentStep:
    return AgentStep(
        thought=thought,
        tool_name=tool_name,
        tool_args=dict(tool_args),
        tokens_used=12,
    )


def test_direct_tool_loop_is_caught_on_turn_three_and_nudged() -> None:
    """A mock agent keeps reading a missing file until the third attempt is blocked."""

    calls: list[str] = []

    def read_file(path: str) -> str:
        calls.append(path)
        raise FileNotFoundError(path)

    shield = LoopShield()

    @shield.guard
    def execute(agent_step: AgentStep) -> str:
        if agent_step.tool_name != "read_file":
            raise AssertionError(agent_step.tool_name)
        return read_file(str(agent_step.tool_args["path"]))

    thoughts = [
        "Open the service configuration.",
        "The last read failed because that file was absent.",
        "Try that same path once more before giving up.",
    ]
    outcomes: list[object] = []
    for turn, thought in enumerate(thoughts, start=1):
        step = _call("read_file", thought, path="config.txt")
        try:
            outcomes.append(execute(step))
        except FileNotFoundError as exc:
            assert turn < 3
            assert exc.filename == "config.txt" or str(exc) == "config.txt"
            outcomes.append(exc)

    assert len(outcomes) == 3
    assert all(isinstance(outcome, FileNotFoundError) for outcome in outcomes[:2])
    verdict = outcomes[2]
    assert isinstance(verdict, LoopVerdict)
    assert verdict.is_loop is True
    assert verdict.loop_type == "graph_cycle"
    assert verdict.intervention_prompt is not None
    assert "`read_file`" in verdict.intervention_prompt
    assert "temporarily locked" in verdict.intervention_prompt
    assert "different strategy" in verdict.intervention_prompt
    assert calls == ["config.txt", "config.txt"]


def test_multi_step_graph_cycle_search_parse_and_log() -> None:
    """search_db -> parse_data -> write_log -> search_db is a directed cycle."""

    observed: list[str] = []

    def search_db() -> list[dict[str, int]]:
        observed.append("search_db")
        return [{"id": 7}]

    def parse_data(rows: list[dict[str, int]]) -> dict[str, int]:
        observed.append("parse_data")
        return rows[0]

    def write_log(record: dict[str, int]) -> str:
        observed.append("write_log")
        return f"logged {record['id']}"

    tools = {
        "search_db": lambda step: search_db(),
        "parse_data": lambda step: parse_data([{"id": 7}]),
        "write_log": lambda step: write_log({"id": 7}),
    }
    names = ["search_db", "parse_data", "write_log", "search_db"]
    steps = [
        _call(name, f"Continue the accounts pipeline with {name}.", resource="accounts")
        for name in names
    ]
    for step in steps:
        tools[step.tool_name](step)

    analyzer = GraphAnalyzer()
    graph = analyzer.build_graph(steps)
    signatures = [tool_signature(step.tool_name, step.tool_args) for step in steps]
    cycle_edges = {(source, target) for source, target, *_rest in nx.find_cycle(graph)}
    expected_edges = {
        (signatures[0], signatures[1]),
        (signatures[1], signatures[2]),
        (signatures[2], signatures[0]),
    }

    assert observed == ["search_db", "parse_data", "write_log", "search_db"]
    assert not nx.is_directed_acyclic_graph(graph)
    assert cycle_edges == expected_edges
    assert {node for edge in cycle_edges for node in edge} == set(signatures[:3])
    assert analyzer.repeat_count(steps) == 1
    assert GraphAnalyzer(max_cycles=0).detect(steps) is True

    second_lap = steps + [
        _call("parse_data", "Parse the account row again.", resource="accounts"),
        _call("write_log", "Log the account row again.", resource="accounts"),
        _call("search_db", "Search the account table again.", resource="accounts"),
    ]
    assert analyzer.detect(second_lap) is True


def test_semantic_loop_rephrases_one_failing_plan_without_tools() -> None:
    """Four paraphrases of one failing plan trip the semantic analyzer and no tool runs."""

    plans = [
        "Retry the failing plan: read the missing config and deploy again.",
        "The same failing plan, rephrased: read the missing config, then deploy again.",
        "Once more, the failing plan is to read the missing config and deploy again.",
        "Rephrasing the failing plan again: read the missing config and deploy again.",
    ]

    def embed(text: str) -> list[float]:
        slot = plans.index(text)
        return [1.0, 0.01 * slot, 0.0]

    tool_runs: list[str] = []
    shield = LoopShield(embed_fn=embed, max_semantic_repeats=3)
    verdicts: list[LoopVerdict] = []
    for attempt, plan in enumerate(plans, start=1):
        # Reasoning only: a distinct placeholder signature so the graph stays acyclic.
        step = _call("reason", plan, attempt=attempt)
        verdicts.append(shield.step(step))

    assert tool_runs == []
    assert all(verdict.is_loop is False for verdict in verdicts[:3])
    assert shield.graph_analyzer.detect(list(shield.history)) is False
    assert shield.semantic_analyzer.detect(list(shield.history)) is True
    caught = verdicts[3]
    assert caught.is_loop is True
    assert caught.loop_type == "semantic"
    assert caught.intervention_prompt is not None
    assert "repeated line of thought" in caught.intervention_prompt
    assert "different strategy" in caught.intervention_prompt


def test_self_healing_recovery_switches_tools_after_the_nudge() -> None:
    """After read_file is locked, the agent follows the nudge and finishes another way."""

    files = {"readme.txt": "PORT=8080"}
    calls: list[str] = []

    def read_file(path: str) -> str:
        calls.append(f"read_file:{path}")
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    def read_readme() -> str:
        calls.append("read_readme")
        return files["readme.txt"]

    shield = LoopShield()

    @shield.guard
    def execute(agent_step: AgentStep) -> str:
        if agent_step.tool_name == "read_file":
            return read_file(str(agent_step.tool_args["path"]))
        if agent_step.tool_name == "read_readme":
            return read_readme()
        raise AssertionError(agent_step.tool_name)

    nudge: str | None = None
    for turn, thought in enumerate(
        [
            "Open the service configuration.",
            "The last read failed because that file was absent.",
            "Try that same path once more before giving up.",
        ],
        start=1,
    ):
        try:
            outcome = execute(_call("read_file", thought, path="config.txt"))
        except FileNotFoundError:
            assert turn < 3
            continue
        assert turn == 3
        assert isinstance(outcome, LoopVerdict)
        nudge = outcome.intervention_prompt

    assert nudge is not None
    assert "`read_file`" in nudge
    assert "must not call `read_file`" in nudge

    recovered = execute(
        _call(
            "read_readme",
            "read_file is locked, so follow the readme and load the published port.",
        )
    )

    assert recovered == "PORT=8080"
    assert calls == ["read_file:config.txt", "read_file:config.txt", "read_readme"]
    assert shield.history[-1].tool_name == "read_readme"
