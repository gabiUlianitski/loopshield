"""Middleware wrapper that inspects each agent step and blocks looping tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from loopshield.adapters import as_embed_fn
from loopshield.graph_analyzer import GraphAnalyzer, tool_signature
from loopshield.mitigator import Mitigator
from loopshield.models import AgentStep, LoopVerdict
from loopshield.semantic_analyzer import EmbedFn, SemanticAnalyzer

P = ParamSpec("P")
R = TypeVar("R")

_INTERVENTIONS: dict[str, str] = {
    "graph_cycle": (
        "LoopShield detected a repeated tool cycle. "
        "Do not call the same tools with the same arguments again. "
        "Use the observations you already have, try a different approach, "
        "or give the final answer."
    ),
    "semantic": (
        "LoopShield detected a repeated line of thought. "
        "Stop rephrasing the same idea. "
        "Use the observations you already have, try a different approach, "
        "or give the final answer."
    ),
    "both": (
        "LoopShield detected a repeated tool cycle and a repeated line of thought. "
        "Do not repeat the same action or reasoning. "
        "Use the observations you already have, try a different approach, "
        "or give the final answer."
    ),
}


class LoopShield:
    """Watch an agent loop, block a tool that starts repeating, and prune context.

    ``step`` records one :class:`AgentStep` and returns the verdict for the
    trajectory so far. ``guard`` wraps the function an agent calls on each
    iteration: a looping call is not executed and the verdict is returned
    instead. A guarded function that receives the whole ``list[AgentStep]``
    is refused before it runs when any step in that list closes a loop.

    Pass an initialized client straight through ``embedding``::

        LoopShield(embedding=OpenAI())
        LoopShield(embedding=Client(host="http://127.0.0.1:11434"))
    """

    def __init__(
        self,
        *,
        max_cycles: int = 1,
        similarity_threshold: float = 0.92,
        max_semantic_repeats: int = 2,
        embed_fn: EmbedFn | None = None,
        embedding: Any | None = None,
        embedding_model: str | None = None,
        max_tokens_estimate: int | None = None,
    ) -> None:
        if max_tokens_estimate is not None and max_tokens_estimate < 0:
            raise ValueError("max_tokens_estimate must be zero or greater")
        if embed_fn is not None and embedding is not None:
            raise ValueError("Pass either embed_fn or embedding, not both.")
        if embedding is None and embedding_model is not None:
            raise ValueError("embedding_model is only used with an OpenAI or Ollama client.")
        resolved_embed_fn = embed_fn if embedding is None else as_embed_fn(embedding, model=embedding_model)
        self.graph_analyzer = GraphAnalyzer(max_cycles=max_cycles)
        self.semantic_analyzer = SemanticAnalyzer(
            resolved_embed_fn,
            similarity_threshold=similarity_threshold,
            max_semantic_repeats=max_semantic_repeats,
        )
        self.mitigator = Mitigator()
        self.max_tokens_estimate = max_tokens_estimate
        self._history: list[AgentStep] = []

    @property
    def history(self) -> tuple[AgentStep, ...]:
        """Steps recorded by :meth:`step` and :meth:`guard`, in call order."""

        return tuple(self._history)

    def step(self, agent_step: AgentStep) -> LoopVerdict:
        """Record ``agent_step`` and judge the trajectory collected so far.

        A loop locks ``agent_step.tool_name``. The verdict's
        ``intervention_prompt`` is the system nudge that tells the model to
        change strategy.
        """

        self._history.append(agent_step)
        verdict = self._verdict_for_latest(self._history)
        if not verdict.is_loop:
            return verdict
        nudge = self.mitigator.generate_nudge(verdict, agent_step.tool_name)
        return verdict.model_copy(update={"intervention_prompt": nudge["content"]})

    def guard(self, func: Callable[P, R]) -> Callable[P, R | LoopVerdict]:
        """Decorate an agent step or an agent loop so a detected loop is not executed.

        Per-step form::

            @shield.guard
            def execute(agent_step: AgentStep) -> str:
                return run_tool(agent_step)

        Loop form::

            @shield.guard
            def run(steps: list[AgentStep]) -> list[str]:
                return [run_tool(step) for step in steps]
        """

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | LoopVerdict:
            agent_step = _find_step(args, kwargs)
            if agent_step is not None:
                verdict = self.step(agent_step)
                if verdict.is_loop:
                    return verdict
                return self._maybe_prune(func(*args, **kwargs))

            sequence = _find_step_sequence(args, kwargs)
            if sequence is not None:
                for item in sequence:
                    verdict = self.step(item)
                    if verdict.is_loop:
                        return verdict
                return self._maybe_prune(func(*args, **kwargs))

            result = func(*args, **kwargs)
            if isinstance(result, AgentStep):
                verdict = self.step(result)
                if verdict.is_loop:
                    return verdict
            return self._maybe_prune(result)

        return wrapper

    def inspect(self, steps: Sequence[AgentStep]) -> LoopVerdict:
        """Return a verdict for ``steps``. The given trajectory is not modified."""

        return _assemble(
            self.graph_analyzer.detect(steps),
            self.semantic_analyzer.detect(steps),
        )

    def _verdict_for_latest(self, steps: Sequence[AgentStep]) -> LoopVerdict:
        """Flag a loop only when the newest step is still inside it.

        A tool that is not part of the repeated cycle can run after a nudge,
        which is what lets the agent leave the loop and finish the task.
        """

        if not steps:
            return _assemble(False, False)
        latest = tool_signature(steps[-1].tool_name, steps[-1].tool_args)
        graph_loop = latest in self.graph_analyzer.repeated_signatures(steps)
        semantic_loop = self.semantic_analyzer.detect(steps)
        return _assemble(graph_loop, semantic_loop)

    def protect(
        self,
        steps: Sequence[AgentStep],
    ) -> tuple[LoopVerdict, list[AgentStep]]:
        """Inspect ``steps`` and drop older copies of a repeated tool call.

        When a loop is present, ``intervention_prompt`` is the tool-lock nudge
        for the latest step.
        """

        verdict = self.inspect(steps)
        if not verdict.is_loop or not steps:
            return verdict, list(steps)
        nudge = self.mitigator.generate_nudge(verdict, steps[-1].tool_name)
        verdict = verdict.model_copy(update={"intervention_prompt": nudge["content"]})
        return verdict, self.mitigator.prune(steps)

    def __call__(
        self,
        steps: Sequence[AgentStep],
    ) -> tuple[LoopVerdict, list[AgentStep]]:
        """Run :meth:`protect` so the shield can be used as middleware."""

        return self.protect(steps)

    def evaluate(self, steps: Sequence[AgentStep]) -> LoopVerdict:
        """Judge a complete trajectory by its newest step, without recording it.

        A loop is reported only while the newest tool call is still inside the
        repeated cycle, so an agent that switches tools after a nudge is let
        through. Framework integrations call this on every message pipeline pass.
        """

        return self._verdict_for_latest(steps)

    def langchain_middleware(self, config: Any | None = None) -> Any:
        """Return a LangChain ``AgentMiddleware`` that screens model and tool calls.

        ``config`` is a :class:`loopshield.frameworks.IntegrationConfig` or a
        mapping of its fields. Pass the result in ``create_agent(middleware=[...])``.
        """

        from loopshield.frameworks import LoopShieldMiddleware

        return LoopShieldMiddleware(self, config)

    def crewai_agent(self) -> Callable[[Any], Any]:
        """Return a decorator for CrewAI's ``Agent.execute_task`` or an ``Agent`` class."""

        from loopshield.frameworks import crewai_agent

        return crewai_agent(self)

    def langgraph_node(
        self,
        node: Callable[..., Any] | None = None,
        config: Any | None = None,
    ) -> Any:
        """Wrap a LangGraph node so a flagged ``messages`` state gets the system nudge.

        Use ``@shield.langgraph_node()`` or ``shield.langgraph_node(node)``.
        """

        from loopshield.frameworks import langgraph_node

        return langgraph_node(self, node, config)

    def _maybe_prune(self, result: R) -> R:
        if self.max_tokens_estimate is None or not _is_message_list(result):
            return result
        return self.mitigator.prune_context(result, self.max_tokens_estimate)  # type: ignore[return-value]


def _assemble(graph_loop: bool, semantic_loop: bool) -> LoopVerdict:
    if graph_loop and semantic_loop:
        loop_type = "both"
    elif graph_loop:
        loop_type = "graph_cycle"
    elif semantic_loop:
        loop_type = "semantic"
    else:
        return LoopVerdict(is_loop=False, loop_type=None, intervention_prompt=None)
    return LoopVerdict(
        is_loop=True,
        loop_type=loop_type,
        intervention_prompt=_INTERVENTIONS[loop_type],
    )


def _find_step(args: tuple[Any, ...], kwargs: dict[str, Any]) -> AgentStep | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, AgentStep):
            return value
    return None


def _find_step_sequence(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[AgentStep] | None:
    for value in (*args, *kwargs.values()):
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, AgentStep) for item in value)
        ):
            return list(value)
    return None


def _is_message_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)
