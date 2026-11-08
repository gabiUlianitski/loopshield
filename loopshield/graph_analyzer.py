"""Directed-graph detection of repeating tool-call cycles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import networkx as nx

from loopshield.models import AgentStep


def tool_signature(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Identify a tool call by its name and normalized arguments."""

    return f"{tool_name.strip()}|{normalize_tool_args(tool_args)}"


def normalize_tool_args(tool_args: dict[str, Any]) -> str:
    """Serialize arguments so key order and string formatting do not matter."""

    return json.dumps(
        _normalize(dict(tool_args)),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class GraphAnalyzer:
    """Track sequential tool calls and flag a cycle that repeats too often.

    Each step becomes a node labeled ``tool_name`` plus normalized ``tool_args``.
    Consecutive steps become directed edges. A cycle counts once each time that
    exact tool sequence is walked and returns to its start. ``detect`` returns
    True when that count is greater than ``max_cycles``.

    With the default ``max_cycles=1``, one completed lap is allowed and the
    second lap is a loop. ``max_cycles=0`` flags the first time a cycle closes.
    """

    def __init__(self, max_cycles: int = 1) -> None:
        if max_cycles < 0:
            raise ValueError("max_cycles must be zero or greater")
        self.max_cycles = max_cycles

    def build_graph(self, steps: Sequence[AgentStep]) -> nx.DiGraph:
        """Return a directed graph of successive tool calls.

        Edge ``weight`` is how many times that transition occurred.
        """

        graph: nx.DiGraph = nx.DiGraph()
        signatures = [_signature(step) for step in steps]
        for signature in signatures:
            graph.add_node(signature)
        for source, target in zip(signatures, signatures[1:], strict=False):
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
            else:
                graph.add_edge(source, target, weight=1)
        return graph

    def repeat_count(self, steps: Sequence[AgentStep]) -> int:
        """Return how many times the most-repeated exact tool cycle completed."""

        if len(steps) < 2:
            return 0
        graph = self.build_graph(steps)
        signatures = [_signature(step) for step in steps]
        counts = [
            _completion_count(signatures, cycle) for cycle in _directed_cycles(graph)
        ]
        return max(counts, default=0)

    def detect(self, steps: Sequence[AgentStep]) -> bool:
        """Return True when one exact tool cycle repeats more than ``max_cycles`` times."""

        return self.repeat_count(steps) > self.max_cycles

    def repeated_signatures(self, steps: Sequence[AgentStep]) -> set[str]:
        """Return tool signatures that sit on a cycle repeated more than ``max_cycles`` times."""

        if len(steps) < 2:
            return set()
        graph = self.build_graph(steps)
        signatures = [_signature(step) for step in steps]
        locked: set[str] = set()
        for cycle in _directed_cycles(graph):
            if _completion_count(signatures, cycle) > self.max_cycles:
                locked.update(cycle)
        return locked


def _signature(step: AgentStep) -> str:
    return tool_signature(step.tool_name, step.tool_args)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _directed_cycles(graph: nx.DiGraph) -> list[tuple[str, ...]]:
    """Yield simple directed cycles, including self-loops."""

    cycles = [tuple(str(node) for node in cycle) for cycle in nx.simple_cycles(graph)]
    for node in graph.nodes:
        if graph.has_edge(node, node):
            loop = (str(node),)
            if loop not in cycles:
                cycles.append(loop)
    return cycles


def _completion_count(signatures: Sequence[str], cycle: tuple[str, ...]) -> int:
    """Count contiguous walks of ``cycle`` that return to the start node.

    Rotations describe the same directed cycle. The maximum is used so one lap
    is not counted once per starting node.
    """

    if not cycle:
        return 0
    best = 0
    for rotation in _rotations(cycle):
        needed = (*rotation, rotation[0])
        width = len(needed)
        matches = sum(
            1
            for index in range(len(signatures) - width + 1)
            if tuple(signatures[index : index + width]) == needed
        )
        best = max(best, matches)
    return best


def _rotations(cycle: tuple[str, ...]) -> list[tuple[str, ...]]:
    if len(cycle) <= 1:
        return [cycle]
    return [cycle[index:] + cycle[:index] for index in range(len(cycle))]
