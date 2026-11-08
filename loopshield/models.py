"""Pydantic schemas for agent steps and loop verdicts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentStep(BaseModel):
    """One tool-using step in an agent trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thought: str
    tool_name: str = Field(min_length=1)
    tool_args: dict[str, Any]
    tokens_used: int = Field(ge=0)


class LoopVerdict(BaseModel):
    """Result of checking a trajectory for a repeating loop.

    ``loop_type`` is ``"graph_cycle"``, ``"semantic"``, or ``"both"`` when a
    loop is present. ``intervention_prompt`` is the nudge to send next.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_loop: bool
    loop_type: str | None = None
    intervention_prompt: str | None = None
