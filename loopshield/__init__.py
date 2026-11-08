"""LoopShield: loop detection and self-healing middleware for AI agents."""

from loopshield.adapters import BaseEmbeddingAdapter, OllamaEmbeddingAdapter, OpenAIEmbeddingAdapter
from loopshield.graph_analyzer import GraphAnalyzer
from loopshield.mitigator import Mitigator, generate_nudge, prune_context
from loopshield.models import AgentStep, LoopVerdict
from loopshield.semantic_analyzer import SemanticAnalyzer
from loopshield.shield import LoopShield

__version__ = "0.1.0"

__all__ = [
    "AgentStep",
    "BaseEmbeddingAdapter",
    "GraphAnalyzer",
    "LoopShield",
    "LoopVerdict",
    "Mitigator",
    "OllamaEmbeddingAdapter",
    "OpenAIEmbeddingAdapter",
    "SemanticAnalyzer",
    "generate_nudge",
    "prune_context",
    "__version__",
]
