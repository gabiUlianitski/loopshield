"""Cosine-similarity detection of repeating agent thoughts."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from loopshield.models import AgentStep

EmbedFn = Callable[[str], list[float]]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""

    return _TOKEN.findall(text.casefold())


def tfidf_embed(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts with a shared bag-of-words TF-IDF vocabulary.

    This is the lightweight fallback used when no embedding model is injected.
    Identical thoughts land on the same vector. A text with no tokens becomes
    a one-dimensional zero vector.
    """

    documents = [tokenize(text) for text in texts]
    vocabulary: dict[str, int] = {}
    for tokens in documents:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    if not vocabulary:
        return [[0.0] for _ in texts]

    document_frequency = [0] * len(vocabulary)
    for tokens in documents:
        for token in set(tokens):
            document_frequency[vocabulary[token]] += 1

    document_count = len(documents)
    vectors: list[list[float]] = []
    for tokens in documents:
        counts = Counter(tokens)
        length = len(tokens) or 1
        vector = [0.0] * len(vocabulary)
        for token, count in counts.items():
            term_frequency = count / length
            inverse_document_frequency = math.log(
                (1.0 + document_count) / (1.0 + document_frequency[vocabulary[token]])
            ) + 1.0
            vector[vocabulary[token]] = term_frequency * inverse_document_frequency
        vectors.append(vector)
    return vectors


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, clipped to [-1, 1]."""

    left_vector = np.asarray(left, dtype=np.float64).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_vector.size == 0 or right_vector.size == 0:
        raise ValueError("Embedding must contain at least one dimension.")
    if left_vector.shape != right_vector.shape:
        raise ValueError(
            "Embedding dimensions differ: "
            f"{left_vector.shape[0]} and {right_vector.shape[0]}"
        )
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0.0:
        return 0.0
    score = float(np.dot(left_vector, right_vector) / denominator)
    if not math.isfinite(score):
        return 0.0
    return float(np.clip(score, -1.0, 1.0))


class SemanticAnalyzer:
    """Flag a thought that keeps matching the consecutive thoughts before it.

    Pass ``embed_fn`` to plug in any model that maps text to a vector. When it
    is omitted, thoughts in the call are embedded together with TF-IDF
    bag-of-words.

    Walking backward from the current thought, each recent thought whose cosine
    similarity exceeds ``similarity_threshold`` extends a streak. The streak
    stops at the first thought that does not exceed it. A semantic loop is
    flagged when that streak is at least ``max_semantic_repeats`` long.
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        *,
        similarity_threshold: float = 0.92,
        max_semantic_repeats: int = 2,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if max_semantic_repeats < 1:
            raise ValueError("max_semantic_repeats must be at least 1")
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_semantic_repeats = max_semantic_repeats

    def similarity_streak(self, steps: Sequence[AgentStep]) -> int:
        """Return how many consecutive thoughts before the latest one exceed the threshold.

        Comparison starts at the current thought and walks backward. The first
        recent thought that does not exceed the threshold ends the streak.
        """

        if len(steps) < 2:
            return 0
        vectors = self._embed([step.thought for step in steps])
        streak = 0
        current = vectors[-1]
        for previous in range(len(vectors) - 2, -1, -1):
            similarity = cosine_similarity(current, vectors[previous])
            if similarity > self.similarity_threshold:
                streak += 1
            else:
                break
        return streak

    def detect(self, steps: Sequence[AgentStep]) -> bool:
        """Return True when a similar streak reaches ``max_semantic_repeats``."""

        return self.similarity_streak(steps) >= self.max_semantic_repeats

    def _embed(self, thoughts: Sequence[str]) -> list[np.ndarray]:
        if self.embed_fn is None:
            embedded = tfidf_embed(thoughts)
        else:
            embedded = [self.embed_fn(thought) for thought in thoughts]
        return [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in embedded]
