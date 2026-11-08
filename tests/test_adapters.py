"""Client wrappers for OpenAI and Ollama embedding calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from loopshield import AgentStep, LoopShield, OllamaEmbeddingAdapter, OpenAIEmbeddingAdapter
from loopshield.adapters import as_embed_fn


class _OpenAIClient:
    def __init__(self) -> None:
        self.models: list[str] = []
        self.inputs: list[str] = []
        self.embeddings = self

    def create(self, *, model: str, input: str) -> SimpleNamespace:
        self.models.append(model)
        self.inputs.append(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(input)), 1.0, 0.0])])


class _OllamaClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def embed(self, *, model: str, input: str) -> dict[str, list[list[float]]]:
        self.calls.append((model, input))
        return {"embeddings": [[0.0, float(len(input)), 1.0]]}


class _LegacyOllamaClient:
    def embeddings(self, *, model: str, prompt: str) -> dict[str, list[float]]:
        self.model = model
        self.prompt = prompt
        return {"embedding": [0.4, 0.5, 0.6]}


def test_openai_adapter_calls_embeddings_create() -> None:
    client = _OpenAIClient()

    vector = OpenAIEmbeddingAdapter(client).embed("config")

    assert client.models == ["text-embedding-3-small"]
    assert client.inputs == ["config"]
    assert vector == [6.0, 1.0, 0.0]


def test_openai_adapter_reads_a_dictionary_response() -> None:
    class DictionaryClient:
        embeddings = SimpleNamespace(
            create=lambda *, model, input: {"data": [{"embedding": [1.0, 2.0]}]}
        )

    assert OpenAIEmbeddingAdapter(DictionaryClient(), model="text-embedding-3-large").embed("x") == [
        1.0,
        2.0,
    ]


def test_ollama_adapter_calls_embed() -> None:
    client = _OllamaClient()

    vector = OllamaEmbeddingAdapter(client, model="nomic-embed-text").embed("plan")

    assert client.calls == [("nomic-embed-text", "plan")]
    assert vector == [0.0, 4.0, 1.0]


def test_legacy_ollama_adapter_calls_embeddings() -> None:
    client = _LegacyOllamaClient()

    vector = OllamaEmbeddingAdapter(client).embed("plan")

    assert client.model == "nomic-embed-text"
    assert client.prompt == "plan"
    assert vector == [0.4, 0.5, 0.6]


def test_shield_accepts_an_openai_client_directly() -> None:
    client = _OpenAIClient()
    shield = LoopShield(embedding=client, embedding_model="text-embedding-3-small", max_semantic_repeats=1)
    steps = [
        AgentStep(thought="Retry the failing plan", tool_name="reason", tool_args={"n": 1}, tokens_used=1),
        AgentStep(thought="Rephrase the failing plan", tool_name="reason", tool_args={"n": 2}, tokens_used=1),
    ]

    verdict = shield.inspect(steps)

    assert verdict.is_loop is True
    assert verdict.loop_type == "semantic"
    assert client.inputs == ["Retry the failing plan", "Rephrase the failing plan"]


def test_shield_accepts_an_ollama_client_directly() -> None:
    client = _OllamaClient()
    shield = LoopShield(embedding=client, max_semantic_repeats=1)

    verdict = shield.step(
        AgentStep(thought="first plan", tool_name="reason", tool_args={"n": 1}, tokens_used=1)
    )
    assert verdict.is_loop is False
    caught = shield.step(
        AgentStep(thought="same plan again", tool_name="reason", tool_args={"n": 2}, tokens_used=1)
    )

    assert caught.loop_type == "semantic"
    assert [call[0] for call in client.calls] == ["nomic-embed-text", "nomic-embed-text"]


def test_embedding_and_embed_fn_cannot_both_be_set() -> None:
    with pytest.raises(ValueError, match="embed_fn or embedding"):
        LoopShield(embed_fn=lambda text: [1.0], embedding=_OpenAIClient())


def test_as_embed_fn_rejects_an_unknown_object() -> None:
    with pytest.raises(TypeError, match="OpenAI or Ollama"):
        as_embed_fn(object())
