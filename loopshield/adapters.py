"""Embedding adapters for clients the application already constructs.

The core package does not import OpenAI or Ollama. Pass an initialized client
and these wrappers call ``client.embeddings.create`` or the local Ollama
embedding API.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Protocol


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


class BaseEmbeddingAdapter(Protocol):
    """Any embedding backend that maps one string to a dense vector."""

    def embed(self, text: str) -> list[float]:
        """Return a dense vector for ``text``."""


class OpenAIEmbeddingsResource(Protocol):
    """The ``client.embeddings`` resource on an ``openai.OpenAI`` client."""

    def create(self, *, model: str, input: str, **kwargs: Any) -> Any:
        """Create an embedding for ``input``."""


class OpenAIClient(Protocol):
    """Initialized OpenAI client with an embeddings resource."""

    embeddings: OpenAIEmbeddingsResource


class OllamaEmbedClient(Protocol):
    """Ollama client that exposes the current ``embed`` call."""

    def embed(self, *, model: str, input: str, **kwargs: Any) -> Any:
        """Embed ``input`` with ``model``."""


class OllamaLegacyClient(Protocol):
    """Ollama client that still exposes the legacy ``embeddings`` call."""

    def embeddings(self, *, model: str, prompt: str, **kwargs: Any) -> Any:
        """Embed ``prompt`` with ``model``."""


class OpenAIEmbeddingAdapter:
    """Call ``client.embeddings.create`` and return the first vector."""

    def __init__(
        self,
        client: OpenAIClient,
        *,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    ) -> None:
        self._client = client
        self.model = _require_model(model)

    def embed(self, text: str) -> list[float]:
        """Embed ``text`` with the configured OpenAI model."""

        response = self._client.embeddings.create(model=self.model, input=text)
        return openai_embedding_vector(response)

    def __call__(self, text: str) -> list[float]:
        """Allow the adapter to be used anywhere an ``embed_fn`` is accepted."""

        return self.embed(text)


class OllamaEmbeddingAdapter:
    """Call a local Ollama client and return the first vector.

    Prefers ``client.embed(model=..., input=...)``. Falls back to the legacy
    ``client.embeddings(model=..., prompt=...)`` call when that is all the
    client provides.
    """

    def __init__(
        self,
        client: OllamaEmbedClient | OllamaLegacyClient,
        *,
        model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL,
    ) -> None:
        self._client = client
        self.model = _require_model(model)

    def embed(self, text: str) -> list[float]:
        """Embed ``text`` with the configured local model."""

        embed = getattr(self._client, "embed", None)
        if callable(embed):
            return ollama_embedding_vector(embed(model=self.model, input=text))
        embeddings = getattr(self._client, "embeddings", None)
        if callable(embeddings):
            return ollama_embedding_vector(embeddings(model=self.model, prompt=text))
        raise TypeError("Ollama client must provide embed() or embeddings().")

    def __call__(self, text: str) -> list[float]:
        """Allow the adapter to be used anywhere an ``embed_fn`` is accepted."""

        return self.embed(text)


def as_embed_fn(source: Any, *, model: str | None = None) -> Callable[[str], list[float]]:
    """Turn an adapter, embed callable, or OpenAI/Ollama client into an embed function.

    ``model`` is used only when ``source`` is a raw client. Adapters already
    carry their model.
    """

    if _is_text_embedding_adapter(source):
        if model is not None:
            raise ValueError("Set the model on the adapter, or pass the client with embedding_model.")
        if callable(source):
            return source
        return source.embed
    if _is_openai_client(source):
        return OpenAIEmbeddingAdapter(source, model=model or DEFAULT_OPENAI_EMBEDDING_MODEL)
    if _is_ollama_client(source):
        return OllamaEmbeddingAdapter(source, model=model or DEFAULT_OLLAMA_EMBEDDING_MODEL)
    if callable(source):
        if model is not None:
            raise ValueError("embedding_model applies to an OpenAI or Ollama client, not a plain embed function.")
        return source
    raise TypeError(
        "embedding must be an embed callable, an embedding adapter, or an initialized OpenAI or Ollama client."
    )


def openai_embedding_vector(response: Any) -> list[float]:
    """Read the first vector from an OpenAI embeddings response."""

    data = response["data"] if isinstance(response, Mapping) else getattr(response, "data", None)
    if not data:
        raise ValueError("OpenAI embedding response did not include any vectors.")
    first = data[0]
    embedding = first["embedding"] if isinstance(first, Mapping) else getattr(first, "embedding", None)
    return _as_floats(embedding)


def ollama_embedding_vector(response: Any) -> list[float]:
    """Read the first vector from a current or legacy Ollama embeddings response."""

    if isinstance(response, Mapping):
        if "embeddings" in response:
            return _first_vector(response["embeddings"], provider="Ollama")
        if "embedding" in response:
            return _as_floats(response["embedding"])
    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        return _first_vector(embeddings, provider="Ollama")
    embedding = getattr(response, "embedding", None)
    if embedding:
        return _as_floats(embedding)
    raise ValueError("Ollama embedding response did not include a vector.")


def _require_model(model: str) -> str:
    cleaned = model.strip()
    if not cleaned:
        raise ValueError("model must be a non-empty string")
    return cleaned


def _as_floats(values: Any) -> list[float]:
    if values is None:
        raise ValueError("Embedding response did not include a vector.")
    try:
        vector = [float(item) for item in values]
    except TypeError as exc:
        raise ValueError("Embedding response did not contain a numeric vector.") from exc
    if not vector:
        raise ValueError("Embedding response was empty.")
    return vector


def _first_vector(vectors: Any, *, provider: str) -> list[float]:
    try:
        first = vectors[0]
    except (IndexError, TypeError, KeyError) as exc:
        raise ValueError(f"{provider} embedding response did not include any vectors.") from exc
    return _as_floats(first)


def _is_text_embedding_adapter(source: object) -> bool:
    embed = getattr(source, "embed", None)
    if not callable(embed):
        return False
    try:
        signature = inspect.signature(embed)
    except (TypeError, ValueError):
        return False
    parameters = [parameter for name, parameter in signature.parameters.items() if name != "self"]
    if not parameters:
        return False
    first = parameters[0]
    return first.name == "text" and first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def _is_openai_client(source: object) -> bool:
    embeddings = getattr(source, "embeddings", None)
    if embeddings is None or callable(embeddings):
        return False
    return callable(getattr(embeddings, "create", None))


def _is_ollama_client(source: object) -> bool:
    if _is_openai_client(source):
        return False
    if callable(getattr(source, "embed", None)):
        return True
    return callable(getattr(source, "embeddings", None))
