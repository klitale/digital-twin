"""Embedding providers behind one Protocol.

``openai_compatible`` (default) calls the gateway's ``/v1/embeddings``; ``local_e5``
runs ``intfloat/multilingual-e5-base`` through sentence-transformers and applies the
``query:`` / ``passage:`` prefixes the model expects. Every provider reports the
identity (``provider``, ``model``, ``dimension``) that the index manifest records.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import openai
from pydantic import SecretStr

from twin.config import EmbedProvider, Settings
from twin.logsetup import get_logger

log = get_logger("twin.embeddings")

TextKind = Literal["query", "passage"]
E5_MODEL = "intfloat/multilingual-e5-base"


class EmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    dimension: int


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> EmbeddingIdentity: ...

    def embed(self, texts: Sequence[str], kind: TextKind) -> list[list[float]]: ...


class OpenAICompatibleEmbeddings:
    """``/v1/embeddings`` on an OpenAI-compatible gateway; no prefixes."""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | str,
        model: str,
        batch_size: int = 10,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self._client = openai.OpenAI(
            base_url=base_url, api_key=key, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.batch_size = batch_size
        self._dimension: int | None = None

    @property
    def identity(self) -> EmbeddingIdentity:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"], "query")[0])
        return EmbeddingIdentity("openai_compatible", self.model, self._dimension)

    def embed(self, texts: Sequence[str], kind: TextKind) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
            except openai.OpenAIError as exc:
                raise EmbeddingError(f"{self.model}: {exc}") from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise EmbeddingError(f"{self.model}: {len(ordered)} vectors for {len(batch)} texts")
            vectors.extend([list(map(float, item.embedding)) for item in ordered])
            usage = response.usage
            log.debug(
                "embeddings.batch",
                model=self.model,
                texts=len(batch),
                prompt_tokens=usage.prompt_tokens if usage else None,
            )
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors


Encoder = Callable[[list[str]], Any]


class LocalE5Embeddings:
    """multilingual-e5 via sentence-transformers (optional extra ``local-embed``)."""

    def __init__(self, model: str = E5_MODEL, encoder: Encoder | None = None) -> None:
        self.model = model
        self._encoder = encoder
        self._dimension: int | None = None

    def _load(self) -> Encoder:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the optional extra
                raise EmbeddingError(
                    "local_e5 needs `uv sync --extra local-embed` (sentence-transformers)"
                ) from exc
            transformer = SentenceTransformer(self.model)
            self._encoder = lambda texts: transformer.encode(texts, normalize_embeddings=True)
        return self._encoder

    @staticmethod
    def prefixed(texts: Sequence[str], kind: TextKind) -> list[str]:
        return [f"{kind}: {t}" for t in texts]

    @property
    def identity(self) -> EmbeddingIdentity:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"], "query")[0])
        return EmbeddingIdentity("local_e5", self.model, self._dimension)

    def embed(self, texts: Sequence[str], kind: TextKind) -> list[list[float]]:
        raw = self._load()(self.prefixed(texts, kind))
        vectors = [list(map(float, row)) for row in raw]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors


class QueryCache:
    """Remembers the last query vectors, so the examples and the statements retrievers
    embed one incoming message once instead of twice."""

    def __init__(self, inner: EmbeddingProvider) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self._key: tuple[str, ...] | None = None
        self._vectors: list[list[float]] = []

    @property
    def identity(self) -> EmbeddingIdentity:
        return self.inner.identity

    def embed(self, texts: Sequence[str], kind: TextKind) -> list[list[float]]:
        if kind != "query":
            return self.inner.embed(texts, kind)
        key = tuple(texts)
        with self._lock:
            if key != self._key:
                self._vectors = self.inner.embed(texts, kind)
                self._key = key
            return [list(vector) for vector in self._vectors]


def embeddings_from_settings(settings: Settings) -> EmbeddingProvider:
    if settings.embed_provider is EmbedProvider.LOCAL_E5:
        return LocalE5Embeddings(settings.embed_model or E5_MODEL)
    settings.require_embeddings()
    return OpenAICompatibleEmbeddings(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key_effective,  # type: ignore[arg-type]
        model=settings.embed_model,
        batch_size=settings.embed_batch_size,
    )
