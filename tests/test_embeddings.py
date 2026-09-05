from __future__ import annotations

import pytest

from fake_openai_server import FakeOpenAIServer
from twin.config import EmbedProvider, Settings
from twin.core.embeddings import (
    EmbeddingError,
    EmbeddingIdentity,
    LocalE5Embeddings,
    OpenAICompatibleEmbeddings,
    embeddings_from_settings,
)


def test_openai_compatible_batches_and_reports_identity() -> None:
    with FakeOpenAIServer(embedding_dim=6) as server:
        provider = OpenAICompatibleEmbeddings(server.base_url, "k", "m", batch_size=2)
        vectors = provider.embed(["а", "б", "в"], "passage")
        assert provider.identity == EmbeddingIdentity("openai_compatible", "m", 6)
        assert vectors[0] == provider.embed(["а"], "query")[0]  # deterministic, no prefixes
    assert len(vectors) == 3 and all(len(v) == 6 for v in vectors)
    assert [len(r["input"]) for r in server.requests[:2]] == [2, 1]


def test_openai_compatible_replaces_empty_text_and_wraps_errors() -> None:
    with FakeOpenAIServer() as server:
        provider = OpenAICompatibleEmbeddings(server.base_url, "k", "m")
        provider.embed(["", "x"], "query")
        assert server.requests[0]["input"] == [" ", "x"]
    with FakeOpenAIServer(fail_times=5) as server:
        provider = OpenAICompatibleEmbeddings(server.base_url, "k", "m", max_retries=0)
        with pytest.raises(EmbeddingError):
            provider.embed(["x"], "query")


def test_local_e5_applies_prefixes() -> None:
    seen: list[list[str]] = []

    def encoder(texts: list[str]) -> list[list[float]]:
        seen.append(texts)
        return [[1.0, 0.0] for _ in texts]

    provider = LocalE5Embeddings(encoder=encoder)
    provider.embed(["вопрос"], "query")
    provider.embed(["ответ"], "passage")
    assert seen == [["query: вопрос"], ["passage: ответ"]]
    assert provider.identity.provider == "local_e5" and provider.identity.dimension == 2


def test_embeddings_from_settings() -> None:
    settings = Settings(_env_file=None, llm_api_key="k")  # type: ignore[call-arg]
    assert isinstance(embeddings_from_settings(settings), OpenAICompatibleEmbeddings)
    local = Settings(_env_file=None, embed_provider=EmbedProvider.LOCAL_E5)  # type: ignore[call-arg]
    assert isinstance(embeddings_from_settings(local), LocalE5Embeddings)
    with pytest.raises(Exception, match="EMBED_API_KEY"):
        embeddings_from_settings(Settings(_env_file=None))  # type: ignore[call-arg]
