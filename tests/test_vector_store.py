from __future__ import annotations

from pathlib import Path

import pytest

from fake_openai_server import HashEmbeddings
from twin.core.embeddings import EmbeddingIdentity
from twin.core.vector_store import ChromaVectorStore, IndexMismatchError, new_manifest


def test_chroma_add_query_filter_reset(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma")
    emb = HashEmbeddings()
    docs = ["привет как дела", "пойдём гулять", "что делаешь"]
    store.add(
        ids=["a", "b", "c"],
        embeddings=emb.embed(docs, "passage"),
        documents=docs,
        metadatas=[
            {"ts": 10, "pair_id": "a"},
            {"ts": 20, "pair_id": "b"},
            {"ts": 30, "pair_id": "c"},
        ],
    )
    assert store.count() == 3
    hits = store.query(emb.embed(["привет как дела"], "query")[0], k=3)
    assert hits[0].id == "a" and hits[0].distance < hits[1].distance
    assert hits[0].document == docs[0] and hits[0].metadata["ts"] == 10
    older = store.query(emb.embed(["привет"], "query")[0], k=3, where={"ts": {"$lt": 25}})
    assert {h.id for h in older} == {"a", "b"}
    without = store.query(
        emb.embed(["привет"], "query")[0],
        k=3,
        where={"$and": [{"ts": {"$lt": 35}}, {"pair_id": {"$nin": ["a"]}}]},
    )
    assert {h.id for h in without} == {"b", "c"}
    assert store.query([0.0] * 8, k=0) == []

    manifest = new_manifest(emb.identity, "ds1", "rule", 3)
    store.write_manifest(manifest)
    assert store.read_manifest() == manifest
    store.reset()
    assert store.count() == 0 and store.read_manifest() is None

    reopened = ChromaVectorStore(tmp_path / "chroma")
    assert reopened.count() == 0


def test_manifest_check_rejects_other_embeddings() -> None:
    manifest = new_manifest(EmbeddingIdentity("openai_compatible", "m", 1024), "ds", "rule", 1)
    manifest.check(EmbeddingIdentity("openai_compatible", "m", 1024))
    for other in (
        EmbeddingIdentity("local_e5", "m", 1024),
        EmbeddingIdentity("openai_compatible", "other", 1024),
        EmbeddingIdentity("openai_compatible", "m", 768),
    ):
        with pytest.raises(IndexMismatchError):
            manifest.check(other)
