"""Vector store Protocol with a persistent ChromaDB implementation.

The store directory carries ``index_manifest.json`` describing which embedding
provider/model/dimension and dataset built it; opening the index with a different
embedding identity fails loudly instead of returning garbage neighbours.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from twin.core.embeddings import EmbeddingIdentity

MANIFEST_FILE = "index_manifest.json"
COLLECTION = "pairs"
Metadata = Mapping[str, str | int | float | bool]


class IndexMismatchError(RuntimeError):
    """The index was built with another embedding provider, model or dimension."""


class IndexManifest(BaseModel):
    provider: str
    model: str
    dimension: int
    dataset_version: str
    embedded_text: str = Field(description="What was embedded (rule name).")
    count: int
    built_at: str

    def check(self, identity: EmbeddingIdentity) -> None:
        if (self.provider, self.model, self.dimension) != (
            identity.provider,
            identity.model,
            identity.dimension,
        ):
            raise IndexMismatchError(
                f"index built with {self.provider}/{self.model} (dim {self.dimension}), "
                f"query uses {identity.provider}/{identity.model} (dim {identity.dimension})"
            )


@dataclass(frozen=True)
class Hit:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float


class VectorStore(Protocol):
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Metadata],
    ) -> None: ...

    def query(
        self, embedding: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[Hit]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def read_manifest(self) -> IndexManifest | None: ...

    def write_manifest(self, manifest: IndexManifest) -> None: ...


def write_index_manifest(path: Path, manifest: IndexManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_index_manifest(path: Path) -> IndexManifest | None:
    if not path.is_file():
        return None
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def new_manifest(
    identity: EmbeddingIdentity, dataset_version: str, embedded_text: str, count: int
) -> IndexManifest:
    return IndexManifest(
        provider=identity.provider,
        model=identity.model,
        dimension=identity.dimension,
        dataset_version=dataset_version,
        embedded_text=embedded_text,
        count=count,
        built_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


class ChromaVectorStore:
    def __init__(self, directory: Path) -> None:
        import chromadb

        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(directory),
            settings=chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_FILE

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Metadata],
    ) -> None:
        if not ids:
            return
        self._collection.add(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],  # type: ignore[arg-type]
            documents=list(documents),
            metadatas=[dict(m) for m in metadatas],  # type: ignore[arg-type]
        )

    def query(
        self, embedding: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[Hit]:
        if k <= 0 or self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(embedding)],  # type: ignore[arg-type]
            n_results=min(k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
        )
        ids = result["ids"][0]
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        return [
            Hit(
                id=hit_id,
                document=documents[i] if i < len(documents) else "",
                metadata=dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
                distance=float(distances[i]) if i < len(distances) else 0.0,
            )
            for i, hit_id in enumerate(ids)
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def read_manifest(self) -> IndexManifest | None:
        return read_index_manifest(self.manifest_path)

    def write_manifest(self, manifest: IndexManifest) -> None:
        write_index_manifest(self.manifest_path, manifest)


def metadata_json(value: Any) -> str:
    """Chroma metadata holds scalars only; nested context travels as a JSON string."""
    return json.dumps(value, ensure_ascii=False)
