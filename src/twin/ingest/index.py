"""Build the retrieval index from training pairs (never from holdout).

Each pair is stored once: the embedded document is the context side (the last two
partner turns), the reply and the full context travel in the metadata. The index
manifest records the embedding identity and the dataset version.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from twin.core.embeddings import EmbeddingProvider
from twin.core.schemas import Pair
from twin.core.vector_store import IndexManifest, VectorStore, metadata_json, new_manifest

EMBEDDED_TEXT_RULE = "last_two_partner_turns"


class IndexExistsError(RuntimeError):
    """The store already holds an index; pass ``--rebuild`` to replace it."""


def context_text_for_index(pair: Pair) -> str:
    partner_turns = [turn.text for turn in pair.context if not turn.is_me]
    if not partner_turns:
        partner_turns = [turn.text for turn in pair.context]
    return "\n".join(partner_turns[-2:])


def pair_metadata(pair: Pair) -> dict[str, str | int | float | bool]:
    return {
        "pair_id": pair.pair_id,
        "chat_id": pair.chat_id,
        "conversation_id": pair.conversation_id,
        "ts": pair.ts,
        "period": pair.period,
        "reply": pair.reply,
        "context_json": metadata_json([turn.model_dump() for turn in pair.context]),
    }


def build_index(
    store: VectorStore,
    embeddings: EmbeddingProvider,
    pairs: Sequence[Pair],
    dataset_version: str,
    batch_size: int = 10,
    progress: bool = False,
    workers: int = 1,
) -> IndexManifest:
    """Embed in batches (``workers`` batches in flight), add to the store in order."""
    if store.count() > 0:
        raise IndexExistsError(f"index already holds {store.count()} records; use --rebuild")
    if any(pair.eval_sample for pair in pairs):
        raise ValueError("refusing to index pairs flagged as evaluation sample (holdout leak)")
    identity = embeddings.identity
    batches = [pairs[start : start + batch_size] for start in range(0, len(pairs), batch_size)]
    documents_per_batch = [[context_text_for_index(p) for p in batch] for batch in batches]

    def embed(documents: list[str]) -> list[list[float]]:
        return embeddings.embed(documents, "passage")

    total = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        vectors_iter = pool.map(embed, documents_per_batch)
        if progress:
            from tqdm import tqdm

            vectors_iter = tqdm(vectors_iter, total=len(batches), desc="index", unit="batch")
        for batch, documents, vectors in zip(
            batches, documents_per_batch, vectors_iter, strict=True
        ):
            store.add(
                ids=[p.pair_id for p in batch],
                embeddings=vectors,
                documents=documents,
                metadatas=[pair_metadata(p) for p in batch],
            )
            total += len(batch)
    manifest = new_manifest(identity, dataset_version, EMBEDDED_TEXT_RULE, total)
    store.write_manifest(manifest)
    return manifest
