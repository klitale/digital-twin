"""Retrieve real replies similar to the incoming message, with leakage guards.

Filters: an upper time bound (never show the future during evaluation), excluded pair
ids (never the target reply) and an optional chat. Near-identical replies are
de-duplicated so eight examples are eight different reactions. Retrieved ids are
logged with every call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from twin.core.embeddings import EmbeddingProvider
from twin.core.schemas import ContextTurn
from twin.core.vector_store import IndexMismatchError, VectorStore
from twin.logsetup import get_logger

log = get_logger("twin.retriever")
_NORMALIZE_RE = re.compile(r"[\W_]+")


@dataclass(frozen=True)
class RetrievedExample:
    pair_id: str
    chat_id: int
    ts: int
    context_text: str
    reply: str
    distance: float
    context: list[ContextTurn]

    @property
    def last_partner_text(self) -> str:
        for turn in reversed(self.context):
            if not turn.is_me:
                return turn.text
        return self.context_text


def normalize_reply(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.lower())


def build_where(
    ts_before: int | None, chat_id: int | None, exclude_pair_ids: Sequence[str]
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if ts_before is not None:
        clauses.append({"ts": {"$lt": ts_before}})
    if chat_id is not None:
        clauses.append({"chat_id": {"$eq": chat_id}})
    if exclude_pair_ids:
        clauses.append({"pair_id": {"$nin": list(exclude_pair_ids)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingProvider,
        k: int = 8,
        overfetch: int = 4,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.k = k
        self.overfetch = overfetch
        manifest = store.read_manifest()
        if manifest is None:
            raise IndexMismatchError("no index manifest found; run `twin index` first")
        manifest.check(embeddings.identity)
        self.manifest = manifest

    def retrieve(
        self,
        incoming: str,
        previous_partner_text: str | None = None,
        chat_id: int | None = None,
        ts_before: int | None = None,
        exclude_pair_ids: Sequence[str] = (),
        k: int | None = None,
    ) -> list[RetrievedExample]:
        k = self.k if k is None else k
        query = "\n".join(part for part in (previous_partner_text, incoming) if part)
        vector = self.embeddings.embed([query], "query")[0]
        where = build_where(ts_before, chat_id, exclude_pair_ids)
        hits = self.store.query(vector, k * self.overfetch, where)
        excluded = set(exclude_pair_ids)
        seen: set[str] = set()
        examples: list[RetrievedExample] = []
        for hit in hits:
            meta = hit.metadata
            if hit.id in excluded or (ts_before is not None and int(meta["ts"]) >= ts_before):
                continue
            key = normalize_reply(str(meta["reply"]))
            if key in seen:
                continue
            seen.add(key)
            context = [ContextTurn.model_validate(t) for t in json.loads(meta["context_json"])]
            examples.append(
                RetrievedExample(
                    pair_id=hit.id,
                    chat_id=int(meta["chat_id"]),
                    ts=int(meta["ts"]),
                    context_text=hit.document,
                    reply=str(meta["reply"]),
                    distance=hit.distance,
                    context=context,
                )
            )
            if len(examples) >= k:
                break
        log.info(
            "retrieval",
            k=k,
            hits=len(hits),
            returned=len(examples),
            ts_before=ts_before,
            excluded=len(excluded),
            retrieved_ids=[e.pair_id for e in examples],
        )
        return examples
