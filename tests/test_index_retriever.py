from __future__ import annotations

from pathlib import Path

import pytest

from fake_openai_server import HashEmbeddings
from twin.core.retriever import Retriever, build_where, normalize_reply
from twin.core.schemas import ContextTurn, Pair
from twin.core.vector_store import ChromaVectorStore, IndexMismatchError
from twin.ingest.index import IndexExistsError, build_index, context_text_for_index


def pair(index: int, partner: str, reply: str, ts: int, eval_sample: bool = False) -> Pair:
    return Pair(
        pair_id=f"1002:{index}",
        chat_id=1002,
        conversation_id="1002:0",
        ts=ts,
        period="2024-03",
        context=[
            ContextTurn(sender_name="Злата", is_me=False, text="старая реплика"),
            ContextTurn(sender_name="Радомир", is_me=True, text="моё"),
            ContextTurn(sender_name="Злата", is_me=False, text=partner),
        ],
        reply=reply,
        reply_message_ids=[index],
        eval_sample=eval_sample,
    )


PAIRS = [
    pair(1, "пойдёшь гулять?", "Ок)", 100),
    pair(2, "пойдёшь гулять сегодня?", "ок", 200),  # near-identical reply -> deduplicated
    pair(3, "что делаешь", "ничего", 300),
    pair(4, "как дела", "норм", 400),
]


def test_context_text_uses_last_two_partner_turns() -> None:
    assert context_text_for_index(PAIRS[0]) == "старая реплика\nпойдёшь гулять?"
    only_me = PAIRS[0].model_copy(
        update={"context": [ContextTurn(sender_name="Р", is_me=True, text="x")]}
    )
    assert context_text_for_index(only_me) == "x"


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[ChromaVectorStore, HashEmbeddings]:
    store = ChromaVectorStore(tmp_path / "chroma")
    emb = HashEmbeddings()
    manifest = build_index(store, emb, PAIRS, "ds1", batch_size=3)
    assert manifest.count == 4 and manifest.embedded_text == "last_two_partner_turns"
    assert emb.calls == [("passage", 3), ("passage", 1)]  # fake identity needs no probe
    return store, emb


def test_build_index_guards(
    tmp_path: Path, indexed: tuple[ChromaVectorStore, HashEmbeddings]
) -> None:
    store, emb = indexed
    with pytest.raises(IndexExistsError):
        build_index(store, emb, PAIRS, "ds1")
    fresh = ChromaVectorStore(tmp_path / "other")
    with pytest.raises(ValueError, match="holdout"):
        build_index(fresh, emb, [pair(9, "x", "y", 1, eval_sample=True)], "ds1")


def test_retriever_filters_and_dedupes(indexed: tuple[ChromaVectorStore, HashEmbeddings]) -> None:
    store, emb = indexed
    retriever = Retriever(store, emb, k=8)
    examples = retriever.retrieve("пойдёшь гулять?", previous_partner_text="старая реплика")
    ids = [e.pair_id for e in examples]
    assert ids[0] == "1002:1" and "1002:2" not in ids  # dedupe keeps the best-ranked twin
    assert len(examples) == 3
    assert examples[0].reply == "Ок)" and examples[0].last_partner_text == "пойдёшь гулять?"
    assert examples[0].context[0].text == "старая реплика"

    older = retriever.retrieve("пойдёшь гулять?", ts_before=250)
    assert all(e.ts < 250 for e in older)
    assert len(older) == 1 and older[0].pair_id in {"1002:1", "1002:2"}  # the two dedupe to one

    without_target = retriever.retrieve("пойдёшь гулять?", exclude_pair_ids=["1002:1"])
    assert "1002:1" not in {e.pair_id for e in without_target}
    assert "1002:2" in {e.pair_id for e in without_target}

    assert len(retriever.retrieve("как дела", k=1)) == 1


def test_retriever_requires_matching_index(
    tmp_path: Path, indexed: tuple[ChromaVectorStore, HashEmbeddings]
) -> None:
    store, _emb = indexed
    with pytest.raises(IndexMismatchError, match="dim"):
        Retriever(store, HashEmbeddings(dim=4))
    with pytest.raises(IndexMismatchError, match="no index"):
        Retriever(ChromaVectorStore(tmp_path / "empty"), HashEmbeddings())


def test_where_and_normalize() -> None:
    assert build_where(None, None, []) is None
    assert build_where(5, None, []) == {"ts": {"$lt": 5}}
    assert build_where(5, 1, ["a"]) == {
        "$and": [{"ts": {"$lt": 5}}, {"chat_id": {"$eq": 1}}, {"pair_id": {"$nin": ["a"]}}]
    }
    assert normalize_reply("Ок)") == normalize_reply(" ок ") == "ок"
