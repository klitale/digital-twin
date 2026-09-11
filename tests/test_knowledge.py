from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fake_openai_server import FakeOpenAIServer, HashEmbeddings
from test_eval import HOLDOUT, JUDGE_JSON
from test_index_retriever import PAIRS
from test_index_retriever import pair as make_pair
from twin.core.backends import GenerationRequest, GenerationResult, RagBackend
from twin.core.embeddings import QueryCache
from twin.core.llm_client import LLMClient, LLMError
from twin.core.memory import MemoryTurn
from twin.core.prompt import build_rag_messages, render_knowledge, uses_knowledge
from twin.core.prompts import load_prompt
from twin.core.retriever import RetrievedExample, Retriever
from twin.core.validate import split_plan, validate_reply
from twin.core.vector_store import STATEMENTS_COLLECTION, STATEMENTS_MANIFEST, ChromaVectorStore
from twin.eval.duel import DuelJudge, DuelVote, combine, parse_duel_json, run_duel, sign_test_p
from twin.eval.harness import AuditingRetriever, EvalConfig, LeakageError, run_eval
from twin.eval.judge import Judge
from twin.eval.schemas import EvalRecord, EvalRun, RunMetadata, RunSummary
from twin.ingest.dossier import (
    DossierExistsError,
    generate_dossier,
    informative_pairs,
    read_dossier,
    write_dossier,
)
from twin.ingest.index import STATEMENTS_RULE, build_index, reply_text_for_index, statement_pairs

LONG = [
    make_pair(5, "как выходные?", "ездил с братом на рыбалку, поймали трёх щук", 500),
    make_pair(6, "что с учёбой", "завалил матан, пересдача в июне", 600),
    make_pair(7, "ок", "ок", 700),
]


def statements_store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(tmp_path / "chroma", STATEMENTS_COLLECTION, STATEMENTS_MANIFEST)


# --- embeddings and the statements collection ---------------------------------------------


def test_query_cache_embeds_one_message_once() -> None:
    inner = HashEmbeddings()
    cache = QueryCache(inner)
    first = cache.embed(["как дела"], "query")
    assert cache.embed(["как дела"], "query") == first
    cache.embed(["как дела"], "passage")  # passages are never cached
    cache.embed(["другое"], "query")
    assert inner.calls == [("query", 1), ("passage", 1), ("query", 1)]
    assert cache.identity == inner.identity


def test_statements_collection_is_separate_and_embeds_replies(tmp_path: Path) -> None:
    emb = HashEmbeddings()
    pairs_store = ChromaVectorStore(tmp_path / "chroma")
    build_index(pairs_store, emb, PAIRS, "ds1")
    store = statements_store(tmp_path)
    selected = statement_pairs(LONG)
    assert [p.pair_id for p in selected] == ["1002:5", "1002:6"]  # "ок" says nothing
    manifest = build_index(
        store, emb, selected, "ds1", text_for=reply_text_for_index, rule=STATEMENTS_RULE
    )
    assert manifest.embedded_text == STATEMENTS_RULE and manifest.count == 2
    assert pairs_store.count() == 4 and store.count() == 2
    assert pairs_store.read_manifest().embedded_text == "last_two_partner_turns"  # type: ignore[union-attr]
    retriever = Retriever(store, emb, k=1)
    # the document is the reply itself, so the identical text is the nearest neighbour
    assert retriever.retrieve("завалил матан, пересдача в июне")[0].pair_id == "1002:6"
    assert retriever.retrieve("завалил матан, пересдача в июне", ts_before=600)[0].pair_id == (
        "1002:5"
    )


# --- prompt ----------------------------------------------------------------------------------


def test_knowledge_sections_only_when_present() -> None:
    assert render_knowledge("Радомир", "", [], "") == ""
    example = RetrievedExample(
        pair_id="1002:6",
        chat_id=1002,
        ts=1717200000,  # 2024-06-01 UTC
        context_text="что с учёбой",
        reply="завалил матан\nпересдача",
        distance=0.1,
        context=[],
    )
    text = render_knowledge("Радомир", "- учится на физтехе\n", [example], "")
    assert "### Что Радомир знает о себе\n" in text and "- учится на физтехе" in text
    assert "### Что Радомир уже говорил на похожие темы" in text
    assert "- [2024-06] завалил матан / пересдача" in text
    assert "про этого собеседника" not in text
    assert "про этого собеседника" in render_knowledge("Радомир", "", [], "- переехала")


@pytest.mark.parametrize("name", ["rag_v4", "rag_v5"])
def test_knowledge_templates_render(name: str) -> None:
    template = load_prompt(name)
    assert uses_knowledge(template) and not uses_knowledge(load_prompt("rag_v2"))
    history = [MemoryTurn(is_me=False, text="привет", ts=1)]
    bundle = build_rag_messages(
        template, "Радомир", "- коротко", [], history, "как дела", dossier="- учится"
    )
    system = bundle.messages[0]["content"]
    assert "### Что Радомир знает о себе" in system and "${" not in system
    assert "\n\n\n" not in system and bundle.version == name
    empty = build_rag_messages(template, "Радомир", "- коротко", [], [], "ку")
    assert "знает о себе" not in empty.messages[0]["content"]
    assert "\n\n\n" not in empty.messages[0]["content"]


def test_plan_is_split_from_the_message() -> None:
    assert split_plan("Замысел: подколоть\nОтвет: ну ты даёшь)") == ("подколоть", "ну ты даёшь)")
    assert split_plan("замысел: рассказать\n\nОтвет:\nездил на рыбалку\nпоймал щуку") == (
        "рассказать",
        "ездил на рыбалку\nпоймал щуку",
    )
    assert split_plan("просто ответ") == (None, "просто ответ")
    assert split_plan("Замысел: подколоть") == ("подколоть", None)
    verdict = validate_reply("Замысел: отмахнуться\nОтвет: не, лень", 600)
    assert verdict.ok and verdict.text == "не, лень" and verdict.plan == "отмахнуться"
    broken = validate_reply("Замысел: отмахнуться", 600)
    assert not broken.ok and broken.reason == "plan_unparsed" and broken.text == ""


def test_rag_backend_with_knowledge_and_plan(tmp_path: Path) -> None:
    emb = QueryCache(HashEmbeddings())
    pairs_store = ChromaVectorStore(tmp_path / "chroma")
    build_index(pairs_store, emb, PAIRS, "ds1")
    # 1002:3 is also an example (reply "ничего"); the statements must not repeat it
    overlap = make_pair(3, "что делаешь", "ничего особенного, сижу дома и читаю книжку", 300)
    stmt_store = statements_store(tmp_path)
    build_index(
        stmt_store,
        emb,
        statement_pairs([overlap, *LONG]),
        "ds1",
        text_for=reply_text_for_index,
        rule=STATEMENTS_RULE,
    )
    answer = "Замысел: рассказать своё\nОтвет: да норм, матан пересдаю"
    with FakeOpenAIServer(reply=answer) as server:
        backend = RagBackend(
            llm=LLMClient(server.base_url, "k", model="fake/model", max_retries=0),
            retriever=Retriever(pairs_store, emb, k=4),
            template=load_prompt("rag_v5"),
            name="Радомир",
            style_profile="- коротко\n",
            statements_retriever=Retriever(stmt_store, emb, k=3),
            dossier="- учится на физтехе\n",
        )
        result = backend.generate(GenerationRequest(partner_id=1, text="как учёба?"))
    assert result.text == "да норм, матан пересдаю" and result.plan == "рассказать своё"
    assert result.prompt_version == "rag_v5"
    assert "1002:3" in result.retrieved_ids and "1002:3" not in result.statement_ids
    assert sorted(result.statement_ids) == ["1002:5", "1002:6"]
    system = server.requests[0]["messages"][0]["content"]
    assert "- учится на физтехе" in system and "уже говорил на похожие темы" in system
    assert "поймали трёх щук" in system


# --- dossier ---------------------------------------------------------------------------------


def test_dossier_map_reduce(tmp_path: Path) -> None:
    train = [
        make_pair(i, "как ты?", f"длинный ответ номер {i} про футбол и учёбу", 100 * i)
        for i in range(1, 5)
    ] + [make_pair(9, "ок?", "ок", 900)]

    def reply(body: dict[str, Any]) -> str:
        user = body["messages"][1]["content"]
        if "### Заметки" in user:
            return "- любит футбол\n- учится в универе\nитог"
        return "- любит футбол (2024-03)"

    with FakeOpenAIServer(reply=reply) as server:
        result = generate_dossier(
            LLMClient(server.base_url, "k", model="fake/map"),
            LLMClient(server.base_url, "k", model="fake/reduce"),
            load_prompt("dossier_map_v1"),
            load_prompt("dossier_reduce_v1"),
            "Радомир",
            train,
            chunk_size=2,
            workers=1,
        )
    chats = [r for r in server.requests if r["path"].endswith("/chat/completions")]
    assert [r["model"] for r in chats] == ["fake/map", "fake/map", "fake/reduce"]
    assert result.pairs_used == 4 and result.chunks == 2 and result.facts == 2
    assert "Радомир: длинный ответ номер 1" in chats[0]["messages"][1]["content"]
    assert (
        "Радомир: ок" not in chats[0]["messages"][1]["content"] + chats[1]["messages"][1]["content"]
    )
    assert "- любит футбол (2024-03)" in chats[2]["messages"][1]["content"]
    target = tmp_path / "self_dossier.md"
    write_dossier(target, result, "ds1", force=False)
    assert read_dossier(target).startswith("- любит футбол")
    with pytest.raises(DossierExistsError):
        write_dossier(target, result, "ds1", force=False)
    assert read_dossier(tmp_path / "missing.md") == ""
    with pytest.raises(ValueError):
        informative_pairs([make_pair(1, "x", "длинный ответ про жизнь", 1, eval_sample=True)])


def test_dossier_splits_a_refused_chunk_and_counts_what_it_drops() -> None:
    train = [
        make_pair(i, "как ты?", f"длинный ответ номер {i} про футбол и учёбу", 100 * i)
        for i in range(1, 5)
    ]
    refusal = "InternalError.Algo.DataInspectionFailed: Input text data may contain inappropriate"

    def run(server: FakeOpenAIServer, min_split: int) -> Any:
        llm = LLMClient(server.base_url, "k", model="fake/map", max_retries=0)
        return generate_dossier(
            llm,
            llm,
            load_prompt("dossier_map_v1"),
            load_prompt("dossier_reduce_v1"),
            "Радомир",
            train,
            chunk_size=4,
            workers=1,
            min_split=min_split,
        )

    with FakeOpenAIServer(
        reply="- любит футбол", fail_times=1, fail_status=400, fail_message=refusal
    ) as server:
        result = run(server, 1)
    assert (result.chunks, result.skipped_pairs, result.pairs_used) == (2, 0, 4)
    with FakeOpenAIServer(
        reply="- любит футбол", fail_times=2, fail_status=400, fail_message=refusal
    ) as server:
        result = run(server, 2)
    # the whole chunk, then its first half were refused: two pairs dropped and counted
    assert (result.chunks, result.skipped_pairs, result.pairs_used) == (1, 2, 2)
    quota = "You have exceeded your usage limit"
    with (
        FakeOpenAIServer(fail_times=1, fail_status=422, fail_message=quota) as server,
        pytest.raises(LLMError),  # never mistaken for a refusal
    ):
        run(server, 1)


# --- evaluation --------------------------------------------------------------------------------


class StatementsBackend:
    mode = "rag"

    def __init__(self, statements: AuditingRetriever) -> None:
        self.statements = statements

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.statements.retrieve(request.text, ts_before=request.ts_before)
        return GenerationResult(
            text="ок",
            mode="rag",
            model="fake",
            prompt_version="rag_v4",
            retrieved_ids=[],
            params={},
            latency_ms=1,
            attempts=1,
            rejected=[],
            raw_texts=["ок"],
        )


class FixedRetriever:
    """A statements retriever that returns one record at a chosen time, right or wrong."""

    k = 3
    manifest = None

    def __init__(self, ts: int) -> None:
        self.ts = ts

    def retrieve(self, *_args: Any, **_kwargs: Any) -> list[RetrievedExample]:
        return [
            RetrievedExample(
                pair_id="1002:77",
                chat_id=1002,
                ts=self.ts,
                context_text="что с учёбой",
                reply="завалил матан",
                distance=0.1,
                context=[],
            )
        ]


@pytest.mark.parametrize(("ts", "leaks"), [(50, False), (10**9, True)])
def test_statements_are_leakage_checked(ts: int, leaks: bool) -> None:
    statements = AuditingRetriever(FixedRetriever(ts))  # type: ignore[arg-type]
    with FakeOpenAIServer(reply=JUDGE_JSON) as server:
        judge = Judge(LLMClient(server.base_url, "k", model="m"), load_prompt("judge_v1"), "Р")

        def evaluate() -> EvalRun:
            return run_eval(
                HOLDOUT,
                {p.pair_id for p in HOLDOUT},
                StatementsBackend(statements),  # type: ignore[arg-type]
                None,
                judge,
                EvalConfig(),
                "ds",
                "ms",
                statements_retriever=statements,
                dossier_sha256="abc",
            )

        if leaks:
            with pytest.raises(LeakageError):
                evaluate()
        else:
            run = evaluate()
            assert run.records[0].statements[0].reply == "завалил матан"
            assert run.metadata.dossier_sha256 == "abc"


def fake_run(run_id: str, prompt: str, outputs: dict[str, str | None]) -> EvalRun:
    records = [
        EvalRecord(
            pair_id=pair_id,
            chat_id=1002,
            ts=1000 + i,
            period="2026-06",
            context=[{"sender_name": "Злата", "is_me": False, "text": "как ты?"}],
            incoming="как ты?",
            reference="норм",
            output=output,
            raw_texts=[output or ""],
            rejected=[],
            error=None,
            fallback_from=None,
            mode="rag",
            model="fake",
            prompt_version=prompt,
            params={},
            latency_ms=1,
            retrieved=[],
            judge=None,
        )
        for i, (pair_id, output) in enumerate(outputs.items())
    ]
    metadata = RunMetadata(
        run_id=run_id,
        started_at="2026-09-11T00:00:00+00:00",
        finished_at="2026-09-11T00:01:00+00:00",
        git_commit=None,
        dataset_version="ds",
        messages_dataset_version="ms",
        mode="rag",
        model="fake",
        prompt_version=prompt,
        judge_model="j",
        judge_prompt_version="judge_v1",
        eval_config={},
        eval_config_version=1,
        holdout_ids_sha256="abc",
        n=len(records),
    )
    summary = RunSummary(
        n=len(records),
        judged=0,
        silent=0,
        fallbacks=0,
        errors=0,
        means={},
        overall=None,
        latency_ms_p50=1,
        per_period={},
    )
    return EvalRun(metadata=metadata, summary=summary, records=records)


def prefer_alive(body: dict[str, Any]) -> str:
    """A judge that likes "живо" and otherwise always picks the first variant."""
    user = body["messages"][1]["content"]
    first = user.split("### Вариант 1\n", 1)[1].split("\n\n### Вариант 2", 1)[0]
    second = user.split("### Вариант 2\n", 1)[1]
    if "живо" in first:
        return '{"winner": "1", "reason": "живее"}'
    if "живо" in second:
        return '{"winner": "2", "reason": "живее"}'
    return '{"winner": "1", "reason": "первый"}'


def test_duel_counts_only_order_consistent_wins() -> None:
    a = fake_run("run_a", "rag_v2", {"p1": "ок", "p2": "норм", "p3": None})
    b = fake_run("run_b", "rag_v5", {"p1": "живо рассказал про рыбалку", "p2": "ага", "p3": "живо"})
    with FakeOpenAIServer(reply=prefer_alive) as server:
        duel = run_duel(
            a,
            b,
            DuelJudge(
                LLMClient(server.base_url, "k", model="j"), load_prompt("duel_v1"), "Радомир"
            ),
        )
    # p2: the judge only ever picks the first slot, so the two orders disagree -> tie
    assert [r.outcome for r in duel.records] == ["b", "tie", "b"]
    assert (duel.summary.a_wins, duel.summary.b_wins, duel.summary.ties) == (0, 2, 1)
    assert duel.prompt_b == "rag_v5" and duel.summary.errors == 0
    user = server.requests[0]["messages"][1]["content"]
    assert "### Реальный ответ Радомир\nнорм" in user
    assert "### Вариант 1\n(нет ответа)" in server.requests[4]["messages"][1]["content"]


def test_duel_helpers() -> None:
    assert parse_duel_json('итог: {"winner": "tie", "reason": "равны"}') == ("tie", "равны")
    with pytest.raises(ValueError):
        parse_duel_json('{"winner": "3"}')
    assert combine(DuelVote(winner="1"), DuelVote(winner="2")) == "a"
    assert combine(DuelVote(winner="1"), DuelVote(winner="1")) == "tie"
    assert combine(DuelVote(winner=None, error="x"), DuelVote(winner="2")) == "tie"
    assert sign_test_p(0, 0) == 1.0 and sign_test_p(5, 5) == 1.0
    assert sign_test_p(0, 10) == pytest.approx(0.002, abs=1e-3)
    other = fake_run("x", "p", {"p1": "a"})
    other.metadata.holdout_ids_sha256 = "zzz"
    with pytest.raises(ValueError):
        run_duel(fake_run("y", "p", {"p1": "b"}), other, None)  # type: ignore[arg-type]
