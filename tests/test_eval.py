from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fake_openai_server import FakeOpenAIServer, HashEmbeddings
from twin.cli import app
from twin.core.backends import GenerationRequest, GenerationResult
from twin.core.llm_client import LLMClient
from twin.core.prompts import load_prompt
from twin.core.retriever import RetrievedExample, Retriever
from twin.core.schemas import ContextTurn, Pair
from twin.core.vector_store import ChromaVectorStore
from twin.eval.compare import compare_runs, render_markdown
from twin.eval.harness import (
    AuditingRetriever,
    EvalConfig,
    LeakageError,
    assert_no_leakage,
    load_eval_config,
    read_run,
    run_eval,
    split_pair,
    summarize,
    write_run,
)
from twin.eval.judge import Judge, parse_judge_json
from twin.eval.report import render_report
from twin.eval.schemas import EvalRun
from twin.ingest.index import build_index

REPO = Path(__file__).parent.parent
JUDGE_JSON = (
    '{"style_similarity": {"score": 4, "reason": "коротко"}, "appropriateness": {"score": 5, "reason": "в тему"}, '
    '"not_assistant_like": {"score": 5, "reason": "живо"}, "consistency": {"score": 3, "reason": "ок"}}'
)


def pair(
    index: int, ts: int, reply: str = "ага)", eval_sample: bool = False, period: str = "2026-06"
) -> Pair:
    return Pair(
        pair_id=f"1002:{index}",
        chat_id=1002,
        conversation_id="1002:0",
        ts=ts,
        period=period,
        context=[
            ContextTurn(sender_name="Злата", is_me=False, text="ты где?"),
            ContextTurn(sender_name="Радомир", is_me=True, text="дома"),
            ContextTurn(sender_name="Злата", is_me=False, text=f"выйдешь {index}?"),
        ],
        reply=reply,
        reply_message_ids=[index],
        eval_sample=eval_sample,
    )


TRAIN = [pair(1, 100, "нее"), pair(2, 200, "ща"), pair(3, 300, "ок)")]
HOLDOUT = [
    pair(10, 1000, "выйду", True),
    pair(11, 1100, "не сегодня", True, period="2026-07"),
    pair(12, 1200, "потом", False),
]


def test_parse_judge_json() -> None:
    scores, reasons = parse_judge_json("Вот оценка:\n" + JUDGE_JSON + "\nконец")
    assert scores == {
        "style_similarity": 4,
        "appropriateness": 5,
        "not_assistant_like": 5,
        "consistency": 3,
    }
    assert reasons["style_similarity"] == "коротко"
    scores, _ = parse_judge_json(
        '{"style_similarity": 2, "appropriateness": 2, "not_assistant_like": 2, "consistency": 2}'
    )
    assert scores["consistency"] == 2
    for bad in (
        "нет json",
        '{"style_similarity": {"score": 9}}',
        '{"style_similarity": {"score": true}}',
    ):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_judge_json(bad)


def test_judge_handles_errors_and_silence() -> None:
    with FakeOpenAIServer(reply=JUDGE_JSON) as server:
        judge = Judge(
            LLMClient(server.base_url, "k", model="openai/gpt-5.4-mini"),
            load_prompt("judge_v1"),
            "Радомир",
        )
        verdict = judge.judge(HOLDOUT[0].context, "выйду", None)
        assert verdict.mean == 4.25 and verdict.error is None
        user = server.requests[-1]["messages"][1]["content"]
        assert "(нет ответа)" in user and "Собеседник: ты где?" in user and "Радомир: дома" in user
        assert (
            server.requests[-1]["temperature"] == 0.0
            and server.requests[-1]["reasoning_effort"] == "none"
        )
    with FakeOpenAIServer(reply="не могу оценить") as server:
        verdict = Judge(
            LLMClient(server.base_url, "k", model="m"), load_prompt("judge_v1"), "Р"
        ).judge(HOLDOUT[0].context, "a", "b")
        assert verdict.error and verdict.error.startswith("unparsable") and verdict.mean is None
    with FakeOpenAIServer(fail_times=9) as server:
        verdict = Judge(
            LLMClient(server.base_url, "k", model="m", max_retries=0), load_prompt("judge_v1"), "Р"
        ).judge(HOLDOUT[0].context, "a", "b")
        assert verdict.error and verdict.scores == {}


def test_split_pair_and_leakage_assertions() -> None:
    history, incoming, previous = split_pair(HOLDOUT[0])
    assert incoming == "выйдешь 10?" and previous == "ты где?"
    assert [(t.is_me, t.text) for t in history] == [(False, "ты где?"), (True, "дома")]
    with pytest.raises(ValueError, match="partner"):
        split_pair(
            HOLDOUT[0].model_copy(
                update={"context": [ContextTurn(sender_name="Р", is_me=True, text="x")]}
            )
        )

    def example(pair_id: str, ts: int) -> RetrievedExample:
        return RetrievedExample(
            pair_id=pair_id,
            chat_id=1002,
            ts=ts,
            context_text="",
            reply="",
            distance=0.0,
            context=[],
        )

    holdout_ids = {p.pair_id for p in HOLDOUT}
    assert_no_leakage(HOLDOUT[0], [example("1002:1", 100)], holdout_ids)
    with pytest.raises(LeakageError, match="target"):
        assert_no_leakage(HOLDOUT[0], [example("1002:10", 100)], holdout_ids)
    with pytest.raises(LeakageError, match="holdout"):
        assert_no_leakage(HOLDOUT[0], [example("1002:11", 100)], holdout_ids)
    with pytest.raises(LeakageError, match="not earlier"):
        assert_no_leakage(HOLDOUT[0], [example("1002:1", 1000)], holdout_ids)


class FakeBackend:
    mode = "rag"

    def __init__(self, retriever: AuditingRetriever, reply: str | None = "выйду)") -> None:
        self.retriever = retriever
        self.reply = reply
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        examples = self.retriever.retrieve(
            request.text,
            previous_partner_text=request.previous_partner_text,
            ts_before=request.ts_before,
            exclude_pair_ids=request.exclude_pair_ids,
        )
        return GenerationResult(
            text=self.reply,
            mode="rag",
            model="fake",
            prompt_version="rag_v1",
            retrieved_ids=[e.pair_id for e in examples],
            params={"temperature": 0.8},
            latency_ms=5,
            attempts=1,
            rejected=[] if self.reply else ["assistant_speak:x"],
            raw_texts=[self.reply or ""],
        )


@pytest.fixture
def retriever(tmp_path: Path) -> AuditingRetriever:
    store = ChromaVectorStore(tmp_path / "chroma")
    emb = HashEmbeddings()
    build_index(store, emb, TRAIN, "ds1")
    return AuditingRetriever(Retriever(store, emb, k=3))


def make_run(
    retriever: AuditingRetriever, reply: str | None = "выйду)", limit: int | None = None
) -> EvalRun:
    with FakeOpenAIServer(reply=JUDGE_JSON) as server:
        judge = Judge(
            LLMClient(server.base_url, "k", model="judge"), load_prompt("judge_v1"), "Радомир"
        )
        return run_eval(
            HOLDOUT,
            {p.pair_id for p in HOLDOUT},
            FakeBackend(retriever, reply),
            retriever,
            judge,
            EvalConfig(limit=limit),
            "ds1",
            "m1",
        )


def test_run_eval_records_and_summary(retriever: AuditingRetriever) -> None:
    run = make_run(retriever)
    assert run.summary.n == 2  # only eval_sample rows
    assert run.summary.judged == 2 and run.summary.silent == 0 and run.summary.overall == 4.25
    assert run.metadata.mode == "rag" and run.metadata.model == "fake" and run.metadata.n == 2
    assert run.metadata.judge_model == "judge" and run.metadata.judge_prompt_version == "judge_v1"
    assert run.metadata.eval_config_version == 1 and len(run.metadata.holdout_ids_sha256) == 12
    record = run.records[0]
    assert (
        record.pair_id == "1002:10"
        and record.incoming == "выйдешь 10?"
        and record.reference == "выйду"
    )
    assert (
        record.output == "выйду)" and record.judge and record.judge.scores["appropriateness"] == 5
    )
    assert record.retrieved and all(r.ts < record.ts for r in record.retrieved)
    assert set(run.summary.per_period) == {"2026-06", "2026-07"}
    silent = make_run(retriever, reply=None)
    assert silent.summary.silent == 2 and silent.records[0].output is None
    assert make_run(retriever, limit=1).summary.n == 1
    with pytest.raises(ValueError, match="no holdout"):
        run_eval(TRAIN, set(), FakeBackend(retriever), retriever, None, EvalConfig(), "d", "m")  # type: ignore[arg-type]


def test_run_roundtrip_compare_and_report(tmp_path: Path, retriever: AuditingRetriever) -> None:
    run = make_run(retriever)
    path = write_run(run, tmp_path / "eval")
    assert path.name.startswith(run.metadata.run_id) and read_run(path) == run
    other = run.model_copy(
        update={
            "metadata": run.metadata.model_copy(
                update={"mode": "hybrid", "run_id": "later_hybrid", "started_at": "2099"}
            )
        }
    )
    comparison = compare_runs([other, run])
    assert comparison.baseline == run.metadata.run_id
    assert comparison.rows[1]["mode"] == "hybrid" and comparison.rows[1]["delta_overall"] == 0.0
    assert comparison.caveats == ["none."]
    markdown = render_markdown(comparison)
    assert "| run | mode |" in markdown and "later_hybrid" in markdown
    html = render_report([run, other], "Радомир")
    assert "color-scheme: light dark" in html and "<details>" in html and "выйдешь 10?" in html
    assert (
        "<script src=" not in html
        and "https://" not in html.split("</style>")[1].split("<h2>Метаданные")[0]
    )
    assert "Худшие 10" in html

    mismatched = run.model_copy(
        update={
            "metadata": run.metadata.model_copy(
                update={"holdout_ids_sha256": "other", "run_id": "x"}
            )
        }
    )
    assert any("different holdout" in c for c in compare_runs([run, mismatched]).caveats)


def test_summarize_with_judge_errors(retriever: AuditingRetriever) -> None:
    run = make_run(retriever)
    broken = run.records[0].model_copy(
        update={"judge": run.records[0].judge.model_copy(update={"scores": {}, "error": "boom"})}
    )
    summary = summarize([broken, run.records[1]])
    assert summary.judged == 1 and summary.errors == 1 and summary.means["consistency"] == 3.0


def test_eval_config_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_export_path: Path
) -> None:
    config = load_eval_config(REPO / "configs" / "eval" / "default.yaml")
    assert config.judge.temperature == 0.0 and config.generation.k == 8 and config.limit is None

    runner = CliRunner()
    data = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    monkeypatch.setenv("TWIN_NAME", "Радомир")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "fake/chat")
    monkeypatch.setenv("JUDGE_MODEL", "fake/judge")
    monkeypatch.setenv("EMBED_MODEL", "fake/embed")
    monkeypatch.delenv("RAW_EXPORT_PATH", raising=False)
    # a tiny split with a real holdout: 2 pairs per chat is enough with these thresholds
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        "filters:\n  min_date: null\nsplit:\n  tail_fraction: 0.5\n  min_pairs_per_chat: 2\n  eval_sample_size: 2\n",
        encoding="utf-8",
    )
    assert (
        runner.invoke(
            app, ["ingest", "--export", str(synthetic_export_path), "--config", str(data_config)]
        ).exit_code
        == 0
    )
    (data / "processed" / "style_profile.md").write_text("- коротко\n", encoding="utf-8")

    def reply(body: dict) -> str:
        return JUDGE_JSON if body["model"] == "fake/judge" else "ага)"

    with FakeOpenAIServer(reply=reply) as server:
        monkeypatch.setenv("LLM_BASE_URL", server.base_url)
        monkeypatch.setenv("JUDGE_BASE_URL", server.base_url)
        monkeypatch.setenv("EMBED_BASE_URL", server.base_url)
        assert runner.invoke(app, ["index"]).exit_code == 0
        result = runner.invoke(app, ["eval", "--mode", "rag", "--limit", "2"])
        assert result.exit_code == 0, result.output
        assert "overall=4.25" in result.output
        runs = list((data / "eval").glob("*_rag_*.json"))
        assert len(runs) == 1
        judge_calls = [r for r in server.requests if r.get("model") == "fake/judge"]
        assert judge_calls and all(c["temperature"] == 0.0 for c in judge_calls)
    compare = runner.invoke(app, ["compare"])
    assert compare.exit_code == 0 and "| run | mode |" in compare.output
    assert (data / "eval" / "compare.md").exists()
    report = runner.invoke(app, ["report"])
    assert report.exit_code == 0, report.output
    html = (data / "eval" / "report.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html and "Радомир" in html
