from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fake_openai_server import FakeOpenAIServer, HashEmbeddings
from test_index_retriever import PAIRS
from twin.cli import app
from twin.core.backends import GenerationRequest, RagBackend
from twin.core.llm_client import LLMClient
from twin.core.memory import MemoryTurn
from twin.core.prompts import load_prompt
from twin.core.retriever import Retriever
from twin.core.vector_store import ChromaVectorStore
from twin.ingest.index import build_index


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    store = ChromaVectorStore(tmp_path / "chroma")
    emb = HashEmbeddings()
    build_index(store, emb, PAIRS, "ds1")
    return Retriever(store, emb, k=3)


def make_backend(server: FakeOpenAIServer, retriever: Retriever) -> RagBackend:
    return RagBackend(
        llm=LLMClient(server.base_url, "k", model="fake/model", max_retries=0),
        retriever=retriever,
        template=load_prompt("rag_v1"),
        name="Радомир",
        style_profile="- коротко\n",
        temperature=0.7,
        max_reply_chars=100,
    )


def test_rag_backend_generates_with_examples(retriever: Retriever) -> None:
    with FakeOpenAIServer(reply="Радомир: норм, а ты?") as server:
        backend = make_backend(server, retriever)
        request = GenerationRequest(
            partner_id=1,
            chat_id=1002,
            message_id=5,
            text="как дела",
            history=[MemoryTurn(is_me=False, text="привет", ts=1)],
        )
        result = backend.generate(request)
    assert result.text == "норм, а ты?"  # label echo stripped
    assert result.mode == "rag" and result.prompt_version == "rag_v1"
    assert result.retrieved_ids and result.attempts == 1 and result.rejected == []
    assert result.raw_texts == ["Радомир: норм, а ты?"]
    assert result.params["temperature"] == 0.7
    body = server.requests[0]
    assert body["temperature"] == 0.7 and "Радомир: норм" in body["messages"][0]["content"]
    assert "как дела" in body["messages"][1]["content"]


def test_rag_backend_regenerates_once_then_stays_silent(retriever: Retriever) -> None:
    replies = iter(["Как ИИ, я не могу", "норм"])
    with FakeOpenAIServer(reply=lambda _body: next(replies)) as server:
        result = make_backend(server, retriever).generate(
            GenerationRequest(partner_id=1, text="ку")
        )
    assert result.text == "норм" and result.attempts == 2
    assert result.rejected == ["assistant_speak:как ии"]

    with FakeOpenAIServer(reply="Извините за беспокойство") as server:
        result = make_backend(server, retriever).generate(
            GenerationRequest(partner_id=1, text="ку")
        )
    assert result.text is None and result.attempts == 2
    assert result.rejected == ["assistant_speak:извините за"] * 2 and len(server.requests) == 2

    with FakeOpenAIServer(fail_times=5) as server:
        result = make_backend(server, retriever).generate(
            GenerationRequest(partner_id=1, text="ку")
        )
    assert result.text is None and result.rejected[0].startswith("llm_error")


def test_holdout_guard_in_backend(retriever: Retriever) -> None:
    with FakeOpenAIServer(reply="ок") as server:
        result = make_backend(server, retriever).generate(
            GenerationRequest(
                partner_id=1, text="пойдёшь гулять?", ts_before=250, exclude_pair_ids=["1002:1"]
            )
        )
    assert result.retrieved_ids == ["1002:2"]  # target excluded, only pairs before ts 250 remain


def test_index_and_chat_cli_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_export_path: Path
) -> None:
    runner = CliRunner()
    data = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    monkeypatch.setenv("TWIN_NAME", "Радомир")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1002,1003")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "fake/chat")
    monkeypatch.setenv("EMBED_MODEL", "fake/embed")
    monkeypatch.delenv("RAW_EXPORT_PATH", raising=False)
    config = tmp_path / "data.yaml"
    config.write_text("filters:\n  min_date: null\n", encoding="utf-8")
    assert (
        runner.invoke(
            app, ["ingest", "--export", str(synthetic_export_path), "--config", str(config)]
        ).exit_code
        == 0
    )
    (data / "processed" / "style_profile.md").write_text(
        "<!-- h -->\n- коротко\n", encoding="utf-8"
    )

    with FakeOpenAIServer(reply="норм)", embedding_dim=8) as server:
        monkeypatch.setenv("LLM_BASE_URL", server.base_url)
        monkeypatch.setenv("EMBED_BASE_URL", server.base_url)
        first = runner.invoke(app, ["index"])
        assert first.exit_code == 0, first.output
        assert "(6 records)" in first.output
        manifest = json.loads((data / "chroma" / "index_manifest.json").read_text())
        assert manifest["model"] == "fake/embed" and manifest["dimension"] == 8

        again = runner.invoke(app, ["index"])
        assert again.exit_code == 1 and "--rebuild" in again.output
        rebuilt = runner.invoke(app, ["index", "--rebuild"])
        assert rebuilt.exit_code == 0 and "dropping 6" in rebuilt.output

        chat = runner.invoke(app, ["chat"], input="как дела\n/reset\nещё раз\n/quit\n")
        assert chat.exit_code == 0, chat.output
        assert chat.output.count("Радомир: норм)") == 2 and "(memory cleared)" in chat.output
        memory = json.loads((data / "state" / "memory" / "1002.json").read_text())
        assert [t["text"] for t in memory] == ["ещё раз", "норм)"]  # reset cleared the first pair
        chat_bodies = [r for r in server.requests if r["path"].endswith("/chat/completions")]
        assert chat_bodies and "Радомир" in chat_bodies[0]["messages"][0]["content"]
