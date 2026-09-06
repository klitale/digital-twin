from __future__ import annotations

import json
from pathlib import Path

import pytest
from training.chat_format import IGNORE_INDEX, build_example, render_chatml
from training.prepare_dataset import (
    TRAIN_FILE,
    TRAIN_MANIFEST,
    TrainManifest,
    format_context,
    pair_to_messages,
    prepare_dataset,
    read_train_jsonl,
)
from training.train_config import load_train_config
from training.train_lora import encode_rows, split_examples
from typer.testing import CliRunner

from fake_openai_server import FakeOpenAIServer, HashEmbeddings
from twin.cli import app
from twin.core.backends import FallbackBackend, FinetunedBackend, GenerationRequest, HybridBackend
from twin.core.llm_client import LLMClient
from twin.core.memory import MemoryTurn
from twin.core.prompts import load_prompt
from twin.core.retriever import Retriever
from twin.core.schemas import ContextTurn, Pair
from twin.core.vector_store import ChromaVectorStore
from twin.ingest.index import build_index

REPO = Path(__file__).parent.parent


def char_encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def pair(index: int, reply: str = "ага)", eval_sample: bool = False) -> Pair:
    return Pair(
        pair_id=f"1002:{index}",
        chat_id=1002,
        conversation_id="1002:0",
        ts=1709290800 + index,
        period="2024-03",
        context=[
            ContextTurn(sender_name="Радомир", is_me=True, text="я дома"),
            ContextTurn(sender_name="Злата", is_me=False, text="выйдешь?"),
        ],
        reply=reply,
        reply_message_ids=[index],
        eval_sample=eval_sample,
    )


def test_chatml_render_and_completion_mask() -> None:
    messages = [
        {"role": "system", "content": "сис"},
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]
    text = render_chatml(messages)
    assert text == (
        "<|im_start|>system\nсис<|im_end|>\n<|im_start|>user\nвопрос<|im_end|>\n"
        "<|im_start|>assistant\nответ<|im_end|>\n"
    )
    assert render_chatml(messages[:2], add_generation_prompt=True).endswith(
        "<|im_start|>assistant\n"
    )
    ids, labels = build_example(messages, char_encode)
    assert ids == char_encode(text)
    prompt_len = len(render_chatml(messages[:2], add_generation_prompt=True))
    assert labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert labels[prompt_len:] == char_encode("ответ<|im_end|>\n")
    with pytest.raises(ValueError, match="assistant"):
        build_example(messages[:2], char_encode)


def test_pair_to_messages_and_context_format() -> None:
    persona = load_prompt("persona_train_v1")
    messages = pair_to_messages(pair(1), persona, "Радомир")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert "Ты — Радомир" in messages[0]["content"]
    assert messages[1]["content"] == "Радомир: я дома\nСобеседник: выйдешь?"
    assert messages[2]["content"] == "ага)"
    assert format_context(pair(1), "Радомир").startswith("Радомир: я дома")


def test_prepare_dataset_writes_manifest_and_refuses_holdout(tmp_path: Path) -> None:
    persona = load_prompt("persona_train_v1")
    pairs = [pair(1), pair(2, reply="очень " * 60), pair(3, reply="нет")]
    manifest = prepare_dataset(
        pairs, persona, "Радомир", char_encode, tmp_path, "ds1", max_seq_length=400
    )
    assert isinstance(manifest, TrainManifest)
    assert manifest.examples == 2 and manifest.dropped_too_long == 1
    assert (
        manifest.pairs_dataset_version == "ds1"
        and manifest.persona_prompt_version == "persona_train_v1"
    )
    assert manifest.tokens_max <= 400 and manifest.reply_tokens_p50 > 0
    rows = read_train_jsonl(tmp_path / TRAIN_FILE)
    assert [r["pair_id"] for r in rows] == ["1002:1", "1002:3"]
    assert json.loads((tmp_path / TRAIN_MANIFEST).read_text())["examples"] == 2
    with pytest.raises(ValueError, match="holdout"):
        prepare_dataset([pair(9, eval_sample=True)], persona, "Р", char_encode, tmp_path, "ds1")
    limited = prepare_dataset(pairs, persona, "Р", char_encode, tmp_path, "ds1", max_examples=1)
    assert limited.examples == 1


def test_train_configs_load() -> None:
    full = load_train_config(REPO / "configs" / "train" / "full.yaml")
    assert full.base_model == "Qwen/Qwen2.5-7B-Instruct" and full.max_seq_length == 2048
    assert full.lora.r == 16 and full.train.epochs == 2 and full.load_in_4bit
    assert set(full.lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    dry = load_train_config(REPO / "configs" / "train" / "dry_run.yaml")
    assert (
        dry.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
        and dry.train.max_steps == 2
        and dry.max_examples == 5
    )
    assert not dry.load_in_4bit


def test_train_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nbase_model: m\nlora:\n  rank: 4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_train_config(path)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": char_encode(text)}


def test_encode_rows_and_split() -> None:
    rows = [
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u" * n},
                {"role": "assistant", "content": "a"},
            ]
        }
        for n in (1, 5, 500)
    ]
    examples = encode_rows(rows, FakeTokenizer(), max_seq_length=200)
    assert len(examples) == 2 and all(IGNORE_INDEX in e["labels"] for e in examples)
    assert split_examples(examples, 0.5, 1) == (examples, [])  # too few rows to hold out
    many = [{"input_ids": [i], "labels": [i]} for i in range(40)]
    train, held = split_examples(many, 0.1, 7)
    assert len(train) == 36 and len(held) == 4
    assert split_examples(many, 0.1, 7) == (train, held)


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    store = ChromaVectorStore(tmp_path / "chroma")
    emb = HashEmbeddings()
    build_index(store, emb, [pair(1), pair(2, reply="нее"), pair(3, reply="ща")], "ds1")
    return Retriever(store, emb, k=3)


def test_finetuned_and_hybrid_prompts(retriever: Retriever) -> None:
    request = GenerationRequest(
        partner_id=1, text="выйдешь?", history=[MemoryTurn(is_me=False, text="привет", ts=1)]
    )
    with FakeOpenAIServer(reply="ага)") as server:
        llm = LLMClient(server.base_url, "k", model="twin", reasoning_off=False)
        ft = FinetunedBackend(llm, load_prompt("finetuned_v1"), "Радомир", "- коротко\n")
        result = ft.generate(request)
        assert result.text == "ага)" and result.mode == "finetuned" and result.retrieved_ids == []
        assert result.prompt_version == "finetuned_v1"
        system = server.requests[-1]["messages"][0]["content"]
        assert "- коротко" in system and "<<<" not in system
        user = server.requests[-1]["messages"][1]["content"]
        assert user.endswith("Собеседник: выйдешь?") and "Собеседник: привет" in user

        hybrid = HybridBackend(llm, retriever, load_prompt("hybrid_v1"), "Радомир")
        result = hybrid.generate(request)
        assert result.mode == "hybrid" and result.prompt_version == "hybrid_v1"
        assert len(result.retrieved_ids) == 3
        system = server.requests[-1]["messages"][0]["content"]
        assert "<<<" in system and "Радомир: ага)" in system and "- коротко" not in system


def test_fallback_backend_switches_on_endpoint_error(retriever: Retriever) -> None:
    request = GenerationRequest(partner_id=1, text="выйдешь?")
    with FakeOpenAIServer(reply="норм") as rag_server, FakeOpenAIServer(fail_times=5) as ft_server:
        ft = FinetunedBackend(
            LLMClient(ft_server.base_url, "k", model="twin", max_retries=0),
            load_prompt("finetuned_v1"),
            "Р",
            "-",
        )
        rag = HybridBackend(
            LLMClient(rag_server.base_url, "k", model="gw"),
            retriever,
            load_prompt("hybrid_v1"),
            "Р",
        )
        backend = FallbackBackend(ft, rag)
        assert backend.mode == "finetuned"
        result = backend.generate(request)
        assert (
            result.text == "норм"
            and result.mode == "hybrid"
            and result.fallback_from == "finetuned"
        )
        assert result.error is None
    with (
        FakeOpenAIServer(reply="Как ИИ я не могу") as ft_server,
        FakeOpenAIServer(reply="норм") as rag_server,
    ):
        ft = FinetunedBackend(
            LLMClient(ft_server.base_url, "k", model="twin"), load_prompt("finetuned_v1"), "Р", "-"
        )
        rag = HybridBackend(
            LLMClient(rag_server.base_url, "k", model="gw"),
            retriever,
            load_prompt("hybrid_v1"),
            "Р",
        )
        result = FallbackBackend(ft, rag).generate(request)
        assert (
            result.text is None and result.fallback_from is None
        )  # validation silence is not an endpoint error
        assert rag_server.requests == []


def test_smoke_test_and_train_prepare_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_export_path: Path
) -> None:
    runner = CliRunner()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    monkeypatch.setenv("TWIN_NAME", "Радомир")
    monkeypatch.delenv("RAW_EXPORT_PATH", raising=False)
    monkeypatch.delenv("FT_BASE_URL", raising=False)
    missing = runner.invoke(app, ["smoke-test-model"])
    assert missing.exit_code == 1 and "FT_BASE_URL" in missing.output
    with FakeOpenAIServer(reply="а ты кто?") as server:
        monkeypatch.setenv("FT_BASE_URL", server.base_url)
        monkeypatch.setenv("FT_API_KEY", "tok")
        result = runner.invoke(app, ["smoke-test-model", "--model", "base"])
        assert result.exit_code == 0, result.output
        assert "model base: а ты кто?" in result.output and "warm" in result.output
        assert server.requests[0]["model"] == "base" and server.auth_headers == ["Bearer tok"]

    config = tmp_path / "data.yaml"
    config.write_text("filters:\n  min_date: null\n", encoding="utf-8")
    assert (
        runner.invoke(
            app, ["ingest", "--export", str(synthetic_export_path), "--config", str(config)]
        ).exit_code
        == 0
    )
    import training.prepare_dataset as prep

    monkeypatch.setattr(prep, "load_qwen_encoder", lambda: char_encode)
    result = runner.invoke(
        app,
        ["train", "--prepare-only", "--config", str(REPO / "configs" / "train" / "dry_run.yaml")],
    )
    assert result.exit_code == 0, result.output
    rows = read_train_jsonl(tmp_path / "data" / "train" / TRAIN_FILE)
    assert len(rows) == 6 and rows[0]["messages"][2]["role"] == "assistant"
    manifest = json.loads((tmp_path / "data" / "train" / TRAIN_MANIFEST).read_text())
    assert manifest["examples"] == 6 and manifest["max_seq_length"] == 512
