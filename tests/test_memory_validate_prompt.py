from __future__ import annotations

from pathlib import Path

from twin.core.memory import ConversationMemory, MemoryTurn
from twin.core.prompt import build_rag_messages, format_examples, format_history
from twin.core.prompts import load_prompt
from twin.core.retriever import RetrievedExample
from twin.core.schemas import ContextTurn
from twin.core.validate import clean_reply, validate_reply


def test_memory_keeps_last_turns_and_persists(tmp_path: Path) -> None:
    memory = ConversationMemory(tmp_path / "mem", max_turns=3)
    assert memory.turns(7) == []
    for i in range(5):
        memory.append(7, MemoryTurn(is_me=i % 2 == 0, text=f"т{i}", ts=i))
    assert [t.text for t in memory.turns(7)] == ["т2", "т3", "т4"]
    assert [t.text for t in ConversationMemory(tmp_path / "mem", 3).turns(7)] == ["т2", "т3", "т4"]
    assert memory.turns(8) == []
    assert memory.reset(7) is True and memory.turns(7) == [] and memory.reset(7) is False


def test_clean_and_validate_reply() -> None:
    assert clean_reply("Радомир: норм)", "Радомир") == "норм)"
    assert clean_reply("«ну да»") == "ну да"
    assert validate_reply("норм)", 600).ok
    assert validate_reply("   ", 600).reason == "empty"
    assert validate_reply("x" * 601, 600).reason == "too_long:601>600"
    assert validate_reply("Как ИИ, я не могу", 600).reason == "assistant_speak:как ии"
    assert validate_reply("Извините за задержку", 600).reason == "assistant_speak:извините за"
    assert validate_reply("Чем могу помочь?", 600).reason == "assistant_speak:чем могу помочь"
    assert validate_reply("не могу помочь", 600, markers=["никогда"]).ok  # markers configurable


def example(i: int, partner: str, reply: str) -> RetrievedExample:
    return RetrievedExample(
        pair_id=f"1002:{i}",
        chat_id=1002,
        ts=i,
        context_text=partner,
        reply=reply,
        distance=0.1,
        context=[ContextTurn(sender_name="Злата", is_me=False, text=partner)],
    )


def test_prompt_formatting_and_rag_messages() -> None:
    examples = [example(1, "пойдёшь?", "ага"), example(2, "а\nб" * 200, "нее)")]
    text = format_examples(examples, "Радомир")
    assert text.startswith("Собеседник: пойдёшь?\nРадомир: ага\n\nСобеседник: а / б")
    assert "…" in text and format_examples([], "Радомир") == "(примеров нет)"
    history = [
        MemoryTurn(is_me=False, text="привет", ts=1),
        MemoryTurn(is_me=True, text="ку", ts=2),
    ]
    assert format_history(history, "Радомир") == "Собеседник: привет\nРадомир: ку"
    assert format_history([], "Радомир") == "(начало разговора)"

    bundle = build_rag_messages(
        load_prompt("rag_v1"), "Радомир", "- коротко\n", examples, history, "идёшь?"
    )
    assert bundle.version == "rag_v1"
    system, user = bundle.messages[0]["content"], bundle.messages[1]["content"]
    assert "${" not in system and "${" not in user
    assert "Ты — Радомир" in system and "- коротко" in system and "Радомир: ага" in system
    assert user.endswith("идёшь?\n\nОтвет Радомир:") and "Собеседник: привет" in user
