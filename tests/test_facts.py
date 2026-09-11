"""Learning from the live conversation: what is read, what is never read, what is shown."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from tests.test_bot import (
    OWNER,
    PARTNER,
    FakeBackend,
    FakeBot,
    business_message,
    connection,
    direct_message,
    settings,
)

from twin.bot.handlers import TwinBot
from twin.bot.state import StateStore
from twin.core.facts import Fact, FactSheet, FactStore, human_turns, parse_facts, update_sheet
from twin.core.llm_client import LLMError
from twin.core.memory import ConversationMemory, MemoryTurn
from twin.core.prompt import build_rag_messages
from twin.core.prompts import load_prompt

NOW = 1_800_000_000


class FakeLLM:
    """Records the prompt it was given and returns a canned fact list."""

    model = "fake"

    def __init__(self, answer: str = "- ищет 100-150к\n- работает дежурным") -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.calls.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer

        class Result:
            text = self.answer
            model = "fake"
            finish_reason = "stop"

        return Result()


# --- the anti-collapse rule ------------------------------------------------------------


def test_bot_written_turns_are_never_learned_from() -> None:
    turns = [
        MemoryTurn(is_me=False, text="я нашел работу", ts=1),
        MemoryTurn(is_me=True, text="я выдумал что живу в Париже", ts=2, by_bot=True),
        MemoryTurn(is_me=True, text="ща приеду", ts=3, by_bot=False),
    ]
    kept = human_turns(turns)
    assert [t.text for t in kept] == ["я нашел работу", "ща приеду"]
    llm = FakeLLM()
    sheet = update_sheet(llm, load_prompt("facts_v1"), FactSheet(partner_id=1), turns, "Р", NOW)
    assert sheet is not None
    prompt = "\n".join(m["content"] for m in llm.calls[0])
    assert "я нашел работу" in prompt and "ща приеду" in prompt
    assert "Париже" not in prompt  # the bot's own invention never reaches the model
    assert [f.text for f in sheet.facts] == ["ищет 100-150к", "работает дежурным"]
    assert sheet.turns_seen == 2 and sheet.updated_at == NOW


def test_update_sheet_without_human_turns_or_with_a_dead_gateway() -> None:
    only_bot = [MemoryTurn(is_me=True, text="ага", ts=1, by_bot=True)]
    assert (
        update_sheet(
            FakeLLM(), load_prompt("facts_v1"), FactSheet(partner_id=1), only_bot, "Р", NOW
        )
        is None
    )
    broken = FakeLLM()
    broken.answer = LLMError("gateway down")
    turns = [MemoryTurn(is_me=False, text="привет", ts=1)]
    assert (
        update_sheet(broken, load_prompt("facts_v1"), FactSheet(partner_id=1), turns, "Р", NOW)
        is None
    )


# --- parsing and storage ---------------------------------------------------------------


def test_parse_facts_keeps_only_list_lines() -> None:
    raw = (
        "Вот памятка:\n- живёт в Москве\n* ищет работу\nпросто текст\n- живёт в Москве\n- "
        + "я" * 300
    )
    facts = parse_facts(raw, NOW)
    assert [f.text for f in facts[:2]] == ["живёт в Москве", "ищет работу"]
    assert len(facts) == 3 and len(facts[2].text) == 160  # deduped and clipped
    assert parse_facts("(пока ничего не известно)", NOW) == []
    assert len(parse_facts("\n".join(f"- факт {i}" for i in range(50)), NOW, limit=15)) == 15


def test_fact_store_roundtrip(tmp_path: Path) -> None:
    store = FactStore(tmp_path / "facts")
    assert store.load(PARTNER).facts == [] and store.load(PARTNER).render().startswith("(пока")
    store.save(
        FactSheet(partner_id=PARTNER, facts=[Fact(text="ищет работу", ts=NOW)], updated_at=NOW)
    )
    assert store.load(PARTNER).render() == "- ищет работу"
    assert store.forget(PARTNER) and not store.forget(PARTNER)


def test_prompt_renders_facts_only_when_the_template_asks() -> None:
    facts = "- ищет 100-150к"
    v3 = build_rag_messages(
        load_prompt("rag_v3"), "Радомир", "- коротко", [], [], "как дела?", facts=facts
    )
    assert facts in v3.messages[0]["content"]
    empty = build_rag_messages(
        load_prompt("rag_v3"), "Радомир", "- коротко", [], [], "как?", facts=""
    )
    assert "(пока ничего не известно)" in empty.messages[0]["content"]
    v2 = build_rag_messages(
        load_prompt("rag_v2"), "Радомир", "- коротко", [], [], "как?", facts=facts
    )
    assert facts not in v2.messages[0]["content"]  # older prompts are untouched


# --- the bot ---------------------------------------------------------------------------


@pytest.fixture
def twin(tmp_path: Path) -> TwinBot:
    async def sleep(seconds: float) -> None:
        return None

    llm = FakeLLM()
    bot = FakeBot()
    backend = FakeBackend()
    t = TwinBot(
        bot=bot,
        settings=settings(facts_min_new_turns=2, facts_interval_hours=1),
        store=StateStore(tmp_path / "state"),
        backend=backend,
        memory=ConversationMemory(tmp_path / "memory", 10),
        sleep=sleep,
        rng=random.Random(1),
        clock=lambda: NOW,
        skip_rate=0.0,
        fact_store=FactStore(tmp_path / "facts"),
        facts_factory=lambda: (llm, load_prompt("facts_v1")),
    )
    t.fake_bot, t.fake_backend, t.fake_llm = bot, backend, llm  # type: ignore[attr-defined]
    return t


@pytest.mark.asyncio
async def test_learning_is_off_until_switched_on(twin: TwinBot) -> None:
    await twin.on_business_connection(connection())
    assert await twin.on_business_message(business_message("привет")) == "sent"
    assert twin.facts_for(PARTNER) == ""
    assert await twin.learning_tick() == {}
    assert twin.fake_llm.calls == []
    assert twin.fake_backend.requests[-1].facts == ""


@pytest.mark.asyncio
async def test_learning_counts_real_turns_and_feeds_the_prompt(twin: TwinBot) -> None:
    await twin.on_business_connection(connection())
    assert await twin.on_direct_message(direct_message("/learn on")) == "learn on"
    assert await twin.on_business_message(business_message("я нашел работу")) == "sent"
    # one partner turn so far: the bot's own reply must not count
    assert twin.state.human_turns_since_facts[str(PARTNER)] == 1
    assert (await twin.learning_tick())[PARTNER] == "not_enough_new_turns"
    # the owner writes by hand: real material, and it pauses the chat as before
    owner = business_message("та он мне сам сказал", sender=OWNER, message_id=77, chat_id=PARTNER)
    assert await twin.on_business_message(owner) == "owner_message_autopause"
    assert twin.state.human_turns_since_facts[str(PARTNER)] == 2
    assert (await twin.learning_tick())[PARTNER] == "updated:2"
    assert twin.fact_store is not None and twin.fact_store.load(PARTNER).facts
    assert twin.state.human_turns_since_facts[str(PARTNER)] == 0
    # the sheet now travels with every generation
    twin.state.unpause(PARTNER)
    assert await twin.on_business_message(business_message("ну как?", message_id=12)) == "sent"
    assert "ищет 100-150к" in twin.fake_backend.requests[-1].facts
    # and it is rate limited
    assert (await twin.learning_tick())[PARTNER] == "not_enough_new_turns"


@pytest.mark.asyncio
async def test_facts_and_forget_commands(twin: TwinBot) -> None:
    await twin.on_business_connection(connection())
    reply = await twin.on_direct_message(direct_message(f"/facts {PARTNER}"))
    assert reply is not None and "/learn on выключен" in reply
    assert await twin.on_direct_message(direct_message("/facts")) == "usage: /facts <user_id>"
    await twin.on_direct_message(direct_message("/learn on"))
    twin.fact_store.save(  # type: ignore[union-attr]
        FactSheet(partner_id=PARTNER, facts=[Fact(text="ищет работу", ts=NOW)], turns_seen=4)
    )
    shown = await twin.on_direct_message(direct_message(f"/facts {PARTNER}"))
    assert shown is not None and "- ищет работу" in shown and "по 4 репликам" in shown
    cleared = await twin.on_direct_message(direct_message(f"/forget {PARTNER}"))
    assert cleared is not None and "cleared" in cleared
    assert twin.facts_for(PARTNER).startswith("(пока")
    status = await twin.on_direct_message(direct_message("/status"))
    assert status is not None and "/learn — on" in status
    assert await twin.on_direct_message(direct_message(f"/facts {PARTNER}", sender=PARTNER)) is None


def test_empty_sheet_keeps_the_cheaper_prompt() -> None:
    """An empty fact section cost 0.08 overall on the holdout, so it is not rendered."""
    from twin.core.backends import GenerationRequest, RagBackend

    calls: list[list[dict[str, str]]] = []

    class Recorder(FakeLLM):
        def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
            calls.append(messages)
            return super().chat(messages, **kwargs)

    class NoRetrieval:
        def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

    backend = RagBackend(
        llm=Recorder("норм)"),
        retriever=NoRetrieval(),
        template=load_prompt("rag_v2"),
        facts_template=load_prompt("rag_v3"),
        name="Радомир",
        style_profile="- коротко",
    )
    empty = backend.generate(GenerationRequest(partner_id=1, text="как дела?", facts="   "))
    assert empty.prompt_version == "rag_v2"
    assert "памятка" not in calls[-1][0]["content"].lower()
    known = backend.generate(
        GenerationRequest(partner_id=1, text="как дела?", facts="- ищет 100-150к")
    )
    assert known.prompt_version == "rag_v3"
    assert "- ищет 100-150к" in calls[-1][0]["content"]
