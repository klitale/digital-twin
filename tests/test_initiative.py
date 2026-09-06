"""Initiative: pure decisions, the scheduler tick, /poke and the command menu."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage
from tests.test_bot import (
    ADMIN,
    CONN,
    OTHER,
    PARTNER,
    FakeBackend,
    FakeBot,
    business_message,
    connection,
    direct_message,
    settings,
)

from twin.bot.aggression import MAX_PARTS, SKIP_RATE
from twin.bot.aggression import resolve as aggression_resolve
from twin.bot.control import COMMANDS, HELP, parse_command
from twin.bot.handlers import TwinBot
from twin.bot.initiative import (
    InitiativeConfig,
    decide,
    decide_followup,
    decide_opener,
    next_window_day,
    parse_range,
    plan_opener,
)
from twin.bot.state import BotState, StateStore
from twin.config import Mode
from twin.core.backends import GenerationRequest, GenerationResult
from twin.core.memory import ConversationMemory
from twin.core.prompt import build_initiative_messages
from twin.core.prompts import load_prompt

TZ = "Europe/Moscow"
# 2026-09-07 12:00 local (Monday), inside the default 10-14 window
NOON = int(datetime(2026, 9, 7, 12, 0, tzinfo=ZoneInfo(TZ)).timestamp())
HOUR = 3600


def cfg(**overrides: Any) -> InitiativeConfig:
    return InitiativeConfig.from_settings(settings(**overrides))


def state_with(**kwargs: Any) -> BotState:
    state = BotState()
    for name in kwargs.pop("features", ()):
        state.set_feature(name, True)
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


# --- config ------------------------------------------------------------------------


def test_parse_range_and_config() -> None:
    assert parse_range("10-14", 3600, "x") == (10 * HOUR, 14 * HOUR)
    with pytest.raises(ValueError):
        parse_range("14-10", 3600, "x")
    with pytest.raises(ValueError):
        parse_range("abc", 60, "x")
    c = cfg()
    assert c.followup_window_s == (20 * 60, 90 * 60) and c.opener_silence_s == 24 * HOUR
    with pytest.raises(ZoneInfoNotFoundError):
        cfg(initiative_tz="Mars/Olympus")


# --- follow-up ----------------------------------------------------------------------


def test_followup_rules() -> None:
    c = cfg(followup_probability=1.0)
    rng = random.Random(0)
    off = state_with()
    assert decide_followup(off, PARTNER, NOON, c, rng).reason == "followup_off"
    s = state_with(features=["followup"])
    assert decide_followup(s, PARTNER, NOON, c, rng).reason == "no_bot_reply_yet"
    s.note_incoming(PARTNER, NOON - 40 * 60)
    s.note_outgoing(PARTNER, NOON - 30 * 60, "reply")
    assert decide_followup(s, PARTNER, NOON - 20 * 60, c, rng).reason == "too_soon"
    assert decide_followup(s, PARTNER, NOON, c, rng).kind == "followup"
    assert decide_followup(s, PARTNER, NOON + 60, c, rng).reason == "already_decided"
    # a new reply resets the decision; the partner speaking last blocks it
    s.note_outgoing(PARTNER, NOON, "reply")
    s.note_incoming(PARTNER, NOON + 60)
    assert decide_followup(s, PARTNER, NOON + HOUR, c, rng).reason == "partner_spoke_last"
    # the owner writing after the bot blocks it too
    s.note_incoming(PARTNER, NOON - 10)
    s.note_outgoing(PARTNER, NOON, "reply")
    s.note_outgoing(PARTNER, NOON + 5, "owner")
    assert decide_followup(s, PARTNER, NOON + HOUR, c, rng).reason == "later_outgoing_message"
    # too late is decided once and never fires
    late = state_with(features=["followup"])
    late.note_incoming(PARTNER, 0)
    late.note_outgoing(PARTNER, NOON - 3 * HOUR, "reply")
    assert decide_followup(late, PARTNER, NOON, c, rng).reason == "too_late"
    assert decide_followup(late, PARTNER, NOON, c, rng).reason == "already_decided"


def test_followup_probability_is_rolled_once() -> None:
    c = cfg(followup_probability=0.0)
    s = state_with(features=["followup"])
    s.note_incoming(PARTNER, 0)
    s.note_outgoing(PARTNER, NOON - 30 * 60, "reply")
    assert decide_followup(s, PARTNER, NOON, c, random.Random(0)).reason == "rolled_no"
    assert decide_followup(s, PARTNER, NOON, c, random.Random(0)).reason == "already_decided"


# --- opener -------------------------------------------------------------------------


def test_opener_planned_once_per_local_day_inside_window() -> None:
    c = cfg(opener_daily_probability=1.0)
    s = state_with(features=["opener"])
    plan = plan_opener(s, PARTNER, NOON, c, random.Random(3))
    assert plan["day"] == "2026-09-07" and isinstance(plan["at"], int)
    local = datetime.fromtimestamp(plan["at"], tz=ZoneInfo(TZ))
    assert 10 <= local.hour < 14
    assert plan_opener(s, PARTNER, NOON + HOUR, c, random.Random(9)) is plan  # same day, same plan
    tomorrow = plan_opener(s, PARTNER, NOON + 24 * HOUR, c, random.Random(9))
    assert tomorrow["day"] == "2026-09-08"
    none = plan_opener(
        state_with(), OTHER, NOON, cfg(opener_daily_probability=0.0), random.Random(0)
    )
    assert none["at"] is None


def test_plan_after_the_window_rolls_to_tomorrow() -> None:
    """A bot started after the window used to roll a time in the past and miss the day."""
    c = cfg(opener_daily_probability=1.0)
    evening = int(datetime(2026, 9, 7, 21, 0, tzinfo=ZoneInfo(TZ)).timestamp())
    assert next_window_day(NOON, TZ, 14 * HOUR) == ("2026-09-07", NOON - 12 * HOUR)
    day, midnight = next_window_day(evening, TZ, 14 * HOUR)
    assert day == "2026-09-08"
    s = state_with(features=["opener"])
    plan = plan_opener(s, PARTNER, evening, c, random.Random(3))
    assert plan["day"] == "2026-09-08"
    assert isinstance(plan["at"], int) and plan["at"] > evening
    local = datetime.fromtimestamp(plan["at"], tz=ZoneInfo(TZ))
    assert 10 <= local.hour < 14
    # the plan survives the rest of the evening instead of being re-rolled or missed
    assert decide_opener(s, PARTNER, evening + HOUR, c, random.Random(0)).reason == "not_yet"
    assert plan_opener(s, PARTNER, evening + HOUR, c, random.Random(0)) is plan
    assert plan_opener(s, PARTNER, midnight + 9 * HOUR, c, random.Random(0)) is plan
    assert decide_opener(s, PARTNER, plan["at"] + 60, c, random.Random(0)).kind == "opener"


def test_opener_rules() -> None:
    c = cfg(opener_daily_probability=1.0)
    rng = random.Random(0)
    assert decide_opener(state_with(), PARTNER, NOON, c, rng).reason == "opener_off"
    s = state_with(features=["opener"])
    s.opener_plan[str(PARTNER)] = {"day": "2026-09-07", "at": NOON + 600, "done": False}
    assert decide_opener(s, PARTNER, NOON, c, rng).reason == "not_yet"
    # alive chat: no opener, plan closed for the day
    s.note_incoming(PARTNER, NOON - 2 * HOUR)
    assert decide_opener(s, PARTNER, NOON + 600, c, rng).reason == "not_silent"
    assert decide_opener(s, PARTNER, NOON + 700, c, rng).reason == "done_today"
    # silent chat fires exactly once
    quiet = state_with(features=["opener"])
    quiet.note_incoming(PARTNER, NOON - 48 * HOUR)
    quiet.opener_plan[str(PARTNER)] = {"day": "2026-09-07", "at": NOON, "done": False}
    assert decide_opener(quiet, PARTNER, NOON + 60, c, rng).kind == "opener"
    assert decide_opener(quiet, PARTNER, NOON + 120, c, rng).reason == "done_today"
    # a plan missed by more than the grace period (bot was down) is dropped
    missed = state_with(features=["opener"])
    missed.opener_plan[str(PARTNER)] = {"day": "2026-09-07", "at": NOON - 2 * HOUR, "done": False}
    assert decide_opener(missed, PARTNER, NOON, c, rng).reason == "missed_window"
    # a chat with no known activity counts as silent
    fresh = state_with(features=["opener"])
    fresh.opener_plan[str(PARTNER)] = {"day": "2026-09-07", "at": NOON, "done": False}
    assert decide_opener(fresh, PARTNER, NOON, c, rng).kind == "opener"


def test_decide_prefers_followup_and_reports_both_reasons() -> None:
    c = cfg(followup_probability=1.0, opener_daily_probability=1.0)
    s = state_with(features=["followup", "opener"])
    s.note_incoming(PARTNER, 0)
    s.note_outgoing(PARTNER, NOON - 30 * 60, "reply")
    s.opener_plan[str(PARTNER)] = {"day": "2026-09-07", "at": NOON, "done": False}
    assert decide(s, PARTNER, NOON, c, random.Random(0)).kind == "followup"
    assert decide(state_with(), PARTNER, NOON, c, random.Random(0)).reason == (
        "followup_off,opener_off"
    )


# --- prompt -------------------------------------------------------------------------


def test_initiative_prompt_renders_task_and_history() -> None:
    template = load_prompt("initiative_v1")
    bundle = build_initiative_messages(template, "Радомир", "- коротко", [], [], "opener")
    assert bundle.version == "initiative_v1"
    assert "пишет первым" in bundle.messages[1]["content"]
    assert "(начало разговора)" in bundle.messages[1]["content"]
    assert "Радомир" in bundle.messages[0]["content"] and "${" not in bundle.messages[0]["content"]
    with pytest.raises(ValueError):
        build_initiative_messages(template, "Р", "", [], [], "reply")


# --- scheduler tick, poke, commands ---------------------------------------------------


class FakeInitiative(FakeBackend):
    mode = "initiative"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = super().generate(request)
        return result.model_copy(update={"mode": f"initiative.{request.intent}"})


@pytest.fixture
def twin(tmp_path: Path) -> TwinBot:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    clock = {"now": NOON}
    bot = FakeBot()
    reply_backend = FakeBackend()
    initiative = FakeInitiative(reply="ну чо, как оно?")
    t = TwinBot(
        bot=bot,
        settings=settings(followup_probability=1.0, opener_daily_probability=1.0),
        store=StateStore(tmp_path / "state"),
        backend=reply_backend,
        memory=ConversationMemory(tmp_path / "memory", 10),
        sleep=sleep,
        rng=random.Random(1),
        clock=lambda: clock["now"],
        skip_rate=0.0,
        initiative_backend=initiative,
    )
    t.fake_bot = bot  # type: ignore[attr-defined]
    t.fake_backend = reply_backend  # type: ignore[attr-defined]
    t.fake_initiative = initiative  # type: ignore[attr-defined]
    t.clock_box = clock  # type: ignore[attr-defined]
    return t


@pytest.mark.asyncio
async def test_tick_fails_closed_then_follows_up_once(twin: TwinBot) -> None:
    assert await twin.initiative_tick() == {PARTNER: "no_connection", OTHER: "no_connection"}
    await twin.on_business_connection(connection())
    twin.state.set_feature("followup", True)
    assert await twin.initiative_tick() == {
        PARTNER: "no_bot_reply_yet,opener_off",
        OTHER: "no_bot_reply_yet,opener_off",
    }
    assert await twin.on_business_message(business_message("как дела?")) == "sent"
    assert len(twin.fake_bot.sent) == 2  # the reply, two parts
    twin.clock_box["now"] = NOON + 30 * 60
    assert (await twin.initiative_tick())[PARTNER] == "sent"
    request = twin.fake_initiative.requests[-1]
    assert request.intent == "followup" and request.text == "как дела?"
    assert request.history[-1].is_me  # the bot's own reply is in the history
    assert twin.fake_bot.sent[-1]["business_connection_id"] == CONN
    assert twin.fake_bot.sent[-1]["text"] == "ну чо, как оно?"
    assert twin.memory.turns(PARTNER)[-1].text == "ну чо, как оно?"
    # never twice for the same reply, and never right after an initiative
    twin.clock_box["now"] = NOON + 40 * 60
    assert (await twin.initiative_tick())[PARTNER] == "later_outgoing_message,opener_off"
    # gates still apply
    twin.state.pause(PARTNER, 30, twin._now())
    assert (await twin.initiative_tick())[PARTNER] == "paused"
    twin.state.unpause(PARTNER)
    twin.state.enabled = False
    assert (await twin.initiative_tick())[PARTNER] == "disabled"


@pytest.mark.asyncio
async def test_opener_in_dry_run_logs_and_does_not_send(twin: TwinBot) -> None:
    await twin.on_business_connection(connection())
    twin.state.dry_run_override = True
    twin.state.set_feature("opener", True)
    twin.state.last_incoming_ts.clear()
    twin.state.last_outgoing_ts.clear()
    twin.state.opener_plan[str(PARTNER)] = {"day": "2026-09-07", "at": NOON, "done": False}
    twin.state.opener_plan[str(OTHER)] = {"day": "2026-09-07", "at": None, "done": False}
    assert await twin.initiative_tick() == {
        PARTNER: "dry_run",
        OTHER: "followup_off,no_opener_today",
    }
    assert twin.fake_bot.sent == []
    assert twin.fake_initiative.requests[-1].intent == "opener"
    assert twin.fake_initiative.requests[-1].text == ""  # nothing to retrieve for
    assert (await twin.initiative_tick())[PARTNER] == "followup_off,done_today"
    assert twin.store.load_state().opener_plan[str(PARTNER)]["done"] is True


@pytest.mark.asyncio
async def test_silent_generation_and_missing_backend(twin: TwinBot) -> None:
    await twin.on_business_connection(connection())
    twin.fake_initiative.reply = None
    assert await twin.poke(PARTNER, "opener") == "silent"
    assert twin.fake_bot.sent == []
    twin._initiative_backend = None
    assert await twin.poke(PARTNER, "opener") == "backend_unavailable"
    assert await twin.poke(9999, "opener") == "not_allowed"


@pytest.mark.asyncio
async def test_peer_telegram_refuses_is_marked_and_cleared(twin: TwinBot) -> None:
    """A first message to a chat the business account cannot write to fails deterministically."""
    await twin.on_business_connection(connection())
    twin.state.set_feature("opener", True)
    refusal = TelegramBadRequest(
        method=SendMessage(chat_id=PARTNER, text="x"),
        message="Bad Request: BUSINESS_PEER_USAGE_MISSING",
    )
    twin.fake_bot.fail_with = [refusal]
    assert await twin.poke(PARTNER, "opener") == "peer_unavailable"
    assert twin.store.load_state().is_peer_blocked(PARTNER)
    # no second attempt: the gate answers before the model is called again
    calls = len(twin.fake_initiative.requests)
    assert await twin.poke(PARTNER, "opener") == "peer_unavailable"
    assert len(twin.fake_initiative.requests) == calls
    assert (await twin.initiative_tick())[PARTNER] == "peer_unavailable"
    # an incoming message proves the dialog exists and clears the mark
    assert await twin.on_business_message(business_message("ты тут?")) == "sent"
    assert not twin.state.is_peer_blocked(PARTNER)
    assert await twin.poke(PARTNER, "opener") == "sent"


@pytest.mark.asyncio
async def test_commands_switch_features_and_poke(twin: TwinBot) -> None:
    await twin.on_business_connection(connection())
    assert await twin.on_direct_message(direct_message("/followup on")) == "followup on"
    assert await twin.on_direct_message(direct_message("/twin opener on")) == "opener on"
    assert twin.store.load_state().features == {"followup": True, "opener": True}
    assert await twin.on_direct_message(direct_message("/opener maybe")) == "usage: /opener on|off"
    status = await twin.on_direct_message(direct_message("/status"))
    assert status and "/followup — on" in status and "/opener — on" in status
    assert "окно опенера 10:00-14:00 Europe/Moscow" in status
    assert "/aggro — normal" in status
    assert await twin.on_direct_message(direct_message("/poke")) == (
        "usage: /poke <user_id> [followup|opener]"
    )
    reply = await twin.on_direct_message(direct_message(f"/poke {PARTNER}"))
    assert reply == "poke opener -> sent"
    assert twin.fake_bot.sent[-2]["chat_id"] == PARTNER  # the initiative itself
    assert twin.fake_bot.sent[-1]["chat_id"] == ADMIN  # the confirmation
    assert twin.fake_initiative.requests[-1].intent == "opener"
    reply = await twin.on_direct_message(direct_message(f"/twin poke {PARTNER} followup"))
    assert reply == "poke followup -> sent"
    assert await twin.on_direct_message(direct_message(f"/poke {PARTNER}", sender=PARTNER)) is None
    assert await twin.on_direct_message(direct_message("/help")) == HELP
    assert await twin.on_direct_message(direct_message("/start@some_bot")) == HELP


def test_parse_command_and_menu() -> None:
    assert parse_command("/twin") == ("help", [])
    assert parse_command("/twin status") == ("status", [])
    assert parse_command("/status@some_bot") == ("status", [])
    assert parse_command("/pause 5 10") == ("pause", ["5", "10"])
    assert parse_command("/unknown") is None
    assert parse_command("привет /status") is None
    for name, description in COMMANDS:
        assert name.isascii() and name.islower() and 1 <= len(name) <= 32
        assert 1 <= len(description) <= 256
    assert [name for name, _ in COMMANDS] == [
        line.split(" — ")[0][1:] for line in HELP.splitlines() if line.startswith("/")
    ]


# --- aggression -----------------------------------------------------------------------


def test_aggression_scales_volume_not_style() -> None:
    base = dict(followup_probability=0.5, opener_daily_probability=0.25)
    normal = aggression_resolve("normal", **base)
    assert normal.level == "normal"
    assert (normal.followup_probability, normal.opener_daily_probability) == (0.5, 0.25)
    assert normal.skip_rate == pytest.approx(SKIP_RATE) and normal.max_parts == MAX_PARTS
    low, high = aggression_resolve("low", **base), aggression_resolve("high", **base)
    assert low.followup_probability < normal.followup_probability < high.followup_probability
    assert low.opener_daily_probability < normal.opener_daily_probability
    assert high.skip_rate < normal.skip_rate < low.skip_rate
    assert low.max_parts == 2 and high.max_parts == MAX_PARTS
    # probabilities stay in range and an unknown level falls back to normal
    loud = aggression_resolve("high", followup_probability=0.8, opener_daily_probability=0.9)
    assert loud.followup_probability == 1.0 and loud.opener_daily_probability == 1.0
    assert aggression_resolve("ЛЮТЫЙ", **base) == normal
    assert aggression_resolve(None, **base) == normal
    assert "пропуск" in normal.describe() and "до 5 сообщений" in normal.describe()


@pytest.mark.asyncio
async def test_aggro_command_changes_behaviour(twin: TwinBot) -> None:
    await twin.on_business_connection(connection())
    twin.skip_rate = None  # follow the level instead of the test override
    assert twin.aggression().level == "normal"
    assert await twin.on_direct_message(direct_message("/aggro ЛЮТО")) == (
        "usage: /aggro low|normal|high"
    )
    reply = await twin.on_direct_message(direct_message("/aggro low"))
    assert reply is not None and reply.startswith("aggro low")
    assert twin.store.load_state().aggression == "low"
    assert twin.aggression().level == "low"
    # a quiet twin ignores more and never floods: two messages per reply at most
    assert twin.aggression().skip_rate > 0.1
    twin.fake_backend.reply = "раз\nдва\nтри\nчетыре"
    assert await twin.on_business_message(business_message("как оно?")) == "sent"
    to_partner = [m for m in twin.fake_bot.sent if m["chat_id"] == PARTNER]
    assert [m["text"] for m in to_partner] == ["раз", "два\nтри\nчетыре"]
    # the initiative config follows the same switch
    assert twin.initiative_config().followup_probability < twin.initiative_cfg.followup_probability
    await twin.on_direct_message(direct_message("/aggro high"))
    assert twin.initiative_config().followup_probability == 1.0
    status = await twin.on_direct_message(direct_message("/status"))
    assert status and "/aggro — high" in status


# --- a dead gateway -------------------------------------------------------------------


class BrokenBackend(FakeBackend):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise self.error


@pytest.mark.asyncio
async def test_gateway_failure_is_logged_not_raised(twin: TwinBot) -> None:
    """Retrieval or the gateway dying must not kill the handler or the initiative loop."""
    await twin.on_business_connection(connection())
    quota = RuntimeError(
        "dashscope/text-embedding-v4: Error code: 422 - You have exceeded your usage limit."
    )
    twin._backends[Mode.RAG] = BrokenBackend(quota)
    assert await twin.on_business_message(business_message("привет")) == "quota_exceeded"
    assert twin.fake_bot.sent == []
    twin._initiative_backend = BrokenBackend(ValueError("index is gone"))
    assert await twin.poke(PARTNER, "opener") == "generation_failed"
    twin.state.set_feature("opener", True)
    twin.state.note_incoming(PARTNER, NOON - 48 * HOUR)  # silent long enough for an opener
    twin.state.opener_plan[str(PARTNER)] = {"day": "2026-09-07", "at": NOON, "done": False}
    assert (await twin.initiative_tick())[PARTNER] == "generation_failed"
    assert twin.fake_bot.sent == []
