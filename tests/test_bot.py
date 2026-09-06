from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter
from aiogram.methods import SendMessage
from aiogram.types import (
    BusinessBotRights,
    BusinessConnection,
    BusinessMessagesDeleted,
    Chat,
    Message,
    User,
)

from twin.bot.business import evaluate_connection, record_from_update
from twin.bot.control import HELP, ControlContext, handle_control
from twin.bot.handlers import TwinBot
from twin.bot.humanize import reply_delay_seconds, should_skip, split_parts
from twin.bot.state import BotState, StateStore
from twin.config import Mode, Settings
from twin.core.backends import GenerationRequest, GenerationResult
from twin.core.memory import ConversationMemory, MemoryTurn

OWNER, PARTNER, OTHER, ADMIN = 5001, 5002, 5003, 5004
CONN = "conn-1"
NOW = 1_800_000_000
DATE = datetime.fromtimestamp(NOW, tz=UTC)


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "tg_bot_token": "1:abc",
        "business_owner_id": OWNER,
        "admin_user_ids": f"{ADMIN}",
        "allowed_user_ids": f"{PARTNER},{OTHER}",
        "twin_name": "Радомир",
        "pause_minutes": 30,
        "dry_run": False,
        "data_dir": "data",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def user(user_id: int) -> User:
    return User(id=user_id, is_bot=False, first_name=f"u{user_id}")


def connection(
    user_id: int = OWNER, can_reply: bool = True, enabled: bool = True
) -> BusinessConnection:
    return BusinessConnection(
        id=CONN,
        user=user(user_id),
        user_chat_id=user_id,
        date=DATE,
        is_enabled=enabled,
        rights=BusinessBotRights(can_reply=can_reply),
    )


def business_message(
    text: str | None = "привет",
    sender: int = PARTNER,
    message_id: int = 10,
    conn_id: str | None = CONN,
    chat_type: str = "private",
    chat_id: int | None = None,
) -> Message:
    if chat_id is None:
        chat_id = sender if chat_type == "private" else -100
    return Message(
        message_id=message_id,
        date=DATE,
        chat=Chat(id=chat_id, type=chat_type),
        from_user=user(sender),
        text=text,
        business_connection_id=conn_id,
    )


def direct_message(text: str, sender: int = ADMIN) -> Message:
    return Message(
        message_id=1,
        date=DATE,
        chat=Chat(id=sender, type="private"),
        from_user=user(sender),
        text=text,
    )


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.fail_with: list[Exception] = []
        self._next_id = 100

    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> Message:
        if self.fail_with:
            raise self.fail_with.pop(0)
        self._next_id += 1
        self.sent.append({"chat_id": chat_id, "text": text, **kwargs})
        return Message(
            message_id=self._next_id, date=DATE, chat=Chat(id=chat_id, type="private"), text=text
        )

    async def send_chat_action(self, chat_id: int, action: str, **kwargs: Any) -> bool:
        self.actions.append({"chat_id": chat_id, "action": action, **kwargs})
        return True


class FakeBackend:
    mode = "rag"

    def __init__(self, reply: str | None = "норм)\nа ты как?") -> None:
        self.reply = reply
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self.reply,
            mode="rag",
            model="fake",
            prompt_version="rag_v1",
            retrieved_ids=["1002:1"],
            params={},
            latency_ms=1,
            attempts=1 if self.reply else 2,
            rejected=[] if self.reply else ["assistant_speak:как ии"] * 2,
            raw_texts=[self.reply or ""],
        )


@pytest.fixture
def parts(tmp_path: Path) -> dict[str, Any]:
    store = StateStore(tmp_path / "state")
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def make(settings_obj: Settings | None = None, **kwargs: Any) -> TwinBot:
        bot = FakeBot()
        backend = kwargs.pop("backend", FakeBackend())
        skip_rate = kwargs.pop("skip_rate", 0.0)
        twin = TwinBot(
            bot=bot,
            settings=settings_obj or settings(),
            store=store,
            backend=backend,
            memory=ConversationMemory(tmp_path / "memory", 10),
            sleep=sleep,
            rng=random.Random(1),
            clock=lambda: NOW,
            skip_rate=skip_rate,
            **kwargs,
        )
        twin.fake_bot = bot  # type: ignore[attr-defined]
        twin.fake_backend = backend  # type: ignore[attr-defined]
        return twin

    return {"store": store, "sleeps": sleeps, "make": make, "tmp": tmp_path}


# --- connection ----------------------------------------------------------------------


def test_connection_record_and_evaluation() -> None:
    record = record_from_update(connection(), now=NOW)
    assert record.business_connection_id == CONN and record.user_id == OWNER
    assert record.can_reply and record.is_enabled and record.updated_at == NOW
    assert evaluate_connection(record, OWNER).operational
    assert (
        evaluate_connection(record, OWNER + 1).reason
        == "connection user_id is not BUSINESS_OWNER_ID"
    )
    assert not evaluate_connection(
        record_from_update(connection(can_reply=False)), OWNER
    ).operational
    assert not evaluate_connection(record_from_update(connection(enabled=False)), OWNER).operational
    assert evaluate_connection(None, OWNER).reason == "no connection stored"
    legacy = BusinessConnection(
        id=CONN, user=user(OWNER), user_chat_id=OWNER, date=DATE, is_enabled=True, can_reply=True
    )
    assert record_from_update(legacy).can_reply is True


@pytest.mark.asyncio
async def test_connection_is_persisted_and_survives_restart(parts: dict[str, Any]) -> None:
    twin = parts["make"]()
    assert "No business connection stored" in twin.startup_report()
    status = await twin.on_business_connection(connection())
    assert status.operational
    assert parts["store"].load_connection().business_connection_id == CONN
    restarted = parts["make"]()
    assert restarted.connection_status().operational
    assert "connection: ok" in restarted.startup_report()


@pytest.mark.asyncio
async def test_wrong_owner_connection_is_refused(parts: dict[str, Any]) -> None:
    twin = parts["make"]()
    status = await twin.on_business_connection(connection(user_id=OWNER + 7))
    assert not status.operational
    assert await twin.on_business_message(business_message()) == "no_connection"
    assert twin.fake_bot.sent == []


# --- fail-closed gates ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_gates_in_order(parts: dict[str, Any]) -> None:
    twin = parts["make"]()
    await twin.on_business_connection(connection())
    assert (
        await twin.on_business_message(business_message(conn_id="other")) == "connection_mismatch"
    )
    assert await twin.on_business_message(business_message(chat_type="group")) == "not_private"
    assert await twin.on_business_message(business_message(sender=OWNER + 99)) == "not_allowed"
    twin.state.enabled = False
    assert await twin.on_business_message(business_message()) == "disabled"
    twin.state.enabled = True
    twin.state.set_chat_enabled(PARTNER, False)
    assert await twin.on_business_message(business_message()) == "disabled"
    twin.state.set_chat_enabled(PARTNER, True)
    twin.state.pause(PARTNER, 30, NOW)
    assert await twin.on_business_message(business_message()) == "paused"
    twin.state.unpause(PARTNER)
    assert await twin.on_business_message(business_message(text=None)) == "non_text"
    assert twin.fake_bot.sent == [] and twin.fake_backend.requests == []


@pytest.mark.asyncio
async def test_reply_is_delivered_in_parts_with_typing(parts: dict[str, Any]) -> None:
    twin = parts["make"]()
    await twin.on_business_connection(connection())
    assert await twin.on_business_message(business_message("как дела?")) == "sent"
    sent = twin.fake_bot.sent
    assert [m["text"] for m in sent] == ["норм)", "а ты как?"]
    assert all(m["business_connection_id"] == CONN and m["chat_id"] == PARTNER for m in sent)
    assert twin.fake_bot.actions and all(
        a["business_connection_id"] == CONN for a in twin.fake_bot.actions
    )
    assert parts["sleeps"] and 2.0 <= sum(parts["sleeps"]) <= 25.0
    state = parts["store"].load_state()
    assert state.was_sent_by_bot(PARTNER, 101) and state.was_sent_by_bot(PARTNER, 102)
    request = twin.fake_backend.requests[0]
    assert (
        request.text == "как дела?" and request.partner_id == PARTNER and request.chat_id == PARTNER
    )
    memory = twin.memory.turns(PARTNER)
    assert [(t.is_me, t.text) for t in memory] == [(False, "как дела?"), (True, "норм)\nа ты как?")]
    # second message carries history and the previous partner text
    await twin.on_business_message(business_message("ок", message_id=11))
    second = twin.fake_backend.requests[1]
    assert second.previous_partner_text == "как дела?" and len(second.history) == 2


@pytest.mark.asyncio
async def test_dry_run_generates_but_never_sends(parts: dict[str, Any]) -> None:
    twin = parts["make"](cli_dry_run=True)
    await twin.on_business_connection(connection())
    assert await twin.on_business_message(business_message()) == "dry_run"
    assert twin.fake_backend.requests[0].dry_run is True
    assert twin.fake_bot.sent == [] and twin.fake_bot.actions == []
    env_dry = parts["make"](settings(dry_run=True))
    assert await env_dry.on_business_message(business_message()) == "dry_run"


@pytest.mark.asyncio
async def test_validation_failure_means_silence(parts: dict[str, Any]) -> None:
    twin = parts["make"](backend=FakeBackend(reply=None))
    await twin.on_business_connection(connection())
    assert await twin.on_business_message(business_message()) == "silent"
    assert twin.fake_bot.sent == []
    assert [t.is_me for t in twin.memory.turns(PARTNER)] == [False]


@pytest.mark.asyncio
async def test_skip_rate(parts: dict[str, Any]) -> None:
    twin = parts["make"](skip_rate=1.0)
    await twin.on_business_connection(connection())
    assert await twin.on_business_message(business_message("просто текст")) == "skipped"
    assert await twin.on_business_message(business_message("а это вопрос?")) == "sent"


# --- autopause -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autopause_when_owner_writes_and_bot_messages_are_recognised(
    parts: dict[str, Any],
) -> None:
    twin = parts["make"]()
    await twin.on_business_connection(connection())
    assert await twin.on_business_message(business_message("привет")) == "sent"
    bot_message_id = twin.fake_bot.sent and 101
    # Telegram echoes the bot's own message as an owner message: recognised, no pause
    echo = business_message("норм)", sender=OWNER, message_id=bot_message_id, chat_id=PARTNER)
    assert await twin.on_business_message(echo) == "own_bot_message"
    assert not twin.state.is_paused(PARTNER, NOW)
    # a message typed by the owner in that chat pauses it
    human = business_message("сам отвечу", sender=OWNER, message_id=555, chat_id=PARTNER)
    assert await twin.on_business_message(human) == "owner_message_autopause"
    assert (
        twin.state.is_paused(PARTNER, NOW) and twin.state.pause_remaining(PARTNER, NOW) == 30 * 60
    )
    assert await twin.on_business_message(business_message("ещё", message_id=12)) == "paused"
    assert parts["store"].load_state().is_paused(PARTNER, NOW)
    assert not twin.state.is_paused(PARTNER, NOW + 31 * 60)


@pytest.mark.asyncio
async def test_manual_pause_fallback_when_owner_messages_are_not_delivered(
    parts: dict[str, Any],
) -> None:
    twin = parts["make"]()
    await twin.on_business_connection(connection())
    reply = await twin.on_direct_message(direct_message(f"/twin pause {PARTNER} 15"))
    assert reply == f"chat {PARTNER} paused for 15 min"
    assert await twin.on_business_message(business_message()) == "paused"
    assert (
        await twin.on_direct_message(direct_message(f"/twin pause {PARTNER} 0"))
        == f"chat {PARTNER} unpaused"
    )
    assert await twin.on_business_message(business_message(message_id=13)) == "sent"


# --- control commands --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_control_commands_only_from_admins(parts: dict[str, Any]) -> None:
    twin = parts["make"]()
    await twin.on_business_connection(connection())
    assert await twin.on_direct_message(direct_message("/twin status", sender=PARTNER)) is None
    assert twin.fake_bot.sent == []
    status = await twin.on_direct_message(direct_message("/twin status"))
    assert status and "bot: on" in status and "mode: rag" in status and f"user {OWNER}" in status
    assert (
        twin.fake_bot.sent[-1]["chat_id"] == ADMIN
        and "business_connection_id" not in twin.fake_bot.sent[-1]
    )
    assert await twin.on_direct_message(direct_message("привет")) is None
    assert await twin.on_direct_message(direct_message("/twin")) == HELP
    assert await twin.on_direct_message(direct_message("/twin off")) == "bot disabled"
    assert parts["store"].load_state().enabled is False
    assert await twin.on_business_message(business_message()) == "disabled"
    assert await twin.on_direct_message(direct_message("/twin on")) == "bot enabled"
    assert await twin.on_direct_message(direct_message("/twin dryrun on")) == "dry_run now True"
    assert await twin.on_business_message(business_message(message_id=14)) == "dry_run"
    assert await twin.on_direct_message(direct_message("/twin dryrun auto")) == "dry_run now False"
    hybrid = await twin.on_direct_message(direct_message("/twin mode hybrid"))
    assert hybrid is not None and hybrid.startswith("mode set to hybrid")
    assert await twin.on_business_message(business_message(message_id=15)) == "backend_unavailable"
    back = await twin.on_direct_message(direct_message("/twin mode rag"))
    assert back is not None and back.startswith("mode set to rag")
    twin.memory.append(PARTNER, MemoryTurn(is_me=False, text="x", ts=1))
    assert (
        await twin.on_direct_message(direct_message(f"/twin reset {PARTNER}"))
        == f"memory for {PARTNER} cleared"
    )
    assert (
        await twin.on_direct_message(direct_message("/twin pause x y"))
        == "usage: /pause <user_id> <minutes>"
    )
    business = business_message("/twin off", sender=ADMIN)
    assert (
        await twin.on_direct_message(business) is None
    )  # through the business connection: ignored


def test_handle_control_pure() -> None:
    ctx = ControlContext(
        state=BotState(),
        memory=ConversationMemory(Path("nope"), 10),
        connection=None,
        settings_mode=Mode.RAG,
        settings_dry_run=True,
        allowed_user_ids=[1, 2],
        now=NOW,
    )
    assert handle_control("hello", ctx) is None
    assert handle_control("/twin@twin_bot status", ctx).startswith("bot: on")
    assert "connection: none" in handle_control("/twin status", ctx)
    assert handle_control("/twin mode nope", ctx) == "usage: /mode rag|finetuned|hybrid"
    assert handle_control("/twin dryrun off", ctx) == "dry_run now False"
    assert handle_control("/twin whatever", ctx) == HELP


# --- delivery errors ---------------------------------------------------------------------


def _retry_after(seconds: int) -> TelegramRetryAfter:
    return TelegramRetryAfter(
        method=SendMessage(chat_id=1, text="x"), message="flood", retry_after=seconds
    )


@pytest.mark.asyncio
async def test_retry_after_is_honoured_once(parts: dict[str, Any]) -> None:
    twin = parts["make"](backend=FakeBackend(reply="ок"))
    await twin.on_business_connection(connection())
    twin.fake_bot.fail_with = [_retry_after(3)]
    assert await twin.on_business_message(business_message()) == "sent"
    assert 3.0 in parts["sleeps"] and len(twin.fake_bot.sent) == 1
    twin.fake_bot.fail_with = [_retry_after(3), _retry_after(3)]
    assert await twin.on_business_message(business_message(message_id=11)) == "failed"


@pytest.mark.asyncio
async def test_forbidden_disables_the_chat(parts: dict[str, Any]) -> None:
    twin = parts["make"](backend=FakeBackend(reply="ок"))
    await twin.on_business_connection(connection())
    twin.fake_bot.fail_with = [
        TelegramForbiddenError(method=SendMessage(chat_id=1, text="x"), message="blocked")
    ]
    assert await twin.on_business_message(business_message()) == "forbidden"
    assert parts["store"].load_state().is_chat_enabled(PARTNER) is False
    assert await twin.on_business_message(business_message(message_id=11)) == "disabled"


@pytest.mark.asyncio
async def test_network_errors_are_retried_then_given_up(parts: dict[str, Any]) -> None:
    twin = parts["make"](backend=FakeBackend(reply="ок"))
    await twin.on_business_connection(connection())
    twin.fake_bot.fail_with = [ConnectionError("net")]
    assert await twin.on_business_message(business_message()) == "sent"
    twin.fake_bot.fail_with = [ConnectionError("net")] * 5
    assert await twin.on_business_message(business_message(message_id=11)) == "failed"
    assert len(twin.fake_bot.sent) == 1


@pytest.mark.asyncio
async def test_edited_and_deleted_updates_only_log_ids(parts: dict[str, Any]) -> None:
    twin = parts["make"](backend=FakeBackend(reply="ок"))
    await twin.on_business_connection(connection())
    await twin.on_business_message(business_message())
    edited = business_message("ок", sender=OWNER, message_id=101, chat_id=PARTNER)
    assert await twin.on_edited_business_message(edited) == "bot_message_edited"
    assert (
        await twin.on_edited_business_message(business_message(message_id=7))
        == "human_message_edited"
    )
    deleted = BusinessMessagesDeleted(
        business_connection_id=CONN, chat=Chat(id=PARTNER, type="private"), message_ids=[101, 7]
    )
    assert await twin.on_deleted_business_messages(deleted) == "deleted:2:1"


# --- humanize ------------------------------------------------------------------------------


def test_humanize_helpers() -> None:
    rng = random.Random(0)
    assert 2.0 <= reply_delay_seconds("ок", rng) <= 4.0
    assert reply_delay_seconds("x" * 1000, rng) == 15.0
    assert split_parts("а\n\nб\n в ") == ["а", "б", "в"]
    assert split_parts("\n".join("1234567")) == ["1", "2", "3", "4", "5\n6\n7"]
    assert split_parts("   ") == []
    assert should_skip("вопрос?", random.Random(0), rate=1.0) is False
    assert should_skip("не вопрос", random.Random(0), rate=1.0) is True
    assert should_skip("не вопрос", random.Random(0), rate=0.0) is False
