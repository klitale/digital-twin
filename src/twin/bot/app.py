"""aiogram wiring: long polling for business updates and control messages."""

from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand

from twin.bot.control import COMMANDS
from twin.bot.handlers import TwinBot
from twin.bot.state import StateStore
from twin.config import Mode, Settings
from twin.core.factory import build_backend, build_initiative_backend, build_memory
from twin.logsetup import get_logger

log = get_logger("twin.bot.app")
ALLOWED_UPDATES = [
    "message",
    "business_connection",
    "business_message",
    "edited_business_message",
    "deleted_business_messages",
]


def build_twin_bot(bot: Bot, settings: Settings, cli_dry_run: bool) -> TwinBot:
    store = StateStore(settings.state_dir)
    twin = TwinBot(
        bot=bot,
        settings=settings,
        store=store,
        backend=None,
        memory=build_memory(settings),
        cli_dry_run=cli_dry_run,
        backend_factory=lambda mode: build_backend(settings, mode),
        initiative_factory=lambda: build_initiative_backend(settings),
    )
    twin.backend_for(twin.mode)  # fail fast on a broken index / missing profile
    return twin


async def run_bot(settings: Settings, cli_dry_run: bool = False) -> None:
    settings.require_bot()
    assert settings.tg_bot_token is not None
    bot = Bot(
        token=settings.tg_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=None),
    )
    twin = build_twin_bot(bot, settings, cli_dry_run)
    log.info("bot.starting", report=twin.startup_report())
    dp = Dispatcher()
    dp.business_connection.register(twin.on_business_connection)
    dp.business_message.register(twin.on_business_message)
    dp.edited_business_message.register(twin.on_edited_business_message)
    dp.deleted_business_messages.register(twin.on_deleted_business_messages)
    dp.message.register(twin.on_direct_message)
    try:
        await bot.set_my_commands([BotCommand(command=c, description=d) for c, d in COMMANDS])
    except Exception as exc:  # cosmetic: the commands still work without the menu
        log.warning("set_my_commands.failed", error=str(exc))
    initiative = asyncio.create_task(twin.initiative_loop(), name="initiative")
    try:
        await dp.start_polling(bot, allowed_updates=ALLOWED_UPDATES)
    finally:
        initiative.cancel()
        await bot.session.close()


def main(settings: Settings, cli_dry_run: bool = False) -> None:
    asyncio.run(run_bot(settings, cli_dry_run))


__all__ = ["Mode", "main", "run_bot"]
