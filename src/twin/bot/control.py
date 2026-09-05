"""``/twin`` control commands in the direct chat with the bot (section 8.3).

Accepted only from ``ADMIN_USER_IDS``. Commands change persisted state, so they
survive restarts; ``/twin status`` summarises everything a dry-run operator needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from twin.bot.state import BotState, BusinessConnectionRecord
from twin.config import Mode
from twin.core.memory import ConversationMemory

HELP = (
    "/twin on|off — global switch\n"
    "/twin status — connection, mode, pauses\n"
    "/twin mode rag|finetuned|hybrid\n"
    "/twin reset <user_id> — clear that partner's memory\n"
    "/twin pause <user_id> <minutes> — 0 to unpause\n"
    "/twin dryrun on|off|auto — override DRY_RUN (auto = from .env)"
)


@dataclass(frozen=True)
class ControlContext:
    state: BotState
    memory: ConversationMemory
    connection: BusinessConnectionRecord | None
    settings_mode: Mode
    settings_dry_run: bool
    allowed_user_ids: list[int]
    now: int


def effective_dry_run(state: BotState, settings_dry_run: bool, cli_dry_run: bool) -> bool:
    if cli_dry_run:
        return True
    if state.dry_run_override is not None:
        return state.dry_run_override
    return settings_dry_run


def effective_mode(state: BotState, settings_mode: Mode) -> Mode:
    return Mode(state.mode) if state.mode else settings_mode


def status_text(ctx: ControlContext, cli_dry_run: bool = False) -> str:
    conn = ctx.connection
    if conn is None:
        connection = "connection: none"
    else:
        connection = (
            f"connection: user {conn.user_id}, enabled={conn.is_enabled}, "
            f"can_reply={conn.can_reply}"
        )
    pauses = [
        f"{chat_id}: {max(0, until - ctx.now) // 60} min"
        for chat_id, until in ctx.state.paused_until.items()
        if until > ctx.now
    ]
    disabled = [chat for chat, on in ctx.state.chat_enabled.items() if not on]
    return "\n".join(
        [
            f"bot: {'on' if ctx.state.enabled else 'off'}",
            f"mode: {effective_mode(ctx.state, ctx.settings_mode).value}",
            f"dry_run: {effective_dry_run(ctx.state, ctx.settings_dry_run, cli_dry_run)}"
            + (
                f" (override {ctx.state.dry_run_override})"
                if ctx.state.dry_run_override is not None
                else ""
            ),
            connection,
            f"allowed users: {len(ctx.allowed_user_ids)}",
            "paused: " + (", ".join(pauses) if pauses else "none"),
            "disabled chats: " + (", ".join(disabled) if disabled else "none"),
        ]
    )


def handle_control(text: str, ctx: ControlContext, cli_dry_run: bool = False) -> str | None:
    """Return the reply text, or ``None`` when the message is not a /twin command.

    The caller persists ``ctx.state`` after a non-None result.
    """
    parts = text.strip().split()
    if not parts or parts[0].split("@")[0] != "/twin":
        return None
    args = parts[1:]
    if not args or args[0] in ("help", "?"):
        return HELP
    command, rest = args[0], args[1:]
    state = ctx.state
    if command in ("on", "off"):
        state.enabled = command == "on"
        return f"bot {'enabled' if state.enabled else 'disabled'}"
    if command == "status":
        return status_text(ctx, cli_dry_run)
    if command == "mode":
        if len(rest) != 1 or rest[0] not in Mode.__members__.values():
            return "usage: /twin mode rag|finetuned|hybrid"
        state.mode = rest[0]
        return f"mode set to {rest[0]} (takes effect on the next message)"
    if command == "reset":
        if len(rest) != 1 or not rest[0].lstrip("-").isdigit():
            return "usage: /twin reset <user_id>"
        cleared = ctx.memory.reset(int(rest[0]))
        return f"memory for {rest[0]} {'cleared' if cleared else 'was already empty'}"
    if command == "pause":
        if len(rest) != 2 or not rest[0].lstrip("-").isdigit() or not rest[1].isdigit():
            return "usage: /twin pause <user_id> <minutes>"
        chat_id, minutes = int(rest[0]), int(rest[1])
        if minutes == 0:
            state.unpause(chat_id)
            return f"chat {chat_id} unpaused"
        state.pause(chat_id, minutes, ctx.now)
        return f"chat {chat_id} paused for {minutes} min"
    if command == "dryrun":
        if len(rest) != 1 or rest[0] not in ("on", "off", "auto"):
            return "usage: /twin dryrun on|off|auto"
        state.dry_run_override = None if rest[0] == "auto" else rest[0] == "on"
        return f"dry_run now {effective_dry_run(state, ctx.settings_dry_run, cli_dry_run)}"
    return HELP


def is_admin(user_id: int | None, admin_ids: list[int]) -> bool:
    return user_id is not None and user_id in admin_ids


def now_ts() -> int:
    return int(time.time())
