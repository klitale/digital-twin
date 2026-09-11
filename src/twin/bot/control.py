"""Control commands in the direct chat with the bot (section 8.3).

Accepted only from ``ADMIN_USER_IDS``, as ``/twin <command>`` or as the bare
``/<command>`` that Telegram's command menu offers (``COMMANDS`` is registered with
``setMyCommands`` at startup). Commands change persisted state, so they survive
restarts; ``/status`` summarises everything an operator needs. ``poke`` needs the
Telegram layer and is executed in ``handlers.py``; it is only parsed here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from twin.bot.aggression import LEVELS, Aggression
from twin.bot.initiative import FEATURES
from twin.bot.state import BotState, BusinessConnectionRecord
from twin.config import Mode
from twin.core.memory import ConversationMemory

# every on|off switch, in the order the command list shows them
SWITCHES = (*FEATURES, "learn")

# (command, description shown in Telegram's menu). User-facing, hence Russian.
COMMANDS: list[tuple[str, str]] = [
    ("status", "Состояние: связь, режим, dry-run, паузы, инициатива"),
    ("on", "Включить бота"),
    ("off", "Выключить бота (ничего не отвечает)"),
    ("mode", "Режим генерации: rag | finetuned | hybrid"),
    ("dryrun", "on|off|auto — генерировать, но не отправлять"),
    ("pause", "<user_id> <минуты> — пауза в чате, 0 = снять"),
    ("reset", "<user_id> — забыть последние сообщения собеседника"),
    ("followup", "on|off — дожимать, если собеседник замолчал после ответа"),
    ("opener", "on|off — иногда писать первым после долгой тишины"),
    ("aggro", "low|normal|high — насколько бот наваливает"),
    ("learn", "on|off — запоминать факты из живой переписки"),
    ("facts", "<user_id> — показать, что бот запомнил про собеседника"),
    ("forget", "<user_id> — стереть запомненные факты"),
    ("poke", "<user_id> [followup|opener] — написать собеседнику сейчас"),
    ("help", "Список команд"),
]
KNOWN = {name for name, _ in COMMANDS} | {"start", "?", "aggression"}

HELP = "\n".join(f"/{name} — {description}" for name, description in COMMANDS) + (
    "\n\nКаждая команда работает и как /twin <команда>."
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
    aggression: Aggression | None = None
    initiative_status: str = ""
    guard_status: str = ""


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """``/twin status`` and ``/status@bot`` both give ``("status", [])``; None otherwise."""
    parts = text.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None
    head = parts[0][1:].split("@")[0].lower()
    if head == "twin":
        if len(parts) == 1:
            return ("help", [])
        return (parts[1].lower(), parts[2:])
    if head in KNOWN:
        return (head, parts[1:])
    return None


def effective_dry_run(state: BotState, settings_dry_run: bool, cli_dry_run: bool) -> bool:
    if cli_dry_run:
        return True
    if state.dry_run_override is not None:
        return state.dry_run_override
    return settings_dry_run


def effective_mode(state: BotState, settings_mode: Mode) -> Mode:
    return Mode(state.mode) if state.mode else settings_mode


def dry_run_text(ctx: ControlContext, cli_dry_run: bool) -> str:
    """Say it in the words of ``/dryrun on|off|auto`` instead of a bare boolean."""
    if cli_dry_run:
        return "on (бот запущен с --dry-run)"
    if ctx.state.dry_run_override is not None:
        return ("on" if ctx.state.dry_run_override else "off") + " (задано командой)"
    return ("on" if ctx.settings_dry_run else "off") + " (из .env)"


def status_text(ctx: ControlContext, cli_dry_run: bool = False) -> str:
    """One line per switch, in the same order and wording as the command list."""
    state = ctx.state
    aggression = ctx.aggression
    switches = [
        f"/on /off — бот: {'on' if state.enabled else 'off'}",
        f"/mode — {effective_mode(state, ctx.settings_mode).value}",
        f"/dryrun — {dry_run_text(ctx, cli_dry_run)}",
        *(f"/{name} — {'on' if state.feature_on(name) else 'off'}" for name in SWITCHES),
        f"/aggro — {aggression.describe() if aggression else 'normal'}",
    ]
    conn = ctx.connection
    if conn is None:
        connection = "связь: нет"
    else:
        connection = (
            f"связь: user {conn.user_id}, "
            f"{'включена' if conn.is_enabled else 'выключена'}, "
            f"{'отвечать можно' if conn.can_reply else 'отвечать нельзя'}"
        )
    pauses = [
        f"{chat_id} — {max(0, until - ctx.now) // 60} мин"
        for chat_id, until in state.paused_until.items()
        if until > ctx.now
    ]
    blocked = [chat for chat, on in state.peer_write_blocked.items() if on]
    disabled = [chat for chat, on in state.chat_enabled.items() if not on]
    lines = [
        *switches,
        "",
        connection,
        f"собеседников: {len(ctx.allowed_user_ids)}",
        "паузы: " + (", ".join(pauses) if pauses else "нет"),
        "выключенные чаты: " + (", ".join(disabled) if disabled else "нет"),
    ]
    if blocked:
        lines.append("нельзя написать первым (Telegram): " + ", ".join(blocked))
    if ctx.initiative_status:
        lines.append(ctx.initiative_status)
    if ctx.guard_status:
        lines.append(ctx.guard_status)
    return "\n".join(lines)


def handle_control(text: str, ctx: ControlContext, cli_dry_run: bool = False) -> str | None:
    """Return the reply text, or ``None`` when the message is not a control command.

    The caller persists ``ctx.state`` after a non-None result. ``poke`` is not handled
    here (it sends a message): the caller checks ``parse_command`` first.
    """
    parsed = parse_command(text)
    if parsed is None:
        return None
    command, rest = parsed
    state = ctx.state
    if command in ("help", "?", "start"):
        return HELP
    if command in ("on", "off"):
        state.enabled = command == "on"
        return f"bot {'enabled' if state.enabled else 'disabled'}"
    if command == "status":
        return status_text(ctx, cli_dry_run)
    if command == "mode":
        if len(rest) != 1 or rest[0] not in Mode.__members__.values():
            return "usage: /mode rag|finetuned|hybrid"
        state.mode = rest[0]
        return f"mode set to {rest[0]} (takes effect on the next message)"
    if command == "reset":
        if len(rest) != 1 or not rest[0].lstrip("-").isdigit():
            return "usage: /reset <user_id>"
        cleared = ctx.memory.reset(int(rest[0]))
        return f"memory for {rest[0]} {'cleared' if cleared else 'was already empty'}"
    if command == "pause":
        if len(rest) != 2 or not rest[0].lstrip("-").isdigit() or not rest[1].isdigit():
            return "usage: /pause <user_id> <minutes>"
        chat_id, minutes = int(rest[0]), int(rest[1])
        if minutes == 0:
            state.unpause(chat_id)
            return f"chat {chat_id} unpaused"
        state.pause(chat_id, minutes, ctx.now)
        return f"chat {chat_id} paused for {minutes} min"
    if command == "dryrun":
        if len(rest) != 1 or rest[0] not in ("on", "off", "auto"):
            return "usage: /dryrun on|off|auto"
        state.dry_run_override = None if rest[0] == "auto" else rest[0] == "on"
        return f"dry_run now {effective_dry_run(state, ctx.settings_dry_run, cli_dry_run)}"
    if command in ("facts", "forget"):
        return None  # needs the fact store; handled in handlers.py
    if command in ("aggro", "aggression"):
        if len(rest) != 1 or rest[0] not in LEVELS:
            return "usage: /aggro " + "|".join(LEVELS)
        state.aggression = rest[0]
        return f"aggro {rest[0]} (действует со следующего сообщения)"
    if command in SWITCHES:
        if len(rest) != 1 or rest[0] not in ("on", "off"):
            return f"usage: /{command} on|off"
        state.set_feature(command, rest[0] == "on")
        return f"{command} {rest[0]}"
    if command == "poke":
        return "usage: /poke <user_id> [followup|opener]"
    return HELP


def is_admin(user_id: int | None, admin_ids: list[int]) -> bool:
    return user_id is not None and user_id in admin_ids


def now_ts() -> int:
    return int(time.time())
