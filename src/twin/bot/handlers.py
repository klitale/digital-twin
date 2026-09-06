"""The Telegram layer: business updates in, ``IncomingMessage -> text | None`` out.

``TwinBot`` holds the dependencies (settings, state store, backend, memory) and exposes
one coroutine per update type. Every gate in ``on_business_message`` fails closed and
returns a short reason string (used by tests and logs); message bodies of ignored
updates are never logged.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter

from twin.bot.business import (
    CONNECT_INSTRUCTIONS,
    ConnectionStatus,
    evaluate_connection,
    store_connection_update,
)
from twin.bot.control import (
    ControlContext,
    effective_dry_run,
    effective_mode,
    handle_control,
    is_admin,
    parse_command,
)
from twin.bot.humanize import (
    parts_summary,
    pause_between_parts,
    reply_delay_seconds,
    should_skip,
    split_parts,
)
from twin.bot.initiative import (
    FEATURES,
    Decision,
    InitiativeConfig,
    decide,
    describe_plan,
    window_text,
)
from twin.bot.state import BotState, StateStore
from twin.config import ConfigError, Mode, Settings
from twin.core.backends import GenerationBackend, GenerationRequest
from twin.core.memory import ConversationMemory, MemoryTurn
from twin.logsetup import get_logger

log = get_logger("twin.bot")
TYPING_INTERVAL_S = 4.0
SEND_RETRIES = 2


class BotLike(Protocol):
    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> Any: ...

    async def send_chat_action(self, chat_id: int, action: str, **kwargs: Any) -> Any: ...


Sleep = Callable[[float], Awaitable[None]]
BackendFactory = Callable[[Mode], GenerationBackend]
InitiativeFactory = Callable[[], GenerationBackend]


class TwinBot:
    def __init__(
        self,
        bot: BotLike,
        settings: Settings,
        store: StateStore,
        backend: GenerationBackend | None,
        memory: ConversationMemory,
        cli_dry_run: bool = False,
        backend_factory: BackendFactory | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.time,
        skip_rate: float | None = None,
        initiative_backend: GenerationBackend | None = None,
        initiative_factory: InitiativeFactory | None = None,
    ) -> None:
        self.bot = bot
        self.settings = settings
        self.store = store
        self.memory = memory
        self.cli_dry_run = cli_dry_run
        self.backend_factory = backend_factory
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.clock = clock
        self.skip_rate = skip_rate
        self.state: BotState = store.load_state()
        self.connection = store.load_connection()
        self._backends: dict[Mode, GenerationBackend] = {}
        if backend is not None:
            self._backends[Mode(backend.mode)] = backend
        self._initiative_backend = initiative_backend
        self.initiative_factory = initiative_factory
        self.initiative_cfg = InitiativeConfig.from_settings(settings)

    # --- helpers -------------------------------------------------------------------

    @property
    def dry_run(self) -> bool:
        return effective_dry_run(self.state, self.settings.dry_run, self.cli_dry_run)

    @property
    def mode(self) -> Mode:
        return effective_mode(self.state, self.settings.twin_mode)

    def connection_status(self) -> ConnectionStatus:
        return evaluate_connection(self.connection, self.settings.business_owner_id)

    def backend_for(self, mode: Mode) -> GenerationBackend:
        if mode not in self._backends:
            if self.backend_factory is None:
                raise ConfigError(f"no backend for mode {mode.value}")
            self._backends[mode] = self.backend_factory(mode)
        return self._backends[mode]

    def _now(self) -> int:
        return int(self.clock())

    def initiative_backend(self) -> GenerationBackend:
        if self._initiative_backend is None:
            if self.initiative_factory is None:
                raise ConfigError("no initiative backend")
            self._initiative_backend = self.initiative_factory()
        return self._initiative_backend

    def initiative_status(self) -> str:
        now = self._now()
        plans = ", ".join(
            f"{chat_id}: {describe_plan(self.state, chat_id, now, self.initiative_cfg.tz)}"
            for chat_id in self.settings.allowed_user_ids
        )
        return f"opener window {window_text(self.initiative_cfg)}; plans: {plans or 'none'}"

    def startup_report(self) -> str:
        status = self.connection_status()
        lines = [
            f"mode={self.mode.value} dry_run={self.dry_run} enabled={self.state.enabled} "
            f"allowed_users={len(self.settings.allowed_user_ids)}",
            f"connection: {status.reason}",
        ]
        if status.record is None:
            lines.append(CONNECT_INSTRUCTIONS)
        return "\n".join(lines)

    # --- update handlers -------------------------------------------------------------

    async def on_business_connection(self, connection: Any) -> ConnectionStatus:
        status = store_connection_update(connection, self.store, self.settings.business_owner_id)
        self.connection = status.record
        return status

    async def on_edited_business_message(self, message: Any) -> str:
        chat_id = message.chat.id
        by_bot = self.state.was_sent_by_bot(chat_id, message.message_id)
        log.info(
            "business_message.edited", chat_id=chat_id, message_id=message.message_id, by_bot=by_bot
        )
        return "bot_message_edited" if by_bot else "human_message_edited"

    async def on_deleted_business_messages(self, deleted: Any) -> str:
        chat_id = deleted.chat.id
        ids = list(deleted.message_ids)
        by_bot = [i for i in ids if self.state.was_sent_by_bot(chat_id, i)]
        log.info("business_message.deleted", chat_id=chat_id, count=len(ids), by_bot=len(by_bot))
        return f"deleted:{len(ids)}:{len(by_bot)}"

    async def on_direct_message(self, message: Any) -> str | None:
        """Control commands in the direct chat (never through the business connection)."""
        if getattr(message, "business_connection_id", None) or message.chat.type != "private":
            return None
        user = message.from_user
        if not is_admin(user.id if user else None, self.settings.admin_user_ids):
            log.info("control.ignored", user_id=user.id if user else None)
            return None
        text = message.text or ""
        parsed = parse_command(text)
        if parsed is not None and parsed[0] == "poke" and self._poke_args(parsed[1]):
            chat_id, kind = self._poke_args(parsed[1])  # type: ignore[misc]
            log.info("control", user_id=user.id, command=["poke", kind])
            outcome = await self.poke(chat_id, kind)
            reply = f"poke {kind} -> {outcome}"
            await self.bot.send_message(chat_id=message.chat.id, text=reply)
            return reply
        reply = handle_control(
            text,
            ControlContext(
                state=self.state,
                memory=self.memory,
                connection=self.connection,
                settings_mode=self.settings.twin_mode,
                settings_dry_run=self.settings.dry_run,
                allowed_user_ids=self.settings.allowed_user_ids,
                now=self._now(),
                initiative_status=self.initiative_status(),
            ),
            self.cli_dry_run,
        )
        if reply is None:
            return None
        self.store.save_state(self.state)
        log.info("control", user_id=user.id, command=text.split()[:2])
        await self.bot.send_message(chat_id=message.chat.id, text=reply)
        return reply

    @staticmethod
    def _poke_args(rest: list[str]) -> tuple[int, str] | None:
        if not rest or not rest[0].lstrip("-").isdigit():
            return None
        kind = rest[1] if len(rest) > 1 else "opener"
        if kind not in FEATURES:
            return None
        return int(rest[0]), kind

    async def on_business_message(self, message: Any) -> str:
        """Fail-closed pipeline for one incoming message in a connected chat."""
        status = self.connection_status()
        conn_id = getattr(message, "business_connection_id", None)
        if not status.operational or status.record is None:
            log.info("business_message.ignored", reason=status.reason)
            return "no_connection"
        if conn_id != status.record.business_connection_id:
            log.info("business_message.ignored", reason="connection id mismatch")
            return "connection_mismatch"
        chat = message.chat
        if chat.type != "private":
            return "not_private"
        user = message.from_user
        if user is None:
            return "no_sender"
        now = self._now()
        if user.id == self.settings.business_owner_id:
            if self.state.was_sent_by_bot(chat.id, message.message_id):
                return "own_bot_message"
            until = self.state.pause(chat.id, self.settings.pause_minutes, now)
            self.state.note_outgoing(chat.id, now, "owner")
            self.store.save_state(self.state)
            log.info("autopause", chat_id=chat.id, until=until, minutes=self.settings.pause_minutes)
            return "owner_message_autopause"
        if user.id not in self.settings.allowed_user_ids:
            log.info("business_message.ignored", reason="user not allowed", user_id=user.id)
            return "not_allowed"
        self.state.note_incoming(chat.id, now)
        self.store.save_state(self.state)
        if not self.state.enabled or not self.state.is_chat_enabled(chat.id):
            return "disabled"
        if self.state.is_paused(chat.id, now):
            return "paused"
        text = message.text
        if not text:
            return "non_text"
        partner = user.id
        history = self.memory.turns(partner)
        previous = next((t.text for t in reversed(history) if not t.is_me), None)
        self.memory.append(partner, MemoryTurn(is_me=False, text=text, ts=now))
        rate = self.skip_rate
        if should_skip(text, self.rng, rate if rate is not None else 1 / 20):
            log.info("business_message.skipped", chat_id=chat.id, message_id=message.message_id)
            return "skipped"
        try:
            backend = self.backend_for(self.mode)
        except ConfigError as exc:
            log.error("backend.unavailable", mode=self.mode.value, error=str(exc))
            return "backend_unavailable"
        request = GenerationRequest(
            partner_id=partner,
            chat_id=chat.id,
            message_id=message.message_id,
            text=text,
            previous_partner_text=previous,
            history=history,
            dry_run=self.dry_run,
        )
        result = await asyncio.to_thread(backend.generate, request)
        if result.text is None:
            log.info("business_message.silent", chat_id=chat.id, rejected=result.rejected)
            return "silent"
        self.memory.append(partner, MemoryTurn(is_me=True, text=result.text, ts=self._now()))
        if self.dry_run:
            log.info(
                "business_message.dry_run",
                chat_id=chat.id,
                message_id=message.message_id,
                parts=parts_summary(split_parts(result.text)),
                response=result.text,
            )
            return "dry_run"
        outcome = await self._deliver(chat.id, status.record.business_connection_id, result.text)
        if outcome == "sent":
            self.state.note_outgoing(chat.id, self._now(), "reply")
            self.store.save_state(self.state)
        return outcome

    # --- initiative --------------------------------------------------------------------

    def _initiative_gate(self, chat_id: int, now: int) -> str | None:
        status = self.connection_status()
        if not status.operational or status.record is None:
            return "no_connection"
        if not self.state.enabled or not self.state.is_chat_enabled(chat_id):
            return "disabled"
        if self.state.is_paused(chat_id, now):
            return "paused"
        return None

    async def initiative_tick(self) -> dict[int, str]:
        """One scheduler pass over the allowed chats; returns chat_id -> outcome/reason."""
        now = self._now()
        results: dict[int, str] = {}
        for chat_id in self.settings.allowed_user_ids:
            gate = self._initiative_gate(chat_id, now)
            if gate is not None:
                results[chat_id] = gate
                continue
            decision: Decision = decide(self.state, chat_id, now, self.initiative_cfg, self.rng)
            self.store.save_state(self.state)  # plans and roll marks changed
            if decision.kind is None:
                results[chat_id] = decision.reason
                continue
            results[chat_id] = await self._initiate(chat_id, decision.kind)
        return results

    async def initiative_loop(self) -> None:
        """Background task next to polling; never raises."""
        while True:
            try:
                await self.initiative_tick()
            except Exception as exc:  # the loop must survive a bad tick
                log.error("initiative.tick_failed", error=str(exc))
            await self.sleep(float(self.settings.initiative_tick_seconds))

    async def poke(self, chat_id: int, kind: str) -> str:
        """``/poke``: an initiative now, past the schedule but never past the gates."""
        if chat_id not in self.settings.allowed_user_ids:
            return "not_allowed"
        gate = self._initiative_gate(chat_id, self._now())
        return gate if gate is not None else await self._initiate(chat_id, kind)

    async def _initiate(self, chat_id: int, kind: str) -> str:
        status = self.connection_status()
        if status.record is None:
            return "no_connection"
        history = self.memory.turns(chat_id)
        partner_texts = [t.text for t in history if not t.is_me]
        query = partner_texts[-1] if partner_texts else (history[-1].text if history else "")
        try:
            backend = self.initiative_backend()
        except ConfigError as exc:
            log.error("initiative.backend_unavailable", error=str(exc))
            return "backend_unavailable"
        request = GenerationRequest(
            partner_id=chat_id,
            chat_id=chat_id,
            text=query,
            previous_partner_text=partner_texts[-2] if len(partner_texts) > 1 else None,
            history=history,
            dry_run=self.dry_run,
            intent=kind,
        )
        result = await asyncio.to_thread(backend.generate, request)
        if result.text is None:
            log.info("initiative.silent", chat_id=chat_id, kind=kind, rejected=result.rejected)
            return "silent"
        self.memory.append(chat_id, MemoryTurn(is_me=True, text=result.text, ts=self._now()))
        if self.dry_run:
            log.info(
                "initiative.dry_run",
                chat_id=chat_id,
                kind=kind,
                parts=parts_summary(split_parts(result.text)),
                response=result.text,
            )
            return "dry_run"
        outcome = await self._deliver(chat_id, status.record.business_connection_id, result.text)
        if outcome == "sent":
            self.state.note_outgoing(chat_id, self._now(), "initiative")
            self.store.save_state(self.state)
            log.info("initiative.sent", chat_id=chat_id, kind=kind)
        return outcome

    # --- delivery --------------------------------------------------------------------

    async def _typing(self, chat_id: int, conn_id: str, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            try:
                await self.bot.send_chat_action(chat_id, "typing", business_connection_id=conn_id)
            except Exception as exc:  # typing is cosmetic; never fail the reply for it
                log.warning("typing.failed", error=str(exc))
            step = min(TYPING_INTERVAL_S, remaining)
            await self.sleep(step)
            remaining -= step

    async def _send_part(self, chat_id: int, conn_id: str, part: str) -> str:
        """Send one message; returns 'sent' | 'forbidden' | 'failed'."""
        attempt = 0
        retried_after = False
        while True:
            attempt += 1
            try:
                sent = await self.bot.send_message(
                    chat_id=chat_id, text=part, business_connection_id=conn_id
                )
                message_id = getattr(sent, "message_id", None)
                if isinstance(message_id, int):
                    self.state.remember_sent(chat_id, message_id)
                    self.store.save_state(self.state)
                return "sent"
            except TelegramRetryAfter as exc:
                if retried_after:
                    log.error("send.retry_after_twice", chat_id=chat_id)
                    return "failed"
                retried_after = True
                log.warning("send.retry_after", chat_id=chat_id, seconds=exc.retry_after)
                await self.sleep(float(exc.retry_after))
            except TelegramForbiddenError as exc:
                self.state.set_chat_enabled(chat_id, False)
                self.store.save_state(self.state)
                log.error("send.forbidden", chat_id=chat_id, error=str(exc))
                return "forbidden"
            except Exception as exc:
                if attempt > SEND_RETRIES:
                    log.error("send.failed", chat_id=chat_id, error=str(exc), attempts=attempt)
                    return "failed"
                log.warning("send.error", chat_id=chat_id, error=str(exc), attempt=attempt)
                await self.sleep(2.0 * attempt)

    async def _deliver(self, chat_id: int, conn_id: str, text: str) -> str:
        parts = split_parts(text)
        await self._typing(chat_id, conn_id, reply_delay_seconds(text, self.rng))
        for index, part in enumerate(parts):
            if index:
                await self._typing(chat_id, conn_id, pause_between_parts(part, self.rng))
            outcome = await self._send_part(chat_id, conn_id, part)
            if outcome != "sent":
                return outcome
        log.info("business_message.sent", chat_id=chat_id, parts=parts_summary(parts))
        return "sent"
