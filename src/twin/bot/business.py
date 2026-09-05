"""Business connection handling (section 8.1).

The bot operates only through a connection whose ``user_id`` equals
``BUSINESS_OWNER_ID`` and which allows replies. Anything else is stored, logged and
refused. Every outgoing call passes ``business_connection_id`` explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from twin.bot.state import BusinessConnectionRecord, StateStore
from twin.logsetup import get_logger

log = get_logger("twin.bot.business")

CONNECT_INSTRUCTIONS = (
    "No business connection stored. Connect the bot: Telegram -> Settings -> Business -> "
    "Chatbots -> pick this bot, restrict 'Selected chats' to the two allowed users, and "
    "allow replies. The bot keeps polling and will pick the connection up automatically."
)


def _can_reply(connection: Any) -> bool:
    """Bot API 9 moved ``can_reply`` under ``rights``; accept both shapes."""
    rights = getattr(connection, "rights", None)
    if rights is not None and getattr(rights, "can_reply", None) is not None:
        return bool(rights.can_reply)
    return bool(getattr(connection, "can_reply", False))


def record_from_update(connection: Any, now: int | None = None) -> BusinessConnectionRecord:
    return BusinessConnectionRecord(
        business_connection_id=str(connection.id),
        user_id=int(connection.user.id),
        user_chat_id=getattr(connection, "user_chat_id", None),
        can_reply=_can_reply(connection),
        is_enabled=bool(connection.is_enabled),
        updated_at=now or int(time.time()),
    )


@dataclass(frozen=True)
class ConnectionStatus:
    record: BusinessConnectionRecord | None
    operational: bool
    reason: str


def evaluate_connection(
    record: BusinessConnectionRecord | None, owner_id: int | None
) -> ConnectionStatus:
    if record is None:
        return ConnectionStatus(None, False, "no connection stored")
    if owner_id is None or record.user_id != owner_id:
        return ConnectionStatus(record, False, "connection user_id is not BUSINESS_OWNER_ID")
    if not record.is_enabled:
        return ConnectionStatus(record, False, "connection disabled by the owner")
    if not record.can_reply:
        return ConnectionStatus(record, False, "connection does not allow replies")
    return ConnectionStatus(record, True, "ok")


def store_connection_update(
    connection: Any, store: StateStore, owner_id: int | None
) -> ConnectionStatus:
    """Persist a ``business_connection`` update and report whether the bot may operate."""
    record = record_from_update(connection)
    store.save_connection(record)
    status = evaluate_connection(record, owner_id)
    log.info(
        "business_connection",
        user_id=record.user_id,
        can_reply=record.can_reply,
        is_enabled=record.is_enabled,
        operational=status.operational,
        reason=status.reason,
    )
    return status
