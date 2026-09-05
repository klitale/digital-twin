"""Runtime state of the bot, persisted as JSON under ``data/state``.

``connection.json`` holds the business connection (section 8.1); ``bot_state.json``
holds the switches (global/per-chat enable, dry-run override, mode), autopause
deadlines and the ids of messages the bot itself sent, so edits, deletions and the
owner's own messages can be told apart from bot output.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field

CONNECTION_FILE = "connection.json"
STATE_FILE = "bot_state.json"
REMEMBERED_MESSAGE_IDS = 200


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class BusinessConnectionRecord(BaseModel):
    business_connection_id: str
    user_id: int
    user_chat_id: int | None = None
    can_reply: bool
    is_enabled: bool
    updated_at: int

    @property
    def operational(self) -> bool:
        return self.can_reply and self.is_enabled


class BotState(BaseModel):
    enabled: bool = True
    dry_run_override: bool | None = Field(default=None, description="/twin dryrun on|off")
    mode: str | None = Field(default=None, description="/twin mode ...; None = settings")
    chat_enabled: dict[str, bool] = Field(default_factory=dict)
    paused_until: dict[str, int] = Field(default_factory=dict, description="chat_id -> unix ts")
    sent_message_ids: dict[str, list[int]] = Field(default_factory=dict)

    # --- queries -----------------------------------------------------------------

    def is_chat_enabled(self, chat_id: int) -> bool:
        return self.chat_enabled.get(str(chat_id), True)

    def is_paused(self, chat_id: int, now: int | None = None) -> bool:
        until = self.paused_until.get(str(chat_id))
        return until is not None and until > (now or int(time.time()))

    def pause_remaining(self, chat_id: int, now: int | None = None) -> int:
        until = self.paused_until.get(str(chat_id), 0)
        return max(0, until - (now or int(time.time())))

    def was_sent_by_bot(self, chat_id: int, message_id: int) -> bool:
        return message_id in self.sent_message_ids.get(str(chat_id), [])

    # --- mutations ---------------------------------------------------------------

    def pause(self, chat_id: int, minutes: int, now: int | None = None) -> int:
        until = (now or int(time.time())) + minutes * 60
        self.paused_until[str(chat_id)] = until
        return until

    def unpause(self, chat_id: int) -> None:
        self.paused_until.pop(str(chat_id), None)

    def remember_sent(self, chat_id: int, message_id: int) -> None:
        ids = self.sent_message_ids.setdefault(str(chat_id), [])
        ids.append(message_id)
        del ids[:-REMEMBERED_MESSAGE_IDS]

    def set_chat_enabled(self, chat_id: int, enabled: bool) -> None:
        self.chat_enabled[str(chat_id)] = enabled


class StateStore:
    """Load/save helpers; every mutation goes through ``save`` so restarts are safe."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def connection_path(self) -> Path:
        return self.directory / CONNECTION_FILE

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_FILE

    def load_connection(self) -> BusinessConnectionRecord | None:
        if not self.connection_path.is_file():
            return None
        return BusinessConnectionRecord.model_validate_json(
            self.connection_path.read_text(encoding="utf-8")
        )

    def save_connection(self, record: BusinessConnectionRecord) -> None:
        _atomic_write(self.connection_path, record.model_dump_json(indent=2) + "\n")

    def load_state(self) -> BotState:
        if not self.state_path.is_file():
            return BotState()
        return BotState.model_validate(json.loads(self.state_path.read_text(encoding="utf-8")))

    def save_state(self, state: BotState) -> None:
        _atomic_write(self.state_path, state.model_dump_json(indent=2) + "\n")
