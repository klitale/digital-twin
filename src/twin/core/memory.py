"""Per-partner short-term memory: the last N turns, persisted as JSON on disk."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class MemoryTurn(BaseModel):
    is_me: bool
    text: str
    ts: int


class ConversationMemory:
    def __init__(self, directory: Path, max_turns: int = 10) -> None:
        self.directory = directory
        self.max_turns = max_turns

    def _path(self, partner_id: int) -> Path:
        return self.directory / f"{partner_id}.json"

    def turns(self, partner_id: int) -> list[MemoryTurn]:
        path = self._path(partner_id)
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [MemoryTurn.model_validate(item) for item in raw][-self.max_turns :]

    def append(self, partner_id: int, turn: MemoryTurn) -> list[MemoryTurn]:
        turns = [*self.turns(partner_id), turn][-self.max_turns :]
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(partner_id).write_text(
            json.dumps([t.model_dump() for t in turns], ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        return turns

    def reset(self, partner_id: int) -> bool:
        path = self._path(partner_id)
        if path.is_file():
            path.unlink()
            return True
        return False
