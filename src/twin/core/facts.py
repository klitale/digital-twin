"""What the twin has learned about a partner from the live conversation.

The retrieval index is frozen at the export, and the per-partner memory holds only the
last few turns, so the twin knows the person's *style* but nothing about what is going
on in their life right now. A fact sheet closes that: every few hours the gateway model
rewrites a short list of durable facts from the real turns of one chat, and the list
goes into the prompt.

One rule makes this safe: **only human-written turns are ever read**. A turn the bot
generated is excluded, so the twin cannot learn from its own inventions and reinforce
them. ``MemoryTurn.by_bot`` carries that flag and ``human_turns`` enforces it.

Fact sheets are personal data: they live under ``data/state/facts/`` and are gitignored.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from twin.core.llm_client import LLMClient, LLMError
from twin.core.memory import MemoryTurn
from twin.core.prompts import PromptTemplate
from twin.logsetup import get_logger

log = get_logger("twin.facts")

MAX_FACTS = 15
MAX_FACT_CHARS = 160
EMPTY = "(пока ничего не известно)"


class Fact(BaseModel):
    text: str
    ts: int = Field(description="when the fact was written down")


class FactSheet(BaseModel):
    partner_id: int
    facts: list[Fact] = Field(default_factory=list)
    updated_at: int = 0
    turns_seen: int = Field(default=0, description="human turns folded in so far")

    def render(self) -> str:
        return "\n".join(f"- {fact.text}" for fact in self.facts) if self.facts else EMPTY


class FactStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, partner_id: int) -> Path:
        return self.directory / f"{partner_id}.json"

    def load(self, partner_id: int) -> FactSheet:
        path = self._path(partner_id)
        if not path.is_file():
            return FactSheet(partner_id=partner_id)
        return FactSheet.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, sheet: FactSheet) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(sheet.partner_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(sheet.model_dump_json(indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)

    def forget(self, partner_id: int) -> bool:
        path = self._path(partner_id)
        if path.is_file():
            path.unlink()
            return True
        return False


def human_turns(turns: Sequence[MemoryTurn]) -> list[MemoryTurn]:
    """Everything a person actually wrote; never what the bot generated."""
    return [turn for turn in turns if not turn.by_bot]


def parse_facts(text: str, now: int, limit: int = MAX_FACTS) -> list[Fact]:
    """One fact per '- ' line; anything else in the answer is ignored."""
    facts: list[Fact] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip().lstrip("-•*").strip()
        if not stripped or not line.strip().startswith(("-", "•", "*")):
            continue
        if stripped.lower().startswith(("пока ничего", "(пока ничего", "нет фактов")):
            continue
        clipped = stripped[:MAX_FACT_CHARS]
        key = clipped.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(Fact(text=clipped, ts=now))
        if len(facts) >= limit:
            break
    return facts


def format_turns(turns: Sequence[MemoryTurn], name: str) -> str:
    return "\n".join(f"{name if t.is_me else 'Собеседник'}: {t.text}" for t in turns)


def update_sheet(
    llm: LLMClient,
    template: PromptTemplate,
    sheet: FactSheet,
    turns: Sequence[MemoryTurn],
    name: str,
    now: int,
    limit: int = MAX_FACTS,
) -> FactSheet | None:
    """Rewrite the sheet from the real turns; ``None`` when the model could not be used."""
    real = human_turns(turns)
    if not real:
        return None
    system, user = template.render(
        name=name,
        facts=sheet.render(),
        conversation=format_turns(real, name),
        limit=limit,
    )
    try:
        result = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
    except LLMError as exc:
        log.error("facts.llm_failed", partner_id=sheet.partner_id, error=str(exc)[:200])
        return None
    facts = parse_facts(result.text, now, limit)
    updated = FactSheet(
        partner_id=sheet.partner_id,
        facts=facts,
        updated_at=now,
        turns_seen=sheet.turns_seen + len(real),
    )
    log.info(
        "facts.updated",
        partner_id=sheet.partner_id,
        before=len(sheet.facts),
        after=len(facts),
        from_turns=len(real),
    )
    return updated


def load_facts_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
