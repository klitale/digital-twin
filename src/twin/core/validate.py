"""Reply validation: the twin stays silent rather than sounding like an assistant."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_ASSISTANT_MARKERS: tuple[str, ...] = (
    "как ии",
    "как искусственный интеллект",
    "языковая модель",
    "языковой модел",
    "я ассистент",
    "как ассистент",
    "виртуальный помощник",
    "чем могу помочь",
    "чем я могу помочь",
    "могу помочь вам",
    "рад помочь",
    "извините за",
    "приношу извинения",
    "прошу прощения за",
    "я не могу помочь",
    "не могу выполнить",
    "я бот",
    "я нейросеть",
    "as an ai",
    "language model",
)


_LIST_ITEM = re.compile(r"^\s*(?:\d{1,3}[.)]|[-•*])\s+\S")
MAX_LIST_ITEMS = 3  # a numbered or bulleted list this long is a service answering, not a person


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    text: str
    reason: str | None = None


def clean_reply(text: str, name: str | None = None) -> str:
    """Strip label echoes (``Радомир: ...``), wrapping quotes and stray whitespace."""
    cleaned = text.strip()
    if name:
        cleaned = re.sub(rf"^\s*{re.escape(name)}\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:Ты|Ответ)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if len(cleaned) >= 2 and cleaned[0] in "\"«'" and cleaned[-1] in "\"»'":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def validate_reply(
    text: str,
    max_chars: int,
    markers: Sequence[str] = DEFAULT_ASSISTANT_MARKERS,
    name: str | None = None,
) -> ValidationResult:
    cleaned = clean_reply(text, name)
    if not cleaned:
        return ValidationResult(False, cleaned, "empty")
    if len(cleaned) > max_chars:
        return ValidationResult(False, cleaned, f"too_long:{len(cleaned)}>{max_chars}")
    items = sum(1 for line in cleaned.splitlines() if _LIST_ITEM.match(line))
    if items >= MAX_LIST_ITEMS:
        return ValidationResult(False, cleaned, f"list_output:{items}")
    lowered = cleaned.lower()
    for marker in markers:
        if marker.lower() in lowered:
            return ValidationResult(False, cleaned, f"assistant_speak:{marker}")
    return ValidationResult(True, cleaned)
