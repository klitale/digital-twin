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


# rag_v5 answers "Замысел: <intent>" then "Ответ: <message>"; only the message is sent
_PLAN_HEAD = re.compile(r"^\s*замысел\s*:", re.IGNORECASE)
_PLAN_RE = re.compile(
    r"^\s*замысел\s*:(?P<plan>.*?)^\s*ответ\s*:(?P<reply>.*)\Z",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    text: str
    reason: str | None = None
    plan: str | None = None


def split_plan(text: str) -> tuple[str | None, str | None]:
    """``(intent, message)``. No intent line: the text is the message. An intent without
    an "Ответ:" line gives no message at all, so the intent can never reach the chat."""
    if not _PLAN_HEAD.match(text):
        return None, text
    match = _PLAN_RE.match(text)
    if match is None:
        return _PLAN_HEAD.sub("", text, count=1).strip(), None
    return match["plan"].strip(), match["reply"].strip()


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
    plan, message = split_plan(text)
    if message is None:
        return ValidationResult(False, "", "plan_unparsed", plan)
    cleaned = clean_reply(message, name)
    if not cleaned:
        return ValidationResult(False, cleaned, "empty", plan)
    if len(cleaned) > max_chars:
        return ValidationResult(False, cleaned, f"too_long:{len(cleaned)}>{max_chars}", plan)
    items = sum(1 for line in cleaned.splitlines() if _LIST_ITEM.match(line))
    if items >= MAX_LIST_ITEMS:
        return ValidationResult(False, cleaned, f"list_output:{items}", plan)
    lowered = cleaned.lower()
    for marker in markers:
        if marker.lower() in lowered:
            return ValidationResult(False, cleaned, f"assistant_speak:{marker}", plan)
    return ValidationResult(True, cleaned, plan=plan)
