"""Prompt construction for the generation backends (templates live in ``prompts/``)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from twin.core.memory import MemoryTurn
from twin.core.prompts import PromptTemplate
from twin.core.retriever import RetrievedExample

MAX_EXAMPLE_CONTEXT_CHARS = 300
MAX_STATEMENT_CHARS = 300
FACTS_EMPTY = "(пока ничего не известно)"
KNOWLEDGE = "${knowledge}"
# Knowledge sections (rag_v4+) are rendered only when they have content: an empty
# section measurably cost 0.08 overall on the holdout (rag_v3 with an empty fact sheet).
DOSSIER_SECTION = (
    "### Что {name} знает о себе\n"
    "Это его жизнь по его же прошлой переписке. Опирайся на это, когда разговор касается "
    "его самого."
)
STATEMENTS_SECTION = (
    "### Что {name} уже говорил на похожие темы\n"
    "Его реальные прошлые реплики, с месяцем. Это его мнения и факты, а не шаблоны: не "
    "повторяй их дословно."
)
FACTS_SECTION = (
    "### Что {name} знает про этого собеседника\n"
    "Короткая памятка из прошлых разговоров. Если она противоречит переписке ниже, верь "
    "переписке."
)
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PromptBundle:
    messages: list[dict[str, str]]
    version: str


def _clip(text: str, limit: int) -> str:
    flat = text.replace("\n", " / ")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def format_examples(examples: Sequence[RetrievedExample], name: str) -> str:
    if not examples:
        return "(примеров нет)"
    blocks = [
        f"Собеседник: {_clip(e.last_partner_text, MAX_EXAMPLE_CONTEXT_CHARS)}\n{name}: {e.reply}"
        for e in examples
    ]
    return "\n\n".join(blocks)


def format_history(history: Sequence[MemoryTurn], name: str) -> str:
    if not history:
        return "(начало разговора)"
    return "\n".join(f"{name if t.is_me else 'Собеседник'}: {t.text}" for t in history)


def uses_knowledge(template: PromptTemplate) -> bool:
    """rag_v4+ carry one ``${knowledge}`` slot for the dossier, statements and facts."""
    return KNOWLEDGE in template.system + template.user


def format_statements(statements: Sequence[RetrievedExample]) -> str:
    return "\n".join(
        f"- [{datetime.fromtimestamp(s.ts, tz=UTC).strftime('%Y-%m')}] "
        f"{_clip(s.reply, MAX_STATEMENT_CHARS)}"
        for s in statements
    )


def render_knowledge(
    name: str, dossier: str, statements: Sequence[RetrievedExample], facts: str
) -> str:
    sections = []
    if dossier.strip():
        sections.append(DOSSIER_SECTION.format(name=name) + "\n" + dossier.strip())
    if statements:
        sections.append(STATEMENTS_SECTION.format(name=name) + "\n" + format_statements(statements))
    if facts.strip():
        sections.append(FACTS_SECTION.format(name=name) + "\n" + facts.strip())
    return "\n\n".join(sections)


def build_rag_messages(
    template: PromptTemplate,
    name: str,
    style_profile: str,
    examples: Sequence[RetrievedExample],
    history: Sequence[MemoryTurn],
    incoming: str,
    facts: str = "",
    dossier: str = "",
    statements: Sequence[RetrievedExample] = (),
) -> PromptBundle:
    values = {
        "name": name,
        "style_profile": style_profile.strip(),
        "examples": format_examples(examples, name),
        "history": format_history(history, name),
        "incoming": incoming,
    }
    knowledge = uses_knowledge(template)
    if knowledge:
        values["knowledge"] = render_knowledge(name, dossier, statements, facts)
    elif "${facts}" in template.system + template.user:
        values["facts"] = facts.strip() or FACTS_EMPTY
    system, user = template.render(**values)
    if knowledge:  # an absent section must not leave a hole in the prompt
        system, user = _BLANK_LINES.sub("\n\n", system), _BLANK_LINES.sub("\n\n", user)
    return PromptBundle(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        version=template.version,
    )


INITIATIVE_TASKS = {
    "followup": (
        "Собеседник не ответил на последнее сообщение {name}. Напиши одну короткую реплику-"
        "продолжение, как делает {name}, когда ему не отвечают: подтолкнуть, переспросить или "
        "добавить мысль к тому, что он уже написал."
    ),
    "opener": (
        "Переписки давно не было, {name} пишет первым. Напиши короткое первое сообщение: "
        "зацепись за последнюю тему из переписки, если она есть, иначе начни так, как {name} "
        "обычно начинает разговор."
    ),
}


def build_initiative_messages(
    template: PromptTemplate,
    name: str,
    style_profile: str,
    examples: Sequence[RetrievedExample],
    history: Sequence[MemoryTurn],
    intent: str,
) -> PromptBundle:
    if intent not in INITIATIVE_TASKS:
        raise ValueError(f"unknown initiative intent {intent!r}")
    system, user = template.render(
        name=name,
        style_profile=style_profile.strip(),
        examples=format_examples(examples, name),
        history=format_history(history, name),
        task=INITIATIVE_TASKS[intent].format(name=name),
    )
    return PromptBundle(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        version=template.version,
    )
