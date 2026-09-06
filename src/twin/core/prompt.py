"""Prompt construction for the generation backends (templates live in ``prompts/``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from twin.core.memory import MemoryTurn
from twin.core.prompts import PromptTemplate
from twin.core.retriever import RetrievedExample

MAX_EXAMPLE_CONTEXT_CHARS = 300


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


def build_rag_messages(
    template: PromptTemplate,
    name: str,
    style_profile: str,
    examples: Sequence[RetrievedExample],
    history: Sequence[MemoryTurn],
    incoming: str,
) -> PromptBundle:
    system, user = template.render(
        name=name,
        style_profile=style_profile.strip(),
        examples=format_examples(examples, name),
        history=format_history(history, name),
        incoming=incoming,
    )
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
