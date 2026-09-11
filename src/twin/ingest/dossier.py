"""Self dossier: what the twin knows about *himself*, distilled from the training replies.

The style profile says how he writes and the retrieved examples show how he reacts, but
nothing tells the model what his life is: what he studies, whom he means by a first
name, what he thinks about things. Asked "how's the project?", the twin could only
shrug. The dossier is a map-reduce over the informative training replies (at least
``min_chars`` long): the gateway model writes notes per chunk of the chat, a stronger
model merges them into at most ``MAX_FACTS`` lines.

Training pairs only. In this export the holdout is the time tail of the only chat, so
the dossier holds nothing from the evaluated period. It is personal data:
``data/processed/self_dossier.md`` is gitignored, hand-editable and never overwritten
without ``--force``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from twin.core.llm_client import LLMClient, LLMError, LLMResult, is_content_refusal
from twin.core.prompts import PromptTemplate
from twin.core.schemas import Pair
from twin.ingest.style_profile import read_style_profile
from twin.logsetup import get_logger

log = get_logger("twin.dossier")

DOSSIER_MAP_PROMPT = "dossier_map_v1"
DOSSIER_REDUCE_PROMPT = "dossier_reduce_v1"
MIN_REPLY_CHARS = 20
CHUNK_PAIRS = 400
MIN_SPLIT_PAIRS = 25  # a refused chunk is halved down to this size, then skipped (counted)
MAX_CONTEXT_CHARS = 200
MAX_FACTS = 30
_FACT_RE = re.compile(r"^\s*-\s+\S")
ChunkNotes = list[tuple[list[Pair], LLMResult]]


class DossierExistsError(FileExistsError):
    """The dossier exists and ``--force`` was not given."""


@dataclass(frozen=True)
class DossierResult:
    text: str
    facts: int
    pairs_used: int
    chunks: int
    map_model: str
    reduce_model: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    skipped_pairs: int = 0  # refused by the map model's content filter, never silently


def informative_pairs(pairs: Sequence[Pair], min_chars: int = MIN_REPLY_CHARS) -> list[Pair]:
    """Replies long enough to say something, in time order; "ок" carries no facts."""
    if any(pair.eval_sample for pair in pairs):
        raise ValueError("refusing to build the dossier from evaluation pairs (holdout leak)")
    return sorted(
        (pair for pair in pairs if len(pair.reply.strip()) >= min_chars),
        key=lambda pair: (pair.ts, pair.pair_id),
    )


def chunked(pairs: Sequence[Pair], size: int) -> list[list[Pair]]:
    return [list(pairs[start : start + size]) for start in range(0, len(pairs), size)]


def _flat(text: str, limit: int | None = None) -> str:
    flat = text.replace("\n", " / ").strip()
    if limit is not None and len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat


def render_chunk(pairs: Sequence[Pair], name: str) -> str:
    blocks = []
    for pair in pairs:
        partner = next((turn.text for turn in reversed(pair.context) if not turn.is_me), "")
        blocks.append(
            f"[{pair.period}] Собеседник: {_flat(partner, MAX_CONTEXT_CHARS)}\n"
            f"{name}: {_flat(pair.reply)}"
        )
    return "\n".join(blocks)


def count_facts(text: str) -> int:
    return sum(1 for line in text.splitlines() if _FACT_RE.match(line))


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_dossier(
    map_llm: LLMClient,
    reduce_llm: LLMClient,
    map_template: PromptTemplate,
    reduce_template: PromptTemplate,
    name: str,
    train_pairs: Sequence[Pair],
    chunk_size: int = CHUNK_PAIRS,
    min_chars: int = MIN_REPLY_CHARS,
    workers: int = 4,
    limit: int = MAX_FACTS,
    min_split: int = MIN_SPLIT_PAIRS,
) -> DossierResult:
    selected = informative_pairs(train_pairs, min_chars)
    if not selected:
        raise ValueError(f"no training reply is at least {min_chars} characters long")

    def call(chunk: list[Pair]) -> LLMResult:
        system, user = map_template.render(
            name=name,
            period=f"{chunk[0].period} — {chunk[-1].period}",
            examples=render_chunk(chunk, name),
        )
        return map_llm.chat(_messages(system, user), temperature=0.2, max_tokens=1200)

    def notes_for(chunk: list[Pair]) -> tuple[ChunkNotes, int]:
        """Notes for one chunk. A chunk the content filter refuses is halved until the
        halves pass; a piece of ``min_split`` pairs or fewer is skipped and counted."""
        try:
            return [(chunk, call(chunk))], 0
        except LLMError as exc:
            if not is_content_refusal(exc):  # only refusals are split; a quota error is not
                raise
            if len(chunk) <= min_split:
                log.warning("dossier.refused", pairs=len(chunk), period=chunk[0].period)
                return [], len(chunk)
            middle = len(chunk) // 2
            left, left_skipped = notes_for(chunk[:middle])
            right, right_skipped = notes_for(chunk[middle:])
            return left + right, left_skipped + right_skipped

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(notes_for, chunked(selected, chunk_size)))
    notes = [note for found, _skipped in outcomes for note in found]
    skipped = sum(count for _found, count in outcomes)
    if not notes:
        raise ValueError("the content filter refused every chunk; try another --map-model")
    merged = "\n\n".join(
        f"### {chunk[0].period} — {chunk[-1].period}\n{result.text.strip()}"
        for chunk, result in notes
    )
    system, user = reduce_template.render(name=name, notes=merged, limit=limit)
    final = reduce_llm.chat(_messages(system, user), temperature=0.2, max_tokens=2500)
    text = final.text.strip() + "\n"
    calls = [result for _chunk, result in notes] + [final]
    return DossierResult(
        text=text,
        facts=count_facts(text),
        pairs_used=len(selected) - skipped,
        chunks=len(notes),
        map_model=notes[0][1].model,
        reduce_model=final.model,
        prompt_version=f"{map_template.version}+{reduce_template.version}",
        prompt_tokens=sum(result.prompt_tokens or 0 for result in calls),
        completion_tokens=sum(result.completion_tokens or 0 for result in calls),
        skipped_pairs=skipped,
    )


def write_dossier(path: Path, result: DossierResult, dataset_version: str, force: bool) -> None:
    if path.exists() and not force:
        raise DossierExistsError(
            f"{path} exists (possibly hand-edited); rerun with --force to overwrite"
        )
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"<!-- generated by `twin dossier` at {stamp}; map {result.map_model}, reduce "
        f"{result.reduce_model}; prompt {result.prompt_version}; dataset {dataset_version}; "
        f"{result.pairs_used} train replies in {result.chunks} chunks, "
        f"{result.skipped_pairs} refused by the content filter. Personal data, "
        "gitignored. Edit freely: never overwritten without --force. -->\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + result.text, encoding="utf-8")


def read_dossier(path: Path) -> str:
    """Dossier text without the generator header; empty when there is none yet."""
    return read_style_profile(path) if path.is_file() else ""
