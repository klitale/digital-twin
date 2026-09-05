"""Style profile: statistics + sampled replies -> LLM -> 20-30 Russian rules.

The sample comes from *training* pairs only (holdout is never used for prompt work),
stratified by year with a seeded generator so the run is reproducible. The output
file is hand-editable and is never overwritten without ``--force``.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from twin.core.llm_client import LLMClient, LLMResult
from twin.core.prompts import PromptTemplate
from twin.core.schemas import Pair
from twin.ingest.profile_dataset import _EMOJI_RE
from twin.ingest.split import allocate_proportionally
from twin.ingest.stats import quantile

DEFAULT_SAMPLE_SIZE = 300
DEFAULT_SEED = 20260905
MAX_CONTEXT_CHARS = 300
RULES_MIN, RULES_MAX = 20, 30

_WORD_RE = re.compile(r"[а-яёa-z]{2,}")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
# Function words that carry no style signal (top-50 should show vocabulary, not grammar).
STOPWORDS = frozenset(
    [
        "а",
        "без",
        "бы",
        "был",
        "была",
        "были",
        "было",
        "быть",
        "в",
        "вам",
        "вас",
        "весь",
        "во",
        "вот",
        "все",
        "всё",
        "всего",
        "всех",
        "вы",
        "где",
        "да",
        "даже",
        "для",
        "до",
        "его",
        "ее",
        "её",
        "ей",
        "ему",
        "если",
        "есть",
        "еще",
        "ещё",
        "же",
        "за",
        "здесь",
        "и",
        "из",
        "или",
        "им",
        "их",
        "к",
        "как",
        "ко",
        "когда",
        "кто",
        "ли",
        "либо",
        "мне",
        "может",
        "мой",
        "моя",
        "мою",
        "мы",
        "на",
        "над",
        "надо",
        "наш",
        "не",
        "него",
        "нее",
        "неё",
        "нет",
        "ни",
        "них",
        "но",
        "ну",
        "о",
        "об",
        "один",
        "он",
        "она",
        "они",
        "оно",
        "от",
        "по",
        "под",
        "при",
        "про",
        "с",
        "сам",
        "свой",
        "себе",
        "себя",
        "со",
        "так",
        "также",
        "такой",
        "там",
        "тебе",
        "тебя",
        "тем",
        "то",
        "того",
        "тоже",
        "той",
        "только",
        "том",
        "тот",
        "тут",
        "ты",
        "у",
        "уж",
        "уже",
        "хоть",
        "чего",
        "чем",
        "что",
        "чтобы",
        "чья",
        "эта",
        "эти",
        "это",
        "этого",
        "этой",
        "этом",
        "этот",
        "эту",
        "я",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "is",
        "it",
        "be",
        "this",
        "that",
    ]
)


@dataclass(frozen=True)
class StyleProfileResult:
    text: str
    rules: int
    sample_size: int
    prompt_version: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


class StyleProfileExistsError(FileExistsError):
    """The profile file exists and ``--force`` was not given."""


def _year(pair: Pair) -> str:
    return pair.period[:4]


def _share(count: int, total: int) -> float:
    return round(100 * count / total, 1) if total else 0.0


def sample_pairs(pairs: Sequence[Pair], size: int, seed: int) -> list[Pair]:
    """Stratified by year (proportional, largest remainder), seeded, sorted by time."""
    strata: dict[str, list[Pair]] = {}
    for pair in pairs:
        strata.setdefault(_year(pair), []).append(pair)
    allocation = allocate_proportionally({y: len(rows) for y, rows in strata.items()}, size)
    rng = random.Random(seed)
    chosen: list[Pair] = []
    for year in sorted(strata):
        rows = sorted(strata[year], key=lambda p: p.pair_id)
        chosen.extend(rng.sample(rows, allocation[year]))
    return sorted(chosen, key=lambda p: (p.ts, p.pair_id))


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def top_words(pairs: Sequence[Pair], n: int = 50) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for pair in pairs:
        counts.update(w for w in tokenize(pair.reply) if w not in STOPWORDS)
    return counts.most_common(n)


def top_emoji(pairs: Sequence[Pair], n: int = 10) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for pair in pairs:
        counts.update(_EMOJI_RE.findall(pair.reply))
    return counts.most_common(n)


def reply_style_stats(pairs: Sequence[Pair]) -> dict[str, float | int]:
    """Numbers the LLM gets alongside the examples (all replies, not just the sample)."""
    replies = [p.reply for p in pairs]
    total = len(replies)
    lengths = sorted(len(r) for r in replies)
    words = [len(tokenize(r)) for r in replies]
    letters_start = [r.lstrip()[:1] for r in replies if r.strip()]
    return {
        "replies": total,
        "chars_p50": quantile(lengths, 0.5),
        "chars_p90": quantile(lengths, 0.9),
        "words_p50": quantile(sorted(words), 0.5),
        "le_10_chars_pct": _share(sum(1 for n in lengths if n <= 10), total),
        "multiline_pct": _share(sum(1 for r in replies if "\n" in r), total),
        "lines_p90": quantile(sorted(r.count("\n") + 1 for r in replies), 0.9),
        "emoji_pct": _share(sum(1 for r in replies if _EMOJI_RE.search(r)), total),
        "question_pct": _share(sum(1 for r in replies if "?" in r), total),
        "exclamation_pct": _share(sum(1 for r in replies if "!" in r), total),
        "ellipsis_pct": _share(sum(1 for r in replies if "..." in r or "…" in r), total),
        "ends_with_period_pct": _share(sum(1 for r in replies if r.rstrip().endswith(".")), total),
        "bracket_smile_pct": _share(sum(1 for r in replies if ")" in r and "(" not in r), total),
        "comma_pct": _share(sum(1 for r in replies if "," in r), total),
        "starts_uppercase_pct": _share(
            sum(1 for ch in letters_start if ch.isalpha() and ch.isupper()), len(letters_start)
        ),
        "all_caps_word_pct": _share(
            sum(1 for r in replies if re.search(r"\b[А-ЯЁA-Z]{3,}\b", r)), total
        ),
        "latin_letters_pct": _share(
            sum(1 for r in replies if re.search(r"[a-zA-Z]{3,}", r)), total
        ),
        "digits_pct": _share(sum(1 for r in replies if re.search(r"\d", r)), total),
    }


_STAT_LABELS = {
    "replies": "всего ответов",
    "chars_p50": "медиана длины, символов",
    "chars_p90": "90-й перцентиль длины, символов",
    "words_p50": "медиана длины, слов",
    "le_10_chars_pct": "доля ответов до 10 символов, %",
    "multiline_pct": "доля ответов из нескольких сообщений/строк, %",
    "lines_p90": "90-й перцентиль числа строк",
    "emoji_pct": "доля ответов с эмодзи, %",
    "question_pct": "доля ответов с вопросом, %",
    "exclamation_pct": "доля ответов с восклицанием, %",
    "ellipsis_pct": "доля ответов с многоточием, %",
    "ends_with_period_pct": "доля ответов с точкой в конце, %",
    "bracket_smile_pct": "доля ответов со скобкой-улыбкой, %",
    "comma_pct": "доля ответов с запятой, %",
    "starts_uppercase_pct": "доля ответов, начинающихся с заглавной буквы, %",
    "all_caps_word_pct": "доля ответов со словом капсом, %",
    "latin_letters_pct": "доля ответов с латиницей, %",
    "digits_pct": "доля ответов с цифрами, %",
}


def render_stats(
    stats: dict[str, float | int],
    words: Sequence[tuple[str, int]],
    emoji: Sequence[tuple[str, int]],
) -> str:
    lines = [f"- {_STAT_LABELS.get(key, key)}: {value}" for key, value in stats.items()]
    lines.append("- топ-50 слов без стоп-слов: " + ", ".join(f"{w} ({c})" for w, c in words))
    if emoji:
        lines.append("- топ эмодзи: " + " ".join(f"{e} ({c})" for e, c in emoji))
    return "\n".join(lines)


def _last_partner_text(pair: Pair) -> str:
    for turn in reversed(pair.context):
        if not turn.is_me:
            return turn.text
    return pair.context[-1].text if pair.context else ""


def render_examples(pairs: Sequence[Pair], name: str) -> str:
    blocks = []
    for index, pair in enumerate(pairs, start=1):
        context = _last_partner_text(pair).replace("\n", " / ")
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[: MAX_CONTEXT_CHARS - 1] + "…"
        reply = pair.reply.replace("\n", " / ")
        blocks.append(f"{index}. Собеседник: {context}\n   {name}: {reply}")
    return "\n".join(blocks)


def count_rules(text: str) -> int:
    return sum(1 for line in text.splitlines() if _BULLET_RE.match(line))


def build_messages(
    template: PromptTemplate,
    name: str,
    train_pairs: Sequence[Pair],
    sample: Sequence[Pair],
) -> list[dict[str, str]]:
    stats = render_stats(
        reply_style_stats(train_pairs), top_words(train_pairs), top_emoji(train_pairs)
    )
    system, user = template.render(
        name=name,
        n_examples=len(sample),
        stats=stats,
        examples=render_examples(sample, name),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_style_profile(
    client: LLMClient,
    template: PromptTemplate,
    name: str,
    train_pairs: Sequence[Pair],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    temperature: float = 0.3,
) -> StyleProfileResult:
    sample = sample_pairs(train_pairs, sample_size, seed)
    messages = build_messages(template, name, train_pairs, sample)
    result: LLMResult = client.chat(messages, temperature=temperature)
    return StyleProfileResult(
        text=result.text.strip() + "\n",
        rules=count_rules(result.text),
        sample_size=len(sample),
        prompt_version=template.version,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
    )


def write_style_profile(
    path: Path, result: StyleProfileResult, dataset_version: str, force: bool
) -> None:
    if path.exists() and not force:
        raise StyleProfileExistsError(
            f"{path} exists (possibly hand-edited); rerun with --force to overwrite"
        )
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"<!-- generated by `twin style-profile` at {stamp}; model {result.model}; "
        f"prompt {result.prompt_version}; dataset {dataset_version}; sample {result.sample_size} "
        "train replies. Edit freely: this file is never overwritten without --force. -->\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + result.text, encoding="utf-8")


def read_style_profile(path: Path) -> str:
    """Profile text without the generator header (what goes into prompts)."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :]
    return text.strip() + "\n"
