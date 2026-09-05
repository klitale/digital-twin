"""Make delivery look human: typing delay, one message per line, occasional silence."""

from __future__ import annotations

import random
from collections.abc import Sequence

MIN_DELAY_S = 2.0
MAX_DELAY_S = 15.0
CHARS_PER_SECOND = 12.0  # "typing speed" used to scale the delay with reply length
MAX_PARTS = 5
SKIP_RATE = 1 / 20


def reply_delay_seconds(text: str, rng: random.Random | None = None) -> float:
    """2-15 s, longer for longer replies, with a little jitter."""
    rng = rng or random.Random()
    base = MIN_DELAY_S + len(text) / CHARS_PER_SECOND
    jitter = rng.uniform(-0.5, 1.5)
    return round(min(MAX_DELAY_S, max(MIN_DELAY_S, base + jitter)), 2)


def split_parts(text: str, max_parts: int = MAX_PARTS) -> list[str]:
    """One outgoing message per line, like the twin's own bursts; tail merged if too many."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) <= max_parts:
        return lines
    head = lines[: max_parts - 1]
    return [*head, "\n".join(lines[max_parts - 1 :])]


def pause_between_parts(part: str, rng: random.Random | None = None) -> float:
    rng = rng or random.Random()
    return round(min(6.0, 0.8 + len(part) / CHARS_PER_SECOND + rng.uniform(0, 0.7)), 2)


def should_skip(text: str, rng: random.Random | None = None, rate: float = SKIP_RATE) -> bool:
    """Skip about one in twenty incoming messages that are not questions."""
    if "?" in text:
        return False
    return (rng or random.Random()).random() < rate


def parts_summary(parts: Sequence[str]) -> str:
    return f"{len(parts)} part(s), {sum(len(p) for p in parts)} chars"
