"""How hard the twin pushes, as one switch (``/aggro``).

The level scales four edge behaviours and nothing else: how often an incoming message
is ignored, how often a follow-up and an opener fire, and how many messages one reply
may be split into. Wording, style and the model are untouched, so a level change is
never a style change; every gate keeps working exactly as before.

Base values come from the settings, so ``normal`` reproduces the configured behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from twin.bot.humanize import MAX_PARTS, SKIP_RATE

LEVELS = ("low", "normal", "high")
DEFAULT = "normal"

# level -> (skip-rate factor, initiative-probability factor, messages per reply)
FACTORS: dict[str, tuple[float, float, int]] = {
    "low": (3.0, 0.4, 2),
    "normal": (1.0, 1.0, MAX_PARTS),
    "high": (0.2, 2.0, MAX_PARTS),
}
MAX_SKIP_RATE = 0.9


@dataclass(frozen=True)
class Aggression:
    level: str
    skip_rate: float
    followup_probability: float
    opener_daily_probability: float
    max_parts: int

    def describe(self) -> str:
        return (
            f"{self.level} (пропуск {self.skip_rate:.0%}, дожим {self.followup_probability:.0%}, "
            f"опенер {self.opener_daily_probability:.0%}/день, до {self.max_parts} сообщений)"
        )


def normalise(level: str | None) -> str:
    return level if level in LEVELS else DEFAULT


def resolve(
    level: str | None,
    followup_probability: float,
    opener_daily_probability: float,
    skip_rate: float = SKIP_RATE,
) -> Aggression:
    name = normalise(level)
    skip_factor, initiative_factor, max_parts = FACTORS[name]
    clamp = lambda value: min(1.0, max(0.0, value))  # noqa: E731
    return Aggression(
        level=name,
        skip_rate=min(MAX_SKIP_RATE, max(0.0, skip_rate * skip_factor)),
        followup_probability=clamp(followup_probability * initiative_factor),
        opener_daily_probability=clamp(opener_daily_probability * initiative_factor),
        max_parts=max_parts,
    )
