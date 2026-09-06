"""Initiative: when the twin writes without being written to (opt-in, per feature).

Two features, each switched at runtime with ``/followup on|off`` and ``/opener on|off``:

* **followup** — the twin replied, the partner stayed silent for ``followup_minutes``:
  with ``followup_probability`` one short nudge, decided once per reply.
* **opener** — the chat has been silent for ``opener_silence_hours``: on
  ``opener_daily_probability`` of days one first message at a random minute inside
  ``opener_hours`` (local time), planned once per local day and never twice.

Every function here is pure over ``BotState`` + a clock + an rng, so the scheduler in
``handlers.py`` stays a thin loop and the rules are unit-tested without Telegram.
In the export the twin started about a quarter of all conversations, almost all of
them late morning, which is where the defaults come from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from twin.bot.state import BotState
from twin.config import Settings

FEATURES = ("followup", "opener")
OPENER_GRACE_S = 30 * 60  # a planned opener older than this (bot was down) is dropped


def parse_range(value: str, unit: int, name: str) -> tuple[int, int]:
    """``"10-14"`` -> ``(10*unit, 14*unit)``; fails loudly on anything else."""
    try:
        lo, hi = (int(part) for part in value.split("-", 1))
    except ValueError as exc:
        raise ValueError(f"{name} must look like 'A-B', got {value!r}") from exc
    if lo < 0 or hi <= lo:
        raise ValueError(f"{name} must satisfy 0 <= A < B, got {value!r}")
    return lo * unit, hi * unit


@dataclass(frozen=True)
class InitiativeConfig:
    tz: str
    opener_window_s: tuple[int, int]  # seconds since local midnight
    opener_daily_probability: float
    opener_silence_s: int
    followup_window_s: tuple[int, int]
    followup_probability: float

    @classmethod
    def from_settings(cls, settings: Settings) -> InitiativeConfig:
        ZoneInfo(settings.initiative_tz)  # fail fast on an unknown zone
        return cls(
            tz=settings.initiative_tz,
            opener_window_s=parse_range(settings.opener_hours, 3600, "OPENER_HOURS"),
            opener_daily_probability=settings.opener_daily_probability,
            opener_silence_s=settings.opener_silence_hours * 3600,
            followup_window_s=parse_range(settings.followup_minutes, 60, "FOLLOWUP_MINUTES"),
            followup_probability=settings.followup_probability,
        )


@dataclass(frozen=True)
class Decision:
    kind: str | None  # "followup" | "opener" | None
    reason: str


def local_midnight(now: int, tz: str) -> tuple[str, int]:
    """Return (YYYY-MM-DD, unix ts of local midnight) for ``now`` in ``tz``."""
    zone = ZoneInfo(tz)
    local = datetime.fromtimestamp(now, tz=zone)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return local.date().isoformat(), int(midnight.timestamp())


def decide_followup(
    state: BotState, chat_id: int, now: int, cfg: InitiativeConfig, rng: random.Random
) -> Decision:
    if not state.feature_on("followup"):
        return Decision(None, "followup_off")
    key = str(chat_id)
    reply_ts = state.last_bot_reply_ts.get(key)
    if reply_ts is None:
        return Decision(None, "no_bot_reply_yet")
    if state.last_outgoing_ts.get(key) != reply_ts:
        return Decision(None, "later_outgoing_message")  # owner wrote, or an initiative went out
    if state.last_incoming_ts.get(key, 0) > reply_ts:  # a tie = the bot answered that message
        return Decision(None, "partner_spoke_last")
    if state.followup_rolled_for.get(key) == reply_ts:
        return Decision(None, "already_decided")
    silence = now - reply_ts
    lo, hi = cfg.followup_window_s
    if silence < lo:
        return Decision(None, "too_soon")
    state.followup_rolled_for[key] = reply_ts  # one decision per reply, whatever it is
    if silence > hi:
        return Decision(None, "too_late")
    if rng.random() >= cfg.followup_probability:
        return Decision(None, "rolled_no")
    return Decision("followup", "ok")


def plan_opener(
    state: BotState, chat_id: int, now: int, cfg: InitiativeConfig, rng: random.Random
) -> dict[str, object]:
    """Ensure today's plan exists for the chat and return it."""
    key = str(chat_id)
    day, midnight = local_midnight(now, cfg.tz)
    plan = state.opener_plan.get(key)
    if plan is not None and plan.get("day") == day:
        return plan
    at: int | None = None
    if rng.random() < cfg.opener_daily_probability:
        lo, hi = cfg.opener_window_s
        at = midnight + rng.randint(lo, hi - 1)
    plan = {"day": day, "at": at, "done": False}
    state.opener_plan[key] = plan
    return plan


def decide_opener(
    state: BotState, chat_id: int, now: int, cfg: InitiativeConfig, rng: random.Random
) -> Decision:
    if not state.feature_on("opener"):
        return Decision(None, "opener_off")
    plan = plan_opener(state, chat_id, now, cfg, rng)
    at = plan.get("at")
    if plan.get("done"):
        return Decision(None, "done_today")
    if not isinstance(at, int):
        return Decision(None, "no_opener_today")
    if now < at:
        return Decision(None, "not_yet")
    if now - at > OPENER_GRACE_S:
        plan["done"] = True
        return Decision(None, "missed_window")
    last = state.last_activity(chat_id)
    if last is not None and now - last < cfg.opener_silence_s:
        plan["done"] = True  # the chat is alive today; no first message needed
        return Decision(None, "not_silent")
    plan["done"] = True
    return Decision("opener", "ok")


def decide(
    state: BotState, chat_id: int, now: int, cfg: InitiativeConfig, rng: random.Random
) -> Decision:
    """Follow-ups first (they are tied to a fresh reply), then the daily opener."""
    followup = decide_followup(state, chat_id, now, cfg, rng)
    if followup.kind:
        return followup
    opener = decide_opener(state, chat_id, now, cfg, rng)
    if opener.kind:
        return opener
    return Decision(None, f"{followup.reason},{opener.reason}")


def describe_plan(state: BotState, chat_id: int, now: int, tz: str) -> str:
    plan = state.opener_plan.get(str(chat_id))
    if not plan:
        return "not planned"
    at = plan.get("at")
    if plan.get("done"):
        return "done today"
    if not isinstance(at, int):
        return "no opener today"
    local = datetime.fromtimestamp(at, tz=ZoneInfo(tz))
    return "today at " + local.strftime("%H:%M") + (" (due)" if at <= now else "")


def window_text(cfg: InitiativeConfig) -> str:
    lo, hi = cfg.opener_window_s
    zero = datetime(2000, 1, 1)
    return f"{(zero + timedelta(seconds=lo)):%H:%M}-{(zero + timedelta(seconds=hi)):%H:%M} {cfg.tz}"
