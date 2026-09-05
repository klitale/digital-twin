"""Deterministic time-based train / holdout split with leakage assertions.

Per chat, the last ``tail_fraction`` of pairs (by time) is the holdout tail; everything
before the cutoff is training material. A fixed-size evaluation sample is drawn from
the tail, stratified by chat and period with a seeded generator. Chats with fewer than
``min_pairs_per_chat`` pairs contribute no holdout and are marked as such.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from twin.core.schemas import ChatSplitSummary, Pair
from twin.ingest.dataconfig import SplitConfig
from twin.ingest.reconstruct import period_of
from twin.ingest.stats import iso_utc


class LeakageError(AssertionError):
    """Holdout material would reach training or retrieval."""


@dataclass
class SplitResult:
    train: list[Pair]
    holdout: list[Pair]
    chats: list[ChatSplitSummary] = field(default_factory=list)
    eval_by_period: Counter[str] = field(default_factory=Counter)


def _largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` draws across strata proportionally to their sizes."""
    population = sum(sizes.values())
    if population == 0 or total <= 0:
        return dict.fromkeys(sizes, 0)
    total = min(total, population)
    exact = {key: size * total / population for key, size in sizes.items()}
    allocation = {key: min(sizes[key], int(value)) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    by_remainder = sorted(sizes, key=lambda key: (-(exact[key] - int(exact[key])), key))
    for key in by_remainder:
        if remaining == 0:
            break
        if allocation[key] < sizes[key]:
            allocation[key] += 1
            remaining -= 1
    return allocation


def choose_eval_sample(holdout: Sequence[Pair], config: SplitConfig) -> set[str]:
    """Pair ids selected for evaluation: stratified by (chat, period), seeded."""
    strata: dict[str, list[str]] = {}
    for pair in holdout:
        key = f"{pair.chat_id}:{period_of(pair.ts, config.stratify_by)}"
        strata.setdefault(key, []).append(pair.pair_id)
    allocation = _largest_remainder(
        {key: len(ids) for key, ids in strata.items()}, config.eval_sample_size
    )
    rng = random.Random(config.seed)
    chosen: set[str] = set()
    for key in sorted(strata):
        ids = sorted(strata[key])
        chosen.update(rng.sample(ids, allocation[key]))
    return chosen


def split_pairs(pairs: Sequence[Pair], config: SplitConfig) -> SplitResult:
    by_chat: dict[int, list[Pair]] = {}
    for pair in pairs:
        by_chat.setdefault(pair.chat_id, []).append(pair)

    result = SplitResult(train=[], holdout=[])
    for index, chat_pairs in enumerate(by_chat.values(), start=1):
        ordered = sorted(chat_pairs, key=lambda p: (p.ts, p.pair_id))
        count = len(ordered)
        if count < config.min_pairs_per_chat:
            result.train.extend(ordered)
            result.chats.append(
                ChatSplitSummary(
                    chat_index=index,
                    pairs=count,
                    train=count,
                    holdout=0,
                    cutoff_date=None,
                    too_short_for_holdout=True,
                )
            )
            continue
        tail_count = max(1, math.floor(count * config.tail_fraction + 0.5))
        cutoff = ordered[count - tail_count].ts
        chat_train = [p for p in ordered if p.ts < cutoff]
        chat_holdout = [p for p in ordered if p.ts >= cutoff]
        result.train.extend(chat_train)
        result.holdout.extend(chat_holdout)
        result.chats.append(
            ChatSplitSummary(
                chat_index=index,
                pairs=count,
                train=len(chat_train),
                holdout=len(chat_holdout),
                cutoff_date=iso_utc(cutoff)[:10],  # type: ignore[index]
                too_short_for_holdout=False,
            )
        )

    chosen = choose_eval_sample(result.holdout, config)
    result.holdout = [
        p.model_copy(update={"eval_sample": p.pair_id in chosen}) for p in result.holdout
    ]
    for pair in result.holdout:
        if pair.eval_sample:
            result.eval_by_period[period_of(pair.ts, config.stratify_by)] += 1
    assert_no_leakage(result.train, result.holdout)
    return result


def assert_no_leakage(train: Sequence[Pair], holdout: Sequence[Pair]) -> None:
    """Fail loudly if ids overlap or any holdout pair precedes a training pair of its chat."""
    train_ids = {p.pair_id for p in train}
    overlap = [p.pair_id for p in holdout if p.pair_id in train_ids]
    if overlap:
        raise LeakageError(f"{len(overlap)} pair id(s) present in both train and holdout")
    latest_train: dict[int, int] = {}
    for pair in train:
        latest_train[pair.chat_id] = max(latest_train.get(pair.chat_id, pair.ts), pair.ts)
    for pair in holdout:
        latest = latest_train.get(pair.chat_id)
        if latest is not None and pair.ts <= latest:
            raise LeakageError(
                f"holdout pair {pair.pair_id} is not later than every training pair of its chat"
            )
