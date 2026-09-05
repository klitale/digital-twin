from __future__ import annotations

import pytest

from twin.core.schemas import ContextTurn, Pair
from twin.ingest.dataconfig import SplitConfig
from twin.ingest.split import LeakageError, _largest_remainder, assert_no_leakage, split_pairs

BASE = 1704067200  # 2024-01-01T00:00:00Z
DAY = 86400


def pair(index: int, chat_id: int = 1002, ts: int | None = None) -> Pair:
    stamp = BASE + index * DAY if ts is None else ts
    return Pair(
        pair_id=f"{chat_id}:{index}",
        chat_id=chat_id,
        conversation_id=f"{chat_id}:0",
        ts=stamp,
        period="2024-01",
        context=[ContextTurn(sender_name="Злата", is_me=False, text="q")],
        reply=f"ответ {index}",
        reply_message_ids=[index],
    )


def test_largest_remainder_allocation() -> None:
    assert _largest_remainder({"a": 50, "b": 30, "c": 20}, 10) == {"a": 5, "b": 3, "c": 2}
    assert _largest_remainder({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert sum(_largest_remainder({"a": 7, "b": 3, "c": 3}, 8).values()) == 8
    assert _largest_remainder({"a": 2, "b": 2}, 10) == {"a": 2, "b": 2}  # capped by population
    assert _largest_remainder({}, 5) == {}


def test_tail_split_by_time_and_eval_sample() -> None:
    pairs = [pair(i) for i in range(100)]  # one per day
    config = SplitConfig(tail_fraction=0.1, min_pairs_per_chat=20, eval_sample_size=4, seed=1)
    result = split_pairs(pairs, config)
    assert len(result.train) == 90 and len(result.holdout) == 10
    assert max(p.ts for p in result.train) < min(p.ts for p in result.holdout)
    assert result.chats[0].model_dump() == {
        "chat_index": 1,
        "pairs": 100,
        "train": 90,
        "holdout": 10,
        "cutoff_date": "2024-03-31",
        "too_short_for_holdout": False,
    }
    chosen = [p for p in result.holdout if p.eval_sample]
    assert len(chosen) == 4
    assert sum(result.eval_by_period.values()) == 4
    assert all(not p.eval_sample for p in result.train)


def test_ties_at_the_cutoff_go_to_holdout() -> None:
    pairs = [pair(i, ts=BASE + (i // 2) * DAY) for i in range(40)]  # two pairs per day
    result = split_pairs(pairs, SplitConfig(tail_fraction=0.05, min_pairs_per_chat=20))
    # round(40 * 0.05) = 2 tail pairs, but the cutoff day holds 2 pairs already: all of them go
    assert len(result.holdout) == 2
    assert max(p.ts for p in result.train) < min(p.ts for p in result.holdout)


def test_short_chat_contributes_no_holdout() -> None:
    pairs = [pair(i) for i in range(10)] + [pair(i, chat_id=2) for i in range(50)]
    result = split_pairs(pairs, SplitConfig(min_pairs_per_chat=20, eval_sample_size=3))
    assert [(c.chat_index, c.too_short_for_holdout, c.holdout) for c in result.chats] == [
        (1, True, 0),
        (2, False, 3),
    ]
    assert all(p.chat_id == 2 for p in result.holdout)


def test_split_is_deterministic_and_seed_dependent() -> None:
    pairs = [pair(i) for i in range(200)]
    a = split_pairs(pairs, SplitConfig(seed=7, eval_sample_size=5))
    b = split_pairs(list(reversed(pairs)), SplitConfig(seed=7, eval_sample_size=5))
    assert [p.pair_id for p in a.holdout if p.eval_sample] == [
        p.pair_id for p in b.holdout if p.eval_sample
    ]
    c = split_pairs(pairs, SplitConfig(seed=8, eval_sample_size=5))
    assert {p.pair_id for p in a.holdout if p.eval_sample} != {
        p.pair_id for p in c.holdout if p.eval_sample
    }


def test_eval_sample_is_stratified_by_period() -> None:
    pairs = [pair(i) for i in range(0, 120)]  # Jan..Apr 2024
    result = split_pairs(pairs, SplitConfig(tail_fraction=0.5, eval_sample_size=6, seed=3))
    assert len(result.holdout) == 60  # Mar 1 .. Apr 29
    assert sum(result.eval_by_period.values()) == 6
    assert set(result.eval_by_period) == {"2024-03", "2024-04"}
    assert result.eval_by_period["2024-03"] == 3  # 31 vs 29 days -> 3 / 3


def test_eval_sample_capped_by_tail_size() -> None:
    result = split_pairs([pair(i) for i in range(40)], SplitConfig(eval_sample_size=80))
    assert len(result.holdout) == 2 and sum(p.eval_sample for p in result.holdout) == 2


def test_assert_no_leakage() -> None:
    train = [pair(0), pair(1)]
    assert_no_leakage(train, [pair(2)])
    with pytest.raises(LeakageError, match="present in both"):
        assert_no_leakage(train, [pair(1)])
    with pytest.raises(LeakageError, match="not later"):
        assert_no_leakage(train, [pair(5, ts=BASE + DAY)])
    assert_no_leakage(train, [pair(5, chat_id=2, ts=BASE)])  # another chat: no constraint
