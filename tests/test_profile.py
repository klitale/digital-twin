from __future__ import annotations

from twin.core.schemas import ChatSplitSummary, ContextTurn, DatasetManifest, Pair
from twin.ingest.dataconfig import DataConfig
from twin.ingest.profile_dataset import caveats, pair_stats, render_report, style_by_year

CFG = DataConfig()
BASE = 1704067200


def pair(index: int, reply: str, ts: int | None = None, eval_sample: bool = False) -> Pair:
    return Pair(
        pair_id=f"1002:{index}",
        chat_id=1002,
        conversation_id="1002:0",
        ts=BASE + index * 3600 if ts is None else ts,
        period="2024-01",
        context=[ContextTurn(sender_name="Злата", is_me=False, text="вопрос?")],
        reply=reply,
        reply_message_ids=[index],
        eval_sample=eval_sample,
    )


def manifest(**overrides: object) -> DatasetManifest:
    base: dict[str, object] = {
        "dataset_version": "abc",
        "messages_dataset_version": "def",
        "config": {},
        "config_source": "built-in defaults",
        "messages": 10,
        "turns": 8,
        "twin_turns": 4,
        "conversations": 1,
        "pairs_kept": 3,
        "pairs_dropped": {"reply_too_long": 2},
        "anonymized": {"email": 1},
        "train": 3,
        "holdout_tail": 0,
        "eval_sample": 0,
        "eval_sample_by_period": {},
        "chats": [
            ChatSplitSummary(
                chat_index=1,
                pairs=3,
                train=3,
                holdout=0,
                cutoff_date=None,
                too_short_for_holdout=True,
            )
        ],
        "per_year": {"2024": {"train": 3, "holdout": 0}},
        "reply_chars": {"p50": 5, "p90": 9, "p99": 9, "max": 9, "over_max": 0},
        "context_turns": {"p50": 1, "p90": 1, "max": 1},
    }
    base.update(overrides)
    return DatasetManifest(**base)  # type: ignore[arg-type]


def test_pair_stats_counts_shares_and_gaps() -> None:
    pairs = [
        pair(1, "ахах"),
        pair(2, "ахах"),
        pair(3, "как дела?\nнорм 😂"),
        pair(4, "x" * 2001, ts=BASE + 40 * 86400),
    ]
    stats = pair_stats(pairs, max_reply_chars=2000)
    assert stats["rows"] == 4
    assert stats["duplicate_reply_rows"] == 1 and stats["distinct_replies"] == 3
    assert stats["replies_over_max"] == 1
    assert stats["replies_with_emoji_pct"] == 25.0
    assert stats["replies_with_question_pct"] == 25.0
    assert stats["replies_multiline_pct"] == 25.0
    assert stats["replies_le_10_chars_pct"] == 50.0
    assert stats["gaps_over_30d"] == 1
    assert stats["per_year"] == {"2024": 4}
    assert pair_stats([], 2000)["rows"] == 0


def test_style_by_year_table() -> None:
    pairs = [pair(1, "да"), pair(2, "нет?", ts=BASE + 366 * 86400)]
    table = style_by_year(pairs)
    assert list(table) == ["2024", "2025"]
    assert table["2025"]["question_pct"] == 100.0


def test_caveats_list_every_anomaly() -> None:
    train = [pair(i, "ахах") for i in range(10)]
    notes = caveats(pair_stats(train, 2000), pair_stats([], 2000), manifest(), CFG)
    joined = "\n".join(notes)
    assert "2 twin turns longer than 2000" in joined
    assert "repeat an identical reply" in joined
    assert "holdout: no rows" in joined
    assert "contributes no holdout" in joined
    assert "fewer than the requested 80" in joined
    assert "year 2024 holds 100.0%" in joined


def test_caveats_none_when_clean() -> None:
    # 40 train pairs 30 days apart (~3.3 years) and 80 holdout pairs 20 days apart (~4.4 years):
    # no year dominates, no gap exceeds 30 days, no duplicates.
    train = [pair(i, f"ответ номер {i}", ts=BASE + i * 86400 * 30) for i in range(40)]
    holdout = [
        pair(100 + i, f"ответ номер {100 + i}", ts=BASE + (1300 + 20 * i) * 86400)
        for i in range(80)
    ]
    m = manifest(
        pairs_dropped={},
        eval_sample=80,
        holdout_tail=80,
        chats=[
            ChatSplitSummary(
                chat_index=1,
                pairs=120,
                train=40,
                holdout=80,
                cutoff_date="x",
                too_short_for_holdout=False,
            )
        ],
    )
    assert caveats(pair_stats(train, 2000), pair_stats(holdout, 2000), m, CFG) == ["none."]


def test_render_report_has_sections_and_no_reply_text() -> None:
    train = [pair(i, f"секретный ответ {i}") for i in range(5)]
    report = render_report(train, [], manifest(), CFG)
    for heading in (
        "## Caveats",
        "## Pipeline counts",
        "## Train",
        "## Holdout tail",
        "## Split per chat",
        "## Style drift by year",
    ):
        assert heading in report
    assert "секретный" not in report
    assert "Злата" not in report
