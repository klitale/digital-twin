from __future__ import annotations

from typing import Any

from twin.core.schemas import Message
from twin.ingest.stats import message_stats, quantile

DAY = 86400


def msg(**overrides: Any) -> Message:
    base: dict[str, Any] = {
        "message_id": 1,
        "chat_id": 1,
        "chat_type": "personal_chat",
        "ts": 1709290800,
        "date_local": "2024-03-01T14:00:00",
        "sender_id": 1001,
        "sender_raw_id": "user1001",
        "sender_name": "Радомир",
        "is_me": True,
        "text": "привет",
    }
    base.update(overrides)
    return Message(**base)


def test_quantile() -> None:
    values = list(range(1, 11))
    assert (quantile(values, 0.5), quantile(values, 0.9), quantile(values, 0.99)) == (5, 9, 10)
    assert quantile([1, 2, 3, 4], 0.5) == 3
    assert quantile([], 0.5) == 0


def test_empty_input_gives_zero_stats() -> None:
    stats = message_stats([])
    assert stats["rows"] == 0
    assert stats["ts_min"] is None and stats["ts_max"] is None
    assert stats["rows_per_year"] == {}
    assert stats["max_gap_days"] == 0


def test_ordering_is_tracked_per_chat() -> None:
    messages = [
        msg(message_id=1, chat_id=1, ts=100),
        msg(message_id=1, chat_id=2, ts=50),  # another chat may start earlier
        msg(message_id=2, chat_id=1, ts=90),  # regression inside chat 1
        msg(message_id=2, chat_id=2, ts=60),
    ]
    stats = message_stats(messages)
    assert stats["out_of_order_rows"] == 1
    assert stats["duplicate_keys"] == 0  # same message_id in different chats is fine


def test_duplicate_keys_count_extra_occurrences() -> None:
    messages = [msg(message_id=5), msg(message_id=5), msg(message_id=5), msg(message_id=6)]
    assert message_stats(messages)["duplicate_keys"] == 2


def test_gap_boundaries() -> None:
    base = 1709290800
    exactly_30 = [msg(message_id=1, ts=base), msg(message_id=2, ts=base + 30 * DAY)]
    assert message_stats(exactly_30)["gaps_over_30d"] == 0
    assert message_stats(exactly_30)["max_gap_days"] == 30.0
    over = [msg(message_id=1, ts=base), msg(message_id=2, ts=base + 31 * DAY)]
    assert message_stats(over)["gaps_over_30d"] == 1
    assert message_stats(over)["max_gap_days"] == 31.0


def test_text_length_boundaries_and_nulls() -> None:
    messages = [
        msg(message_id=1, text="x" * 2000),
        msg(message_id=2, text="x" * 2001),
        msg(message_id=3, text=""),
        msg(message_id=4, sender_name=None, reply_to_message_id=1, is_forwarded=True),
        msg(message_id=5, media_type="photo", edited_ts=1, is_me=False),
    ]
    stats = message_stats(messages)
    assert stats["long_text_rows"] == 1
    assert stats["empty_text_rows"] == 1
    assert stats["text_chars_max"] == 2001
    assert stats["sender_name_null_rows"] == 1
    assert stats["reply_rows"] == 1
    assert stats["forwarded_rows"] == 1
    assert stats["caption_rows"] == 1
    assert stats["edited_rows"] == 1
    assert stats["me_rows"] == 4


def test_time_range_and_rows_per_year() -> None:
    messages = [msg(message_id=1, ts=1704067200), msg(message_id=2, ts=1735689600)]  # 2024, 2025
    stats = message_stats(messages)
    assert stats["ts_min"] == "2024-01-01T00:00:00Z"
    assert stats["ts_max"] == "2025-01-01T00:00:00Z"
    assert stats["rows_per_year"] == {"2024": 1, "2025": 1}
    assert stats["gaps_over_30d"] == 1


def test_stats_are_deterministic() -> None:
    messages = [msg(message_id=i, ts=1709290800 + i) for i in range(5)]
    assert message_stats(messages) == message_stats(list(messages))
