"""Sanity statistics for message datasets (project brief, section 5.4).

Everything here is counts and quantiles; no text leaves this module. The output feeds
the CLI report and the manifests, so it must stay deterministic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from twin.core.schemas import Message

LONG_TEXT_CHARS = 2000
GAP_DAYS = 30


def quantile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def iso_utc(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def message_stats(messages: Sequence[Message]) -> dict[str, object]:
    """Row counts, nulls, length magnitudes, duplicates, ordering and temporal gaps."""
    lengths = sorted(len(m.text) for m in messages)
    keys = Counter((m.chat_id, m.message_id) for m in messages)
    duplicates = sum(count - 1 for count in keys.values() if count > 1)

    out_of_order = 0
    gaps_over_threshold = 0
    max_gap_seconds = 0
    previous_by_chat: dict[int, int] = {}
    for m in messages:
        previous = previous_by_chat.get(m.chat_id)
        if previous is not None:
            if m.ts < previous:
                out_of_order += 1
            gap = m.ts - previous
            if gap > GAP_DAYS * 86400:
                gaps_over_threshold += 1
            max_gap_seconds = max(max_gap_seconds, gap)
        previous_by_chat[m.chat_id] = m.ts

    ts_values = [m.ts for m in messages]
    return {
        "rows": len(messages),
        "duplicate_keys": duplicates,
        "out_of_order_rows": out_of_order,
        "empty_text_rows": sum(1 for length in lengths if length == 0),
        "long_text_rows": sum(1 for length in lengths if length > LONG_TEXT_CHARS),
        "text_chars_p50": quantile(lengths, 0.5),
        "text_chars_p90": quantile(lengths, 0.9),
        "text_chars_p99": quantile(lengths, 0.99),
        "text_chars_max": lengths[-1] if lengths else 0,
        "sender_name_null_rows": sum(1 for m in messages if m.sender_name is None),
        "reply_rows": sum(1 for m in messages if m.reply_to_message_id is not None),
        "forwarded_rows": sum(1 for m in messages if m.is_forwarded),
        "caption_rows": sum(1 for m in messages if m.media_type is not None),
        "edited_rows": sum(1 for m in messages if m.edited_ts is not None),
        "me_rows": sum(1 for m in messages if m.is_me),
        "ts_min": iso_utc(min(ts_values)) if ts_values else None,
        "ts_max": iso_utc(max(ts_values)) if ts_values else None,
        "gaps_over_30d": gaps_over_threshold,
        "max_gap_days": round(max_gap_seconds / 86400, 1),
        "rows_per_year": dict(
            sorted(Counter(iso_utc(m.ts)[:4] for m in messages).items())  # type: ignore[index]
        ),
    }
