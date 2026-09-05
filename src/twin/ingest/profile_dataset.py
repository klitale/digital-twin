"""Profiling report for the pair dataset (project brief, section 5.4).

Counts, quantiles and shares only: the report never quotes a reply or a name, so it can
be pasted into a chat. Every anomaly the checks find is listed under *Caveats*.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise

from twin.core.schemas import DatasetManifest, Pair
from twin.ingest.dataconfig import DataConfig
from twin.ingest.stats import GAP_DAYS, quantile

_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿]")


def _year(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y")


def _share(count: int, total: int) -> float:
    return round(100 * count / total, 1) if total else 0.0


def pair_stats(pairs: Sequence[Pair], max_reply_chars: int) -> dict[str, object]:
    reply_lengths = sorted(len(p.reply) for p in pairs)
    context_turns = sorted(len(p.context) for p in pairs)
    context_chars = sorted(sum(len(t.text) for t in p.context) for p in pairs)
    id_counts = Counter(p.pair_id for p in pairs)
    reply_counts = Counter(p.reply for p in pairs)
    ordered = sorted(pairs, key=lambda p: (p.chat_id, p.ts))
    gaps_over = 0
    max_gap = 0
    for previous, current in pairwise(ordered):
        if previous.chat_id != current.chat_id:
            continue
        gap = current.ts - previous.ts
        max_gap = max(max_gap, gap)
        if gap > GAP_DAYS * 86400:
            gaps_over += 1
    total = len(pairs)
    return {
        "rows": total,
        "duplicate_pair_ids": sum(c - 1 for c in id_counts.values() if c > 1),
        "duplicate_reply_rows": sum(c - 1 for c in reply_counts.values() if c > 1),
        "distinct_replies": len(reply_counts),
        "empty_replies": sum(1 for length in reply_lengths if length == 0),
        "reply_chars_p50": quantile(reply_lengths, 0.5),
        "reply_chars_p90": quantile(reply_lengths, 0.9),
        "reply_chars_p99": quantile(reply_lengths, 0.99),
        "reply_chars_max": reply_lengths[-1] if reply_lengths else 0,
        "replies_over_max": sum(1 for length in reply_lengths if length > max_reply_chars),
        "replies_le_10_chars_pct": _share(sum(1 for n in reply_lengths if n <= 10), total),
        "replies_with_emoji_pct": _share(sum(1 for p in pairs if _EMOJI_RE.search(p.reply)), total),
        "replies_with_question_pct": _share(sum(1 for p in pairs if "?" in p.reply), total),
        "replies_multiline_pct": _share(sum(1 for p in pairs if "\n" in p.reply), total),
        "context_turns_p50": quantile(context_turns, 0.5),
        "context_turns_p90": quantile(context_turns, 0.9),
        "context_turns_max": context_turns[-1] if context_turns else 0,
        "context_chars_p50": quantile(context_chars, 0.5),
        "context_chars_p90": quantile(context_chars, 0.9),
        "context_sender_name_null_rows": sum(
            1 for p in pairs if any(t.sender_name is None for t in p.context)
        ),
        "gaps_over_30d": gaps_over,
        "max_gap_days": round(max_gap / 86400, 1),
        "per_year": dict(sorted(Counter(_year(p.ts) for p in pairs).items())),
    }


def style_by_year(pairs: Sequence[Pair]) -> dict[str, dict[str, object]]:
    """Style drift table: per year, how the twin's replies look."""
    by_year: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_year.setdefault(_year(pair.ts), []).append(pair)
    table: dict[str, dict[str, object]] = {}
    for year in sorted(by_year):
        rows = by_year[year]
        lengths = sorted(len(p.reply) for p in rows)
        table[year] = {
            "pairs": len(rows),
            "reply_chars_p50": quantile(lengths, 0.5),
            "reply_chars_p90": quantile(lengths, 0.9),
            "le_10_chars_pct": _share(sum(1 for n in lengths if n <= 10), len(rows)),
            "emoji_pct": _share(sum(1 for p in rows if _EMOJI_RE.search(p.reply)), len(rows)),
            "question_pct": _share(sum(1 for p in rows if "?" in p.reply), len(rows)),
            "multiline_pct": _share(sum(1 for p in rows if "\n" in p.reply), len(rows)),
            "context_turns_p50": quantile(sorted(len(p.context) for p in rows), 0.5),
        }
    return table


def caveats(
    train_stats: dict[str, object],
    holdout_stats: dict[str, object],
    manifest: DatasetManifest,
    config: DataConfig,
) -> list[str]:
    notes: list[str] = []
    dropped = manifest.pairs_dropped
    if dropped.get("reply_too_long"):
        notes.append(
            f"{dropped['reply_too_long']} twin turns longer than {config.reply.max_chars} chars "
            "were dropped (section 5.4 flag)."
        )
    for name, stats in (("train", train_stats), ("holdout", holdout_stats)):
        rows = int(stats["rows"])
        if rows == 0:
            notes.append(f"{name}: no rows.")
            continue
        if stats["duplicate_pair_ids"]:
            notes.append(f"{name}: {stats['duplicate_pair_ids']} duplicate pair ids.")
        if stats["empty_replies"]:
            notes.append(f"{name}: {stats['empty_replies']} empty replies.")
        if stats["replies_over_max"]:
            notes.append(f"{name}: {stats['replies_over_max']} replies over the max length.")
        dup_pct = _share(int(stats["duplicate_reply_rows"]), rows)
        if dup_pct > 10:
            notes.append(
                f"{name}: {dup_pct}% of rows repeat an identical reply text "
                "(short reactions dominate; retrieval must de-duplicate)."
            )
        if stats["gaps_over_30d"]:
            notes.append(
                f"{name}: {stats['gaps_over_30d']} gap(s) over {GAP_DAYS} days between "
                f"consecutive pairs (max {stats['max_gap_days']} days)."
            )
        if stats["context_sender_name_null_rows"]:
            notes.append(
                f"{name}: {stats['context_sender_name_null_rows']} rows have a context turn "
                "without a sender name."
            )
        per_year = dict(stats["per_year"])  # type: ignore[call-overload]
        if per_year:
            top_year, top_count = max(per_year.items(), key=lambda kv: kv[1])
            top_share = _share(top_count, rows)
            if top_share > 35:
                notes.append(
                    f"{name}: year {top_year} holds {top_share}% of rows; style drift across "
                    "years is likely (see the drift table) - consider filters.min_date."
                )
    for chat in manifest.chats:
        if chat.too_short_for_holdout:
            notes.append(
                f"chat {chat.chat_index}: only {chat.pairs} pairs, below "
                f"min_pairs_per_chat={config.split.min_pairs_per_chat}; contributes no holdout."
            )
    if manifest.eval_sample < config.split.eval_sample_size:
        notes.append(
            f"evaluation sample has {manifest.eval_sample} pairs, fewer than the requested "
            f"{config.split.eval_sample_size}."
        )
    if len(manifest.chats) > 1:
        total_pairs = sum(c.pairs for c in manifest.chats)
        top = max(manifest.chats, key=lambda c: c.pairs)
        if total_pairs and top.pairs / total_pairs > 0.7:
            notes.append(
                f"chat {top.chat_index} holds {_share(top.pairs, total_pairs)}% of all pairs; "
                "per-chat imbalance."
            )
    if train_stats["rows"] and holdout_stats["rows"]:
        train_p50 = int(train_stats["reply_chars_p50"])
        holdout_p50 = int(holdout_stats["reply_chars_p50"])
        if train_p50 and abs(holdout_p50 - train_p50) / train_p50 > 0.5:
            notes.append(
                f"median reply length differs a lot between train ({train_p50}) and holdout "
                f"({holdout_p50}); the holdout tail may not represent the training style."
            )
        for key, label in (
            ("replies_with_emoji_pct", "emoji"),
            ("replies_with_question_pct", "question"),
            ("replies_le_10_chars_pct", "<= 10 chars"),
            ("replies_multiline_pct", "multi-line"),
        ):
            train_share = float(train_stats[key])  # type: ignore[arg-type]
            holdout_share = float(holdout_stats[key])  # type: ignore[arg-type]
            if abs(train_share - holdout_share) > 5:
                notes.append(
                    f"{label} share differs between train ({train_share}%) and holdout "
                    f"({holdout_share}%); recent style drifted from the training bulk."
                )
    if not notes:
        notes.append("none.")
    return notes


def _table(rows: dict[str, dict[str, object]]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    columns = list(next(iter(rows.values())).keys())
    lines = ["| period | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for key, values in rows.items():
        lines.append(f"| {key} | " + " | ".join(str(values[c]) for c in columns) + " |")
    return lines


def _kv_table(stats: dict[str, object]) -> list[str]:
    lines = ["| metric | value |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in stats.items() if key != "per_year"]
    return lines


def render_report(
    train: Sequence[Pair], holdout: Sequence[Pair], manifest: DatasetManifest, config: DataConfig
) -> str:
    train_stats = pair_stats(train, config.reply.max_chars)
    holdout_stats = pair_stats(holdout, config.reply.max_chars)
    eval_rows = [p for p in holdout if p.eval_sample]
    eval_stats = pair_stats(eval_rows, config.reply.max_chars)
    lines = [
        "# Dataset profile",
        "",
        f"dataset_version `{manifest.dataset_version}` built from messages "
        f"`{manifest.messages_dataset_version}` with data config v{config.version}.",
        "",
        "## Caveats",
        "",
        *[f"- {note}" for note in caveats(train_stats, holdout_stats, manifest, config)],
        "",
        "## Pipeline counts",
        "",
        "| stage | count |",
        "|---|---|",
        f"| messages | {manifest.messages} |",
        f"| turns (after {config.turns.merge_window_seconds}s merge) | {manifest.turns} |",
        f"| twin turns (pair candidates) | {manifest.twin_turns} |",
        f"| conversations (gap > {config.context.max_age_seconds}s) | {manifest.conversations} |",
        f"| pairs kept | {manifest.pairs_kept} |",
        f"| train | {manifest.train} |",
        f"| holdout tail | {manifest.holdout_tail} |",
        f"| evaluation sample | {manifest.eval_sample} |",
        "",
        "### Twin turns dropped by reason",
        "",
        "| reason | count |",
        "|---|---|",
        *[f"| {reason} | {count} |" for reason, count in manifest.pairs_dropped.items()],
        "",
        "### Anonymisation replacements",
        "",
        "| type | count |",
        "|---|---|",
        *[f"| {kind} | {count} |" for kind, count in manifest.anonymized.items()],
        "",
        "## Train",
        "",
        *_kv_table(train_stats),
        "",
        "## Holdout tail",
        "",
        *_kv_table(holdout_stats),
        "",
        "## Evaluation sample",
        "",
        *_kv_table(eval_stats),
        "",
        "sample by period: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(manifest.eval_sample_by_period.items())),
        "",
        "## Split per chat",
        "",
        "| chat | pairs | train | holdout | cutoff (UTC date) | too short |",
        "|---|---|---|---|---|---|",
        *[
            f"| {c.chat_index} | {c.pairs} | {c.train} | {c.holdout} | {c.cutoff_date} | "
            f"{c.too_short_for_holdout} |"
            for c in manifest.chats
        ],
        "",
        "## Rows per year (train / holdout)",
        "",
        "| year | train | holdout |",
        "|---|---|---|",
        *[f"| {y} | {v['train']} | {v['holdout']} |" for y, v in manifest.per_year.items()],
        "",
        "## Style drift by year (train + holdout)",
        "",
        *_table(style_by_year([*train, *holdout])),
        "",
    ]
    return "\n".join(lines)
