"""Orchestration of ``twin ingest`` and ``twin analyze-data``.

``run_parse`` (Phase 1) writes ``messages.jsonl``; ``run_pairs`` (Phase 2) turns it
into ``pairs.jsonl`` / ``holdout.jsonl`` with a counts-only ``dataset_manifest.json``
and a Markdown profiling report.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from twin.config import ConfigError, Settings
from twin.core.schemas import (
    DatasetManifest,
    ExportKind,
    Message,
    MessagesManifest,
    Pair,
    counter_to_dict,
)
from twin.ingest.anonymize import anonymize_text
from twin.ingest.dataconfig import DataConfig
from twin.ingest.parse_export import (
    ParseResult,
    load_export,
    parse_export,
    top_senders,
    write_messages_jsonl,
)
from twin.ingest.profile_dataset import render_report
from twin.ingest.reconstruct import build_pairs, build_turns
from twin.ingest.split import split_pairs
from twin.ingest.stats import message_stats, quantile

MESSAGES_FILE = "messages.jsonl"
MESSAGES_MANIFEST = "messages_manifest.json"
PAIRS_FILE = "pairs.jsonl"
HOLDOUT_FILE = "holdout.jsonl"
DATASET_MANIFEST = "dataset_manifest.json"
PROFILE_REPORT = "profile_report.md"
STYLE_PROFILE_FILE = "style_profile.md"


class SenderNotConfiguredError(ConfigError):
    """``TWIN_SENDER_ID`` is unset or absent from the export; carries a hint."""

    def __init__(self, candidates: list[tuple[str, int]], configured: int | None) -> None:
        self.candidates = candidates
        if configured is None:
            problem = "TWIN_SENDER_ID is not set."
        else:
            problem = f"TWIN_SENDER_ID={configured} matches no sender in the export."
        if candidates:
            hint = ", ".join(f"{raw_id} ({count} messages)" for raw_id, count in candidates)
            hint = f" Most active senders in private chats: {hint}."
        else:
            hint = " The export contains no messages in private chats."
        super().__init__(
            f"{problem}{hint} Put the numeric part of the right from_id into .env and rerun."
        )


# --- shared helpers -------------------------------------------------------------------


def file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def write_jsonl(rows: Iterable[Pair], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json())
            fh.write("\n")
            count += 1
    return count


def read_pairs_jsonl(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                pairs.append(Pair.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid pair record") from exc
    return pairs


# --- Phase 1: parse ---------------------------------------------------------------------


@dataclass
class IngestReport:
    export_path: Path
    export_kind: ExportKind
    messages_path: Path
    manifest_path: Path
    manifest: MessagesManifest
    stats: dict[str, object]
    messages: list[Message]


def resolve_export_path(settings: Settings, explicit: Path | None) -> Path:
    """``--export`` > ``RAW_EXPORT_PATH`` > the single ``result.json`` under ``data/raw``."""
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"export file not found: {explicit}")
        return explicit
    if settings.raw_export_path is not None:
        if not settings.raw_export_path.is_file():
            raise ConfigError(f"RAW_EXPORT_PATH not found: {settings.raw_export_path}")
        return settings.raw_export_path
    raw_dir = settings.data_dir / "raw"
    candidates = sorted(raw_dir.rglob("result.json")) if raw_dir.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ConfigError(
            f"no result.json found under {raw_dir}; set RAW_EXPORT_PATH or pass --export"
        )
    raise ConfigError(
        f"{len(candidates)} result.json files under {raw_dir}; set RAW_EXPORT_PATH or pass --export"
    )


def build_manifest(
    result: ParseResult,
    export_path: Path,
    twin_sender_id: int,
    version: str,
    source_sha256: str,
    source_bytes: int,
) -> MessagesManifest:
    ts_values = [m.ts for m in result.messages]
    return MessagesManifest(
        dataset_version=version,
        export_kind=result.export_kind,
        source_path=str(export_path),
        source_sha256=source_sha256,
        source_bytes=source_bytes,
        twin_sender_id=twin_sender_id,
        raw_messages=result.raw_messages,
        kept_messages=len(result.messages),
        dropped=counter_to_dict(result.dropped),
        dropped_service_by_action=counter_to_dict(result.dropped_service_by_action),
        dropped_media_by_kind=counter_to_dict(result.dropped_media_by_kind),
        dropped_payload_by_kind=counter_to_dict(result.dropped_payload_by_kind),
        skipped_chats_by_type=counter_to_dict(result.skipped_chats_by_type),
        anomalies=counter_to_dict(result.anomalies),
        chats=result.chats,
        sender_counts=counter_to_dict(result.sender_counts),
        ts_min=min(ts_values) if ts_values else None,
        ts_max=max(ts_values) if ts_values else None,
        ts_source_counts=counter_to_dict(result.ts_source_counts),
        local_utc_offsets=counter_to_dict(result.local_utc_offsets),
    )


def run_parse(settings: Settings, explicit_export: Path | None = None) -> IngestReport:
    export_path = resolve_export_path(settings, explicit_export)
    data = load_export(export_path)
    twin = settings.twin_sender_id
    if twin is None:
        raise SenderNotConfiguredError(top_senders(data), None)
    result = parse_export(data, twin_sender_id=twin)
    if not any(chat.me_messages for chat in result.chats):
        raise SenderNotConfiguredError(top_senders(data), twin)
    stats = message_stats(result.messages)  # before writing: a failure leaves no partial output

    processed = settings.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    messages_path = processed / MESSAGES_FILE
    write_messages_jsonl(result.messages, messages_path)
    version, _ = file_sha256(messages_path)
    source_sha256, source_bytes = file_sha256(export_path)
    manifest = build_manifest(result, export_path, twin, version[:12], source_sha256, source_bytes)
    manifest_path = processed / MESSAGES_MANIFEST
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return IngestReport(
        export_path=export_path,
        export_kind=result.export_kind,
        messages_path=messages_path,
        manifest_path=manifest_path,
        manifest=manifest,
        stats=stats,
        messages=result.messages,
    )


def _section(title: str, counts: dict[str, int]) -> list[str]:
    if not counts:
        return []
    return [f"{title}:", *[f"  {key:<28}{value:>8}" for key, value in counts.items()]]


def format_report(report: IngestReport) -> str:
    m = report.manifest
    s = report.stats
    lines = [
        f"export: {report.export_path} ({m.export_kind.value}, {m.source_bytes} bytes)",
        f"messages.jsonl: {report.messages_path} (dataset_version {m.dataset_version})",
        "",
        f"raw messages: {m.raw_messages}   kept: {m.kept_messages}   "
        f"dropped: {m.raw_messages - m.kept_messages}",
        *_section("dropped by reason", m.dropped),
        *_section("service by action", m.dropped_service_by_action),
        *_section("media without caption by kind", m.dropped_media_by_kind),
        *_section("non-text payload by kind", m.dropped_payload_by_kind),
        *_section("skipped chats by type", m.skipped_chats_by_type),
        *_section("anomalies (kept, counted)", m.anomalies),
        "chats:",
        *[
            f"  {c.chat_type} id={c.chat_id}: raw {c.raw_messages}, kept {c.kept_messages}, "
            f"twin {c.me_messages}"
            for c in m.chats
        ],
        "kept messages per sender:",
        *[
            f"  {raw_id:<20}{count:>8}{_twin_marker(raw_id, m.twin_sender_id)}"
            for raw_id, count in m.sender_counts.items()
        ],
        *_section("timestamp source", m.ts_source_counts),
        *_section("local time minus UTC (seconds)", m.local_utc_offsets),
        "",
        "sanity (5.4):",
        *[f"  {key:<24}{value}" for key, value in s.items()],
    ]
    return "\n".join(lines)


def _twin_marker(raw_id: str, twin_sender_id: int) -> str:
    return "   <- twin" if raw_id == f"user{twin_sender_id}" else ""


# --- Phase 2: pairs, split, profile -----------------------------------------------------


@dataclass
class PairsReport:
    pairs_path: Path
    holdout_path: Path
    manifest_path: Path
    report_path: Path
    manifest: DatasetManifest
    report: str


def anonymize_messages(
    messages: Sequence[Message], config: DataConfig
) -> tuple[list[Message], Counter[str]]:
    counts: Counter[str] = Counter()
    out: list[Message] = []
    for message in messages:
        text, replaced = anonymize_text(message.text, config.anonymize)
        counts.update(replaced)
        out.append(message if text == message.text else message.model_copy(update={"text": text}))
    return out, counts


def dataset_version_of(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


class AccountingError(RuntimeError):
    """Counts that must add up do not; the dataset is not written."""


def _check_accounting(name: str, left: int, right: int) -> None:
    if left != right:
        raise AccountingError(f"{name}: {left} != {right}")


def run_pairs(
    settings: Settings,
    config: DataConfig,
    messages: Sequence[Message],
    messages_version: str,
    config_source: str = "built-in defaults",
) -> PairsReport:
    anonymized, anonymized_counts = anonymize_messages(messages, config)
    by_chat: dict[int, list[Message]] = {}
    for message in anonymized:
        by_chat.setdefault(message.chat_id, []).append(message)

    all_pairs: list[Pair] = []
    dropped: Counter[str] = Counter()
    turns_total = twin_turns = conversations = 0
    for chat_messages in by_chat.values():
        chat_messages.sort(key=lambda m: (m.ts, m.message_id))
        turns = build_turns(chat_messages, config.turns.merge_window_seconds)
        built = build_pairs(turns, config)
        all_pairs.extend(built.pairs)
        dropped.update(built.dropped)
        turns_total += len(turns)
        twin_turns += built.twin_turns
        conversations += built.conversations

    split = split_pairs(all_pairs, config.split)
    _check_accounting(
        "pairs + dropped vs twin turns", len(all_pairs) + sum(dropped.values()), twin_turns
    )
    _check_accounting(
        "train + holdout vs pairs", len(split.train) + len(split.holdout), len(all_pairs)
    )
    _check_accounting(
        "eval sample by period vs flags",
        sum(split.eval_by_period.values()),
        sum(1 for p in split.holdout if p.eval_sample),
    )

    processed = settings.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    pairs_path = processed / PAIRS_FILE
    holdout_path = processed / HOLDOUT_FILE
    write_jsonl(split.train, pairs_path)
    write_jsonl(split.holdout, holdout_path)

    per_year: dict[str, dict[str, int]] = {}
    for name, rows in (("train", split.train), ("holdout", split.holdout)):
        for pair in rows:
            year = pair.period[:4]
            per_year.setdefault(year, {"train": 0, "holdout": 0})[name] += 1
    _check_accounting(
        "per-year vs train", sum(v["train"] for v in per_year.values()), len(split.train)
    )
    _check_accounting(
        "per-year vs holdout", sum(v["holdout"] for v in per_year.values()), len(split.holdout)
    )
    kept = [*split.train, *split.holdout]
    reply_lengths = sorted(len(p.reply) for p in kept)
    context_turns = sorted(len(p.context) for p in kept)
    manifest = DatasetManifest(
        dataset_version=dataset_version_of([pairs_path, holdout_path]),
        messages_dataset_version=messages_version,
        config=config.model_dump(mode="json"),
        config_source=config_source,
        messages=len(messages),
        turns=turns_total,
        twin_turns=twin_turns,
        conversations=conversations,
        pairs_kept=len(all_pairs),
        pairs_dropped=counter_to_dict(dropped),
        anonymized=counter_to_dict(anonymized_counts),
        train=len(split.train),
        holdout_tail=len(split.holdout),
        eval_sample=sum(1 for p in split.holdout if p.eval_sample),
        eval_sample_by_period=dict(sorted(split.eval_by_period.items())),
        chats=split.chats,
        per_year=dict(sorted(per_year.items())),
        reply_chars={
            "p50": quantile(reply_lengths, 0.5),
            "p90": quantile(reply_lengths, 0.9),
            "p99": quantile(reply_lengths, 0.99),
            "max": reply_lengths[-1] if reply_lengths else 0,
            "over_max": sum(1 for n in reply_lengths if n > config.reply.max_chars),
        },
        context_turns={
            "p50": quantile(context_turns, 0.5),
            "p90": quantile(context_turns, 0.9),
            "max": context_turns[-1] if context_turns else 0,
        },
    )
    manifest_path = processed / DATASET_MANIFEST
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report = render_report(split.train, split.holdout, manifest, config)
    report_path = processed / PROFILE_REPORT
    report_path.write_text(report, encoding="utf-8")
    return PairsReport(
        pairs_path=pairs_path,
        holdout_path=holdout_path,
        manifest_path=manifest_path,
        report_path=report_path,
        manifest=manifest,
        report=report,
    )


def load_processed(settings: Settings) -> tuple[list[Pair], list[Pair], DatasetManifest]:
    processed = settings.processed_dir
    for name in (PAIRS_FILE, HOLDOUT_FILE, DATASET_MANIFEST):
        if not (processed / name).is_file():
            raise ConfigError(f"{processed / name} not found; run `twin ingest` first")
    manifest = DatasetManifest.model_validate_json(
        (processed / DATASET_MANIFEST).read_text(encoding="utf-8")
    )
    return (
        read_pairs_jsonl(processed / PAIRS_FILE),
        read_pairs_jsonl(processed / HOLDOUT_FILE),
        manifest,
    )


def run_profile(settings: Settings, config: DataConfig) -> Path:
    """``twin analyze-data``: rebuild the report from the files on disk."""
    train, holdout, manifest = load_processed(settings)
    report_path = settings.processed_dir / PROFILE_REPORT
    report_path.write_text(render_report(train, holdout, manifest, config), encoding="utf-8")
    return report_path


def format_pairs_report(report: PairsReport) -> str:
    m = report.manifest
    lines = [
        f"pairs.jsonl: {report.pairs_path} ({m.train} train)",
        f"holdout.jsonl: {report.holdout_path} ({m.holdout_tail} tail, {m.eval_sample} eval)",
        f"dataset_manifest.json: {report.manifest_path} (dataset_version {m.dataset_version}, "
        f"config: {m.config_source})",
        f"profile report: {report.report_path}",
        "",
        f"messages {m.messages} -> turns {m.turns} -> twin turns {m.twin_turns} "
        f"-> pairs {m.pairs_kept}",
        *_section("twin turns dropped by reason", m.pairs_dropped),
        *_section("anonymized", m.anonymized),
        "split per chat:",
        *[
            f"  chat {c.chat_index}: pairs {c.pairs}, train {c.train}, holdout {c.holdout}, "
            f"cutoff {c.cutoff_date}{' (too short for holdout)' if c.too_short_for_holdout else ''}"
            for c in m.chats
        ],
        "eval sample by period: "
        + ", ".join(f"{k}: {v}" for k, v in m.eval_sample_by_period.items()),
        f"reply chars: {m.reply_chars}",
        f"context turns: {m.context_turns}",
    ]
    return "\n".join(lines)
