"""Orchestration of ``twin ingest``: locate the export, parse, write outputs, report.

Phase 1 stops after ``messages.jsonl``; reconstruction, pairs and the split are added in
Phase 2 as further steps of the same pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from twin.config import ConfigError, Settings
from twin.core.schemas import ExportKind, MessagesManifest, counter_to_dict
from twin.ingest.parse_export import (
    ParseResult,
    load_export,
    parse_export,
    top_senders,
    write_messages_jsonl,
)
from twin.ingest.stats import message_stats

MESSAGES_FILE = "messages.jsonl"
MESSAGES_MANIFEST = "messages_manifest.json"


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


@dataclass
class IngestReport:
    export_path: Path
    export_kind: ExportKind
    messages_path: Path
    manifest_path: Path
    manifest: MessagesManifest
    stats: dict[str, object]


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


def file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


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
