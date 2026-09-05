"""Public data contracts shared by ingestion, retrieval, training and evaluation.

Every on-disk dataset (``messages.jsonl``, later ``pairs.jsonl``) is a sequence of these
models serialised with ``model_dump_json``. Changing a field here is a dataset version
change and must be reflected in the manifests.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExportKind(StrEnum):
    SINGLE_CHAT = "single_chat"
    FULL_EXPORT = "full_export"


class DropReason(StrEnum):
    """Why a raw export record did not become a :class:`Message`. Counted, never silent."""

    SERVICE = "service"
    STICKER = "sticker"
    MEDIA_WITHOUT_CAPTION = "media_without_caption"
    NON_TEXT_PAYLOAD = "non_text_payload"
    EMPTY_TEXT = "empty_text"
    BAD_TIMESTAMP = "bad_timestamp"
    UNKNOWN_SENDER = "unknown_sender"
    INVALID_UNICODE = "invalid_unicode"
    MALFORMED = "malformed"
    UNSUPPORTED_TYPE = "unsupported_type"
    CHAT_TYPE_SKIPPED = "chat_type_skipped"


class Message(BaseModel):
    """One text-bearing message from a Telegram Desktop export, normalised."""

    model_config = ConfigDict(frozen=True)

    message_id: int
    chat_id: int
    chat_type: str
    ts: int = Field(description="Unix seconds, UTC (from date_unixtime).")
    date_local: str = Field(description="Original 'date' string from the export (local time).")
    sender_id: int
    sender_raw_id: str = Field(description="Original from_id, e.g. 'user1001'.")
    sender_name: str | None
    is_me: bool
    text: str = Field(description="Flattened text; for media messages this is the caption.")
    reply_to_message_id: int | None = None
    is_forwarded: bool = False
    edited_ts: int | None = None
    media_type: str | None = Field(
        default=None, description="Attachment kind when the text is a caption (photo, ...)."
    )
    ts_source: str = Field(default="date_unixtime", description="date_unixtime | date_local")


class ContextTurn(BaseModel):
    """One preceding turn in a pair's context."""

    model_config = ConfigDict(frozen=True)

    sender_name: str | None
    is_me: bool
    text: str


class Pair(BaseModel):
    """A twin reply with the conversation that preceded it (``pairs.jsonl`` row)."""

    model_config = ConfigDict(frozen=True)

    pair_id: str = Field(description="'<chat_id>:<first reply message_id>'")
    chat_id: int
    conversation_id: str = Field(description="'<chat_id>:<session index>' (gap > context age)")
    ts: int = Field(description="Unix seconds UTC of the first reply message.")
    period: str = Field(description="YYYY-MM of ts (UTC), the stratification unit.")
    context: list[ContextTurn]
    reply: str
    reply_message_ids: list[int]
    eval_sample: bool = Field(default=False, description="Holdout rows chosen for evaluation.")


class ChatSplitSummary(BaseModel):
    """Per-chat split counts keyed by position, never by Telegram id."""

    chat_index: int
    pairs: int
    train: int
    holdout: int
    cutoff_date: str | None = Field(description="YYYY-MM-DD (UTC) of the first holdout pair.")
    too_short_for_holdout: bool


class DatasetManifest(BaseModel):
    """Counts-only, committed description of ``pairs.jsonl`` / ``holdout.jsonl``."""

    dataset_version: str = Field(description="sha256 prefix over pairs.jsonl + holdout.jsonl")
    messages_dataset_version: str
    config: dict[str, object]
    config_source: str = Field(description="YAML path or 'built-in defaults'")
    messages: int
    turns: int
    twin_turns: int
    conversations: int
    pairs_kept: int
    pairs_dropped: dict[str, int]
    anonymized: dict[str, int]
    train: int
    holdout_tail: int
    eval_sample: int
    eval_sample_by_period: dict[str, int]
    chats: list[ChatSplitSummary]
    per_year: dict[str, dict[str, int]] = Field(description="year -> {train, holdout}")
    reply_chars: dict[str, int] = Field(description="p50, p90, p99, max, over_max")
    context_turns: dict[str, int] = Field(description="p50, p90, max")


class ChatSummary(BaseModel):
    """Per-chat counts for the manifest. No names."""

    chat_id: int
    chat_type: str
    raw_messages: int
    kept_messages: int
    me_messages: int = 0


class MessagesManifest(BaseModel):
    """Counts-only description of a ``messages.jsonl`` build (gitignored: carries ids)."""

    dataset_version: str = Field(description="sha256 prefix of messages.jsonl")
    export_kind: ExportKind
    source_path: str
    source_sha256: str
    source_bytes: int
    twin_sender_id: int
    raw_messages: int
    kept_messages: int
    dropped: dict[str, int]
    dropped_service_by_action: dict[str, int]
    dropped_media_by_kind: dict[str, int]
    dropped_payload_by_kind: dict[str, int]
    skipped_chats_by_type: dict[str, int]
    anomalies: dict[str, int] = Field(
        description="Non-drop oddities: dangling replies, entity mismatches, unparsed fields"
    )
    chats: list[ChatSummary]
    sender_counts: dict[str, int] = Field(description="raw from_id -> kept messages")
    ts_min: int | None
    ts_max: int | None
    ts_source_counts: dict[str, int]
    local_utc_offsets: dict[str, int] = Field(
        description="'date' minus date_unixtime in seconds -> kept messages with that offset"
    )


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
