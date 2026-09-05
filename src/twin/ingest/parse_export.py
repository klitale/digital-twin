"""Telegram Desktop JSON export -> normalised :class:`Message` records.

Supports both export shapes:

* a single chat: ``{"name", "type", "id", "messages": [...]}``;
* a full export: ``{"chats": {"list": [...]}, "left_chats": {"list": [...]}, ...}``.

Rules (project brief, section 5.1): flatten ``text`` (string or list of fragments), drop
service messages, stickers and media without a caption, keep captions, identify the twin
via ``twin_sender_id``. Every dropped record is counted by reason; oddities that do not
cause a drop (dangling replies, unparsable optional fields) are counted as anomalies.
Deterministic: output order is the export order; nothing is sorted or de-duplicated.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twin.core.schemas import ChatSummary, DropReason, ExportKind, Message

PRIVATE_CHAT_TYPES: frozenset[str] = frozenset({"personal_chat"})
MIN_TS = 1  # 1970-01-01T00:00:01Z
MAX_TS = 4_102_444_800  # 2100-01-01T00:00:00Z
_SENDER_RE = re.compile(r"(user|channel|chat)(-?\d+)")
_INT_RE = re.compile(r"-?\d+")

# Keys Telegram Desktop writes for non-media, non-text payloads (no "media_type" on them).
PAYLOAD_KINDS: dict[str, str] = {
    "poll": "poll",
    "contact_information": "contact",
    "location_information": "location",
    "live_location_period_seconds": "location",
    "place_name": "location",
    "address": "location",
    "game_title": "game",
    "game_description": "game",
    "invoice_information": "invoice",
    "todo_list": "todo_list",
    "giveaway_information": "giveaway",
    "giveaway_results": "giveaway",
    "paid_stars_amount": "paid_media",
    "rich_message": "rich_message",
}


class ExportFormatError(ValueError):
    """The JSON does not look like a Telegram Desktop export."""


class MalformedRecordError(ValueError):
    """A single record has a shape the exporter never produces; counted, not fatal."""


@dataclass
class RawChat:
    chat_id: int
    chat_type: str
    messages: list[Any]


@dataclass
class ParseResult:
    export_kind: ExportKind
    messages: list[Message]
    dropped: Counter[str] = field(default_factory=Counter)
    dropped_service_by_action: Counter[str] = field(default_factory=Counter)
    dropped_media_by_kind: Counter[str] = field(default_factory=Counter)
    dropped_payload_by_kind: Counter[str] = field(default_factory=Counter)
    skipped_chats_by_type: Counter[str] = field(default_factory=Counter)
    anomalies: Counter[str] = field(default_factory=Counter)
    chats: list[ChatSummary] = field(default_factory=list)
    sender_counts: Counter[str] = field(default_factory=Counter)
    ts_source_counts: Counter[str] = field(default_factory=Counter)
    local_utc_offsets: Counter[str] = field(default_factory=Counter)

    @property
    def raw_messages(self) -> int:
        return sum(chat.raw_messages for chat in self.chats)


# --- loading -----------------------------------------------------------------------


def load_export(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ExportFormatError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(data, dict):
        raise ExportFormatError(f"{path}: top level is {type(data).__name__}, expected an object")
    return data


def detect_kind(data: dict[str, Any]) -> ExportKind:
    if isinstance(data.get("messages"), list):
        return ExportKind.SINGLE_CHAT
    chats = data.get("chats")
    if isinstance(chats, dict) and isinstance(chats.get("list"), list):
        return ExportKind.FULL_EXPORT
    raise ExportFormatError(
        "not a Telegram Desktop export: expected 'messages' (single chat) or 'chats.list'"
    )


def _raw_chat_lists(data: dict[str, Any]) -> list[list[Any]]:
    if detect_kind(data) is ExportKind.SINGLE_CHAT:
        return [[data]]
    lists = [data["chats"]["list"]]
    left = data.get("left_chats")
    if isinstance(left, dict) and isinstance(left.get("list"), list):
        lists.append(left["list"])
    return lists


def iter_chats(data: dict[str, Any]) -> Iterator[RawChat]:
    """Chats in export order; in a full export ``left_chats`` follow ``chats``."""
    for raw_list in _raw_chat_lists(data):
        for raw in raw_list:
            if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
                raise ExportFormatError("chat entry without a 'messages' list")
            chat_id = parse_int_field(raw.get("id"))
            if chat_id is None:
                raise ExportFormatError("chat entry without a numeric 'id'")
            chat_type = raw.get("type")
            yield RawChat(
                chat_id=chat_id,
                chat_type=chat_type if isinstance(chat_type, str) and chat_type else "unknown",
                messages=raw["messages"],
            )


# --- field normalisation -------------------------------------------------------------


def parse_int_field(value: Any) -> int | None:
    """``7`` or ``"7"`` -> ``7``; bools, floats, blanks and junk -> ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _INT_RE.fullmatch(value.strip()):
        return int(value.strip())
    return None


def flatten_text(text: Any) -> str:
    """``text`` is a string or a list of ``str`` / ``{"type", "text", ...}`` fragments.

    Fragments are concatenated verbatim (empty and whitespace-only fragments are real
    separators). Anything else is a malformed record.
    """
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if not isinstance(text, list):
        raise MalformedRecordError(f"text is {type(text).__name__}, expected str or list")
    parts: list[str] = []
    for fragment in text:
        if isinstance(fragment, str):
            parts.append(fragment)
        elif isinstance(fragment, dict):
            value = fragment.get("text")
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise MalformedRecordError("text fragment with a non-string 'text'")
            parts.append(value)
        else:
            raise MalformedRecordError(f"text fragment is {type(fragment).__name__}")
    return "".join(parts)


def parse_sender(raw_from_id: Any) -> tuple[str, int] | None:
    """``"user1001"`` -> ``("user", 1001)``; ``None`` when absent or malformed."""
    if not isinstance(raw_from_id, str):
        return None
    match = _SENDER_RE.fullmatch(raw_from_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _in_range(ts: int) -> bool:
    return MIN_TS <= ts <= MAX_TS


def parse_local_date(value: Any) -> int | None:
    """Naive ``date`` string read as if it were UTC; ``None`` when unparsable."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def parse_timestamp(raw: dict[str, Any]) -> tuple[int, str] | None:
    """Prefer ``date_unixtime`` (UTC seconds); fall back to the local ``date`` string."""
    unix = parse_int_field(raw.get("date_unixtime"))
    if unix is not None and _in_range(unix):
        return unix, "date_unixtime"
    local = parse_local_date(raw.get("date"))
    if local is not None and _in_range(local):
        return local, "date_local"
    return None


def attachment_kind(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """``(media_kind, payload_kind)``: media is photo/file based, payload is poll/contact/..."""
    media: str | None = None
    media_type = raw.get("media_type")
    if isinstance(media_type, str) and media_type:
        media = media_type
    elif "photo" in raw:
        media = "photo"
    elif "file" in raw:
        media = "document"
    payload = next((kind for key, kind in PAYLOAD_KINDS.items() if key in raw), None)
    return media, payload


def _entities_text(raw: dict[str, Any]) -> str | None:
    entities = raw.get("text_entities")
    if not isinstance(entities, list):
        return None
    parts: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            return None
        value = entity.get("text")
        parts.append(value if isinstance(value, str) else "")
    return "".join(parts)


# --- main conversion --------------------------------------------------------------


@dataclass
class Converted:
    message: Message | None
    reason: str | None
    kind: str | None = None
    raw_id: int | None = None


def convert_message(
    raw: Any, chat: RawChat, twin_sender_id: int, anomalies: Counter[str] | None = None
) -> Converted:
    """Classify one raw record: a :class:`Message`, or a drop reason with a sub-kind."""
    anomalies = anomalies if anomalies is not None else Counter()
    if not isinstance(raw, dict):
        return Converted(None, DropReason.MALFORMED, "not_an_object")
    raw_id = parse_int_field(raw.get("id"))
    if raw_id is None:
        return Converted(None, DropReason.MALFORMED, "bad_id")

    record_type = raw.get("type")
    if record_type == "service":
        action = raw.get("action")
        return Converted(
            None, DropReason.SERVICE, action if isinstance(action, str) else "?", raw_id
        )
    if record_type != "message":
        return Converted(None, DropReason.UNSUPPORTED_TYPE, str(record_type), raw_id)

    media, payload = attachment_kind(raw)
    try:
        text = flatten_text(raw.get("text"))
    except MalformedRecordError:
        return Converted(None, DropReason.MALFORMED, "bad_text", raw_id)
    if media == "sticker":
        return Converted(None, DropReason.STICKER, "sticker", raw_id)
    if not text.strip():
        if media is not None:
            return Converted(None, DropReason.MEDIA_WITHOUT_CAPTION, media, raw_id)
        if payload is not None:
            return Converted(None, DropReason.NON_TEXT_PAYLOAD, payload, raw_id)
        return Converted(None, DropReason.EMPTY_TEXT, None, raw_id)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return Converted(None, DropReason.INVALID_UNICODE, None, raw_id)

    sender = parse_sender(raw.get("from_id"))
    if sender is None:
        return Converted(None, DropReason.UNKNOWN_SENDER, None, raw_id)
    prefix, sender_id = sender

    stamp = parse_timestamp(raw)
    if stamp is None:
        return Converted(None, DropReason.BAD_TIMESTAMP, None, raw_id)
    ts, ts_source = stamp

    entities_text = _entities_text(raw)
    if entities_text is not None and entities_text != text:
        anomalies["text_entities_mismatch"] += 1

    edited_ts: int | None = None
    if raw.get("edited_unixtime") is not None:
        edited_ts = parse_int_field(raw["edited_unixtime"])
        if edited_ts is None:
            anomalies["edited_unixtime_unparsed"] += 1

    reply_to: int | None = None
    if raw.get("reply_to_message_id") is not None:
        reply_to = parse_int_field(raw["reply_to_message_id"])
        if reply_to is None:
            anomalies["reply_id_unparsed"] += 1

    sender_name = raw.get("from")
    message = Message(
        message_id=raw_id,
        chat_id=chat.chat_id,
        chat_type=chat.chat_type,
        ts=ts,
        date_local=raw["date"] if isinstance(raw.get("date"), str) else "",
        sender_id=sender_id,
        sender_raw_id=str(raw["from_id"]),
        sender_name=sender_name if isinstance(sender_name, str) else None,
        is_me=prefix == "user" and sender_id == twin_sender_id,
        text=text,
        reply_to_message_id=reply_to,
        is_forwarded="forwarded_from" in raw or "forwarded_from_id" in raw,
        edited_ts=edited_ts,
        media_type=media or payload,
        ts_source=ts_source,
    )
    return Converted(message, None, media or payload, raw_id)


def _record_offset(message: Message, offsets: Counter[str]) -> None:
    if message.ts_source != "date_unixtime":
        return
    local = parse_local_date(message.date_local)
    if local is not None:
        offsets[f"{local - message.ts:+d}"] += 1


def parse_export(
    data: dict[str, Any],
    twin_sender_id: int,
    chat_types: Iterable[str] = PRIVATE_CHAT_TYPES,
) -> ParseResult:
    allowed = frozenset(chat_types)
    result = ParseResult(export_kind=detect_kind(data), messages=[])
    for chat in iter_chats(data):
        if chat.chat_type not in allowed:
            result.dropped[DropReason.CHAT_TYPE_SKIPPED] += len(chat.messages)
            result.skipped_chats_by_type[chat.chat_type] += len(chat.messages)
            result.chats.append(
                ChatSummary(
                    chat_id=chat.chat_id,
                    chat_type=chat.chat_type,
                    raw_messages=len(chat.messages),
                    kept_messages=0,
                )
            )
            continue
        raw_ids: set[int] = set()
        kept_ids: set[int] = set()
        kept: list[Message] = []
        for raw in chat.messages:
            converted = convert_message(raw, chat, twin_sender_id, result.anomalies)
            if converted.raw_id is not None:
                raw_ids.add(converted.raw_id)
            if converted.message is None:
                assert converted.reason is not None
                result.dropped[converted.reason] += 1
                if converted.reason == DropReason.SERVICE:
                    result.dropped_service_by_action[converted.kind or "?"] += 1
                elif converted.reason == DropReason.MEDIA_WITHOUT_CAPTION:
                    result.dropped_media_by_kind[converted.kind or "?"] += 1
                elif converted.reason == DropReason.NON_TEXT_PAYLOAD:
                    result.dropped_payload_by_kind[converted.kind or "?"] += 1
                continue
            message = converted.message
            kept.append(message)
            kept_ids.add(message.message_id)
            result.sender_counts[message.sender_raw_id] += 1
            result.ts_source_counts[message.ts_source] += 1
            _record_offset(message, result.local_utc_offsets)
        for message in kept:
            target = message.reply_to_message_id
            if target is None:
                continue
            if target not in raw_ids:
                result.anomalies["reply_target_missing"] += 1
            elif target not in kept_ids:
                result.anomalies["reply_target_dropped"] += 1
        result.messages.extend(kept)
        result.chats.append(
            ChatSummary(
                chat_id=chat.chat_id,
                chat_type=chat.chat_type,
                raw_messages=len(chat.messages),
                kept_messages=len(kept),
                me_messages=sum(1 for m in kept if m.is_me),
            )
        )
    return result


def top_senders(data: dict[str, Any], n: int = 3) -> list[tuple[str, int]]:
    """Most active ``from_id`` values over ``type == "message"`` records in private chats."""
    counts: Counter[str] = Counter()
    for chat in iter_chats(data):
        if chat.chat_type not in PRIVATE_CHAT_TYPES:
            continue
        for raw in chat.messages:
            if (
                isinstance(raw, dict)
                and raw.get("type") == "message"
                and isinstance(raw.get("from_id"), str)
            ):
                counts[raw["from_id"]] += 1
    return counts.most_common(n)


# --- output -----------------------------------------------------------------------


def write_messages_jsonl(messages: Iterable[Message], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for message in messages:
            fh.write(message.model_dump_json())
            fh.write("\n")
            count += 1
    return count


def read_messages_jsonl(path: Path) -> list[Message]:
    messages: list[Message] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                messages.append(Message.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid message record") from exc
    return messages
