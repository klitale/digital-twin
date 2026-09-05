from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from twin.core.schemas import DropReason, ExportKind, Message
from twin.ingest.parse_export import (
    ExportFormatError,
    MalformedRecordError,
    attachment_kind,
    convert_message,
    detect_kind,
    flatten_text,
    iter_chats,
    load_export,
    parse_export,
    parse_int_field,
    parse_sender,
    parse_timestamp,
    read_messages_jsonl,
    top_senders,
    write_messages_jsonl,
)

ME = 1001
FIXTURES = Path(__file__).parent / "fixtures"


def raw_message(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": 7,
        "type": "message",
        "date": "2024-03-01T14:00:00",
        "date_unixtime": "1709290800",
        "from": "Радомир Ясенев",
        "from_id": "user1001",
        "text": "привет",
        "text_entities": [{"type": "plain", "text": "привет"}],
    }
    base.update(overrides)
    return base


def single_chat(*messages: Any, **chat_overrides: Any) -> dict[str, Any]:
    chat: dict[str, Any] = {
        "name": "Злата",
        "type": "personal_chat",
        "id": 1002,
        "messages": list(messages),
    }
    chat.update(chat_overrides)
    return chat


# --- field helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("просто строка", "просто строка"),
        ("", ""),
        (None, ""),
        ("  \n", "  \n"),
        (["a ", {"type": "bold", "text": "b"}, " c"], "a b c"),
        (["Привет", "\n", " ", "мир"], "Привет\n мир"),
        ([{"type": "link", "text": "https://example.com"}, ""], "https://example.com"),
        ([{"type": "text_link", "text": "тут", "href": "https://example.com"}], "тут"),
        ([{"type": "custom_emoji", "text": "🔥", "document_id": "x"}], "🔥"),
        ([{"type": "blockquote", "text": "раз\nдва", "collapsed": True}], "раз\nдва"),
        ([{"type": "weird"}], ""),
        ([{"type": "weird", "text": None}, "!"], "!"),
    ],
)
def test_flatten_text(raw: Any, expected: str) -> None:
    assert flatten_text(raw) == expected


@pytest.mark.parametrize("raw", [[1, "x"], [["a"]], {"type": "plain", "text": "x"}, False, 5])
def test_flatten_text_rejects_malformed(raw: Any) -> None:
    with pytest.raises(MalformedRecordError):
        flatten_text(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("user1001", ("user", 1001)),
        ("channel42", ("channel", 42)),
        ("chat-7", ("chat", -7)),
        ("user1001\n", None),
        ("1001", None),
        ("user", None),
        (None, None),
        (1001, None),
    ],
)
def test_parse_sender(raw: Any, expected: tuple[str, int] | None) -> None:
    assert parse_sender(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (7, 7),
        ("7", 7),
        (" 7 ", 7),
        ("-3", -3),
        (True, None),
        (7.0, None),
        ("x", None),
        (None, None),
    ],
)
def test_parse_int_field(raw: Any, expected: int | None) -> None:
    assert parse_int_field(raw) == expected


def test_parse_timestamp_prefers_unixtime_and_falls_back_to_local_date() -> None:
    assert parse_timestamp({"date_unixtime": "1709290800", "date": "2000-01-01T00:00:00"}) == (
        1709290800,
        "date_unixtime",
    )
    assert parse_timestamp({"date": "2024-03-01T11:00:00"}) == (1709290800, "date_local")
    assert parse_timestamp({"date": "2024-03-01T14:00:00+03:00"}) == (1709290800, "date_local")
    assert parse_timestamp({"date_unixtime": "x", "date": "2024-03-01T11:00:00"}) == (
        1709290800,
        "date_local",
    )
    assert parse_timestamp({"date": "not a date"}) is None
    assert parse_timestamp({}) is None


def test_parse_timestamp_rejects_out_of_range_values() -> None:
    assert parse_timestamp({"date_unixtime": "1709290800000"}) is None  # milliseconds
    assert parse_timestamp({"date_unixtime": "-100000000000"}) is None
    assert parse_timestamp({"date_unixtime": "0"}) is None
    assert parse_timestamp({"date_unixtime": "1709290800000", "date": "2024-03-01T11:00:00"}) == (
        1709290800,
        "date_local",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, (None, None)),
        ({"media_type": "sticker"}, ("sticker", None)),
        ({"photo": "photos/a.jpg"}, ("photo", None)),
        ({"file": "files/a.pdf"}, ("document", None)),
        ({"file": "files/a.mp4", "media_type": "video_file"}, ("video_file", None)),
        ({"poll": {"question": "?"}}, (None, "poll")),
        ({"contact_information": {}}, (None, "contact")),
        ({"location_information": {}}, (None, "location")),
        ({"game_title": "x", "game_description": "y"}, (None, "game")),
        ({"invoice_information": {}}, (None, "invoice")),
        ({"giveaway_results": {}}, (None, "giveaway")),
        ({"photo": "p", "poll": {}}, ("photo", "poll")),
    ],
)
def test_attachment_kind(raw: dict[str, Any], expected: tuple[str | None, str | None]) -> None:
    assert attachment_kind(raw) == expected


# --- export shapes -----------------------------------------------------------------


def test_detect_kind_and_iter_chats() -> None:
    single = single_chat(raw_message())
    assert detect_kind(single) is ExportKind.SINGLE_CHAT
    assert [c.chat_id for c in iter_chats(single)] == [1002]

    full = {"chats": {"list": [single, {**single, "id": "1003"}]}}
    assert detect_kind(full) is ExportKind.FULL_EXPORT
    assert [c.chat_id for c in iter_chats(full)] == [1002, 1003]

    with_left = {**full, "left_chats": {"list": [{**single, "id": 42, "type": "private_group"}]}}
    assert [(c.chat_id, c.chat_type) for c in iter_chats(with_left)] == [
        (1002, "personal_chat"),
        (1003, "personal_chat"),
        (42, "private_group"),
    ]
    assert next(iter_chats(single_chat(type=""))).chat_type == "unknown"


@pytest.mark.parametrize(
    "data",
    [
        {"foo": "bar"},
        {"chats": {"list": [{"id": 1}]}},
        {"chats": {"list": [{"id": None, "messages": []}]}},
        {"chats": {"list": [{"id": "abc", "messages": []}]}},
        {"chats": {"list": [{"messages": []}]}},
    ],
)
def test_bad_shapes_raise_export_format_error(data: dict[str, Any]) -> None:
    with pytest.raises(ExportFormatError):
        detect_kind(data)
        list(iter_chats(data))


@pytest.mark.parametrize("content", ["[]", "{not json"])
def test_load_export_rejects_invalid_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "result.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ExportFormatError):
        load_export(path)


# --- conversion rules ----------------------------------------------------------------


def convert(
    raw: Any, anomalies: Counter[str] | None = None
) -> tuple[Message | None, str | None, str | None]:
    chat = next(iter_chats(single_chat()))
    converted = convert_message(raw, chat, ME, anomalies)
    return converted.message, converted.reason, converted.kind


def test_plain_message_is_kept_with_all_contract_fields() -> None:
    message, reason, kind = convert(raw_message(reply_to_message_id=3))
    assert reason is None and kind is None
    assert message is not None
    assert message.message_id == 7
    assert message.chat_id == 1002
    assert message.chat_type == "personal_chat"
    assert message.ts == 1709290800
    assert message.date_local == "2024-03-01T14:00:00"
    assert message.sender_id == 1001
    assert message.sender_raw_id == "user1001"
    assert message.sender_name == "Радомир Ясенев"
    assert message.is_me is True
    assert message.text == "привет"
    assert message.reply_to_message_id == 3
    assert message.is_forwarded is False
    assert message.media_type is None
    assert message.ts_source == "date_unixtime"


@pytest.mark.parametrize(
    "overrides",
    [{"from_id": "user1002"}, {"from_id": "channel1001"}, {"from_id": "chat1001"}],
)
def test_is_me_requires_user_prefix_and_matching_id(overrides: dict[str, Any]) -> None:
    message, _, _ = convert(raw_message(**overrides))
    assert message is not None and message.is_me is False


@pytest.mark.parametrize(
    ("overrides", "reason", "kind"),
    [
        ({"type": "service", "action": "phone_call"}, DropReason.SERVICE, "phone_call"),
        ({"type": "service"}, DropReason.SERVICE, "?"),
        ({"type": "something_else"}, DropReason.UNSUPPORTED_TYPE, "something_else"),
        (
            {"text": "", "media_type": "sticker", "sticker_emoji": "😂"},
            DropReason.STICKER,
            "sticker",
        ),
        ({"text": "подпись", "media_type": "sticker"}, DropReason.STICKER, "sticker"),
        ({"text": "", "photo": "photos/a.jpg"}, DropReason.MEDIA_WITHOUT_CAPTION, "photo"),
        (
            {"text": "", "media_type": "voice_message"},
            DropReason.MEDIA_WITHOUT_CAPTION,
            "voice_message",
        ),
        (
            {"text": "", "file": "(File not included.)"},
            DropReason.MEDIA_WITHOUT_CAPTION,
            "document",
        ),
        ({"text": "", "contact_information": {}}, DropReason.NON_TEXT_PAYLOAD, "contact"),
        ({"text": "", "poll": {"question": "?"}}, DropReason.NON_TEXT_PAYLOAD, "poll"),
        ({"text": "", "invoice_information": {}}, DropReason.NON_TEXT_PAYLOAD, "invoice"),
        ({"text": ""}, DropReason.EMPTY_TEXT, None),
        ({"text": "   \n"}, DropReason.EMPTY_TEXT, None),
        ({"text": [], "text_entities": []}, DropReason.EMPTY_TEXT, None),
        ({"text": ["", {"type": "bold", "text": ""}]}, DropReason.EMPTY_TEXT, None),
        ({"from_id": None}, DropReason.UNKNOWN_SENDER, None),
        ({"from_id": "1001"}, DropReason.UNKNOWN_SENDER, None),
        ({"date": None, "date_unixtime": None}, DropReason.BAD_TIMESTAMP, None),
        ({"date": "garbage", "date_unixtime": "x"}, DropReason.BAD_TIMESTAMP, None),
        ({"date": None, "date_unixtime": "1709290800000"}, DropReason.BAD_TIMESTAMP, None),
        ({"id": None}, DropReason.MALFORMED, "bad_id"),
        ({"id": "abc"}, DropReason.MALFORMED, "bad_id"),
        ({"text": [1, "x"]}, DropReason.MALFORMED, "bad_text"),
        ({"text": {"type": "plain", "text": "x"}}, DropReason.MALFORMED, "bad_text"),
        ({"text": "a\ud83db"}, DropReason.INVALID_UNICODE, None),
    ],
)
def test_drop_reasons(overrides: dict[str, Any], reason: str | None, kind: str | None) -> None:
    message, got_reason, got_kind = convert(raw_message(**overrides))
    assert got_reason == reason
    assert got_kind == kind
    assert message is None


def test_non_object_record_is_malformed() -> None:
    for raw in ("oops", None, 5, ["x"]):
        message, reason, kind = convert(raw)
        assert (message, reason, kind) == (None, DropReason.MALFORMED, "not_an_object")


def test_kept_message_with_channel_sender() -> None:
    message, reason, _ = convert(raw_message(from_id="channel5"))
    assert reason is None and message is not None
    assert message.sender_id == 5 and message.sender_raw_id == "channel5"


def test_caption_is_kept_with_media_type() -> None:
    message, _, kind = convert(raw_message(text="смотри", photo="photos/a.jpg", width=1, height=1))
    assert message is not None
    assert message.text == "смотри"
    assert message.media_type == "photo" and kind == "photo"
    message, _, _ = convert(raw_message(text=["a", {"type": "bold", "text": "b"}], file="f"))
    assert message is not None and message.text == "ab" and message.media_type == "document"
    message, _, _ = convert(raw_message(text="игра", game_title="x", game_description="y"))
    assert message is not None and message.media_type == "game"


@pytest.mark.parametrize(
    "overrides",
    [
        {"forwarded_from": "Канал", "forwarded_from_id": "channel12"},
        {"forwarded_from": None, "forwarded_from_id": "user1005"},
        {"forwarded_from": "Скрытый"},
        {"forwarded_from_id": "user1005"},
    ],
)
def test_forwarded_flag_uses_key_presence(overrides: dict[str, Any]) -> None:
    message, _, _ = convert(raw_message(**overrides))
    assert message is not None
    assert message.is_forwarded is True
    assert message.sender_id == 1001  # the forwarding sender, never the origin


def test_optional_fields_and_anomalies() -> None:
    anomalies: Counter[str] = Counter()
    message, _, _ = convert(
        raw_message(
            edited="2024-03-01T15:00:00",
            edited_unixtime="1709294400",
            reply_to_message_id="8",
            **{"from": None},
        ),
        anomalies,
    )
    assert message is not None
    assert message.edited_ts == 1709294400
    assert message.reply_to_message_id == 8
    assert message.sender_name is None
    assert anomalies == {}

    message, _, _ = convert(
        raw_message(
            edited_unixtime="zz",
            reply_to_message_id=True,
            text_entities=[{"type": "plain", "text": "другое"}],
        ),
        anomalies,
    )
    assert message is not None
    assert message.edited_ts is None and message.reply_to_message_id is None
    assert anomalies == {
        "edited_unixtime_unparsed": 1,
        "reply_id_unparsed": 1,
        "text_entities_mismatch": 1,
    }


def test_list_text_is_flattened_in_export_order() -> None:
    text = [
        "глянь ",
        {"type": "link", "text": "https://example.com/p"},
        " там ",
        {"type": "bold", "text": "жесть"},
    ]
    message, _, _ = convert(raw_message(text=text))
    assert message is not None
    assert message.text == "глянь https://example.com/p там жесть"


# --- whole exports -----------------------------------------------------------------


def test_synthetic_single_chat_fixture(synthetic_export_path: Path) -> None:
    result = parse_export(load_export(synthetic_export_path), twin_sender_id=ME)
    assert result.export_kind is ExportKind.SINGLE_CHAT
    assert result.raw_messages == 21
    assert len(result.messages) == 17
    assert dict(result.dropped) == {
        DropReason.SERVICE: 1,
        DropReason.STICKER: 1,
        DropReason.MEDIA_WITHOUT_CAPTION: 2,
    }
    assert dict(result.dropped_service_by_action) == {"phone_call": 1}
    assert dict(result.dropped_media_by_kind) == {"photo": 1, "voice_message": 1}
    assert dict(result.dropped_payload_by_kind) == {}
    assert dict(result.skipped_chats_by_type) == {}
    assert dict(result.anomalies) == {}
    assert result.sender_counts == {"user1001": 10, "user1002": 7}
    assert dict(result.ts_source_counts) == {"date_unixtime": 17}
    assert dict(result.local_utc_offsets) == {"+0": 17}
    by_id = {m.message_id: m for m in result.messages}
    assert by_id[8].media_type == "photo" and by_id[8].text == "смотри какой кот"
    assert by_id[11].reply_to_message_id == 8
    assert by_id[12].is_forwarded is True
    assert by_id[17].text == "ладно\nнапишу ему вечером\nесли не забуду"
    assert by_id[17].edited_ts is not None
    assert [m.message_id for m in result.messages] == sorted(by_id)  # export order kept
    assert result.chats[0].model_dump() == {
        "chat_id": 1002,
        "chat_type": "personal_chat",
        "raw_messages": 21,
        "kept_messages": 17,
        "me_messages": 10,
    }


def test_synthetic_full_export_fixture() -> None:
    result = parse_export(load_export(FIXTURES / "export_full_synthetic.json"), twin_sender_id=ME)
    assert result.export_kind is ExportKind.FULL_EXPORT
    assert result.raw_messages == 13
    assert len(result.messages) == 6
    assert dict(result.dropped) == {
        DropReason.CHAT_TYPE_SKIPPED: 4,
        DropReason.SERVICE: 1,
        DropReason.STICKER: 1,
        DropReason.MEDIA_WITHOUT_CAPTION: 1,
    }
    assert dict(result.skipped_chats_by_type) == {"private_group": 2, "bot_chat": 2}
    assert [(c.chat_id, c.chat_type, c.kept_messages, c.me_messages) for c in result.chats] == [
        (1002, "personal_chat", 4, 2),
        (1003, "personal_chat", 2, 1),
        (2001, "private_group", 0, 0),
        (3001, "bot_chat", 0, 0),
    ]
    assert result.sender_counts == {"user1001": 3, "user1002": 2, "user1003": 1}
    assert dict(result.local_utc_offsets) == {"+10800": 4, "+0": 2}
    assert dict(result.anomalies) == {}
    by_key = {(m.chat_id, m.message_id): m for m in result.messages}
    assert by_key[(1003, 102)].sender_name is None  # deleted account: "from": null
    assert by_key[(1002, 6)].reply_to_message_id == 1
    assert (
        by_key[(1002, 2)].text
        == "Да, обязательно иду. Вот расписание: https://example.invalid/raspisanie"
    )
    assert top_senders(load_export(FIXTURES / "export_full_synthetic.json")) == [
        ("user1001", 3),
        ("user1002", 3),
        ("user1003", 2),
    ]


def test_single_chat_and_full_export_produce_identical_messages() -> None:
    chat = single_chat(raw_message(id=1), raw_message(id=2, from_id="user1002"))
    as_single = parse_export(chat, twin_sender_id=ME)
    as_full = parse_export({"chats": {"list": [chat]}}, twin_sender_id=ME)
    assert as_single.messages == as_full.messages
    assert as_single.chats == as_full.chats


def test_full_export_keeps_only_private_chats_unless_asked() -> None:
    personal = single_chat(raw_message(id=1), raw_message(id=2, from_id="user1002"))
    group = single_chat(
        raw_message(id=1), raw_message(id=2, type="service"), id=555, type="private_group"
    )
    left = single_chat(raw_message(id=9), id=777, type="private_supergroup")
    full = {"chats": {"about": "x", "list": [personal, group]}, "left_chats": {"list": [left]}}
    result = parse_export(full, twin_sender_id=ME)
    assert len(result.messages) == 2
    assert result.dropped[DropReason.CHAT_TYPE_SKIPPED] == 3
    assert dict(result.skipped_chats_by_type) == {"private_group": 2, "private_supergroup": 1}
    assert [(c.chat_type, c.raw_messages, c.kept_messages) for c in result.chats] == [
        ("personal_chat", 2, 2),
        ("private_group", 2, 0),
        ("private_supergroup", 1, 0),
    ]
    assert top_senders(full) == [("user1001", 1), ("user1002", 1)]

    with_groups = parse_export(
        full, twin_sender_id=ME, chat_types={"personal_chat", "private_group"}
    )
    assert len(with_groups.messages) == 3
    assert with_groups.dropped[DropReason.SERVICE] == 1


def test_reply_targets_are_audited_not_dropped() -> None:
    data = single_chat(
        raw_message(id=1, text="", media_type="sticker"),
        raw_message(id=2, from_id="user1002", reply_to_message_id=1),
        raw_message(id=3, reply_to_message_id=999),
        raw_message(id=4, reply_to_message_id=3),
    )
    result = parse_export(data, twin_sender_id=ME)
    assert [m.reply_to_message_id for m in result.messages] == [1, 999, 3]
    assert dict(result.anomalies) == {"reply_target_dropped": 1, "reply_target_missing": 1}


def test_top_senders_ranks_message_records_only() -> None:
    data = single_chat(
        *[raw_message(id=i, from_id="user1001") for i in range(1, 5)],
        *[raw_message(id=i, from_id="user1002") for i in range(5, 8)],
        *[raw_message(id=i, from_id="user1003") for i in range(8, 10)],
        raw_message(id=10, from_id="user1004"),
        raw_message(id=11, type="service", actor_id="user1004"),
        raw_message(id=12, from_id=None),
    )
    assert top_senders(data) == [("user1001", 4), ("user1002", 3), ("user1003", 2)]
    assert top_senders(data, n=1) == [("user1001", 4)]


def test_ts_source_fallback_is_counted_end_to_end() -> None:
    data = single_chat(raw_message(id=1), raw_message(id=2, date_unixtime=None))
    result = parse_export(data, twin_sender_id=ME)
    assert [m.ts_source for m in result.messages] == ["date_unixtime", "date_local"]
    assert result.messages[1].ts == 1709301600  # local string read as UTC
    assert dict(result.ts_source_counts) == {"date_unixtime": 1, "date_local": 1}
    assert dict(result.local_utc_offsets) == {"+10800": 1}


def test_parse_is_deterministic(synthetic_export_path: Path) -> None:
    data = load_export(synthetic_export_path)
    first = parse_export(data, twin_sender_id=ME)
    second = parse_export(json.loads(json.dumps(data)), twin_sender_id=ME)
    for attr in (
        "messages",
        "dropped",
        "dropped_service_by_action",
        "dropped_media_by_kind",
        "dropped_payload_by_kind",
        "skipped_chats_by_type",
        "anomalies",
        "chats",
        "sender_counts",
        "ts_source_counts",
        "local_utc_offsets",
    ):
        assert getattr(first, attr) == getattr(second, attr), attr


def test_jsonl_roundtrip_and_malformed_lines(tmp_path: Path, synthetic_export_path: Path) -> None:
    result = parse_export(load_export(synthetic_export_path), twin_sender_id=ME)
    path = tmp_path / "out" / "messages.jsonl"
    assert write_messages_jsonl(result.messages, path) == 17
    assert read_messages_jsonl(path) == result.messages

    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    assert len(read_messages_jsonl(path)) == 17
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"message_id": "abc"}\n')
    with pytest.raises(ValueError, match=r"messages\.jsonl:20: invalid message record"):
        read_messages_jsonl(path)
