from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from twin.core.schemas import Message
from twin.ingest.dataconfig import DataConfig
from twin.ingest.reconstruct import PairDrop, build_pairs, build_turns, period_of, reply_filter

CFG = DataConfig()
BASE = 1709290800  # 2024-03-01T11:00:00Z


def msg(message_id: int, offset: int, text: str, me: bool = True, **overrides: Any) -> Message:
    fields: dict[str, Any] = {
        "message_id": message_id,
        "chat_id": 1002,
        "chat_type": "personal_chat",
        "ts": BASE + offset,
        "date_local": "",
        "sender_id": 1001 if me else 1002,
        "sender_raw_id": "user1001" if me else "user1002",
        "sender_name": "Радомир" if me else "Злата",
        "is_me": me,
        "text": text,
    }
    fields.update(overrides)
    return Message(**fields)


def test_turns_merge_same_sender_within_window() -> None:
    messages = [
        msg(1, 0, "а", me=False),
        msg(2, 10, "б"),
        msg(3, 190, "в"),  # 180 s after the previous message: still merged
        msg(4, 371, "г"),  # 181 s after: new turn
        msg(5, 380, "д", me=False),
        msg(6, 381, "е"),
    ]
    turns = build_turns(messages, CFG.turns.merge_window_seconds)
    assert [(t.is_me, t.text, t.message_ids) for t in turns] == [
        (False, "а", [1]),
        (True, "б\nв", [2, 3]),
        (True, "г", [4]),
        (False, "д", [5]),
        (True, "е", [6]),
    ]
    assert (turns[1].start_ts, turns[1].end_ts) == (BASE + 10, BASE + 190)


def test_turn_own_text_excludes_forwarded_messages() -> None:
    turns = build_turns([msg(1, 0, "своё"), msg(2, 5, "чужое", is_forwarded=True)], 180)
    assert turns[0].text == "своё\nчужое"
    assert turns[0].own_text == "своё"


def test_build_turns_with_zero_window_never_merges() -> None:
    assert len(build_turns([msg(1, 0, "а"), msg(2, 0, "б")], 0)) == 1  # equal ts still merges
    assert len(build_turns([msg(1, 0, "а"), msg(2, 1, "б")], 0)) == 2


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("нормальный ответ", None),
        ("ок", PairDrop.REPLY_TOO_SHORT),
        ("  ок  ", PairDrop.REPLY_TOO_SHORT),
        ("😂😂😂", PairDrop.REPLY_NO_WORDS),
        (")))", PairDrop.REPLY_NO_WORDS),
        ("!!! ...", PairDrop.REPLY_NO_WORDS),
        ("https://example.com/a", PairDrop.REPLY_BARE_LINK),
        ("www.example.com", PairDrop.REPLY_BARE_LINK),
        ("<url>", PairDrop.REPLY_BARE_LINK),
        ("https://example.com/a\nhttps://example.com/b", PairDrop.REPLY_BARE_LINK),
        ("<url>\n<url>", PairDrop.REPLY_BARE_LINK),
        ("<phone>", PairDrop.REPLY_BARE_LINK),
        ("https://example.com/a )", PairDrop.REPLY_BARE_LINK),
        ("мой <phone>", None),
        ("глянь https://example.com/a", None),
        ("x" * 2001, PairDrop.REPLY_TOO_LONG),
        ("x" * 2000, None),
        ("ок\n😂", None),  # merged multi-message reply with a word
    ],
)
def test_reply_filter(reply: str, expected: PairDrop | None) -> None:
    assert reply_filter(reply, CFG) == expected


def test_reply_filter_flags_can_be_disabled() -> None:
    cfg = DataConfig.model_validate(
        {"reply": {"drop_no_words": False, "drop_bare_links": False, "min_chars": 1}}
    )
    assert reply_filter("😂", cfg) is None
    assert reply_filter("https://example.com", cfg) is None


def test_pairs_context_window_and_ordering() -> None:
    messages = [
        msg(1, 0, "первое", me=False),
        msg(2, 1000, "второе"),
        msg(3, 2000, "третье", me=False),
        msg(4, 3000, "четвёртое"),
        msg(5, 4000, "пятое", me=False),
        msg(6, 5000, "шестое"),
        msg(7, 6000, "седьмое", me=False),
        msg(8, 7000, "восьмое"),
        msg(9, 8000, "девятое", me=False),
        msg(10, 9000, "ответ клона"),
    ]
    built = build_pairs(build_turns(messages, 180), CFG)
    assert built.twin_turns == 5
    last = built.pairs[-1]
    assert last.pair_id == "1002:10"
    assert last.reply == "ответ клона"
    assert [t.text for t in last.context] == [
        "четвёртое",
        "пятое",
        "шестое",
        "седьмое",
        "восьмое",
        "девятое",
    ]
    assert [t.is_me for t in last.context] == [True, False, True, False, True, False]
    assert last.context[0].sender_name == "Радомир" and last.context[1].sender_name == "Злата"
    assert last.reply_message_ids == [10]
    assert last.period == "2024-03"
    assert last.conversation_id == "1002:0"


def test_pairs_context_age_limit_and_conversations() -> None:
    messages = [
        msg(1, 0, "старое", me=False),
        msg(2, 100, "старый ответ"),
        msg(3, 100 + 7201, "новое после паузы", me=False),  # gap > 2 h: new conversation
        msg(4, 100 + 7300, "ответ на новое"),
    ]
    built = build_pairs(build_turns(messages, 180), CFG)
    assert built.conversations == 2
    assert [p.conversation_id for p in built.pairs] == ["1002:0", "1002:1"]
    assert [t.text for t in built.pairs[1].context] == ["новое после паузы"]


def test_pairs_drop_reasons_are_counted() -> None:
    messages = [
        msg(1, 0, "без контекста"),  # first turn: nothing precedes it
        msg(2, 1000, "вопрос", me=False),
        msg(3, 1100, "ок"),  # too short
        msg(4, 1400, "😂"),  # separate twin turn after the partner? no: own previous turn
        msg(5, 2000, "ещё", me=False),
        msg(6, 2100, "https://example.com"),  # bare link
        msg(7, 3000, "и?", me=False),
        msg(8, 3100, "чужое", is_forwarded=True),  # forwarded only
        msg(9, 4000, "ну", me=False),
        msg(10, 4100, "нормальный ответ"),
    ]
    built = build_pairs(build_turns(messages, 180), CFG)
    assert built.twin_turns == 6
    assert dict(built.dropped) == {
        PairDrop.NO_CONTEXT: 1,
        PairDrop.REPLY_TOO_SHORT: 1,
        PairDrop.NO_PARTNER_CONTEXT: 1,
        PairDrop.REPLY_BARE_LINK: 1,
        PairDrop.REPLY_FORWARDED_ONLY: 1,
    }
    assert [p.reply for p in built.pairs] == ["нормальный ответ"]


def test_min_date_filter() -> None:
    cfg = DataConfig.model_validate({"filters": {"min_date": "2024-03-01"}})
    assert cfg.filters.min_ts == 1709251200
    messages = [
        msg(1, -100000, "раньше", me=False),
        msg(2, -99900, "старый"),
        msg(3, 0, "позже", me=False),
        msg(4, 100, "новый"),
    ]
    built = build_pairs(build_turns(messages, 180), cfg)
    assert dict(built.dropped) == {PairDrop.BEFORE_MIN_DATE: 1}
    assert [p.reply for p in built.pairs] == ["новый"]


def test_forwarded_messages_stay_in_context_but_not_in_reply() -> None:
    messages = [
        msg(1, 0, "смотри", me=False),
        msg(2, 10, "переслано", me=False, is_forwarded=True),
        msg(
            3,
            100,
            "моё",
        ),
        msg(4, 110, "и переслано", is_forwarded=True),
        msg(5, 500, "ок?", me=False),
        msg(6, 600, "да, ответ"),
    ]
    built = build_pairs(build_turns(messages, 180), CFG)
    first, second = built.pairs
    assert first.reply == "моё"
    assert first.context[0].text == "смотри\nпереслано"
    assert second.context[1].text == "моё\nи переслано"  # context keeps everything

    cfg = DataConfig.model_validate({"reply": {"exclude_forwarded": False}})
    assert build_pairs(build_turns(messages, 180), cfg).pairs[0].reply == "моё\nи переслано"


def test_out_of_order_input_is_sorted_by_the_pipeline() -> None:
    from twin.config import Settings
    from twin.ingest.pipeline import run_pairs

    messages = [
        msg(3, 400, "третье", me=False),
        msg(1, 0, "первое", me=False),
        msg(4, 500, "ответ"),
        msg(2, 100, "второе"),
    ]
    settings = Settings(_env_file=None, data_dir=Path("data"))  # type: ignore[call-arg]
    report = run_pairs(settings, CFG, messages, "v")
    assert report.manifest.turns == 4  # sorted: 1 partner, 2 twin, 3 partner, 4 twin
    assert report.manifest.pairs_kept == 2


def test_period_of() -> None:
    assert period_of(BASE) == "2024-03"
    assert period_of(BASE, "year") == "2024"
    assert period_of(BASE, "none") == "all"


def test_build_pairs_is_deterministic() -> None:
    messages = [msg(i, i * 500, f"текст {i}", me=(i % 2 == 0)) for i in range(1, 40)]
    first = build_pairs(build_turns(messages, 180), CFG)
    second = build_pairs(build_turns(list(messages), 180), CFG)
    assert first.pairs == second.pairs and first.dropped == second.dropped
