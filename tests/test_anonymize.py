from __future__ import annotations

import pytest

from twin.ingest.anonymize import anonymize_text
from twin.ingest.dataconfig import AnonymizeConfig

ALL = AnonymizeConfig()

# Synthetic samples; the trailing marker keeps the repo privacy guard quiet on these lines.
PHONE_PLUS = "+7 999 123-45-67"  # privacy-check: allow
PHONE_8 = "89991234567"  # privacy-check: allow
PHONE_9 = "9991234567"  # privacy-check: allow
PHONE_7 = "7 999 123 45 67"  # privacy-check: allow
NOT_PHONES = "1709290800 8123456789 12345678"  # timestamp, 10-digit id, short number


@pytest.mark.parametrize(
    ("text", "expected", "kind"),
    [
        (f"позвони {PHONE_PLUS} вечером", "позвони <phone> вечером", "phone"),
        (f"номер {PHONE_8}", "номер <phone>", "phone"),
        (f"номер {PHONE_9}", "номер <phone>", "phone"),
        (f"номер {PHONE_7}", "номер <phone>", "phone"),
        ("пиши на test.user+tag@example.com", "пиши на <email>", "email"),
        ("карта 1234 5678 9012 3456 ок", "карта <card> ок", "card"),
        ("карта 1234-5678-9012-3456", "карта <card>", "card"),
        ("карта 1234 5678 9012 3456 789", "карта <card>", "card"),
        ("вот https://example.com/x?token=abc123 глянь", "вот <url> глянь", "token_url"),
        ("https://t.me/+AbCdEf", "<url>", "token_url"),
        ("https://t.me/joinchat/AbCdEf", "<url>", "token_url"),
        ("профиль https://t.me/some_person", "профиль <url>", "token_url"),
        ("профиль t.me/some_person", "профиль <url>", "token_url"),
        ("спроси @some_user ок", "спроси @user ок", "mention"),
    ],
)
def test_each_rule(text: str, expected: str, kind: str) -> None:
    result, counts = anonymize_text(text, ALL)
    assert result == expected
    assert counts == {kind: 1}


@pytest.mark.parametrize(
    "text",
    [
        "Радомир, привет",  # names stay
        "https://example.com/article/42",  # plain URLs stay
        "https://t.me/some_channel/15",  # a post link, not a profile
        "цена 1500 рублей, в 2024 году",
        NOT_PHONES,
        "в 2020 2021 2022 годах",
        "mail@ без домена",
        "@ab короткий",
    ],
)
def test_untouched(text: str) -> None:
    result, counts = anonymize_text(text, ALL)
    assert result == text
    assert counts == {}


def test_rules_can_be_disabled() -> None:
    config = AnonymizeConfig(phones=False, mentions=False)
    text = f"звони {PHONE_PLUS}, @some_user, mail: a@b.co"
    result, counts = anonymize_text(text, config)
    assert result == f"звони {PHONE_PLUS}, @some_user, mail: <email>"
    assert counts == {"email": 1}


def test_multiple_replacements_are_counted() -> None:
    text = "a@b.co и c@d.co, ещё @one_user и @two_user"
    result, counts = anonymize_text(text, ALL)
    assert result == "<email> и <email>, ещё @user и @user"
    assert counts == {"email": 2, "mention": 2}
