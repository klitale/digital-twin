"""Replace secrets and contact details in message text with placeholders.

Names stay (they are part of the conversational style); only phone numbers, e-mails,
card numbers, URLs carrying tokens or invite codes, and @mentions are replaced. Every
replacement is counted by type. The rules are regular expressions applied in a fixed
order, so the result is deterministic.
"""

from __future__ import annotations

import re
from collections import Counter

from twin.ingest.dataconfig import AnonymizeConfig

PLACEHOLDERS = {
    "phone": "<phone>",
    "email": "<email>",
    "card": "<card>",
    "token_url": "<url>",
    "mention": "@user",
}

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+|\bt\.me/[^\s<>\"']+", re.IGNORECASE)
_TOKEN_QUERY_RE = re.compile(
    r"[?&#](?:token|access_token|auth|key|api_key|apikey|sig|signature|secret|session|"
    r"password|pwd|code|invite|hash)=",
    re.IGNORECASE,
)
_INVITE_PATH_RE = re.compile(r"t\.me/(?:\+|joinchat/)|/invite/|/join/", re.IGNORECASE)
# t.me/<username> with nothing after it is a profile link: the same identity as an @mention.
_PROFILE_LINK_RE = re.compile(r"^(?:https?://)?t\.me/[A-Za-z][A-Za-z0-9_]{3,31}/?$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_MENTION_RE = re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_]{3,31}\b")
_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?:[ -]?\d{1,3})?(?!\d)")
# +CC, a leading 7/8 (11 digits) or a bare Russian mobile starting with 9 (10 digits).
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s\-(]*\d{3}|[78][\s\-(]*\d{3}|9\d{2})"
    r"[\s\-)]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?![\w-])"
)


def _is_token_url(url: str) -> bool:
    return bool(
        _TOKEN_QUERY_RE.search(url) or _INVITE_PATH_RE.search(url) or _PROFILE_LINK_RE.match(url)
    )


def anonymize_text(text: str, config: AnonymizeConfig) -> tuple[str, Counter[str]]:
    """Return the anonymised text and the number of replacements per type."""
    counts: Counter[str] = Counter()

    def sub(pattern: re.Pattern[str], kind: str, value: str) -> None:
        nonlocal text
        text, n = pattern.subn(value, text)
        if n:
            counts[kind] += n

    if config.token_urls:
        placeholder = PLACEHOLDERS["token_url"]

        def replace_url(match: re.Match[str]) -> str:
            if _is_token_url(match.group(0)):
                counts["token_url"] += 1
                return placeholder
            return match.group(0)

        text = _URL_RE.sub(replace_url, text)
    if config.emails:
        sub(_EMAIL_RE, "email", PLACEHOLDERS["email"])
    if config.mentions:
        sub(_MENTION_RE, "mention", PLACEHOLDERS["mention"])
    if config.cards:
        sub(_CARD_RE, "card", PLACEHOLDERS["card"])
    if config.phones:
        sub(_PHONE_RE, "phone", PLACEHOLDERS["phone"])
    return text, counts
