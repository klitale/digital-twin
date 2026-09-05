"""Messages -> turns -> (context, reply) pairs for one chat.

A *turn* merges consecutive messages of one sender that are closer than the merge
window. A *pair* is a twin turn (the reply) plus up to ``max_turns`` preceding turns
that ended no earlier than ``max_age_seconds`` before the reply started. Filters are
applied in a fixed order and every rejected twin turn is counted by reason.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from twin.core.schemas import ContextTurn, Message, Pair
from twin.ingest.dataconfig import DataConfig

_WORD_RE = re.compile(r"\w")
_LINK_OR_PLACEHOLDER_RE = re.compile(
    r"(?:https?://|www\.|t\.me/)\S+|<(?:url|phone|email|card)>", re.IGNORECASE
)


class PairDrop(StrEnum):
    BEFORE_MIN_DATE = "before_min_date"
    REPLY_FORWARDED_ONLY = "reply_forwarded_only"
    NO_CONTEXT = "no_context"
    NO_PARTNER_CONTEXT = "no_partner_context"
    REPLY_TOO_SHORT = "reply_too_short"
    REPLY_NO_WORDS = "reply_no_words"
    REPLY_BARE_LINK = "reply_bare_link"
    REPLY_TOO_LONG = "reply_too_long"


@dataclass
class Turn:
    chat_id: int
    sender_id: int
    is_me: bool
    sender_name: str | None
    start_ts: int
    end_ts: int
    message_ids: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    forwarded: list[bool] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.texts)

    @property
    def own_text(self) -> str:
        """Text written by the sender (forwarded messages excluded)."""
        return "\n".join(t for t, fwd in zip(self.texts, self.forwarded, strict=True) if not fwd)


def build_turns(messages: Sequence[Message], merge_window_seconds: int) -> list[Turn]:
    """Merge consecutive same-sender messages of one chat.

    Messages must be in chronological order (the pipeline sorts by ``(ts, message_id)``).
    The window is measured from the previous message of the turn (chain merge), so a
    long run of quick messages becomes one turn even if it spans more than the window.
    """
    turns: list[Turn] = []
    for message in messages:
        current = turns[-1] if turns else None
        if (
            current is not None
            and current.sender_id == message.sender_id
            and message.ts - current.end_ts <= merge_window_seconds
        ):
            current.end_ts = max(current.end_ts, message.ts)
        else:
            current = Turn(
                chat_id=message.chat_id,
                sender_id=message.sender_id,
                is_me=message.is_me,
                sender_name=message.sender_name,
                start_ts=message.ts,
                end_ts=message.ts,
            )
            turns.append(current)
        current.message_ids.append(message.message_id)
        current.texts.append(message.text)
        current.forwarded.append(message.is_forwarded)
    return turns


def period_of(ts: int, granularity: str = "month") -> str:
    stamp = datetime.fromtimestamp(ts, tz=UTC)
    if granularity == "year":
        return stamp.strftime("%Y")
    if granularity == "none":
        return "all"
    return stamp.strftime("%Y-%m")


def reply_filter(reply: str, config: DataConfig) -> PairDrop | None:
    stripped = reply.strip()
    if len(stripped) < config.reply.min_chars:
        return PairDrop.REPLY_TOO_SHORT
    without_links = _LINK_OR_PLACEHOLDER_RE.sub("", stripped)
    if (
        config.reply.drop_bare_links
        and without_links != stripped
        and not _WORD_RE.search(without_links)
    ):
        return PairDrop.REPLY_BARE_LINK  # only links / placeholders, however many
    if config.reply.drop_no_words and not _WORD_RE.search(stripped):
        return PairDrop.REPLY_NO_WORDS
    if len(stripped) > config.reply.max_chars:
        return PairDrop.REPLY_TOO_LONG
    return None


@dataclass
class PairBuild:
    pairs: list[Pair]
    dropped: Counter[str]
    twin_turns: int
    conversations: int


def build_pairs(turns: Sequence[Turn], config: DataConfig) -> PairBuild:
    """Pairs for one chat. Conversation ids restart after a gap over ``max_age_seconds``."""
    pairs: list[Pair] = []
    dropped: Counter[str] = Counter()
    twin_turns = 0
    conversation = 0
    min_ts = config.filters.min_ts
    for index, turn in enumerate(turns):
        if index > 0 and turn.start_ts - turns[index - 1].end_ts > config.context.max_age_seconds:
            conversation += 1
        if not turn.is_me:
            continue
        twin_turns += 1
        if min_ts is not None and turn.start_ts < min_ts:
            dropped[PairDrop.BEFORE_MIN_DATE] += 1
            continue
        reply = turn.own_text if config.reply.exclude_forwarded else turn.text
        if not reply.strip() and any(turn.forwarded):
            dropped[PairDrop.REPLY_FORWARDED_ONLY] += 1
            continue
        context_turns: list[Turn] = []
        for previous in reversed(turns[max(0, index - config.context.max_turns) : index]):
            if turn.start_ts - previous.end_ts > config.context.max_age_seconds:
                break
            context_turns.append(previous)
        context_turns.reverse()
        if not context_turns:
            dropped[PairDrop.NO_CONTEXT] += 1
            continue
        if config.context.require_partner_last and context_turns[-1].is_me:
            dropped[PairDrop.NO_PARTNER_CONTEXT] += 1
            continue
        reason = reply_filter(reply, config)
        if reason is not None:
            dropped[reason] += 1
            continue
        pairs.append(
            Pair(
                pair_id=f"{turn.chat_id}:{turn.message_ids[0]}",
                chat_id=turn.chat_id,
                conversation_id=f"{turn.chat_id}:{conversation}",
                ts=turn.start_ts,
                period=period_of(turn.start_ts),
                context=[
                    ContextTurn(sender_name=c.sender_name, is_me=c.is_me, text=c.text)
                    for c in context_turns
                ],
                reply=reply,
                reply_message_ids=list(turn.message_ids),
            )
        )
    return PairBuild(
        pairs=pairs,
        dropped=dropped,
        twin_turns=twin_turns,
        conversations=conversation + 1 if turns else 0,
    )
