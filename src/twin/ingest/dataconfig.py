"""Typed view of ``configs/data/*.yaml`` (dataset construction parameters)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from twin.config import ConfigError

DEFAULT_DATA_CONFIG = Path("configs/data/default.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TurnsConfig(_Strict):
    merge_window_seconds: int = Field(180, ge=0)


class ContextConfig(_Strict):
    max_turns: int = Field(6, ge=1)
    max_age_seconds: int = Field(7200, ge=1)
    require_partner_last: bool = True


class ReplyConfig(_Strict):
    min_chars: int = Field(3, ge=1)
    max_chars: int = Field(2000, ge=1)
    drop_no_words: bool = True
    drop_bare_links: bool = True
    exclude_forwarded: bool = True


class AnonymizeConfig(_Strict):
    phones: bool = True
    emails: bool = True
    cards: bool = True
    token_urls: bool = True
    mentions: bool = True


class FiltersConfig(_Strict):
    min_date: date | None = None

    @property
    def min_ts(self) -> int | None:
        if self.min_date is None:
            return None
        return int(datetime.combine(self.min_date, datetime.min.time(), tzinfo=UTC).timestamp())


class SplitConfig(_Strict):
    tail_fraction: float = Field(0.05, gt=0, lt=1)
    min_pairs_per_chat: int = Field(20, ge=1)
    eval_sample_size: int = Field(80, ge=1)
    stratify_by: Literal["month", "year", "none"] = "month"
    seed: int = 20260905


class DataConfig(_Strict):
    version: int = 1
    turns: TurnsConfig = TurnsConfig()
    context: ContextConfig = ContextConfig()
    reply: ReplyConfig = ReplyConfig()
    anonymize: AnonymizeConfig = AnonymizeConfig()
    filters: FiltersConfig = FiltersConfig()
    split: SplitConfig = SplitConfig()


BUILT_IN_SOURCE = "built-in defaults"


def config_source(path: Path = DEFAULT_DATA_CONFIG) -> str:
    return str(path) if path.is_file() else BUILT_IN_SOURCE


def load_data_config(path: Path = DEFAULT_DATA_CONFIG) -> DataConfig:
    """Load a YAML config. The repo default file mirrors the built-in defaults, so when
    it is absent (tests, a bare install) the built-in values are used; any other missing
    path is an error."""
    if not path.is_file():
        if path == DEFAULT_DATA_CONFIG:
            return DataConfig()
        raise ConfigError(f"data config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return DataConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
