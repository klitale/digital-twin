from __future__ import annotations

from pathlib import Path

import pytest

from twin.config import ConfigError
from twin.ingest.dataconfig import (
    DEFAULT_DATA_CONFIG,
    DataConfig,
    FiltersConfig,
    SplitConfig,
    load_data_config,
)

REPO_DEFAULT = Path(__file__).parent.parent / "configs" / "data" / "default.yaml"


def test_repo_default_file_mirrors_built_in_defaults_except_the_date_filter() -> None:
    loaded = load_data_config(REPO_DEFAULT)
    assert loaded.filters.min_date is not None  # project decision: drop the old-style years
    assert loaded.split.tail_fraction > DataConfig().split.tail_fraction  # keeps ~1100 tail pairs
    neutral = loaded.model_copy(
        update={"filters": FiltersConfig(), "split": SplitConfig(seed=loaded.split.seed)}
    )
    assert neutral == DataConfig()


def test_missing_default_path_falls_back_to_built_in_defaults() -> None:
    assert load_data_config(DEFAULT_DATA_CONFIG) == DataConfig()  # cwd is an empty tmp dir


def test_missing_explicit_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_data_config(tmp_path / "nope.yaml")


def test_unknown_keys_and_bad_values_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("reply:\n  min_chars: 3\n  typo: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="typo"):
        load_data_config(path)
    path.write_text("split:\n  tail_fraction: 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="tail_fraction"):
        load_data_config(path)


def test_partial_override(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("filters:\n  min_date: 2021-01-01\nsplit:\n  seed: 1\n", encoding="utf-8")
    config = load_data_config(path)
    assert config.filters.min_ts == 1609459200
    assert config.split.seed == 1 and config.split.eval_sample_size == 80
