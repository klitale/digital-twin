from __future__ import annotations

from pathlib import Path

import pytest

from twin.config import ConfigError, EmbedProvider, Mode, Settings, parse_int_list


def make(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type]


def test_defaults_do_not_need_any_environment() -> None:
    settings = make()
    assert settings.twin_mode is Mode.RAG
    assert settings.dry_run is True
    assert settings.embed_provider is EmbedProvider.OPENAI_COMPATIBLE
    assert settings.allowed_user_ids == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", []), ("1", [1]), ("1, 2", [1, 2]), ("1;2", [1, 2]), ([3, "4"], [3, 4])],
)
def test_parse_int_list(raw: object, expected: list[int]) -> None:
    assert parse_int_list(raw) == expected


def test_id_lists_are_parsed_from_comma_separated_strings() -> None:
    settings = make(allowed_user_ids="1001, 1002", admin_user_ids="1003")
    assert settings.allowed_user_ids == [1001, 1002]
    assert settings.admin_user_ids == [1003]


def test_dotenv_is_read_and_empty_values_are_ignored(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "TWIN_SENDER_ID=1001\nALLOWED_USER_IDS=1002,1003\nHF_TOKEN=\nLLM_API_KEY=sk-test\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env)  # type: ignore[call-arg]
    assert settings.twin_sender_id == 1001
    assert settings.allowed_user_ids == [1002, 1003]
    assert settings.hf_token is None
    assert settings.llm_api_key is not None


def test_secrets_never_appear_in_repr() -> None:
    settings = make(llm_api_key="very-secret-value", tg_bot_token="1:abc")
    assert "very-secret" not in repr(settings)
    assert "very-secret" not in str(settings.model_dump())


def test_judge_and_embed_keys_fall_back_to_llm_key_on_same_gateway() -> None:
    settings = make(llm_api_key="sk-shared")
    assert settings.judge_api_key_effective is not None
    assert settings.embed_api_key_effective is not None
    other = make(llm_api_key="sk-shared", judge_base_url="https://other.example/v1")
    assert other.judge_api_key_effective is None


def test_require_ingest_needs_sender_id() -> None:
    with pytest.raises(ConfigError, match="TWIN_SENDER_ID"):
        make().require_ingest()
    make(twin_sender_id=1001).require_ingest()


def test_require_bot_bounds_the_allowlist() -> None:
    base = {"tg_bot_token": "1:abc", "business_owner_id": 1, "admin_user_ids": "1"}
    with pytest.raises(ConfigError, match="1 to 3 distinct"):
        make(allowed_user_ids="", **base).require_bot()
    with pytest.raises(ConfigError, match="1 to 3 distinct"):
        make(allowed_user_ids="1001,1002,1003,1004", **base).require_bot()
    make(allowed_user_ids="1001", **base).require_bot()
    make(allowed_user_ids="1001,1001", **base).require_bot()  # duplicates collapse
    make(allowed_user_ids="1001,1002", **base).require_bot()
    make(allowed_user_ids="1001,1002,1003", **base).require_bot()


def test_require_bot_reports_missing_token_and_owner() -> None:
    with pytest.raises(ConfigError, match="TG_BOT_TOKEN"):
        make(allowed_user_ids="1001,1002").require_bot()
    with pytest.raises(ConfigError, match="BUSINESS_OWNER_ID"):
        make(tg_bot_token="1:abc", allowed_user_ids="1001,1002").require_bot()
