"""Application settings loaded from environment variables and ``.env``.

Three Telegram identities are deliberately separate, because in this project they
are three different accounts:

* ``twin_sender_id`` - whose messages in the export are the replies we imitate.
* ``business_owner_id`` - the account the bot is connected to via Telegram Business.
* ``admin_user_ids`` - accounts allowed to send ``/twin`` control commands.

Nothing is validated eagerly beyond types. Each entry point calls the ``require_*``
method for the prerequisites it actually needs, so ``twin ingest`` never fails because
a bot token is missing.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ConfigError(ValueError):
    """A command's prerequisites are missing from the configuration."""


class Mode(StrEnum):
    RAG = "rag"
    FINETUNED = "finetuned"
    HYBRID = "hybrid"


class EmbedProvider(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    LOCAL_E5 = "local_e5"


def parse_int_list(value: object) -> list[int]:
    """Parse ``"1, 2"`` / ``"1;2"`` / ``[1, 2]`` into ``[1, 2]``; empty input gives ``[]``."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        return [int(p) for p in parts if p]
    if isinstance(value, list | tuple):
        return [int(v) for v in value]
    raise TypeError(f"cannot parse an id list from {type(value).__name__}")


IntList = Annotated[list[int], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    # Persona
    twin_name: str = ""

    # Telegram identities
    tg_bot_token: SecretStr | None = None
    twin_sender_id: int | None = None
    business_owner_id: int | None = None
    admin_user_ids: IntList = Field(default_factory=list)
    allowed_user_ids: IntList = Field(default_factory=list)
    pause_minutes: int = 30
    twin_mode: Mode = Mode.RAG
    dry_run: bool = True

    # Generation LLM (OpenAI-compatible gateway)
    llm_base_url: str = "https://api.timeweb.ai/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "dashscope/qwen3.5-flash"

    # Fine-tuned model endpoint (Modal, OpenAI-compatible)
    ft_base_url: str | None = None
    ft_api_key: SecretStr | None = None
    ft_model: str = "base"

    # Judge
    judge_base_url: str = "https://api.timeweb.ai/v1"
    judge_api_key: SecretStr | None = None
    judge_model: str = "openai/gpt-5.4-mini"

    style_profile_model: str = "anthropic/claude-sonnet-4-6"

    # Embeddings
    embed_provider: EmbedProvider = EmbedProvider.OPENAI_COMPATIBLE
    embed_base_url: str = "https://api.timeweb.ai/v1"
    embed_api_key: SecretStr | None = None
    embed_model: str = "dashscope/text-embedding-v4"

    # Modal / Hugging Face
    modal_token_id: SecretStr | None = None
    modal_token_secret: SecretStr | None = None
    hf_token: SecretStr | None = None
    hf_repo_id: str | None = None

    # Retrieval and generation
    embed_batch_size: int = 10  # the gateway rejects larger embedding batches (503)
    embed_workers: int = 4
    retrieval_k: int = 8
    history_turns: int = 10
    generation_temperature: float = 0.8

    # Limits and paths
    max_reply_chars: int = 600
    data_dir: Path = Path("data")
    raw_export_path: Path | None = None

    @field_validator("admin_user_ids", "allowed_user_ids", mode="before")
    @classmethod
    def _split_ids(cls, value: object) -> list[int]:
        return parse_int_list(value)

    # --- derived -----------------------------------------------------------------

    @property
    def judge_api_key_effective(self) -> SecretStr | None:
        """Judge key, falling back to the LLM key when both use the same gateway."""
        return self.judge_api_key or self._same_gateway_key(self.judge_base_url)

    @property
    def embed_api_key_effective(self) -> SecretStr | None:
        """Embedding key, falling back to the LLM key when both use the same gateway."""
        return self.embed_api_key or self._same_gateway_key(self.embed_base_url)

    def _same_gateway_key(self, base_url: str) -> SecretStr | None:
        if self.llm_api_key and base_url.rstrip("/") == self.llm_base_url.rstrip("/"):
            return self.llm_api_key
        return None

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def memory_dir(self) -> Path:
        return self.state_dir / "memory"

    # --- prerequisite checks -------------------------------------------------------

    def require_ingest(self) -> None:
        if self.twin_sender_id is None:
            raise ConfigError(
                "TWIN_SENDER_ID is not set: it must be the Telegram id of the person whose "
                "messages are the replies to imitate"
            )

    def require_llm(self) -> None:
        if not self.llm_api_key:
            raise ConfigError("LLM_API_KEY is not set")

    def require_embeddings(self) -> None:
        if (
            self.embed_provider is EmbedProvider.OPENAI_COMPATIBLE
            and not self.embed_api_key_effective
        ):
            raise ConfigError("EMBED_API_KEY is not set and LLM_API_KEY cannot be reused")

    def require_judge(self) -> None:
        if not self.judge_api_key_effective:
            raise ConfigError("JUDGE_API_KEY is not set and LLM_API_KEY cannot be reused")

    def require_bot(self) -> None:
        if not self.tg_bot_token:
            raise ConfigError("TG_BOT_TOKEN is not set")
        if self.business_owner_id is None:
            raise ConfigError("BUSINESS_OWNER_ID is not set")
        if len(set(self.allowed_user_ids)) != 2:
            raise ConfigError("ALLOWED_USER_IDS must contain exactly two distinct user ids")
        if not self.admin_user_ids:
            raise ConfigError("ADMIN_USER_IDS is empty: nobody could control the bot")


def load_settings(**overrides: object) -> Settings:
    """Load settings from the environment (and ``.env`` in the working directory)."""
    return Settings(**overrides)  # type: ignore[arg-type]
