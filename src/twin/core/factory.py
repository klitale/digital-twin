"""Assemble runtime components from settings (shared by ``twin chat``, the bot, eval)."""

from __future__ import annotations

from twin.config import ConfigError, Mode, Settings
from twin.core.backends import (
    FallbackBackend,
    FinetunedBackend,
    GenerationBackend,
    HybridBackend,
    InitiativeBackend,
    RagBackend,
)
from twin.core.embeddings import embeddings_from_settings
from twin.core.llm_client import LLMClient
from twin.core.memory import ConversationMemory
from twin.core.prompts import load_prompt
from twin.core.retriever import Retriever
from twin.core.vector_store import ChromaVectorStore
from twin.ingest.pipeline import STYLE_PROFILE_FILE
from twin.ingest.style_profile import read_style_profile

RAG_PROMPT = "rag_v1"
FINETUNED_PROMPT = "finetuned_v1"
HYBRID_PROMPT = "hybrid_v1"
INITIATIVE_PROMPT = "initiative_v1"


def load_style_profile(settings: Settings) -> str:
    path = settings.processed_dir / STYLE_PROFILE_FILE
    if not path.is_file():
        raise ConfigError(f"{path} not found; run `twin style-profile` first")
    return read_style_profile(path)


def build_retriever(settings: Settings, k: int | None = None) -> Retriever:
    store = ChromaVectorStore(settings.chroma_dir)
    return Retriever(store, embeddings_from_settings(settings), k=k or settings.retrieval_k)


def build_memory(settings: Settings) -> ConversationMemory:
    return ConversationMemory(settings.memory_dir, settings.history_turns)


def build_gateway_llm(settings: Settings) -> LLMClient:
    settings.require_llm()
    return LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,  # type: ignore[arg-type]
        model=settings.llm_model,
        temperature=settings.generation_temperature,
    )


def build_finetuned_llm(settings: Settings) -> LLMClient:
    if not settings.ft_base_url or not settings.ft_api_key:
        raise ConfigError("FT_BASE_URL / FT_API_KEY are not set (deploy serving/modal_app.py)")
    return LLMClient(
        base_url=settings.ft_base_url,
        api_key=settings.ft_api_key,
        model=settings.ft_model,
        temperature=settings.generation_temperature,
        timeout=settings.ft_timeout_seconds,
        max_retries=0,
        reasoning_off=False,
    )


def build_backend(
    settings: Settings,
    mode: Mode | None = None,
    k: int | None = None,
    with_fallback: bool = True,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
) -> GenerationBackend:
    """``retriever``/``llm`` overrides let the eval harness audit retrieval and pin params."""
    mode = mode or settings.twin_mode
    if not settings.twin_name.strip():
        raise ConfigError("TWIN_NAME is not set")
    if mode is Mode.RAG:
        return RagBackend(
            llm=llm or build_gateway_llm(settings),
            retriever=retriever or build_retriever(settings, k),
            template=load_prompt(RAG_PROMPT),
            name=settings.twin_name,
            style_profile=load_style_profile(settings),
            temperature=settings.generation_temperature,
            max_reply_chars=settings.max_reply_chars,
        )
    ft = build_finetuned_llm(settings)
    primary: GenerationBackend
    if mode is Mode.FINETUNED:
        primary = FinetunedBackend(
            llm=ft,
            template=load_prompt(FINETUNED_PROMPT),
            name=settings.twin_name,
            style_profile=load_style_profile(settings),
            temperature=settings.generation_temperature,
            max_reply_chars=settings.max_reply_chars,
        )
    else:
        primary = HybridBackend(
            llm=ft,
            retriever=retriever or build_retriever(settings, k),
            template=load_prompt(HYBRID_PROMPT),
            name=settings.twin_name,
            temperature=settings.generation_temperature,
            max_reply_chars=settings.max_reply_chars,
        )
    if not with_fallback:
        return primary
    return FallbackBackend(primary, build_backend(settings, Mode.RAG, k, retriever=retriever))


def build_initiative_backend(settings: Settings) -> InitiativeBackend:
    """Initiative always speaks through the gateway model: the fine-tuned 7B only knows how
    to answer an incoming message, not how to follow a task."""
    if not settings.twin_name.strip():
        raise ConfigError("TWIN_NAME is not set")
    return InitiativeBackend(
        llm=build_gateway_llm(settings),
        retriever=build_retriever(settings),
        template=load_prompt(INITIATIVE_PROMPT),
        name=settings.twin_name,
        style_profile=load_style_profile(settings),
        temperature=settings.generation_temperature,
        max_reply_chars=min(settings.max_reply_chars, 300),
    )
