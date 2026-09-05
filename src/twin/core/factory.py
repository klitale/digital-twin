"""Assemble runtime components from settings (shared by ``twin chat``, the bot, eval)."""

from __future__ import annotations

from twin.config import ConfigError, Mode, Settings
from twin.core.backends import GenerationBackend, RagBackend
from twin.core.embeddings import embeddings_from_settings
from twin.core.llm_client import LLMClient
from twin.core.memory import ConversationMemory
from twin.core.prompts import load_prompt
from twin.core.retriever import Retriever
from twin.core.vector_store import ChromaVectorStore
from twin.ingest.pipeline import STYLE_PROFILE_FILE
from twin.ingest.style_profile import read_style_profile

RAG_PROMPT = "rag_v1"


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


def build_backend(
    settings: Settings, mode: Mode | None = None, k: int | None = None
) -> GenerationBackend:
    mode = mode or settings.twin_mode
    if not settings.twin_name.strip():
        raise ConfigError("TWIN_NAME is not set")
    if mode is Mode.RAG:
        settings.require_llm()
        llm = LLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            model=settings.llm_model,
            temperature=settings.generation_temperature,
        )
        return RagBackend(
            llm=llm,
            retriever=build_retriever(settings, k),
            template=load_prompt(RAG_PROMPT),
            name=settings.twin_name,
            style_profile=load_style_profile(settings),
            temperature=settings.generation_temperature,
            max_reply_chars=settings.max_reply_chars,
        )
    raise ConfigError(f"mode {mode.value} arrives in Phase 6")
