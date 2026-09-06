"""Generation backends behind one Protocol: ``generate(request) -> GenerationResult``.

The Telegram layer, ``twin chat`` and the evaluation harness only ever see this
interface. Validation with a single regeneration lives here so every backend behaves
the same: a reply that fails validation twice becomes silence (``text is None``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from twin.core.llm_client import LLMClient, LLMError
from twin.core.memory import MemoryTurn
from twin.core.prompt import PromptBundle, build_rag_messages, format_examples, format_history
from twin.core.prompts import PromptTemplate
from twin.core.retriever import Retriever
from twin.core.validate import DEFAULT_ASSISTANT_MARKERS, validate_reply
from twin.logsetup import get_logger

log = get_logger("twin.generation")
MAX_ATTEMPTS = 2


class GenerationRequest(BaseModel):
    partner_id: int
    chat_id: int | None = None
    message_id: int | None = None
    text: str
    previous_partner_text: str | None = None
    history: list[MemoryTurn] = Field(default_factory=list)
    ts_before: int | None = Field(default=None, description="Retrieval upper bound (eval).")
    exclude_pair_ids: list[str] = Field(default_factory=list)
    dry_run: bool = False


class GenerationResult(BaseModel):
    text: str | None
    mode: str
    model: str
    prompt_version: str
    retrieved_ids: list[str]
    params: dict[str, Any]
    latency_ms: int
    attempts: int
    rejected: list[str] = Field(description="validation reasons of rejected attempts")
    raw_texts: list[str] = Field(description="every raw completion, for eval records")
    error: str | None = Field(default=None, description="endpoint failure, if any")
    fallback_from: str | None = Field(default=None, description="mode that failed first")


class GenerationBackend(Protocol):
    mode: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


def generate_validated(
    llm: LLMClient,
    bundle: PromptBundle,
    *,
    mode: str,
    name: str,
    temperature: float,
    max_reply_chars: int,
    markers: Sequence[str],
    retrieved_ids: list[str],
    request: GenerationRequest,
) -> GenerationResult:
    """Call the model, validate, regenerate once, then stay silent."""
    started = time.perf_counter()
    rejected: list[str] = []
    raw_texts: list[str] = []
    text: str | None = None
    error: str | None = None
    model = llm.model
    attempts = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        try:
            result = llm.chat(bundle.messages, temperature=temperature)
        except LLMError as exc:
            error = str(exc)
            rejected.append("llm_error")
            raw_texts.append("")
            break
        model = result.model
        raw_texts.append(result.text)
        verdict = validate_reply(result.text, max_reply_chars, markers, name)
        if verdict.ok:
            text = verdict.text
            break
        rejected.append(verdict.reason or "invalid")
    latency_ms = int((time.perf_counter() - started) * 1000)
    params = {"temperature": temperature, "max_reply_chars": max_reply_chars}
    log.info(
        "generation",
        chat_id=request.chat_id,
        message_id=request.message_id,
        partner_id=request.partner_id,
        mode=mode,
        model=model,
        prompt_version=bundle.version,
        retrieved_example_ids=retrieved_ids,
        params=params,
        latency_ms=latency_ms,
        attempts=attempts,
        rejected=rejected,
        error=error,
        response=text,
        dry_run=request.dry_run,
    )
    return GenerationResult(
        text=text,
        mode=mode,
        model=model,
        prompt_version=bundle.version,
        retrieved_ids=retrieved_ids,
        params=params,
        latency_ms=latency_ms,
        attempts=attempts,
        rejected=rejected,
        raw_texts=raw_texts,
        error=error,
    )


class RagBackend:
    mode = "rag"

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        template: PromptTemplate,
        name: str,
        style_profile: str,
        temperature: float = 0.8,
        max_reply_chars: int = 600,
        markers: Sequence[str] = DEFAULT_ASSISTANT_MARKERS,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.template = template
        self.name = name
        self.style_profile = style_profile
        self.temperature = temperature
        self.max_reply_chars = max_reply_chars
        self.markers = markers

    def generate(self, request: GenerationRequest) -> GenerationResult:
        examples = self.retriever.retrieve(
            request.text,
            previous_partner_text=request.previous_partner_text,
            ts_before=request.ts_before,
            exclude_pair_ids=request.exclude_pair_ids,
        )
        bundle = build_rag_messages(
            self.template, self.name, self.style_profile, examples, request.history, request.text
        )
        return generate_validated(
            self.llm,
            bundle,
            mode=self.mode,
            name=self.name,
            temperature=self.temperature,
            max_reply_chars=self.max_reply_chars,
            markers=self.markers,
            retrieved_ids=[e.pair_id for e in examples],
            request=request,
        )


class FinetunedBackend:
    """Fine-tuned model on the Modal endpoint + style profile, no retrieved examples."""

    mode = "finetuned"

    def __init__(
        self,
        llm: LLMClient,
        template: PromptTemplate,
        name: str,
        style_profile: str,
        temperature: float = 0.8,
        max_reply_chars: int = 600,
        markers: Sequence[str] = DEFAULT_ASSISTANT_MARKERS,
    ) -> None:
        self.llm = llm
        self.template = template
        self.name = name
        self.style_profile = style_profile
        self.temperature = temperature
        self.max_reply_chars = max_reply_chars
        self.markers = markers

    def generate(self, request: GenerationRequest) -> GenerationResult:
        system, user = self.template.render(
            name=self.name,
            style_profile=self.style_profile.strip(),
            history=format_history(request.history, self.name),
            incoming=request.text,
        )
        bundle = PromptBundle(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            version=self.template.version,
        )
        return generate_validated(
            self.llm,
            bundle,
            mode=self.mode,
            name=self.name,
            temperature=self.temperature,
            max_reply_chars=self.max_reply_chars,
            markers=self.markers,
            retrieved_ids=[],
            request=request,
        )


class HybridBackend:
    """Fine-tuned model + retrieved examples in a short prompt (no style profile)."""

    mode = "hybrid"

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        template: PromptTemplate,
        name: str,
        temperature: float = 0.8,
        max_reply_chars: int = 600,
        markers: Sequence[str] = DEFAULT_ASSISTANT_MARKERS,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.template = template
        self.name = name
        self.temperature = temperature
        self.max_reply_chars = max_reply_chars
        self.markers = markers

    def generate(self, request: GenerationRequest) -> GenerationResult:
        examples = self.retriever.retrieve(
            request.text,
            previous_partner_text=request.previous_partner_text,
            ts_before=request.ts_before,
            exclude_pair_ids=request.exclude_pair_ids,
        )
        system, user = self.template.render(
            name=self.name,
            examples=format_examples(examples, self.name),
            history=format_history(request.history, self.name),
            incoming=request.text,
        )
        bundle = PromptBundle(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            version=self.template.version,
        )
        return generate_validated(
            self.llm,
            bundle,
            mode=self.mode,
            name=self.name,
            temperature=self.temperature,
            max_reply_chars=self.max_reply_chars,
            markers=self.markers,
            retrieved_ids=[e.pair_id for e in examples],
            request=request,
        )


class FallbackBackend:
    """Use ``primary``; when its endpoint fails (cold start, error), answer with ``fallback``."""

    def __init__(self, primary: GenerationBackend, fallback: GenerationBackend) -> None:
        self.primary = primary
        self.fallback = fallback
        self.mode = primary.mode

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = self.primary.generate(request)
        if result.error is None:
            return result
        log.warning(
            "generation.fallback",
            from_mode=self.primary.mode,
            to_mode=self.fallback.mode,
            error=result.error,
        )
        fallback = self.fallback.generate(request)
        return fallback.model_copy(update={"fallback_from": self.primary.mode})
