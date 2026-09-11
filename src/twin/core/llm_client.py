"""One chat-completion client for every OpenAI-compatible endpoint the project talks to.

Used for generation, the judge, the style profile (Timeweb AI Gateway) and the
fine-tuned model on Modal. Retries with backoff are bounded (the SDK's ``max_retries``),
every call logs tokens and latency, and the API key never reaches the logs.
Thinking/reasoning is switched off explicitly per model family: the project wants
plain, fast chat replies.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import openai
from pydantic import SecretStr

from twin.logsetup import get_logger

log = get_logger("twin.llm")

Message = dict[str, str]


class LLMError(RuntimeError):
    """The endpoint failed after the bounded retries, or returned no text."""


# A provider's content filter refusing the text (DashScope answers 400 DataInspectionFailed
# on swearing, sex or politics). Deterministic for that text, unlike a timeout or a quota.
REFUSAL_MARKERS = ("datainspectionfailed", "inappropriate content", "content_filter")


def is_content_refusal(error: object) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    finish_reason: str | None
    reasoning_tokens: int | None = None


def reasoning_off_params(model: str) -> dict[str, Any]:
    """Request parameters that disable thinking for the given model id.

    Verified against the Timeweb gateway (Phase 4): ``enable_thinking=false`` switches
    Qwen 3.5 off; GPT-5.4 mini does not reason by default and accepts
    ``reasoning_effort``; DeepSeek V4 Flash keeps reasoning (10-40 tokens on short
    prompts) whatever is sent, so the gateway ignores its switch - the client logs
    ``reasoning_tokens`` so this stays visible.
    """
    lowered = model.lower()
    if lowered.startswith("deepseek/"):
        return {"thinking": {"type": "disabled"}}
    if "qwen" in lowered:
        return {"enable_thinking": False}
    if lowered.startswith("openai/gpt-5"):
        return {"reasoning_effort": "none"}
    return {}


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | str,
        model: str,
        temperature: float = 0.8,
        timeout: float = 60.0,
        max_retries: int = 3,
        reasoning_off: bool = True,
    ) -> None:
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self._client = openai.OpenAI(
            base_url=base_url, api_key=key, timeout=timeout, max_retries=max_retries
        )
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.reasoning_off = reasoning_off

    def __repr__(self) -> str:
        return f"LLMClient(base_url={self.base_url!r}, model={self.model!r})"

    def chat(
        self,
        messages: Sequence[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResult:
        model = model or self.model
        temperature = self.temperature if temperature is None else temperature
        body: dict[str, Any] = {}
        if self.reasoning_off:
            body.update(reasoning_off_params(model))
        if extra_body:
            body.update(extra_body)
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=list(messages),  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=body or None,
            )
        except openai.OpenAIError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            log.error("llm.chat.failed", model=model, latency_ms=latency_ms, error=str(exc))
            raise LLMError(f"{model}: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "") if choice and choice.message else ""
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        result = LLMResult(
            text=text,
            model=response.model or model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason if choice else None,
            reasoning_tokens=reasoning_tokens,
        )
        log.info(
            "llm.chat",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            finish_reason=result.finish_reason,
            temperature=temperature,
        )
        if not text.strip():
            raise LLMError(f"{model}: empty completion (finish_reason={result.finish_reason})")
        return result
