"""LLM-as-a-judge: Russian rubric, structured JSON, temperature 0, blind to the mode."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from twin.core.llm_client import LLMClient, LLMError
from twin.core.prompts import PromptTemplate
from twin.core.schemas import ContextTurn
from twin.eval.schemas import CRITERIA, JudgeScores

SILENCE = "(нет ответа)"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def format_context(context: Sequence[ContextTurn], name: str) -> str:
    return "\n".join(f"{name if t.is_me else 'Собеседник'}: {t.text}" for t in context)


def parse_judge_json(raw: str) -> tuple[dict[str, int], dict[str, str]]:
    """Extract the scores object; raises ``ValueError`` when it is not usable."""
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError("no JSON object in the judge response")
    data = json.loads(match.group(0))
    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for criterion in CRITERIA:
        entry = data.get(criterion)
        if isinstance(entry, dict):
            score, reason = entry.get("score"), entry.get("reason", "")
        else:
            score, reason = entry, ""
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not 1 <= int(score) <= 5
        ):
            raise ValueError(f"criterion {criterion}: score {score!r} is not 1..5")
        scores[criterion] = int(score)
        reasons[criterion] = str(reason)
    return scores, reasons


class Judge:
    def __init__(
        self,
        llm: LLMClient,
        template: PromptTemplate,
        name: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self.llm = llm
        self.template = template
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def prompt_version(self) -> str:
        return self.template.version

    def judge(
        self, context: Sequence[ContextTurn], reference: str, candidate: str | None
    ) -> JudgeScores:
        system, user = self.template.render(
            name=self.name,
            context=format_context(context, self.name),
            reference=reference,
            candidate=candidate if candidate is not None else SILENCE,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            result = self.llm.chat(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
        except LLMError as exc:
            return JudgeScores(scores={}, reasons={}, raw="", error=str(exc))
        try:
            scores, reasons = parse_judge_json(result.text)
        except (ValueError, json.JSONDecodeError) as exc:
            return JudgeScores(scores={}, reasons={}, raw=result.text, error=f"unparsable: {exc}")
        return JudgeScores(scores=scores, reasons=reasons, raw=result.text)
