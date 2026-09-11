"""Evaluation records: one JSON file per run, reproducible from the metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

CRITERIA = ("style_similarity", "appropriateness", "not_assistant_like", "consistency")


class RetrievedRecord(BaseModel):
    pair_id: str
    ts: int
    last_partner_text: str
    reply: str
    distance: float


class JudgeScores(BaseModel):
    scores: dict[str, int] = Field(description="criterion -> 1..5")
    reasons: dict[str, str]
    raw: str
    error: str | None = None

    @property
    def mean(self) -> float | None:
        values = [self.scores[c] for c in CRITERIA if c in self.scores]
        return round(sum(values) / len(values), 3) if values else None


class EvalRecord(BaseModel):
    pair_id: str
    chat_id: int
    ts: int
    period: str
    context: list[dict[str, Any]]
    incoming: str
    reference: str
    output: str | None
    raw_texts: list[str]
    rejected: list[str]
    error: str | None
    fallback_from: str | None
    mode: str
    model: str
    prompt_version: str
    params: dict[str, Any]
    latency_ms: int
    retrieved: list[RetrievedRecord]
    statements: list[RetrievedRecord] = Field(
        default_factory=list, description="his past statements shown (rag_v4+)"
    )
    judge: JudgeScores | None


class RunMetadata(BaseModel):
    run_id: str
    started_at: str
    finished_at: str
    git_commit: str | None
    dataset_version: str
    messages_dataset_version: str
    mode: str
    model: str
    prompt_version: str
    judge_model: str
    judge_prompt_version: str
    eval_config: dict[str, Any]
    eval_config_version: int
    holdout_ids_sha256: str = Field(description="identity of the evaluated sample")
    style_profile_sha256: str | None = Field(
        default=None,
        description="identity of the style profile, which is part of the prompt but is "
        "gitignored personal data; None for runs recorded before it was tracked",
    )
    dossier_sha256: str | None = Field(
        default=None, description="identity of the self dossier (rag_v4+); gitignored"
    )
    n: int


class RunSummary(BaseModel):
    n: int
    judged: int
    silent: int
    fallbacks: int
    errors: int
    means: dict[str, float]
    overall: float | None
    latency_ms_p50: int
    per_period: dict[str, dict[str, float]]


class EvalRun(BaseModel):
    metadata: RunMetadata
    summary: RunSummary
    records: list[EvalRecord]
