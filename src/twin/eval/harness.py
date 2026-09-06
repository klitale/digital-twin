"""Generate and judge every holdout pair for one mode, with leakage assertions.

For each pair the incoming message is the last partner turn of the stored context; the
earlier turns are the history; retrieval is bounded to ``ts < pair.ts`` and never
returns the target pair or any holdout pair (asserted here, loudly).
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from twin.core.backends import GenerationBackend, GenerationRequest
from twin.core.memory import MemoryTurn
from twin.core.retriever import RetrievedExample, Retriever
from twin.core.schemas import Pair
from twin.eval.judge import Judge
from twin.eval.schemas import (
    CRITERIA,
    EvalRecord,
    EvalRun,
    RetrievedRecord,
    RunMetadata,
    RunSummary,
)
from twin.ingest.stats import quantile
from twin.logsetup import get_logger

log = get_logger("twin.eval")


class LeakageError(AssertionError):
    pass


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class JudgeConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        prompt: str = "judge_v1"
        temperature: float = 0.0
        max_tokens: int = 500

    class GenerationConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        temperature: float = 0.8
        k: int = 8

    version: int = 1
    judge: JudgeConfig = JudgeConfig()
    generation: GenerationConfig = GenerationConfig()
    limit: int | None = None


DEFAULT_EVAL_CONFIG = Path("configs/eval/default.yaml")


def load_eval_config(path: Path = DEFAULT_EVAL_CONFIG) -> EvalConfig:
    """The repo default mirrors the built-in defaults; other missing paths are errors."""
    if not path.is_file():
        if path == DEFAULT_EVAL_CONFIG:
            return EvalConfig()
        raise FileNotFoundError(f"eval config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return EvalConfig.model_validate(yaml.safe_load(fh) or {})


class AuditingRetriever:
    """Wraps a retriever and remembers what it returned so the harness can assert on it."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.last: list[RetrievedExample] = []
        self.k = inner.k
        self.manifest = inner.manifest

    def retrieve(self, *args: Any, **kwargs: Any) -> list[RetrievedExample]:
        self.last = self.inner.retrieve(*args, **kwargs)
        return self.last


def split_pair(pair: Pair) -> tuple[list[MemoryTurn], str, str | None]:
    """``(history, incoming, previous_partner_text)`` from a stored pair's context."""
    turns = list(pair.context)
    if not turns or turns[-1].is_me:
        raise ValueError(f"pair {pair.pair_id}: the last context turn must be the partner's")
    incoming = turns[-1].text
    history = [MemoryTurn(is_me=t.is_me, text=t.text, ts=pair.ts) for t in turns[:-1]]
    previous = next((t.text for t in reversed(turns[:-1]) if not t.is_me), None)
    return history, incoming, previous


def assert_no_leakage(
    pair: Pair, retrieved: Sequence[RetrievedExample], holdout_ids: set[str]
) -> None:
    for example in retrieved:
        if example.pair_id == pair.pair_id:
            raise LeakageError(f"{pair.pair_id}: the target reply was retrieved")
        if example.pair_id in holdout_ids:
            raise LeakageError(f"{pair.pair_id}: holdout pair {example.pair_id} was retrieved")
        if example.ts >= pair.ts:
            raise LeakageError(
                f"{pair.pair_id}: retrieved {example.pair_id} is not earlier "
                f"({example.ts} >= {pair.ts})"
            )


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def holdout_identity(pairs: Sequence[Pair]) -> str:
    return hashlib.sha256("\n".join(sorted(p.pair_id for p in pairs)).encode()).hexdigest()[:12]


def summarize(records: Sequence[EvalRecord]) -> RunSummary:
    judged = [r for r in records if r.judge and not r.judge.error]
    means = {
        c: round(sum(r.judge.scores[c] for r in judged) / len(judged), 3) if judged else 0.0  # type: ignore[union-attr]
        for c in CRITERIA
    }
    overall = round(sum(means.values()) / len(CRITERIA), 3) if judged else None
    per_period: dict[str, dict[str, float]] = {}
    for period in sorted({r.period for r in records}):
        rows = [r for r in judged if r.period == period]
        if rows:
            per_period[period] = {
                "n": float(len(rows)),
                **{c: round(sum(r.judge.scores[c] for r in rows) / len(rows), 3) for c in CRITERIA},  # type: ignore[union-attr]
            }
    return RunSummary(
        n=len(records),
        judged=len(judged),
        silent=sum(1 for r in records if r.output is None),
        fallbacks=sum(1 for r in records if r.fallback_from),
        errors=sum(1 for r in records if r.error or (r.judge and r.judge.error)),
        means=means,
        overall=overall,
        latency_ms_p50=quantile(sorted(r.latency_ms for r in records), 0.5),
        per_period=per_period,
    )


def run_eval(
    pairs: Sequence[Pair],
    holdout_ids: set[str],
    backend: GenerationBackend,
    retriever: AuditingRetriever | None,
    judge: Judge,
    config: EvalConfig,
    dataset_version: str,
    messages_dataset_version: str,
    style_profile_sha256: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EvalRun:
    started = datetime.now(tz=UTC)
    sample = [p for p in pairs if p.eval_sample]
    if config.limit is not None:
        sample = sample[: config.limit]
    if not sample:
        raise ValueError("no holdout pairs with eval_sample=true")
    records: list[EvalRecord] = []
    for index, pair in enumerate(sample, start=1):
        history, incoming, previous = split_pair(pair)
        request = GenerationRequest(
            partner_id=pair.chat_id,
            chat_id=pair.chat_id,
            text=incoming,
            previous_partner_text=previous,
            history=history,
            ts_before=pair.ts,
            exclude_pair_ids=[pair.pair_id],
        )
        result = backend.generate(request)
        retrieved = retriever.last if retriever else []
        assert_no_leakage(pair, retrieved, holdout_ids)
        verdict = judge.judge(pair.context, pair.reply, result.text)
        records.append(
            EvalRecord(
                pair_id=pair.pair_id,
                chat_id=pair.chat_id,
                ts=pair.ts,
                period=pair.period,
                context=[t.model_dump() for t in pair.context],
                incoming=incoming,
                reference=pair.reply,
                output=result.text,
                raw_texts=result.raw_texts,
                rejected=result.rejected,
                error=result.error,
                fallback_from=result.fallback_from,
                mode=result.mode,
                model=result.model,
                prompt_version=result.prompt_version,
                params=result.params,
                latency_ms=result.latency_ms,
                retrieved=[
                    RetrievedRecord(
                        pair_id=e.pair_id,
                        ts=e.ts,
                        last_partner_text=e.last_partner_text,
                        reply=e.reply,
                        distance=e.distance,
                    )
                    for e in retrieved
                ],
                judge=verdict,
            )
        )
        if progress:
            progress(index, len(sample))
    finished = datetime.now(tz=UTC)
    modes = {r.mode for r in records}
    models = {r.model for r in records}
    prompt_versions = {r.prompt_version for r in records}
    metadata = RunMetadata(
        run_id=f"{started.strftime('%Y%m%dT%H%M%SZ')}_{backend.mode}",
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        git_commit=git_commit(),
        dataset_version=dataset_version,
        messages_dataset_version=messages_dataset_version,
        mode=backend.mode if len(modes) == 1 else "+".join(sorted(modes)),
        model="+".join(sorted(models)),
        prompt_version="+".join(sorted(prompt_versions)),
        style_profile_sha256=style_profile_sha256,
        judge_model=judge.llm.model,
        judge_prompt_version=judge.prompt_version,
        eval_config=config.model_dump(),
        eval_config_version=config.version,
        holdout_ids_sha256=holdout_identity(sample),
        n=len(records),
    )
    return EvalRun(metadata=metadata, summary=summarize(records), records=records)


def run_filename(run: EvalRun) -> str:
    model_slug = "".join(ch if ch.isalnum() else "-" for ch in run.metadata.model)[:40]
    return f"{run.metadata.run_id}_{model_slug}.json"


def write_run(run: EvalRun, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / run_filename(run)
    path.write_text(run.model_dump_json(indent=1) + "\n", encoding="utf-8")
    log.info("eval.run_written", path=str(path), n=run.summary.n, overall=run.summary.overall)
    return path


def read_run(path: Path) -> EvalRun:
    return EvalRun.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs(directory: Path) -> list[Path]:
    return (
        sorted(p for p in directory.glob("*.json") if p.name != "compare.json")
        if directory.is_dir()
        else []
    )
