"""Blind pairwise judge: which of two runs writes the more alive reply, still in character.

The absolute judge compares every reply with the one real reply, so it rewards the
safest guess and cannot see that an answer is *interesting*. A duel shows the judge two
candidates for the same moment (plus the real reply, so it knows the person) and asks
which is the better message to a friend while still plausibly his. Every pair is judged
in both orders; a side wins only if it wins both, anything else is a tie, which cancels
the judge's position bias. The two-sided sign test over wins and losses says whether
the result is more than a coin flip.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from twin.core.llm_client import LLMClient, LLMError
from twin.core.prompts import PromptTemplate
from twin.core.schemas import ContextTurn
from twin.eval.harness import list_runs
from twin.eval.judge import SILENCE, format_context
from twin.eval.schemas import EvalRun

DUEL_PROMPT = "duel_v1"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
Side = Literal["a", "b", "tie"]


class DuelVote(BaseModel):
    winner: Literal["1", "2", "tie"] | None
    reason: str = ""
    raw: str = ""
    error: str | None = None


class DuelRecord(BaseModel):
    pair_id: str
    incoming: str
    reference: str
    a: str | None
    b: str | None
    a_first: DuelVote
    b_first: DuelVote
    outcome: Side


class DuelSummary(BaseModel):
    n: int
    a_wins: int
    b_wins: int
    ties: int
    errors: int
    p_value: float


class DuelRun(BaseModel):
    run_a: str
    run_b: str
    prompt_a: str
    prompt_b: str
    judge_model: str
    judge_prompt_version: str
    started_at: str
    summary: DuelSummary
    records: list[DuelRecord]


def parse_duel_json(raw: str) -> tuple[str, str]:
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError("no JSON object in the duel response")
    data = json.loads(match.group(0))
    winner = str(data.get("winner", "")).strip().lower()
    if winner not in ("1", "2", "tie"):
        raise ValueError(f"winner {winner!r} is not 1, 2 or tie")
    return winner, str(data.get("reason", ""))


class DuelJudge:
    def __init__(
        self, llm: LLMClient, template: PromptTemplate, name: str, max_tokens: int = 200
    ) -> None:
        self.llm = llm
        self.template = template
        self.name = name
        self.max_tokens = max_tokens

    @property
    def prompt_version(self) -> str:
        return self.template.version

    def vote(
        self,
        context: list[ContextTurn],
        reference: str,
        first: str | None,
        second: str | None,
    ) -> DuelVote:
        system, user = self.template.render(
            name=self.name,
            context=format_context(context, self.name),
            reference=reference,
            first=first if first is not None else SILENCE,
            second=second if second is not None else SILENCE,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            result = self.llm.chat(messages, temperature=0.0, max_tokens=self.max_tokens)
        except LLMError as exc:
            return DuelVote(winner=None, error=str(exc))
        try:
            winner, reason = parse_duel_json(result.text)
        except (ValueError, json.JSONDecodeError) as exc:
            return DuelVote(winner=None, raw=result.text, error=f"unparsable: {exc}")
        return DuelVote(winner=winner, reason=reason, raw=result.text)  # type: ignore[arg-type]


def combine(a_first: DuelVote, b_first: DuelVote) -> Side:
    """A side wins only when it wins in both orders; anything else is a tie."""
    a_says = {"1": "a", "2": "b"}.get(a_first.winner or "", "tie")
    b_says = {"1": "b", "2": "a"}.get(b_first.winner or "", "tie")
    return a_says if a_says == b_says else "tie"  # type: ignore[return-value]


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test; ties are left out, as the test requires."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def run_duel(
    run_a: EvalRun,
    run_b: EvalRun,
    judge: DuelJudge,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> DuelRun:
    if run_a.metadata.holdout_ids_sha256 != run_b.metadata.holdout_ids_sha256:
        raise ValueError("the runs were evaluated on different holdout samples")
    by_b = {record.pair_id: record for record in run_b.records}
    common = [record for record in run_a.records if record.pair_id in by_b]
    if limit is not None:
        common = common[:limit]
    if not common:
        raise ValueError("the runs share no pairs")
    records: list[DuelRecord] = []
    for index, ra in enumerate(common, start=1):
        rb = by_b[ra.pair_id]
        context = [ContextTurn.model_validate(turn) for turn in ra.context]
        a_first = judge.vote(context, ra.reference, ra.output, rb.output)
        b_first = judge.vote(context, ra.reference, rb.output, ra.output)
        records.append(
            DuelRecord(
                pair_id=ra.pair_id,
                incoming=ra.incoming,
                reference=ra.reference,
                a=ra.output,
                b=rb.output,
                a_first=a_first,
                b_first=b_first,
                outcome=combine(a_first, b_first),
            )
        )
        if progress:
            progress(index, len(common))
    a_wins = sum(1 for record in records if record.outcome == "a")
    b_wins = sum(1 for record in records if record.outcome == "b")
    summary = DuelSummary(
        n=len(records),
        a_wins=a_wins,
        b_wins=b_wins,
        ties=len(records) - a_wins - b_wins,
        errors=sum(1 for r in records if r.a_first.error or r.b_first.error),
        p_value=sign_test_p(a_wins, b_wins),
    )
    return DuelRun(
        run_a=run_a.metadata.run_id,
        run_b=run_b.metadata.run_id,
        prompt_a=run_a.metadata.prompt_version,
        prompt_b=run_b.metadata.prompt_version,
        judge_model=judge.llm.model,
        judge_prompt_version=judge.prompt_version,
        started_at=datetime.now(tz=UTC).isoformat(),
        summary=summary,
        records=records,
    )


def write_duel(duel: DuelRun, directory: Path) -> Path:
    """Duels live in a subdirectory so `twin compare` never mistakes them for runs."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{duel.run_a}__vs__{duel.run_b}.json"
    path.write_text(duel.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path


def resolve_run(directory: Path, key: str) -> Path:
    """A run file path, or the unique run in ``directory`` whose name starts with ``key``."""
    path = Path(key)
    if path.is_file():
        return path
    matches = [p for p in list_runs(directory) if p.name.startswith(key)]
    if len(matches) != 1:
        raise ValueError(f"{key!r} matches {len(matches)} run(s) in {directory}")
    return matches[0]
