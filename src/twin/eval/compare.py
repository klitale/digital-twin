"""Compare evaluation runs: one table across runs, deltas against a baseline, per-period
breakdown, and the section 5.4 checks (same sample, missing scores, duplicates)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from twin.eval.schemas import CRITERIA, EvalRun


@dataclass
class Comparison:
    rows: list[dict[str, object]]
    baseline: str | None
    per_period: dict[str, dict[str, dict[str, float]]]
    caveats: list[str] = field(default_factory=list)


def _label(run: EvalRun) -> str:
    return run.metadata.run_id


def compare_runs(runs: Sequence[EvalRun], baseline_mode: str = "rag") -> Comparison:
    if not runs:
        raise ValueError("no evaluation runs found")
    ordered = sorted(runs, key=lambda r: r.metadata.started_at)
    baseline = next((r for r in ordered if r.metadata.mode == baseline_mode), ordered[0])
    caveats: list[str] = []
    samples = {r.metadata.holdout_ids_sha256 for r in ordered}
    if len(samples) > 1:
        caveats.append(
            "runs were evaluated on different holdout samples; deltas are not comparable"
        )
    sizes = {r.summary.n for r in ordered}
    if len(sizes) > 1:
        caveats.append(f"runs have different sizes: {sorted(sizes)}")
    judges = {(r.metadata.judge_model, r.metadata.judge_prompt_version) for r in ordered}
    if len(judges) > 1:
        caveats.append("runs used different judge models or rubrics")
    rows: list[dict[str, object]] = []
    for run in ordered:
        s = run.summary
        ids = [r.pair_id for r in run.records]
        if len(ids) != len(set(ids)):
            caveats.append(f"{_label(run)}: duplicate pair ids in records")
        if s.judged < s.n:
            caveats.append(f"{_label(run)}: {s.n - s.judged} record(s) without judge scores")
        if s.silent:
            caveats.append(f"{_label(run)}: {s.silent} silent reply/replies (scored as silence)")
        if s.fallbacks:
            caveats.append(f"{_label(run)}: {s.fallbacks} reply/replies came from the rag fallback")
        row: dict[str, object] = {
            "run": _label(run),
            "mode": run.metadata.mode,
            "model": run.metadata.model,
            "prompt": run.metadata.prompt_version,
            "n": s.n,
            "silent": s.silent,
            "overall": s.overall,
            **{c: s.means.get(c) for c in CRITERIA},
            "latency_p50_ms": s.latency_ms_p50,
        }
        if run is not baseline and baseline.summary.overall is not None and s.overall is not None:
            row["delta_overall"] = round(s.overall - baseline.summary.overall, 3)
            for c in CRITERIA:
                row[f"delta_{c}"] = round(s.means[c] - baseline.summary.means[c], 3)
        rows.append(row)
    per_period = {_label(run): run.summary.per_period for run in ordered}
    if not caveats:
        caveats.append("none.")
    return Comparison(rows=rows, baseline=_label(baseline), per_period=per_period, caveats=caveats)


def render_markdown(comparison: Comparison) -> str:
    columns = [
        "run",
        "mode",
        "model",
        "prompt",
        "n",
        "silent",
        "overall",
        *CRITERIA,
        "latency_p50_ms",
        "delta_overall",
    ]
    lines = [
        "# Evaluation comparison",
        "",
        f"baseline: `{comparison.baseline}`",
        "",
        "## Caveats",
        "",
        *[f"- {c}" for c in comparison.caveats],
        "",
        "## Runs",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in comparison.rows:
        lines.append(
            "| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in columns) + " |"
        )
    lines += ["", "## Per period (mean of the four criteria)", ""]
    periods = sorted({p for table in comparison.per_period.values() for p in table})
    lines.append("| run | " + " | ".join(periods) + " |")
    lines.append("|---|" + "---|" * len(periods))
    for label, table in comparison.per_period.items():
        cells = []
        for period in periods:
            entry = table.get(period)
            if entry:
                cells.append(
                    f"{sum(entry[c] for c in CRITERIA) / len(CRITERIA):.2f} (n={int(entry['n'])})"
                )
            else:
                cells.append("")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
