"""Self-contained HTML report over evaluation runs (no CDN, inline CSS/JS).

Built after `modern-web-guidance`: ``<details>`` for expandable examples (searchable
by "find in page"), ``color-scheme: light dark`` with ``light-dark()`` tokens, tables in
scrollable wrappers, ``content-visibility`` on the long example lists.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from twin.eval.compare import Comparison, compare_runs
from twin.eval.schemas import CRITERIA, EvalRecord, EvalRun

WORST = 10
MAX_RETRIEVED_SHOWN = 5
CRITERION_LEGEND = {
    "style_similarity": "стиль — длина, лексика, пунктуация, эмодзи как у эталона",
    "appropriateness": "уместность — ответ по месту в разговоре",
    "not_assistant_like": "не ассистент — без вежливых формул и объяснений",
    "consistency": "непротиворечивость — не спорит с фактами из контекста и эталона",
}
CRITERION_LABELS = {
    "style_similarity": "стиль",
    "appropriateness": "уместность",
    "not_assistant_like": "не ассистент",
    "consistency": "непротиворечивость",
}

CSS = """
:root { color-scheme: light dark;
  --bg: light-dark(#fbfaf7, #14161a); --fg: light-dark(#1e2126, #e6e4de);
  --muted: light-dark(#5f6672, #9aa2ad); --line: light-dark(#dcd8ce, #2c3138);
  --card: light-dark(#ffffff, #1b1e24); --accent: light-dark(#2f6fde, #7fa8ff);
  --bar: light-dark(#a9c1ef, #5b7fcf); --good: light-dark(#2e8b57, #6fcf97);
  --bad: light-dark(#c0392b, #ff8a80); }
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem clamp(1rem, 4vw, 3rem); background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 72rem; margin-inline: auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; } h2 { font-size: 1.2rem; margin: 2rem 0 .75rem; }
h3 { font-size: 1rem; margin: 1.25rem 0 .5rem; color: var(--muted); font-weight: 600; }
.meta { color: var(--muted); font-size: .9rem; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: .5rem; background: var(--card); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: .45rem .6rem; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; font-size: .92rem; }
th { color: var(--muted); font-weight: 600; font-size: .8rem; white-space: normal; vertical-align: bottom; }
td.wrap { white-space: normal; min-width: 11rem; }
td small { color: var(--muted); display: block; font-size: .8rem; }
.scroll { scrollbar-width: thin; }
.legend { columns: 2; column-gap: 2rem; font-size: .9rem; color: var(--muted); margin: .25rem 0 0; padding-left: 1.1rem; }
tr:last-child td { border-bottom: 0; }
.delta-pos { color: var(--good); } .delta-neg { color: var(--bad); }
.dist { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: .75rem; }
.dist figure { margin: 0; padding: .75rem; border: 1px solid var(--line); border-radius: .5rem; background: var(--card); }
.dist figcaption { font-size: .85rem; color: var(--muted); margin-bottom: .4rem; }
.bar { display: grid; grid-template-columns: 1.2rem 1fr 2.5rem; align-items: center; gap: .4rem; font-size: .85rem; }
.bar i { display: block; height: .7rem; background: var(--bar); border-radius: .2rem; }
.bar i.zero { background: transparent; }
.examples { content-visibility: auto; contain-intrinsic-size: auto 40rem; }
details { border: 1px solid var(--line); border-radius: .5rem; background: var(--card); margin: .5rem 0; }
summary { cursor: pointer; padding: .6rem .8rem; }
summary .score { font-weight: 700; }
summary .id { color: var(--muted); font-size: .85rem; margin-left: .5rem; }
details > div { padding: 0 .8rem .8rem; }
pre { white-space: pre-wrap; word-break: break-word; margin: .25rem 0 .75rem; padding: .6rem .7rem;
  border-radius: .4rem; background: var(--bg); border: 1px solid var(--line); font: .9rem/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
.caveats li { margin: .2rem 0; }
"""


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def run_label(run: EvalRun) -> str:
    """Human label: mode · model tail · start time; the run id stays in the metadata."""
    model = run.metadata.model.split("/")[-1]
    started = run.metadata.started_at[:16].replace("T", " ")
    return f"{run.metadata.mode} · {model} · {started}"


def _delta_cell(value: object) -> str:
    if value is None:
        return "<td></td>"
    cls = "delta-pos" if float(value) > 0 else "delta-neg" if float(value) < 0 else ""
    return f'<td class="{cls}">{float(value):+.2f}</td>'


def summary_table(comparison: Comparison) -> str:
    head = [
        "прогон",
        "n",
        "молчал",
        "overall (Δ к baseline)",
        *[CRITERION_LABELS[c] for c in CRITERIA],
        "latency p50",
    ]
    rows = []
    for row in comparison.rows:
        delta = row.get("delta_overall")
        delta_html = ""
        if delta is not None:
            cls = "delta-pos" if float(delta) > 0 else "delta-neg" if float(delta) < 0 else ""
            delta_html = f' <span class="{cls}">{float(delta):+.2f}</span>'
        cells = [
            f'<td class="wrap"><strong>{_esc(row["mode"])}</strong> · {_esc(row["model"])}'
            f"<small>{_esc(row['prompt'])} · {_esc(row['run'])}</small></td>",
            f"<td>{_esc(row['n'])}</td>",
            f"<td>{_esc(row['silent'])}</td>",
            f"<td><strong>{_fmt(row['overall'])}</strong>{delta_html}</td>",
            *[f"<td>{_fmt(row.get(c))}</td>" for c in CRITERIA],
            f"<td>{_esc(row['latency_p50_ms'])} ms</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{_esc(h)}</th>" for h in head)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def distributions(runs: Sequence[EvalRun]) -> str:
    figures = []
    for run in runs:
        judged = [r for r in run.records if r.judge and not r.judge.error]
        for criterion in CRITERIA:
            counts = Counter(r.judge.scores[criterion] for r in judged)  # type: ignore[union-attr]
            total = max(1, sum(counts.values()))
            bars = "".join(
                f'<div class="bar"><span>{score}</span>'
                f'<i class="{"zero" if not counts.get(score) else ""}" style="width:{100 * counts.get(score, 0) / total:.0f}%"></i>'
                f"<span>{counts.get(score, 0)}</span></div>"
                for score in (5, 4, 3, 2, 1)
            )
            figures.append(
                f"<figure><figcaption>{_esc(run.metadata.mode)} · {_esc(CRITERION_LABELS[criterion])} · n={len(judged)}</figcaption>{bars}</figure>"
            )
    return f'<div class="dist">{"".join(figures)}</div>'


def per_period_table(comparison: Comparison, runs: Sequence[EvalRun]) -> str:
    periods = sorted({p for table in comparison.per_period.values() for p in table})
    head = "".join(f"<th>{_esc(p)}</th>" for p in periods)
    rows = []
    labels = {run.metadata.run_id: run_label(run) for run in runs}
    for label, table in comparison.per_period.items():
        cells = []
        for period in periods:
            entry = table.get(period)
            if entry:
                mean = sum(entry[c] for c in CRITERIA) / len(CRITERIA)
                cells.append(f"<td>{mean:.2f} <span class='meta'>n={int(entry['n'])}</span></td>")
            else:
                cells.append("<td></td>")
        rows.append(
            f'<tr><td class="wrap">{_esc(labels.get(label, label))}</td>{"".join(cells)}</tr>'
        )
    return f'<div class="scroll"><table><thead><tr><th>прогон</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _turns(record: EvalRecord, name: str) -> str:
    return "\n".join(
        f"{name if t.get('is_me') else 'Собеседник'}: {t.get('text', '')}" for t in record.context
    )


def example_block(record: EvalRecord, name: str) -> str:
    judge = record.judge
    mean = judge.mean if judge else None
    reasons = (
        "\n".join(
            f"{CRITERION_LABELS[c]}: {judge.scores.get(c, '-')} — {judge.reasons.get(c, '')}"
            for c in CRITERIA
        )
        if judge and judge.scores
        else (judge.error if judge else "не оценено")
    )
    shown = record.retrieved[:MAX_RETRIEVED_SHOWN]
    retrieved = "\n\n".join(f"Собеседник: {r.last_partner_text}\n{name}: {r.reply}" for r in shown)
    retrieved = retrieved or "(нет)"
    if len(record.retrieved) > len(shown):
        retrieved += f"\n\n… ещё {len(record.retrieved) - len(shown)}"
    output = record.output if record.output is not None else "(молчание)"
    extra = []
    if record.fallback_from:
        extra.append(f"fallback from {record.fallback_from}")
    if record.rejected:
        extra.append("rejected: " + ", ".join(record.rejected))
    summary = (
        f'<span class="score">{_fmt(mean)}</span> из 5'
        f'<span class="id">{_esc(record.period)} · {_esc(record.pair_id)} · {_esc(record.latency_ms)} ms'
        f"{' · ' + _esc('; '.join(extra)) if extra else ''}</span>"
    )
    return (
        f"<details><summary>{summary}</summary><div>"
        f"<h3>Контекст</h3><pre>{_esc(_turns(record, name))}</pre>"
        f"<h3>Эталон</h3><pre>{_esc(record.reference)}</pre>"
        f"<h3>Ответ ({_esc(record.mode)}, {_esc(record.model)}, {_esc(record.prompt_version)})</h3><pre>{_esc(output)}</pre>"
        f"<h3>Найденные примеры</h3><pre>{_esc(retrieved)}</pre>"
        f"<h3>Оценка judge</h3><pre>{_esc(reasons)}</pre>"
        "</div></details>"
    )


def worst_examples(run: EvalRun, name: str, limit: int = WORST) -> str:
    scored = [r for r in run.records if r.judge and r.judge.mean is not None]
    worst = sorted(scored, key=lambda r: (r.judge.mean, r.pair_id))[:limit]  # type: ignore[union-attr]
    return "".join(example_block(r, name) for r in worst)


def render_report(runs: Sequence[EvalRun], name: str, baseline_mode: str = "rag") -> str:
    comparison = compare_runs(runs, baseline_mode)
    generated = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    meta_lines = [
        f"{_esc(r.metadata.run_id)}: commit {_esc(r.metadata.git_commit)}, dataset {_esc(r.metadata.dataset_version)}, "
        f"judge {_esc(r.metadata.judge_model)} / {_esc(r.metadata.judge_prompt_version)}, "
        f"eval config v{_esc(r.metadata.eval_config_version)}, sample {_esc(r.metadata.holdout_ids_sha256)}"
        for r in runs
    ]
    legend = "".join(f"<li>{_esc(text)}</li>" for text in CRITERION_LEGEND.values())
    worst_sections = "".join(
        f"<h3>{_esc(r.metadata.mode)} · {_esc(r.metadata.model)}</h3>{worst_examples(r, name)}"
        for r in runs
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>digital-twin · оценка режимов</title>
<style>{CSS}</style>
</head>
<body>
<h1>digital-twin · оценка режимов</h1>
<p class="meta">сгенерировано {generated}; baseline {_esc(comparison.baseline)}; шкала 1–5, выше лучше</p>
<h2>Сводка</h2>
{summary_table(comparison)}
<ul class="legend">{legend}</ul>
<h2>Caveats</h2>
<ul class="caveats">{"".join(f"<li>{_esc(c)}</li>" for c in comparison.caveats)}</ul>
<h2>Распределение оценок</h2>
{distributions(runs)}
<h2>По периодам</h2>
{per_period_table(comparison, runs)}
<h2>Худшие {WORST} примеров на режим</h2>
<div class="examples">{worst_sections}</div>
<h2>Метаданные прогонов</h2>
<ul class="meta">{"".join(f"<li>{line}</li>" for line in meta_lines)}</ul>
</body>
</html>
"""
