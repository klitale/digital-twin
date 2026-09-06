---
name: twin-eval
description: Evaluate and compare the digital-twin generation modes (rag, finetuned, hybrid) on the holdout sample with the blind LLM judge, and build the HTML report. Use when the user asks how good a mode is, wants to compare prompts/models/adapters, mentions "eval", "judge", "scores", "report", "compare", or before claiming any improvement (never claim one without a run).
---

# twin-eval

## Workflow

1. Preconditions: `pairs.jsonl`/`holdout.jsonl` + index (see twin-ingest), `JUDGE_MODEL`
   reachable; for `finetuned`/`hybrid` also `FT_BASE_URL`/`FT_API_KEY`/`FT_MODEL` and a deployed
   serving app (`uv run twin smoke-test-model --model <adapter>` first).
2. `uv run twin eval --mode rag` (≈6 min for 80 pairs: generation + judge). Repeat per mode.
   Use `--limit 5` for a smoke check. Every run lands in `data/eval/<ts>_<mode>_<model>.json`
   with per-example records and run metadata (commit, dataset/prompt/eval versions, judge).
3. `uv run twin compare` — table with deltas against the rag baseline plus Caveats
   (different samples, silent replies, fallbacks). Report the caveats with the numbers.
4. `uv run twin report` → `data/eval/report.html` (self-contained). To review visually:
   `uv run --with "playwright==1.49.1" python -m playwright install chromium` then screenshot.

## Reading results

- Scores are 1–5 per criterion: style_similarity, appropriateness, not_assistant_like,
  consistency; `overall` is their mean. Deltas below ~0.1 on 80 pairs are noise.
- `silent` = validation rejected both attempts; `fallbacks` = the Modal endpoint failed and
  rag answered. Both are listed in Caveats; treat them as part of the result.
- Leakage is asserted on every retrieval (target, holdout, future); a `LeakageError` aborts the run.

## Failure behaviour

- No runs → `compare`/`report` exit 1. Judge JSON unparsable → the record keeps `judge.error`
  and is excluded from means (counted in `errors`).
- Endpoint timeouts in `finetuned`/`hybrid` fall back to rag and are flagged; rerun after warming
  the container with `smoke-test-model` if fallbacks are many.

## Test prompts (executed)

1. "Прогони eval для rag и покажи средние" → `twin eval --mode rag`, summary line.
2. "Сравни все прогоны" → `twin compare` (Markdown table + caveats).
3. "Собери HTML-отчёт" → `twin report`, path printed.
