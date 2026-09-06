---
name: twin-ingest
description: Rebuild the digital-twin dataset from a Telegram export and review it. Use whenever the export changed, data filters or the split config changed, the user asks to re-ingest / rebuild pairs / regenerate the style profile or the index, or any dataset number is about to be quoted (run the profiling checks first). Triggers on "ingest", "pairs", "holdout", "split", "style profile", "index", "dataset manifest".
---

# twin-ingest

Turns `data/raw/**/result.json` into `messages.jsonl` → `pairs.jsonl` + `holdout.jsonl` →
style profile → Chroma index, with counts at every step.

## Workflow

1. Preconditions: `.env` has `TWIN_SENDER_ID`, `TWIN_NAME`, `LLM_API_KEY`; the export is under
   `data/raw/` (or `RAW_EXPORT_PATH`). Check `configs/data/default.yaml` (`filters.min_date`,
   `split.*`) before running: changing it changes the dataset version.
2. `uv run twin ingest` — parse, pairs, split, manifest, profiling report. Read the printed
   drop counts and `data/processed/profile_report.md` **Caveats** section before quoting
   any number (section 5.4 discipline).
3. `uv run twin style-profile` only when the persona rules should change; the existing
   file is hand-editable and is never overwritten without `--force`.
4. `uv run twin index --rebuild` after a new `pairs.jsonl` (the index manifest pins the
   dataset version; a stale index is refused by retrieval).
5. Ship: `deploy/sync_index.sh srv-b --restart` (bot), `deploy/sync_index.sh srv-c --processed` (jobs).

## Inputs / outputs

- In: `data/raw/**/result.json`, `configs/data/default.yaml`, `.env`.
- Out (gitignored except the counts-only manifest): `data/processed/{messages,pairs,holdout}.jsonl`,
  `dataset_manifest.json` (committed), `profile_report.md`, `style_profile.md`, `data/chroma/`.

## Failure behaviour

- `TWIN_SENDER_ID` missing or absent from the export → exit 1 with the top senders; nothing written.
- Empty export, malformed JSON, unknown config keys → exit 1 with `error:` and no partial files.
- Leakage assertions (`ingest/split.py`) raise if holdout overlaps train or precedes it.

## Test prompts (executed)

1. "Пересобери датасет с min_date 2022 и покажи, сколько пар осталось" → edit YAML, `twin ingest`, quote counts from the output.
2. "Сколько у клона ответов длиннее 2000 символов?" → `profile_report.md` Caveats / `dataset_manifest.json` `pairs_dropped.reply_too_long`.
3. "Обнови индекс на боте" → `twin index --rebuild` then `deploy/sync_index.sh srv-b --restart`.
