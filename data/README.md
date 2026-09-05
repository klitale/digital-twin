# data/

Everything under `data/` is gitignored except this file and
`data/processed/dataset_manifest.json` (counts only, no names or ids).

| Directory | Contents | Produced by |
|---|---|---|
| `raw/` | `result.json` — Telegram Desktop JSON export (single chat or full export) | you |
| `private/` | `blocklist.txt` — strings that must never be committed | you |
| `processed/` | `messages.jsonl`, `pairs.jsonl`, `holdout.jsonl`, `dataset_manifest.json`, `style_profile.md`, profiling report | `twin ingest`, `twin analyze-data`, `twin style-profile` |
| `chroma/` | persistent ChromaDB index + index manifest | `twin index` |
| `train/` | `train.jsonl` in chat format, token stats | `twin train` (prepare step) |
| `eval/` | `<timestamp>_<mode>_<model>.json` runs, `report.html` | `twin eval`, `twin compare`, `twin report` |
| `state/` | `connection.json`, per-partner memory, bot-sent message ids, pause state | the bot at runtime |

To get the export: Telegram Desktop → Settings → Advanced → Export Telegram data →
format **JSON**, only text messages needed (media can be excluded). Place the
resulting `result.json` in `data/raw/`.
