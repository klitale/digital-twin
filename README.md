# digital-twin

A Telegram bot that answers *as one person would*: connected to a Telegram account
through Telegram Business → Chatbots, it replies to exactly two allowlisted people in
that person's writing style, learned from a Telegram Desktop export.

Three generation modes sit behind one interface and are compared on the same holdout
with the same blind LLM judge:

| Mode | How replies are produced |
|---|---|
| `rag` | base chat model via an OpenAI-compatible gateway + style profile + retrieved examples of real replies |
| `finetuned` | Qwen 2.5 7B Instruct + LoRA (Unsloth), served by vLLM on Modal; style profile only |
| `hybrid` | the fine-tuned model + retrieved examples, short prompt (`docs/hybrid.md`) |

Educational portfolio project. Nothing personal is in this repository (see *Privacy*).

## Architecture

```
Telegram export ─► messages.jsonl ─► pairs.jsonl + holdout.jsonl ─► style profile
                                          │                             │
                                          ├─► Chroma index (train only) ┤
                                          └─► train.jsonl ─► LoRA on Modal ─► vLLM on Modal
Telegram Business update ─► fail-closed gates ─► retrieval ─► prompt ─► GenerationBackend
        ─► validation (regenerate once, else silence) ─► humanized delivery
Evaluation: holdout pair ─► backend (no future, no target, no holdout in retrieval) ─► judge JSON ─► compare / report
```

Key modules: `src/twin/ingest` (parsing, pairs, split, profiling, style profile, index),
`src/twin/core` (embeddings, vector store, retriever, prompts, backends, validation,
memory), `src/twin/bot` (aiogram Business handlers, autopause, control commands),
`src/twin/eval` (harness, judge, compare, HTML report), `training/` (LoRA pipeline and
the Modal job), `serving/` (vLLM on Modal), `deploy/` (three-VPS runbook).

## Requirements

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/); a Telegram bot with Business Mode
  (BotFather) and an account with Business features to connect it to
- An OpenAI-compatible gateway for generation, judge and embeddings (the project was built
  against Timeweb Cloud AI Gateway); a Modal account for training and serving
- Your own Telegram Desktop JSON export (`result.json`)

## Run it with your own export

```bash
git clone https://github.com/klitale/digital-twin && cd digital-twin
uv sync && uv run pre-commit install
cp .env.example .env               # tokens, ids, model ids; see the comments inside
# put the export under data/raw/ (see data/README.md), then:
uv run twin ingest                 # messages -> pairs + time-based holdout, profiling report
uv run twin style-profile          # 20-30 Russian style rules (hand-editable)
uv run twin index                  # Chroma index of the training pairs
uv run twin chat                   # talk to the twin in the terminal (rag mode)
uv run twin run --dry-run          # the bot: replies are logged, nothing is sent
uv run twin eval --mode rag && uv run twin compare && uv run twin report
```

Fine-tuning (Modal GPU, ~1 h): `uv run twin train --remote --detach`, then
`uv run modal deploy serving/modal_app.py` and `uv run twin smoke-test-model --model twin`.
Details: `training/README.md`, `serving/README.md`, `deploy/README.md`.

Dataset filters and the split live in `configs/data/default.yaml`; training in
`configs/train/*.yaml`; evaluation in `configs/eval/default.yaml`. Every dataset,
prompt template, train config and eval run carries a version, and every run records
the git commit, so any number in `data/eval/report.html` is reproducible.

## Results (one run per mode, 80 holdout pairs, blind judge `gpt-5.4-mini`)

| mode | overall | style | appropriateness | not assistant-like | consistency | p50 latency |
|---|---|---|---|---|---|---|
| `rag` (Qwen 3.5 Flash via gateway) | **3.22** | 2.50 | 3.00 | 4.60 | 2.79 | 1.25 s |
| `finetuned` (Qwen 2.5 7B + LoRA, 2 epochs, loss 3.3 -> 2.44) | 2.83 | 2.05 | 2.51 | 4.34 | 2.43 | 1.06 s |
| `hybrid` (the adapter + top-8 examples) | 2.73 | 1.95 | 2.43 | 4.19 | 2.38 | 1.12 s |

Scores are 1-5 per criterion; run-to-run noise on this sample is about ±0.1, so the gaps
are real. What the records show: the adapter reproduces the twin's *form* well (median
reply length 36 chars against 37 in the references, short lines, no assistant tone),
but a 7B model loses to the much larger gateway model on picking the right thing to say.
Hybrid is the worst of the three: with only a short prompt the 7B model tends to
paraphrase the retrieved examples into longer, rambling replies (median 41 chars).
The judge compares against a single reference reply, which caps `style` for every mode:
most replies in this chat are one-line reactions that no model can predict exactly.
`data/eval/report.html` has every reply next to its reference and the judge's reasons.

## Telegram safety

The bot fails closed: it replies only through a verified business connection whose
owner matches `BUSINESS_OWNER_ID`, only in private chats with the two
`ALLOWED_USER_IDS`, only when enabled and not paused, and only when the generated reply
passes validation (no empty, over-long or assistant-sounding text). Restrict the
Telegram-side *Selected chats* to the same two users as a second line of defence.
Control commands (`/twin on|off|status|mode|reset|pause|dryrun`) work only in the direct
chat with the bot and only from `ADMIN_USER_IDS`. A message written by the account owner
in a connected chat pauses the bot there for `PAUSE_MINUTES` (if Telegram delivers such
messages; otherwise `/twin pause`).

## Privacy

The export, derived datasets, the style profile, evaluation outputs, the index and the
runtime state live under `data/` and are gitignored. A pre-commit guard
(`scripts/privacy_check.py`) plus `gitleaks` refuse commits containing Telegram ids,
phone numbers, API keys or any string from a private blocklist; tests and docs use
synthetic fixtures only. The only committed data artefact is the counts-only
`data/processed/dataset_manifest.json`.

## Project skills

`.claude/skills/twin-ingest`, `twin-eval`, `twin-train` describe the recurring workflows
for Claude Code (rebuild the dataset, evaluate and compare, train and serve).

## License

MIT
