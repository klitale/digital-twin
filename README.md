# digital-twin

A Telegram bot that answers *as one person would*: connected to a Telegram account
through Telegram Business → Chatbots, it replies to a short allowlist of people (at most
three) in that person's writing style, learned from a Telegram Desktop export.

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
owner matches `BUSINESS_OWNER_ID`, only in private chats with the
`ALLOWED_USER_IDS` (one to three of them), only when enabled and not paused, and only when the generated reply
passes validation (no empty, over-long or assistant-sounding text). Restrict the
Telegram-side *Selected chats* to the same users as a second line of defence.
Control commands work only in the direct chat with the bot and only from
`ADMIN_USER_IDS`; they are listed in Telegram's command menu (`/help`):

| Command | Effect |
|---|---|
| `/status` | connection, mode, dry-run, pauses, initiative switches and today's opener plan |
| `/on`, `/off` | global switch |
| `/mode rag\|finetuned\|hybrid` | generation mode from the next message on |
| `/dryrun on\|off\|auto` | generate but never send (`auto` = `DRY_RUN` from `.env`) |
| `/pause <user_id> <minutes>` | pause one chat (0 = unpause) |
| `/reset <user_id>` | forget that partner's recent turns |
| `/followup on\|off` | after the twin's reply and 20-90 min of silence, one nudge with p=0.5 |
| `/opener on\|off` | after 24 h of silence, on a quarter of days one first message at a random minute between 10:00 and 14:00 |
| `/aggro low\|normal\|high` | how hard the twin pushes: how often it ignores a message, how often initiative fires, how many messages one reply is split into |
| `/poke <user_id> [followup\|opener]` | send an initiative now (still behind every gate) |

A message written by the account owner in a connected chat pauses the bot there for
`PAUSE_MINUTES` (if Telegram delivers such messages; otherwise `/pause`).

`/status` lists one line per switch in the same order and wording as the commands, so
`/dryrun on|off|auto` reads back as `on`/`off` with its source rather than as a bare
boolean. `/aggro` (`src/twin/bot/aggression.py`) scales volume only, never wording: at
`low` the twin ignores more messages, nudges and opens far less often and never sends
more than two messages in a row; at `high` it almost never ignores a message and its
initiative probabilities double. `normal` reproduces the configured defaults exactly.

**Initiative** (`src/twin/bot/initiative.py`) is off by default. Because the opener fires
on only a quarter of days, `/poke <user_id>` is the way to see one on demand, and
`deploy/state.sh <host>` prints the switches, pauses and today's plan. Telegram allows a
first message only in chats the business account already has a dialog with; elsewhere it
answers `BUSINESS_PEER_USAGE_MISSING`, and the bot marks that peer, stops trying and
clears the mark when the person writes. Replies are unaffected. The defaults come from
the export, where the twin started about a quarter of all conversations, nearly all of
them late morning. Decisions are pure functions of the persisted state, a clock and a
seeded rng (one decision per reply, one opener plan per local day, never twice), the
scheduler is a one-minute loop next to polling, and every initiative passes the same
gates as a reply: verified connection, allowlist, enabled, not paused, validation,
dry-run. Openers and follow-ups always use the gateway model with `prompts/initiative_v1.md`
(the fine-tuned 7B only knows how to answer, not how to follow a task). Tunables:
`INITIATIVE_TZ`, `OPENER_HOURS`, `OPENER_DAILY_PROBABILITY`, `OPENER_SILENCE_HOURS`,
`FOLLOWUP_MINUTES`, `FOLLOWUP_PROBABILITY`.

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
