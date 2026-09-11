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
| `rag`, prompt `rag_v2` (in production) | **3.38** | 2.54 | 3.25 | 4.55 | 3.16 | 1.48 s |
| `rag`, prompt `rag_v1` | 3.22 | 2.50 | 3.00 | 4.60 | 2.79 | 1.25 s |
| `finetuned` (Qwen 2.5 7B + LoRA, 2 epochs, loss 3.3 -> 2.44) | 2.83 | 2.05 | 2.51 | 4.34 | 2.43 | 1.06 s |
| `hybrid` (the adapter + top-8 examples) | 2.73 | 1.95 | 2.43 | 4.19 | 2.38 | 1.12 s |

Scores are 1-5 per criterion; run-to-run noise on this sample is about ±0.1, so the gaps
are real. `rag_v2` replaced a length rule that said "a few words, rarely more" with the
measured distribution and a line-break rule; it gained 0.15 overall, almost all of it in
consistency (+0.38) and appropriateness (+0.25). It did *not* fix the length itself: the
twin still never writes a long message (see below). Length was the one thing `rag_v2` failed to move, and the fix turned out to live in the
style profile rather than in the prompt. Instructions lose to the anchors: eight
retrieved examples with a median reply of 33 characters, and the profile's own "a typical
reply is 1-6 words". Restating the measured tail in the profile itself (one reply in ten
over 90 characters, one in thirty over 150, and what he is doing when he writes long)
moved the distribution without moving the score:

| p50 | p75 | p90 | max | over 100 chars | messages per reply |
|---|---|---|---|---|---|
| 29 / 31 / **40** / *36* | 42 / 46 / **56** / *60* | 51 / 57 / **66** / *90* | 77 / 77 / **111** / *237* | 0% / 0% / **1.2%** / *6.2%* | 2.20 / 2.30 / **2.04** / *1.57* |

Reading: `rag_v1` / `rag_v2` / **`rag_v2` + length-aware profile** / *the real replies*.
The judge scores those three 3.22, 3.38 and 3.34, so the last step is free within noise
while the shape of the output gets closer to the person. The remaining gap is the far
tail: he still writes the occasional several-hundred-character message and the twin does
not. Because the profile is gitignored personal data, every run now records its
`style_profile_sha256` and `twin compare` says when two runs share a prompt version but
not a profile.

What the records show: the adapter reproduces the twin's *form* well (median
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
passes validation (no empty, over-long, list-shaped or assistant-sounding text). Restrict the
Telegram-side *Selected chats* to the same users as a second line of defence.
Control commands work only in the direct chat with the bot and only from
`ADMIN_USER_IDS`; they are listed in Telegram's command menu (`/help`):

| Command | Effect |
|---|---|
| `/status` | connection, mode, dry-run, pauses, initiative switches, today's opener plan and guard counters |
| `/on`, `/off` | global switch |
| `/mode rag\|finetuned\|hybrid` | generation mode from the next message on |
| `/dryrun on\|off\|auto` | generate but never send (`auto` = `DRY_RUN` from `.env`) |
| `/pause <user_id> <minutes>` | pause one chat (0 = unpause) |
| `/reset <user_id>` | forget that partner's recent turns |
| `/followup on\|off` | after the twin's reply and 20-90 min of silence, one nudge with p=0.5 |
| `/opener on\|off` | after 24 h of silence, on a quarter of days one first message at a random minute between 10:00 and 14:00 |
| `/aggro low\|normal\|high` | how hard the twin pushes: how often it ignores a message, how often initiative fires, how many messages one reply is split into |
| `/learn on\|off` | keep a fact sheet per partner, distilled from the live conversation |
| `/facts <user_id>` | show what the twin has learned about that person |
| `/forget <user_id>` | wipe that fact sheet |
| `/poke <user_id> [followup\|opener]` | send an initiative now (still behind every gate) |

A message written by the account owner in a connected chat pauses the bot there for
`PAUSE_MINUTES` (if Telegram delivers such messages; otherwise `/pause`).

`/status` lists one line per switch in the same order and wording as the commands, so
`/dryrun on|off|auto` reads back as `on`/`off` with its source rather than as a bare
boolean. `/aggro` (`src/twin/bot/aggression.py`) scales volume only, never wording: at
`low` the twin ignores more messages, nudges and opens far less often and never sends
more than two messages in a row; at `high` it almost never ignores a message and its
initiative probabilities double. `normal` reproduces the configured defaults exactly.

**Provocations and spend** (`src/twin/core/guard.py`). "Write me 1000 cities", "list a
hundred facts about yourself", "ignore your instructions", "write my essay": a person
shrugs these off in one line, a model writes a two-thousand-token list and the gateway
bills for it. Four layers keep that cheap:

1. *Every* reply has an output cap (`max_tokens`, about half of `MAX_REPLY_CHARS`), and a
   completion cut off by it is dropped without a second attempt, so a runaway list costs
   one capped call instead of two full ones. A reply with three or more numbered or
   bulleted lines is rejected as list-shaped.
2. Cheap patterns (no model call) recognise a bulk request, a do-my-work request and a
   prompt injection. Such a message is answered in character with `prompts/guard_v1.md`
   added to the prompt and a 100-token budget; it stays in the chat history only as a
   200-character stub and is never read by learning.
3. From the third provocation in an hour (`GUARD_PROVOCATIONS_PER_HOUR`) the twin goes
   quiet in that chat without calling the model at all.
4. Every chat has a reply budget (`GUARD_REPLIES_PER_HOUR`, `GUARD_REPLIES_PER_DAY`), and
   incoming messages are clipped to `MAX_INCOMING_CHARS` before the prompt, the memory
   and the embedding query. A wall of text (usually a forwarded post) is clipped and gets
   a short reaction, but never counts as a provocation.

Precision matters more than recall: a false positive turns a normal message into a
brush-off, while a miss is still bounded by layer 1. Against every partner message in
the export's pairs (7343 unique turns) none of the three patterns fires (a first draft
caught a forwarded advert that merely mentioned a "system prompt", so the injection
patterns now need a request aimed at the twin); 16 turns are walls, all forwarded posts.
The list rule rejects one of 7330 real replies. The reply budget of 60 an hour and 300 a
day cuts 4% of the export's active hours and 1% of its days. The normal prompt is
untouched, so the holdout numbers above still hold.

**Learning** (`src/twin/core/facts.py`, off until `/learn on`) is what makes the twin
know anything that happened after the export. The retrieval index is frozen and the
per-partner memory holds ten turns, so without it the twin has style but no news. Every
`FACTS_INTERVAL_HOURS`, and only after `FACTS_MIN_NEW_TURNS` new turns have arrived, the
gateway model rewrites a short list of durable facts for that chat (`prompts/facts_v1.md`)
and `rag_v3` puts it in the prompt, below the style profile and above the rules. A sheet
earns that section only when it has something in it: `rag_v3` with an empty sheet scored
3.26 against 3.34 for the same run without the section, so an empty sheet falls back to
`rag_v2` and the twin pays nothing for a feature it is not using. What a *full* sheet is
worth cannot be measured on the holdout at all, because replaying 2026 messages produces
no live conversation to learn from; that is an honest gap, not a claim.

One rule keeps it from eating its own tail: **only human-written turns are ever read**.
`MemoryTurn.by_bot` marks everything the bot generated and `human_turns()` drops it, so a
detail the twin invented can never come back as something it "knows". What does count is
the partner's messages and the messages the account owner writes by hand (those already
pause the bot in that chat; now they are also kept as real material). Sheets are capped
at `FACTS_MAX` facts of 160 characters, live in `data/state/facts/` and are gitignored.

Note the identity question before switching it on: the account the bot is connected to is
not the person being imitated, so what the owner types by hand is that owner's writing,
not the twin's. It is used as *facts about the situation*, never as a style example, and
nothing is ever added to the retrieval index automatically.

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
