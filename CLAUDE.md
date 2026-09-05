# digital-twin — CLAUDE.md

Telegram Business bot that replies from a connected account in the writing style of one
person from a Telegram export. Three generation modes behind one interface (RAG,
fine-tuned LoRA on Modal, hybrid) and one evaluation harness with a blind LLM judge.
Educational portfolio project; the repository is public and must contain no personal data.
## Process (non-negotiable)

- Work one phase at a time (table below). Before a phase: state goal, files, acceptance.
  After: `uv run pytest`, `uv run ruff check .`, phase checks, commit, report with
  caveats, then stop and wait for approval. Never start the next phase silently.
- Never hide failing tests, data anomalies or unsupported assumptions.
- Decisions on architecture, data validity, eval correctness or Telegram safety go into
  the phase report, never into a silent default.
- `/systematic-debugging` before changing code on any failure; `/analyze` checklist before
  dataset or eval numbers; `/modern-web-guidance` before HTML, `/design-critique` after.
- Push only on explicit command; before the first push: `gitleaks detect` over the full
  history and a blocklist grep over all commits.

| # | Phase | Status |
|---|---|---|
| 0 | Bootstrap (this skeleton) | done |
| 1 | Export parsing → `messages.jsonl` | done |
| 2 | Pairs, time-based split, profiling report | done |
| 3 | Style profile | done |
| 4 | Index, retrieval, prompt, RAG backend, `twin chat` | done |
| 5 | Business bot + VPS inventory/deploy | done |
| 6 | Fine-tuning pipeline + Modal serving + backends | next |
| 7 | Eval harness, judge, compare, HTML report | |
| 8 | Project skills, README | |
| 9 | Real fine-tune and three-mode comparison | |

## Stack and layout
Python 3.11, `uv`, `ruff`, `pytest`, `typer`, `pydantic-settings`, `structlog`, `aiogram` 3
(Business updates, long polling), `openai` SDK, ChromaDB, Unsloth + TRL and vLLM on Modal.
No Docker, no webhook, no userbot, no local model serving.

```
src/twin/  config.py (Settings, require_*), cli.py, logsetup.py
           core/schemas.py (contracts), llm_client.py (one OpenAI-compatible client,
           reasoning off per model family), prompts.py (versioned prompts/*.md templates),
           embeddings.py, vector_store.py (Chroma + index manifest), retriever.py (filters,
           dedupe), prompt.py, backends.py (GenerationBackend, RagBackend, validate-once),
           validate.py, memory.py, factory.py (build_backend from settings)
           bot/handlers.py (TwinBot: fail-closed gates, autopause, delivery), state.py,
           business.py, control.py (/twin commands), humanize.py, app.py (aiogram polling)
deploy/    inventory.sh, push.sh (rsync + install), install.sh <role>, sync_index.sh,
           status.sh, service.sh, roles/*.service, README.md (runbook); hosts.yaml gitignored
           ingest/parse_export.py (parser), reconstruct.py (turns, pairs), anonymize.py,
           split.py (time tail + seeded eval sample, leakage asserts), profile_dataset.py,
           dataconfig.py (configs/data/*.yaml), pipeline.py (twin ingest), stats.py (5.4),
           style_profile.py (twin style-profile: 300 train replies + stats -> Russian rules)
           bot/ eval/                          # added phase by phase
scripts/privacy_check.py  tests/ (synthetic fixtures, fake_openai_server.py)
prompts/<name>_v<N>.md  configs/data/default.yaml (min_date 2021, 15% tail)  training/ serving/
```

## Conventions
- Code, comments, docs, commits: English; anything fed to an LLM: Russian. Persona name
  only via `TWIN_NAME` / `${name}`.
- Dataset formats are contracts (`core/schemas.py`): deterministic transformations, every
  dropped record counted by reason, nothing cleaned silently.
- Holdout is never used for training, indexing, few-shot or prompt tuning; retrieval
  never returns the target reply, holdout records or records later than the query.
  Split: per chat, the last `split.tail_fraction` of pairs by time is the holdout tail
  (`holdout.jsonl`, all of it quarantined); `eval_sample=true` marks the seeded,
  month-stratified evaluation rows. `dataset_manifest.json` is committed: counts only.
- Every generated reply is one structlog record (chat_id, message_id, mode, model,
  prompt_version, retrieved_example_ids, params, latency, response, dry_run). Never log
  tokens, keys or full business-connection payloads (`logsetup` redacts).
- Tests need no credentials or paid APIs (fake server, temporary Chroma) and run in an
  empty cwd, so `.env` is never read by accident.
- Gateway facts (Phase 4): embeddings batches over ~10 texts return 503
  (`EMBED_BATCH_SIZE=10`, parallel workers); Qwen 3.5 Flash honours
  `enable_thinking=false` (default generator), DeepSeek V4 Flash keeps reasoning on,
  GPT-5.4 mini does not reason. Index = train pairs only (last two partner turns);
  `index_manifest.json` pins provider/model/dimension/dataset, mismatch fails loudly.

## Deployment facts (Phase 5)
- Roles: bot = srv-b (`/opt/twin`, systemd `twin`, RSS ~230 MB), jobs = srv-c (`~/twin`,
  no sudo, CLI only), staging = srv-a installed but **disabled**: one bot token allows one
  polling instance and one Business account allows one chatbot, so staging needs its own
  bot + account. Deploy path today: `deploy/push.sh` (rsync); `git pull` once on GitHub.
- The bot starts with `DRY_RUN` from `.env` and idles until a business connection arrives.

## Telegram identities (three different accounts in this deployment)

`TWIN_SENDER_ID` = whose export messages are the replies; `BUSINESS_OWNER_ID` = the
account the bot is connected to (connection `user_id` must match); `ADMIN_USER_IDS` =
who may send `/twin` commands; `ALLOWED_USER_IDS` = exactly two chat partners.
`Settings.require_bot()` / `require_ingest()` / `require_llm()` enforce these per command.

## Privacy rules for the public repo

- Never commit `.env`, `deploy/hosts.yaml`, `data/` (except `data/README.md` and the
  counts-only `dataset_manifest.json`), weights, Chroma data, eval outputs, the style
  profile. No name, Telegram id, phone, city or contact anywhere in the repo, commit
  messages or this file; fixtures are synthetic only.
- `pre-commit`: ruff, gitleaks, `scripts/privacy_check.py` (id/phone/key patterns,
  `.env`-derived ids and secrets, `data/private/blocklist.txt`). A synthetic test line
  may carry `privacy-check: allow`; there is no file-level escape hatch.
- Git identity is set locally to the project handle; no other identity in history.

## Commands
`uv sync` · `uv run twin --help` · `uv run pytest` · `uv run ruff check .` · `uv run ruff format .` · `uv run pre-commit run --all-files`
