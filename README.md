# digital-twin

A Telegram bot that answers *as you would*: connected to your account through
Telegram Business → Chatbots, it replies to a short allowlist of people in your own
writing style, learned from your exported Telegram history.

Three generation modes sit behind one interface and are compared with the same
evaluation harness and the same blind LLM judge:

| Mode | How replies are produced |
|---|---|
| `rag` | base LLM via an OpenAI-compatible gateway + style profile + retrieved examples of your real replies |
| `finetuned` | Qwen 2.5 7B Instruct + LoRA (Unsloth), served by vLLM on Modal; style profile only |
| `hybrid` | the fine-tuned model + retrieved examples |

> Status: work in progress, built phase by phase. Phase 0 (project skeleton) is done.

## Pipeline

```
Telegram export → messages.jsonl → pairs.jsonl (+ time-based holdout) → style profile
      → Chroma index → prompt → GenerationBackend (rag | finetuned | hybrid)
      → response validation → Telegram Business delivery (or --dry-run)
```

## Requirements

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/)
- A Telegram bot with Business Mode enabled (BotFather) and a Telegram account with
  Business features to connect it to
- An OpenAI-compatible LLM gateway (generation, judge, embeddings) and a Modal account
  for training and serving the fine-tuned model
- Your own Telegram Desktop JSON export (`result.json`)

## Quick start

```bash
git clone https://github.com/klitale/digital-twin && cd digital-twin
uv sync
cp .env.example .env            # fill in tokens, ids and model ids
uv run pre-commit install
uv run twin --help
```

Put your export under `data/raw/` (see `data/README.md`), then follow the commands in
the order they are listed by `twin --help`: `ingest`, `analyze-data`, `style-profile`,
`index`, `chat`, `run --dry-run`, and so on. Each command is documented as its phase lands.

## Privacy

Nothing personal is meant to reach this repository: the export, derived datasets, the
style profile, evaluation outputs and the runtime state all live under `data/` and are
gitignored. A pre-commit privacy guard (`scripts/privacy_check.py`) plus `gitleaks`
refuse commits containing Telegram ids, phone numbers, API keys or any string from a
private blocklist. Tests use synthetic fixtures only.

The bot fails closed: it replies only through a verified business connection, only in
private chats with exactly two allowlisted users, and only when the generated reply
passes validation. Restrict the Telegram-side "Selected chats" to the same two users as
well.

## License

MIT
