---
name: twin-train
description: Prepare the training set and run or check LoRA fine-tuning of the digital-twin on Modal, then serve the adapter. Use when the user mentions "train", "fine-tune", "LoRA", "adapter", "Modal", "vLLM", "smoke test", wants a new adapter after a dataset change, or asks whether training finished.
---

# twin-train

## Workflow

1. `uv run twin train --prepare-only` — `data/train/train.jsonl` + counts-only manifest
   (token stats with the Qwen tokenizer; rows over `max_seq_length` dropped and counted;
   holdout rows are refused).
2. Pipeline checks before spending GPU time:
   `uv run --with-requirements training/requirements-cpu.txt twin train --dry-run` (CPU, tiny model)
   and `uv run twin train --remote --dry-run` (Modal T4, Unsloth, 2 steps).
3. Real run: `uv run twin train --remote --detach --config configs/train/full.yaml` prints a
   `call_id`; poll with `uv run modal run training/train_modal.py --result <call_id>`
   (`status: running` until done). Expect ~1 h on A10G, ~$1–2. No local process waits.
4. Serve: `uv run modal deploy serving/modal_app.py` (adapters are discovered at container
   start), `uv run twin smoke-test-model --model twin`, then `FT_MODEL=twin` in `.env`.
5. Evaluate with twin-eval before saying anything about quality.

## Inputs / outputs

- In: `data/processed/pairs.jsonl`, `prompts/persona_train_v1.md`, `configs/train/*.yaml`, Modal tokens.
- Out: `data/train/train.jsonl` (gitignored), adapter in the Modal Volume `digital-twin`
  under `adapters/<name>/` with `train_config.yaml`, `train_manifest.json`, `train_summary.json`.

## Failure behaviour

- Missing torch locally → the CLI prints the exact `uv run --with-requirements` command.
- T4 has no bf16: the trainer picks fp16 automatically; A10G uses bf16.
- vLLM needs the CUDA toolkit: the serving image is a `nvidia/cuda:*-devel` base; a
  "port 8000 never started" error means the engine crashed — read `modal app logs digital-twin-serve`.
- A cold endpoint answers in 2–4 min; the bot falls back to rag within `FT_TIMEOUT_SECONDS`.

## Test prompts (executed)

1. "Обучение закончилось?" → `modal run training/train_modal.py --result <id>`.
2. "Какие адаптеры есть?" → `modal run training/train_modal.py --list-only`.
3. "Проверь, что сервинг отвечает" → `twin smoke-test-model --model base`.
