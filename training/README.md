# Training the twin (LoRA on Qwen 2.5 7B Instruct)

Everything runs from the CLI; the workstation never trains anything real.

| Step | Command | Where |
|---|---|---|
| Build `data/train/train.jsonl` from the training pairs | `uv run twin train --prepare-only` | workstation (tokenizer only) |
| Pipeline check, tiny model, 2 steps | `uv run --with-requirements training/requirements-cpu.txt twin train --dry-run` | workstation CPU (~7 min incl. downloads) |
| Pipeline check on a GPU (2 steps, Unsloth) | `uv run twin train --remote --dry-run` | Modal T4 (~1.5 min + image build) |
| Real run, full dataset | `uv run twin train --remote --config configs/train/full.yaml` | Modal A10G |
| List adapters in the Volume | `uv run modal run training/train_modal.py --list-only` | |

## What `train.jsonl` looks like

One line per pair: `{"messages": [system persona (Russian, `prompts/persona_train_v1.md`),
user = context turns with name prefixes, assistant = the twin's reply], "pair_id"}`.
Holdout rows can never enter: `prepare_dataset` refuses `eval_sample` rows and reads
`pairs.jsonl` (the train split) only. Examples over `max_seq_length` Qwen tokens are
dropped and counted in `data/train/train_manifest.json`.

## How training works

`training/train_lora.py` is one code path: Unsloth `FastLanguageModel` (4-bit, LoRA)
when CUDA + Unsloth are available, plain `transformers` + `peft` otherwise. The loss is
computed on the reply only (`training/chat_format.build_example` masks the prompt with
-100, the equivalent of Unsloth's `train_on_responses_only`). bf16 on Ampere+ (A10G),
fp16 on T4, fp32 on CPU. All hyper-parameters come from `configs/train/*.yaml` and are
copied next to the adapter.

`training/train_modal.py` ships the config and dataset to a GPU container and stores
`adapters/<name>/` (adapter weights, `train_config.yaml`, `train_manifest.json`,
`train_summary.json` with the loss history) in the Modal Volume `digital-twin`; the
serving app (`serving/`) reads adapters from the same Volume. GPU type comes from the
config (`gpu:`), the dry run always uses T4.

## Expected time and cost (full run)

~6.2k examples, 2 epochs, batch 2 x accumulation 4 -> ~1.5k steps. On A10G about an
hour (Qwen 2.5 7B in 4-bit, seq 2048), ~$1-2 of GPU time; T4 works too but 2-3x slower.
The first run also downloads the base model into the `digital-twin-hf-cache` Volume.

## Spend limits and resuming

Modal stops every function the moment the workspace spend limit is exceeded
("Workspace … has exceeded its spend limit"): the serving endpoint answers 4xx and a
running training job is killed. Raise the limit in the Modal dashboard (Settings →
Billing) and simply relaunch the same command: checkpoints are written to the Volume
every `save_steps` steps and the trainer resumes from the newest one.

## After training

```bash
uv run modal deploy serving/modal_app.py      # registers adapters found in the Volume
uv run twin smoke-test-model --model twin
# .env: FT_MODEL=twin ; TWIN_MODE=finetuned|hybrid
```

## Colab fallback

`training/train_lora.ipynb` runs the same module on a Colab GPU if Modal quota is
unavailable: upload `train.jsonl` and a config, install `requirements-gpu.txt`, run the
cells. Checkpoints go to the notebook's `output_dir` (mount Google Drive to keep them).
