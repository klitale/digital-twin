"""LoRA fine-tuning of the twin (Qwen 2.5 Instruct) with completion-only loss.

One code path for every environment:

* GPU with Unsloth available: ``FastLanguageModel`` loads the base model (4-bit when the
  config says so) and wraps it with LoRA;
* otherwise (CPU dry-run, Colab without Unsloth): ``transformers`` + ``peft``.

Training itself is a plain ``transformers.Trainer`` over examples whose labels mask the
prompt (``training.chat_format.build_example``), which is what Unsloth's
``train_on_responses_only`` achieves. Heavy imports stay inside functions so the module
is importable (and testable) without torch.

Usage: ``python -m training.train_lora --config configs/train/full.yaml [--dry-run]
[--dataset PATH] [--output-dir DIR]``
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from training.chat_format import IGNORE_INDEX, build_example
from training.prepare_dataset import read_train_jsonl
from training.train_config import TrainConfig, load_train_config


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _bf16_supported() -> bool:
    """Ampere+ only; T4 (Turing) trains in fp16."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        return False


def _try_unsloth() -> Any | None:
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]

        return FastLanguageModel
    except Exception:
        return None


def load_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any, str]:
    """Returns ``(model, tokenizer, loader)`` where loader is 'unsloth' or 'transformers'."""
    fast = _try_unsloth() if _cuda_available() else None
    if fast is not None:
        model, tokenizer = fast.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=None,
        )
        model = fast.get_peft_model(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
        return model, tokenizer, "unsloth"

    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    kwargs: dict[str, Any] = {}
    if config.load_in_4bit and _cuda_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if _bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    if not _cuda_available():
        dtype = torch.float32
    elif _bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=dtype, **kwargs)
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model, tokenizer, "transformers"


def encode_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int
) -> list[dict[str, list[int]]]:
    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    examples = []
    for row in rows:
        input_ids, labels = build_example(row["messages"], encode)
        if len(input_ids) > max_seq_length:
            continue
        examples.append({"input_ids": input_ids, "labels": labels})
    return examples


class _Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def split_examples(
    examples: list[dict[str, list[int]]], eval_fraction: float, seed: int
) -> tuple[list, list]:
    if eval_fraction <= 0 or len(examples) < 20:
        return examples, []
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_eval = max(1, int(len(examples) * eval_fraction))
    eval_idx = set(order[:n_eval])
    return [e for i, e in enumerate(examples) if i not in eval_idx], [
        examples[i] for i in sorted(eval_idx)
    ]


def train(
    config: TrainConfig, dataset_path: Path, output_dir: Path, dry_run: bool = False
) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    started = time.time()
    rows = read_train_jsonl(dataset_path)
    if config.max_examples is not None:
        rows = rows[: config.max_examples]
    if dry_run:
        rows = rows[:5]
    model, tokenizer, loader = load_model_and_tokenizer(config)
    examples = encode_rows(rows, tokenizer, config.max_seq_length)
    train_examples, eval_examples = split_examples(
        examples, config.train.eval_fraction, config.seed
    )
    if not train_examples:
        raise ValueError("no training examples after encoding")
    params = config.train
    use_cuda = _cuda_available()
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=params.epochs,
        max_steps=2 if dry_run else (params.max_steps or -1),
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.per_device_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        warmup_ratio=params.warmup_ratio,
        lr_scheduler_type=params.lr_scheduler,
        weight_decay=params.weight_decay,
        logging_steps=params.logging_steps,
        save_steps=params.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_examples else "no",
        eval_steps=params.eval_steps,
        bf16=use_cuda and _bf16_supported(),
        fp16=use_cuda and not _bf16_supported(),
        seed=config.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=use_cuda,
        use_cpu=not use_cuda,
    )
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_examples,
        eval_dataset=eval_examples or None,
        data_collator=_Collator(pad_id),
    )
    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    device_name = "cpu"
    if use_cuda:
        import torch

        device_name = torch.cuda.get_device_name(0)
    summary = {
        "name": config.name,
        "loader": loader,
        "device": "cuda" if use_cuda else "cpu",
        "device_name": device_name,
        "examples_train": len(train_examples),
        "examples_eval": len(eval_examples),
        "steps": int(trainer.state.global_step),
        "train_loss": float(result.training_loss),
        "loss_history": history,
        "seconds": round(time.time() - started, 1),
        "adapter_dir": str(adapter_dir),
        "dry_run": dry_run,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune (completion-only loss)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="5 examples, 2 steps")
    args = parser.parse_args(argv)
    config = load_train_config(args.config)
    dataset = args.dataset or Path(config.dataset)
    output_dir = args.output_dir or Path(config.output_dir)
    summary = train(config, dataset, output_dir, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in summary.items() if k != "loss_history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
