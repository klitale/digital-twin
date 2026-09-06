"""Typed training configuration (``configs/train/*.yaml``)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraConfig(_Strict):
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainParams(_Strict):
    epochs: float = 2
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    max_steps: int | None = None
    eval_fraction: float = 0.02
    eval_steps: int = 200


class TrainConfig(_Strict):
    name: str
    base_model: str
    dataset: str = "data/train/train.jsonl"
    dataset_version: str | None = None
    seed: int = 20260905
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora: LoraConfig = LoraConfig()
    train: TrainParams = TrainParams()
    output_dir: str = "outputs/twin"
    gpu: str = "A10G"
    max_examples: int | None = None


def load_train_config(path: Path) -> TrainConfig:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return TrainConfig.model_validate(raw)
