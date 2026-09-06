"""Modal GPU job around ``training.train_lora``.

``twin train --remote [--dry-run]`` runs ``modal run training/train_modal.py`` which
ships the config and ``train.jsonl`` to a GPU container, trains, and stores the adapter
in the ``digital-twin`` Volume under ``adapters/<name>/`` next to the training config,
the dataset manifest and a loss summary. The serving app reads adapters from the same
Volume. GPU type comes from ``TWIN_TRAIN_GPU`` (set by the CLI from the config).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import modal

VOLUME_NAME = "digital-twin"
REMOTE_ROOT = Path("/vol")
GPU = os.environ.get("TWIN_TRAIN_GPU", "A10G")

app = modal.App("digital-twin-train")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("digital-twin-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("pydantic>=2.9", "pyyaml", "tokenizers", "huggingface_hub")
    .pip_install_from_requirements("training/requirements-gpu.txt")
    .add_local_dir("src/twin", remote_path="/root/twin")
    .add_local_dir("training", remote_path="/root/training")
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(REMOTE_ROOT): volume, "/root/.cache/huggingface": hf_cache},
    timeout=6 * 60 * 60,
)
def train_remote(config_yaml: str, dataset_bytes: bytes, manifest_json: str, dry_run: bool) -> dict:
    import yaml

    from training.train_config import TrainConfig
    from training.train_lora import train

    config = TrainConfig.model_validate(yaml.safe_load(config_yaml))
    work = Path("/tmp/twin-train")
    work.mkdir(parents=True, exist_ok=True)
    dataset = work / "train.jsonl"
    dataset.write_bytes(dataset_bytes)
    run_dir = work / "run"
    summary = train(config, dataset, run_dir, dry_run=dry_run)

    target = REMOTE_ROOT / "adapters" / config.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir / "adapter", target)
    (target / "train_config.yaml").write_text(config_yaml, encoding="utf-8")
    (target / "train_manifest.json").write_text(manifest_json, encoding="utf-8")
    (target / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    summary["adapter_path"] = str(target)
    return summary


@app.function(image=image, volumes={str(REMOTE_ROOT): volume})
def list_adapters() -> list[dict]:
    out = []
    root = REMOTE_ROOT / "adapters"
    if root.exists():
        for path in sorted(root.iterdir()):
            summary_file = path / "train_summary.json"
            summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
            out.append(
                {
                    "name": path.name,
                    "steps": summary.get("steps"),
                    "train_loss": summary.get("train_loss"),
                }
            )
    return out


@app.local_entrypoint()
def main(
    config: str = "configs/train/full.yaml",
    dataset: str = "data/train/train.jsonl",
    dry_run: bool = False,
    list_only: bool = False,
) -> None:
    if list_only:
        print(json.dumps(list_adapters.remote(), indent=2))
        return
    config_yaml = Path(config).read_text(encoding="utf-8")
    manifest_path = Path(dataset).with_name("train_manifest.json")
    manifest_json = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else "{}"
    summary = train_remote.remote(config_yaml, Path(dataset).read_bytes(), manifest_json, dry_run)
    print(json.dumps({k: v for k, v in summary.items() if k != "loss_history"}, indent=2))
