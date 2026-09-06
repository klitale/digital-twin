"""OpenAI-compatible vLLM server on Modal with LoRA adapters from the shared Volume.

* base model ``Qwen/Qwen2.5-7B-Instruct`` (weights cached in the ``digital-twin-hf-cache``
  Volume), served as model name ``base``;
* every ``/vol/adapters/<name>/`` with an ``adapter_config.json`` is registered as a
  LoRA module ``<name>`` (``model=<name>`` in requests);
* bearer token from the Modal Secret ``digital-twin-api`` (``MODAL_API_KEY``);
* one A10G, scales to zero after ``SCALEDOWN_SECONDS``.

Deploy: ``modal deploy serving/modal_app.py``; the URL printed for ``serve`` + ``/v1`` is
``FT_BASE_URL``. See serving/README.md.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VOLUME_NAME = "digital-twin"
PORT = 8000
SCALEDOWN_SECONDS = 5 * 60
MAX_MODEL_LEN = 4096

app = modal.App("digital-twin-serve")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("digital-twin-hf-cache", create_if_missing=True)
# vLLM needs the CUDA toolkit (nvcc) at runtime for its JIT kernels: a devel base image.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm", "huggingface_hub")
    .env({"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "0", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)


def discover_adapters(root: Path) -> list[str]:
    modules = []
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if (path / "adapter_config.json").is_file():
                modules.append(f"{path.name}={path}")
    return modules


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("digital-twin-api")],
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=60 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve() -> None:
    volume.reload()
    adapters = discover_adapters(Path("/vol/adapters"))
    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--served-model-name",
        "base",
        "--api-key",
        os.environ["MODAL_API_KEY"],
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-lora",
        "--max-lora-rank",
        "32",
    ]
    if adapters:
        cmd += ["--lora-modules", *adapters]
    names = [a.split("=")[0] for a in adapters]
    print(f"starting vllm for {BASE_MODEL}; adapters: {names}")
    subprocess.Popen(cmd)
