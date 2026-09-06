"""Optional: mirror an adapter from the Modal Volume to a private Hugging Face repo.

Usage: ``uv run python -m training.push_adapter --name twin`` with ``HF_TOKEN`` and
``HF_REPO_ID`` in ``.env``. Not required by the pipeline (the serving app reads the
Volume directly); it exists so the adapter survives outside Modal.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from twin.config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="twin", help="adapter name in the Volume")
    args = parser.parse_args(argv)
    settings = load_settings()
    if not settings.hf_token or not settings.hf_repo_id:
        print("HF_TOKEN / HF_REPO_ID are not set; nothing to do", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / args.name
        subprocess.run(
            ["modal", "volume", "get", "digital-twin", f"adapters/{args.name}", str(local)],
            check=True,
        )
        api = HfApi(token=settings.hf_token.get_secret_value())
        api.create_repo(settings.hf_repo_id, private=True, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(local), repo_id=settings.hf_repo_id, repo_type="model")
    print(f"uploaded adapters/{args.name} to {settings.hf_repo_id} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
