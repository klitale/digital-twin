"""``pairs.jsonl`` (train split only) -> ``data/train/train.jsonl`` in chat format.

Every example is ``[{system: short persona}, {user: context with name prefixes},
{assistant: reply}]``. Holdout rows are refused (asserted twice: the file we read is the
train split, and no row may carry ``eval_sample``). Token statistics use the Qwen
tokenizer; examples over ``max_seq_length`` tokens are dropped and counted.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from training.chat_format import Message, render_chatml
from twin.core.prompts import PromptTemplate
from twin.core.schemas import Pair
from twin.ingest.stats import quantile

TRAIN_FILE = "train.jsonl"
TRAIN_MANIFEST = "train_manifest.json"
TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"

Encode = Callable[[str], list[int]]


class TrainManifest(BaseModel):
    dataset_version: str = Field(description="sha256 prefix of train.jsonl")
    pairs_dataset_version: str
    persona_prompt_version: str
    tokenizer: str
    max_seq_length: int
    examples: int
    dropped_too_long: int
    tokens_p50: int
    tokens_p90: int
    tokens_max: int
    tokens_total: int
    reply_tokens_p50: int


def format_context(pair: Pair, name: str) -> str:
    return "\n".join(f"{name if t.is_me else 'Собеседник'}: {t.text}" for t in pair.context)


def pair_to_messages(pair: Pair, persona: PromptTemplate, name: str) -> list[Message]:
    system, user = persona.render(name=name, context=format_context(pair, name))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": pair.reply},
    ]


def load_qwen_encoder(repo: str = TOKENIZER_REPO) -> Encode:
    """``tokenizers`` only (no transformers/torch); downloads ``tokenizer.json`` once."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(repo)
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def prepare_dataset(
    pairs: Sequence[Pair],
    persona: PromptTemplate,
    name: str,
    encode: Encode,
    out_dir: Path,
    pairs_dataset_version: str,
    max_seq_length: int = 2048,
    max_examples: int | None = None,
) -> TrainManifest:
    if any(p.eval_sample for p in pairs):
        raise ValueError("holdout rows in the training input (eval_sample=true)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TRAIN_FILE
    lengths: list[int] = []
    reply_lengths: list[int] = []
    dropped = 0
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            if max_examples is not None and written >= max_examples:
                break
            messages = pair_to_messages(pair, persona, name)
            n_tokens = len(encode(render_chatml(messages)))
            if n_tokens > max_seq_length:
                dropped += 1
                continue
            lengths.append(n_tokens)
            reply_lengths.append(len(encode(pair.reply)))
            fh.write(
                json.dumps({"messages": messages, "pair_id": pair.pair_id}, ensure_ascii=False)
            )
            fh.write("\n")
            written += 1
    import hashlib

    version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    lengths.sort()
    reply_lengths.sort()
    manifest = TrainManifest(
        dataset_version=version,
        pairs_dataset_version=pairs_dataset_version,
        persona_prompt_version=persona.version,
        tokenizer=TOKENIZER_REPO,
        max_seq_length=max_seq_length,
        examples=written,
        dropped_too_long=dropped,
        tokens_p50=quantile(lengths, 0.5),
        tokens_p90=quantile(lengths, 0.9),
        tokens_max=lengths[-1] if lengths else 0,
        tokens_total=sum(lengths),
        reply_tokens_p50=quantile(reply_lengths, 0.5),
    )
    (out_dir / TRAIN_MANIFEST).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_train_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows
