"""ChatML rendering and completion-only label masking shared by prepare/train.

Qwen 2.5 uses ChatML; rendering it here (instead of ``apply_chat_template``) keeps
token counting in ``prepare_dataset`` and label masking in ``train_lora`` byte-identical,
and keeps both testable without torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Message = dict[str, str]
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
IGNORE_INDEX = -100

Encode = Callable[[str], list[int]]


def render_chatml(messages: Sequence[Message], add_generation_prompt: bool = False) -> str:
    text = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    if add_generation_prompt:
        text += f"{IM_START}assistant\n"
    return text


def build_example(messages: Sequence[Message], encode: Encode) -> tuple[list[int], list[int]]:
    """``(input_ids, labels)`` with the prompt masked out; only the reply is learned."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the assistant reply")
    prompt = render_chatml(messages[:-1], add_generation_prompt=True)
    full = prompt + messages[-1]["content"] + f"{IM_END}\n"
    prompt_ids = encode(prompt)
    full_ids = encode(full)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenizer did not keep the prompt as a prefix of the full text")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return full_ids, labels
