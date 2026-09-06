# Hybrid mode design

`hybrid` = the fine-tuned model (LoRA adapter served by vLLM on Modal) + retrieval, with
a deliberately short prompt (`prompts/hybrid_v1.md`).

| | rag | finetuned | hybrid |
|---|---|---|---|
| model | base chat model via the gateway | Qwen 2.5 7B + LoRA on Modal | Qwen 2.5 7B + LoRA on Modal |
| style profile in prompt | yes | yes | **no** (the weights carry the style) |
| retrieved examples | top-8 | none | top-8 |
| prompt version | `rag_v1` | `finetuned_v1` | `hybrid_v1` |

Why no style profile in hybrid: the adapter already learned lexicon, punctuation and
length; repeating the rules in text costs tokens and tends to make the reply more
"instructed". Retrieved examples add *situational* memory the weights cannot hold:
what the twin actually said in similar exchanges.

All three share `GenerationRequest` / `GenerationResult`, validation with one
regeneration, and the retrieval leakage rules (target reply, holdout, future rows are
never retrieved). In the bot, `finetuned` and `hybrid` are wrapped in a fallback: a
cold Modal container or an endpoint error within `FT_TIMEOUT_SECONDS` falls back to
`rag` for that message and logs `fallback_from`.

## Observed (Phase 9, first adapter)

On the 80-pair holdout hybrid scored 2.73 against 2.83 for `finetuned` and 3.22 for
`rag`. The retrieved examples did not add situational memory for the 7B model; they made
it paraphrase them into longer replies (median 41 chars vs 36 for `finetuned` and 37 in
the references). Next things to try, in order: a `hybrid_v2` prompt that frames the
examples as "how you usually answer" rather than material to reuse, top-4 instead of
top-8, and `temperature` 0.6. Any change is judged with `twin eval` before it is adopted.
