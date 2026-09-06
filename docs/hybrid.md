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
