# Serving the fine-tuned twin on Modal

`serving/modal_app.py` runs vLLM with the base `Qwen/Qwen2.5-7B-Instruct` and every LoRA
adapter found in the `digital-twin` Volume under `adapters/<name>/`, behind an
OpenAI-compatible `/v1` protected by a bearer token.

## One-time setup

```bash
uv run modal token set --token-id ... --token-secret ...   # or MODAL_TOKEN_* in .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # -> FT_API_KEY in .env
uv run modal secret create digital-twin-api MODAL_API_KEY=<that value>
uv run modal deploy serving/modal_app.py
# prints https://<workspace>--digital-twin-serve-serve.modal.run -> FT_BASE_URL=<url>/v1
uv run twin smoke-test-model --model base
```

## Adding an adapter

`twin train --remote --config configs/train/full.yaml` writes `adapters/twin/` into the
Volume. Adapters are discovered at container start, so redeploy (or just let the
container scale to zero and come back): `uv run modal deploy serving/modal_app.py`.
Then `twin smoke-test-model --model twin` and `FT_MODEL=twin` in `.env`.
List adapters: `uv run modal run training/train_modal.py --list-only`.

## Cold start and cost

First start downloads ~15 GB of weights into the cache Volume (5-10 min); later cold
starts load from the Volume in about 1-2 min; warm requests answer in ~1 s. The
container scales to zero after 5 minutes idle, so the bot wraps `finetuned`/`hybrid`
in a fallback to `rag` when the endpoint does not answer within `FT_TIMEOUT_SECONDS`.
A10G is billed per second only while a container is up (order of $1/h).
