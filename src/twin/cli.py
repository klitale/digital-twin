"""``twin`` command line interface.

Commands are wired in phase by phase; until then they exit with code 2 and say which
phase implements them, so the full surface is visible from ``twin --help`` from day one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from twin import __version__
from twin.config import Mode
from twin.logsetup import configure_logging

app = typer.Typer(
    help="Digital twin: a Telegram Business bot that replies in a personal writing style.",
    no_args_is_help=True,
    add_completion=False,
)


def _not_yet(command: str, phase: int) -> None:
    typer.echo(f"`twin {command}` is not implemented yet (planned for Phase {phase}).", err=True)
    raise typer.Exit(code=2)


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option("--log-level", envvar="TWIN_LOG_LEVEL", help="DEBUG, INFO, WARNING, ERROR."),
    ] = "INFO",
) -> None:
    configure_logging(log_level)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def ingest(
    export: Annotated[
        Path | None,
        typer.Option("--export", help="Telegram export result.json (default: RAW_EXPORT_PATH)."),
    ] = None,
    config: Annotated[
        Path, typer.Option("--config", help="Dataset construction parameters (YAML).")
    ] = Path("configs/data/default.yaml"),
    stop_after: Annotated[
        str | None,
        typer.Option("--stop-after", help="'parse' to stop after messages.jsonl."),
    ] = None,
) -> None:
    """Parse the export, build context/reply pairs and the time-based split."""
    from twin.config import ConfigError, load_settings
    from twin.ingest.dataconfig import config_source, load_data_config
    from twin.ingest.parse_export import ExportFormatError
    from twin.ingest.pipeline import format_pairs_report, format_report, run_pairs, run_parse

    if stop_after not in (None, "parse"):
        typer.echo(f"error: --stop-after must be 'parse', got {stop_after!r}", err=True)
        raise typer.Exit(code=2)
    settings = load_settings()
    try:
        data_config = load_data_config(config)
        parsed = run_parse(settings, export)
        typer.echo(format_report(parsed))
        if stop_after == "parse":
            return
        pairs = run_pairs(
            settings,
            data_config,
            parsed.messages,
            parsed.manifest.dataset_version,
            config_source=config_source(config),
        )
    except (ConfigError, ExportFormatError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("")
    typer.echo(format_pairs_report(pairs))


@app.command("analyze-data")
def analyze_data(
    config: Annotated[
        Path, typer.Option("--config", help="Dataset construction parameters (YAML).")
    ] = Path("configs/data/default.yaml"),
) -> None:
    """Rebuild the profiling report (section 5.4) from pairs.jsonl and holdout.jsonl."""
    from twin.config import ConfigError, load_settings
    from twin.ingest.dataconfig import load_data_config
    from twin.ingest.pipeline import run_profile

    settings = load_settings()
    try:
        report_path = run_profile(settings, load_data_config(config))
    except (ConfigError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    text = report_path.read_text(encoding="utf-8")
    caveats = text.split("## Caveats", 1)[1].split("## ", 1)[0].strip()
    typer.echo(f"report: {report_path}")
    typer.echo("Caveats:")
    typer.echo(caveats)


@app.command("style-profile")
def style_profile(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing, possibly hand-edited profile.")
    ] = False,
    sample_size: Annotated[int, typer.Option("--sample-size", min=1)] = 300,
    seed: Annotated[int, typer.Option("--seed")] = 20260905,
    prompt: Annotated[str, typer.Option("--prompt", help="Template name under prompts/.")] = (
        "style_profile_v1"
    ),
) -> None:
    """Derive the style profile (20-30 Russian rules) from sampled training replies."""
    from twin.config import ConfigError, load_settings
    from twin.core.llm_client import LLMClient, LLMError
    from twin.core.prompts import PromptError, load_prompt
    from twin.ingest.pipeline import STYLE_PROFILE_FILE, load_processed
    from twin.ingest.style_profile import (
        RULES_MAX,
        RULES_MIN,
        StyleProfileExistsError,
        generate_style_profile,
        write_style_profile,
    )

    settings = load_settings()
    target = settings.processed_dir / STYLE_PROFILE_FILE
    try:
        if target.exists() and not force:
            raise StyleProfileExistsError(
                f"{target} exists (possibly hand-edited); rerun with --force to overwrite"
            )
        settings.require_llm()
        if not settings.twin_name.strip():
            raise ConfigError("TWIN_NAME is not set: the profile needs the persona's name")
        train, _holdout, manifest = load_processed(settings)
        template = load_prompt(prompt)
        client = LLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            model=settings.style_profile_model,
        )
        result = generate_style_profile(
            client, template, settings.twin_name, train, sample_size=sample_size, seed=seed
        )
        write_style_profile(target, result, manifest.dataset_version, force=force)
    except (
        ConfigError,
        PromptError,
        LLMError,
        StyleProfileExistsError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"style profile: {target}")
    typer.echo(
        f"model {result.model}, prompt {result.prompt_version}, sample {result.sample_size} "
        f"replies, {result.rules} rules, tokens {result.prompt_tokens}+{result.completion_tokens}, "
        f"{result.latency_ms} ms"
    )
    if not RULES_MIN <= result.rules <= RULES_MAX:
        typer.echo(
            f"warning: expected {RULES_MIN}-{RULES_MAX} rules, got {result.rules}; "
            "review the file or rerun with --force",
            err=True,
        )


@app.command()
def dossier(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing, possibly hand-edited dossier.")
    ] = False,
    chunk_size: Annotated[int, typer.Option("--chunk-size", min=1)] = 400,
    min_chars: Annotated[int, typer.Option("--min-chars", min=1)] = 20,
    map_model: Annotated[
        str | None, typer.Option("--map-model", help="Model for the per-chunk notes.")
    ] = None,
) -> None:
    """Distil the self dossier (what the twin knows about himself) from training replies."""
    from twin.config import ConfigError, load_settings
    from twin.core.llm_client import LLMClient, LLMError
    from twin.core.prompts import PromptError, load_prompt
    from twin.ingest.dossier import (
        DOSSIER_MAP_PROMPT,
        DOSSIER_REDUCE_PROMPT,
        DossierExistsError,
        generate_dossier,
        write_dossier,
    )
    from twin.ingest.pipeline import SELF_DOSSIER_FILE, load_processed

    settings = load_settings()
    target = settings.processed_dir / SELF_DOSSIER_FILE
    try:
        if target.exists() and not force:
            raise DossierExistsError(
                f"{target} exists (possibly hand-edited); rerun with --force to overwrite"
            )
        settings.require_llm()
        if not settings.twin_name.strip():
            raise ConfigError("TWIN_NAME is not set: the dossier needs the persona's name")
        train, _holdout, manifest = load_processed(settings)

        def client(model: str) -> LLMClient:
            return LLMClient(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,  # type: ignore[arg-type]
                model=model,
            )

        result = generate_dossier(
            client(map_model or settings.llm_model),
            client(settings.style_profile_model),
            load_prompt(DOSSIER_MAP_PROMPT),
            load_prompt(DOSSIER_REDUCE_PROMPT),
            settings.twin_name,
            train,
            chunk_size=chunk_size,
            min_chars=min_chars,
            workers=settings.embed_workers,
        )
        write_dossier(target, result, manifest.dataset_version, force=force)
    except (ConfigError, PromptError, LLMError, DossierExistsError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"dossier: {target}")
    typer.echo(
        f"{result.facts} facts from {result.pairs_used} replies in {result.chunks} chunks "
        f"({result.skipped_pairs} refused by the content filter); "
        f"map {result.map_model}, reduce {result.reduce_model}; "
        f"tokens {result.prompt_tokens}+{result.completion_tokens}"
    )


@app.command()
def index(
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Drop the existing index and build it again.")
    ] = False,
    statements: Annotated[
        bool,
        typer.Option(
            "--statements", help="Build the statements collection (his reply texts) instead."
        ),
    ] = False,
) -> None:
    """Build the retrieval index from the training pairs (holdout is never indexed)."""
    from twin.config import ConfigError, load_settings
    from twin.core.embeddings import EmbeddingError, embeddings_from_settings
    from twin.core.vector_store import (
        STATEMENTS_COLLECTION,
        STATEMENTS_MANIFEST,
        ChromaVectorStore,
    )
    from twin.ingest.index import (
        STATEMENTS_RULE,
        IndexExistsError,
        build_index,
        reply_text_for_index,
        statement_pairs,
    )
    from twin.ingest.pipeline import load_processed

    settings = load_settings()
    try:
        train, _holdout, manifest = load_processed(settings)
        if statements:
            store = ChromaVectorStore(
                settings.chroma_dir, STATEMENTS_COLLECTION, STATEMENTS_MANIFEST
            )
            pairs, extra = (
                statement_pairs(train),
                {
                    "text_for": reply_text_for_index,
                    "rule": STATEMENTS_RULE,
                },
            )
        else:
            store = ChromaVectorStore(settings.chroma_dir)
            pairs, extra = train, {}
        if rebuild and store.count() > 0:
            typer.echo(f"dropping {store.count()} indexed records")
            store.reset()
        index_manifest = build_index(
            store,
            embeddings_from_settings(settings),
            pairs,
            manifest.dataset_version,
            batch_size=settings.embed_batch_size,
            progress=True,
            workers=settings.embed_workers,
            **extra,  # type: ignore[arg-type]
        )
    except (ConfigError, EmbeddingError, IndexExistsError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"index: {settings.chroma_dir} ({index_manifest.count} records)")
    typer.echo(index_manifest.model_dump_json(indent=2))


@app.command()
def chat(
    partner_id: Annotated[
        int | None, typer.Option("--partner-id", help="Memory slot; default: first allowed user.")
    ] = None,
    k: Annotated[int | None, typer.Option("--k", help="Retrieved examples per turn.")] = None,
) -> None:
    """Talk to the twin in the terminal (type /reset to clear memory, /quit to leave)."""
    import sys
    import time

    from twin.config import ConfigError, load_settings
    from twin.core.backends import GenerationRequest
    from twin.core.factory import build_backend, build_memory
    from twin.core.memory import MemoryTurn
    from twin.core.prompts import PromptError
    from twin.core.vector_store import IndexMismatchError

    settings = load_settings()
    partner = (
        partner_id
        if partner_id is not None
        else (settings.allowed_user_ids[0] if settings.allowed_user_ids else 0)
    )
    try:
        backend = build_backend(settings, k=k)
    except (ConfigError, PromptError, IndexMismatchError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    memory = build_memory(settings)
    typer.echo(f"twin chat: mode {backend.mode}, partner {partner}. /reset, /quit", err=True)
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            memory.reset(partner)
            typer.echo("(memory cleared)", err=True)
            continue
        history = memory.turns(partner)
        previous = next((t.text for t in reversed(history) if not t.is_me), None)
        request = GenerationRequest(
            partner_id=partner, text=text, previous_partner_text=previous, history=history
        )
        result = backend.generate(request)
        now = int(time.time())
        memory.append(partner, MemoryTurn(is_me=False, text=text, ts=now))
        if result.text is None:
            typer.echo(f"({settings.twin_name} молчит: {', '.join(result.rejected)})")
        else:
            typer.echo(f"{settings.twin_name}: {result.text}")
            memory.append(partner, MemoryTurn(is_me=True, text=result.text, ts=now))
        typer.echo(
            f"  [{result.model}, {result.latency_ms} ms, examples {len(result.retrieved_ids)}]",
            err=True,
        )


@app.command()
def run(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Generate replies but never send them.")
    ] = False,
) -> None:
    """Run the Telegram Business bot (long polling)."""
    from twin.bot.app import main as run_main
    from twin.config import ConfigError, load_settings
    from twin.core.prompts import PromptError
    from twin.core.vector_store import IndexMismatchError

    settings = load_settings()
    try:
        run_main(settings, dry_run)
    except (ConfigError, PromptError, IndexMismatchError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def control(
    command: Annotated[
        list[str], typer.Argument(help="A control command as in Telegram: `on`, `learn on`...")
    ],
) -> None:
    """Apply one control command to the persisted bot state (run while the bot is stopped,
    or the running bot overwrites it: deploy/control.sh does that)."""
    import time

    from twin.bot.control import ControlContext, handle_control, parse_command
    from twin.bot.state import StateStore
    from twin.config import load_settings
    from twin.core.factory import build_memory

    settings = load_settings()
    text = "/twin " + " ".join(command)
    parsed = parse_command(text)
    if parsed is None or parsed[0] in ("poke", "facts", "forget"):
        typer.echo("error: use `twin poke` to send; facts/forget need the Telegram bot", err=True)
        raise typer.Exit(code=2)
    store = StateStore(settings.state_dir)
    state = store.load_state()
    context = ControlContext(
        state=state,
        memory=build_memory(settings),
        connection=store.load_connection(),
        settings_mode=settings.twin_mode,
        settings_dry_run=settings.dry_run,
        allowed_user_ids=settings.allowed_user_ids,
        now=int(time.time()),
    )
    reply = handle_control(text, context)
    store.save_state(state)
    typer.echo(reply)


@app.command()
def poke(
    user_id: Annotated[int, typer.Argument(help="An ALLOWED_USER_IDS partner.")],
    kind: Annotated[str, typer.Argument(help="opener | followup")] = "opener",
) -> None:
    """Send one initiative message now, through every gate (connection, allowlist, enabled,
    pause, validation, dry-run). Run while the bot is stopped: deploy/control.sh does that."""
    import asyncio

    from aiogram import Bot
    from aiogram.client.default import DefaultBotProperties

    from twin.bot.app import build_twin_bot
    from twin.bot.initiative import FEATURES
    from twin.config import ConfigError, load_settings

    settings = load_settings()
    if kind not in FEATURES:
        typer.echo(f"error: kind must be one of {', '.join(FEATURES)}", err=True)
        raise typer.Exit(code=2)
    try:
        settings.require_bot()
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    async def send() -> str:
        bot = Bot(
            token=settings.tg_bot_token.get_secret_value(),  # type: ignore[union-attr]
            default=DefaultBotProperties(parse_mode=None),
        )
        try:
            return await build_twin_bot(bot, settings, cli_dry_run=False).poke(user_id, kind)
        finally:
            await bot.session.close()

    typer.echo(f"poke {kind} {user_id} -> {asyncio.run(send())}")


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/train/full.yaml"),
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Tiny model, 5 examples, 2 steps.")
    ] = False,
    remote: Annotated[bool, typer.Option("--remote", help="Run as a Modal GPU job.")] = False,
    prepare_only: Annotated[
        bool, typer.Option("--prepare-only", help="Only build data/train/train.jsonl.")
    ] = False,
    detach: Annotated[
        bool, typer.Option("--detach", help="With --remote: spawn and print the call id.")
    ] = False,
) -> None:
    """Prepare train.jsonl from the training pairs and run LoRA fine-tuning."""
    import os
    import subprocess

    from training import prepare_dataset as prep
    from training.train_config import load_train_config

    from twin.config import ConfigError, load_settings
    from twin.core.prompts import PromptError, load_prompt
    from twin.ingest.pipeline import load_processed

    settings = load_settings()
    try:
        if dry_run and config == Path("configs/train/full.yaml"):
            config = Path("configs/train/dry_run.yaml")
        train_config = load_train_config(config)
        if not settings.twin_name.strip():
            raise ConfigError("TWIN_NAME is not set")
        train_pairs, _holdout, manifest = load_processed(settings)
        persona = load_prompt("persona_train_v1")
        out_dir = settings.data_dir / "train"
        train_manifest = prep.prepare_dataset(
            train_pairs,
            persona,
            settings.twin_name,
            prep.load_qwen_encoder(),
            out_dir,
            manifest.dataset_version,
            max_seq_length=train_config.max_seq_length,
        )
    except (ConfigError, PromptError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"train.jsonl: {out_dir / prep.TRAIN_FILE}")
    typer.echo(train_manifest.model_dump_json(indent=2))
    if prepare_only:
        return

    dataset = out_dir / prep.TRAIN_FILE
    if remote:
        if not (settings.modal_token_id and settings.modal_token_secret):
            typer.echo("error: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are not set", err=True)
            raise typer.Exit(code=1)
        env = {
            **os.environ,
            "MODAL_TOKEN_ID": settings.modal_token_id.get_secret_value(),
            "MODAL_TOKEN_SECRET": settings.modal_token_secret.get_secret_value(),
            "TWIN_TRAIN_GPU": "T4" if dry_run else train_config.gpu,
        }
        cmd = [
            "modal",
            "run",
            *(["--detach"] if detach else []),
            "training/train_modal.py",
            "--config",
            str(config),
            "--dataset",
            str(dataset),
        ]
        if dry_run:
            cmd.append("--dry-run")
        if detach:
            cmd.append("--detach")
        typer.echo("launching: " + " ".join(cmd), err=True)
        raise typer.Exit(code=subprocess.call(cmd, env=env))

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        typer.echo(
            "torch is not installed in this environment. For the CPU dry run use:\n"
            "  uv run --with-requirements training/requirements-cpu.txt twin train --dry-run",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    from training.train_lora import train as train_lora

    summary = train_lora(train_config, dataset, Path(train_config.output_dir), dry_run=dry_run)
    typer.echo(
        f"trained {summary['steps']} step(s) on {summary['device']} via {summary['loader']}: "
        f"loss {summary['train_loss']:.3f}, adapter {summary['adapter_dir']}"
    )


@app.command("smoke-test-model")
def smoke_test_model(
    model: Annotated[
        str | None, typer.Option("--model", help="Adapter name served by Modal (default FT_MODEL).")
    ] = None,
    message: Annotated[str, typer.Option("--message")] = "привет, ты где пропал?",
) -> None:
    """Send one Russian message through the fine-tuned endpoint and time it."""
    from twin.config import ConfigError, load_settings
    from twin.core.factory import build_finetuned_llm
    from twin.core.llm_client import LLMError
    from twin.core.prompts import load_prompt

    settings = load_settings()
    try:
        llm = build_finetuned_llm(settings)
        llm.model = model or settings.ft_model
        llm._client = llm._client.with_options(timeout=20 * 60)  # cold starts take minutes
        template = load_prompt("persona_train_v1")
        system, _ = template.render(name=settings.twin_name or "Он", context="")
        result = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Собеседник: {message}"},
            ],
            temperature=0.7,
            max_tokens=120,
        )
    except (ConfigError, LLMError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    cold = result.latency_ms > 30_000
    typer.echo(f"model {result.model}: {result.text}")
    typer.echo(
        f"latency {result.latency_ms} ms ({'cold start' if cold else 'warm'}), tokens "
        f"{result.prompt_tokens}+{result.completion_tokens}"
    )


@app.command("eval")
def eval_(
    mode: Annotated[Mode, typer.Option("--mode")] = Mode.RAG,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/eval/default.yaml"),
    limit: Annotated[int | None, typer.Option("--limit", help="Evaluate only N pairs.")] = None,
) -> None:
    """Generate and judge every holdout pair for one mode; writes data/eval/<run>.json."""
    from twin.config import ConfigError, load_settings
    from twin.core.embeddings import QueryCache, embeddings_from_settings
    from twin.core.factory import (
        build_backend,
        build_retriever,
        build_statements_retriever,
        dossier_digest,
        style_profile_digest,
    )
    from twin.core.llm_client import LLMClient, LLMError
    from twin.core.prompts import PromptError, load_prompt
    from twin.core.vector_store import IndexMismatchError
    from twin.eval.harness import (
        AuditingRetriever,
        LeakageError,
        load_eval_config,
        run_eval,
        write_run,
    )
    from twin.eval.judge import Judge
    from twin.ingest.pipeline import load_processed

    settings = load_settings()
    try:
        eval_config = load_eval_config(config)
        if limit is not None:
            eval_config = eval_config.model_copy(update={"limit": limit})
        settings.require_judge()
        if not settings.twin_name.strip():
            raise ConfigError("TWIN_NAME is not set")
        _train, holdout, manifest = load_processed(settings)
        embeddings = QueryCache(embeddings_from_settings(settings))
        retriever = AuditingRetriever(
            build_retriever(settings, eval_config.generation.k, embeddings)
        )
        inner_statements = build_statements_retriever(settings, embeddings)
        statements = AuditingRetriever(inner_statements) if inner_statements else None
        gateway = None
        if mode is Mode.RAG:
            settings.require_llm()
            gateway = LLMClient(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,  # type: ignore[arg-type]
                model=settings.llm_model,
                temperature=eval_config.generation.temperature,
            )
        backend = build_backend(
            settings,
            mode,
            retriever=retriever,  # type: ignore[arg-type]
            llm=gateway,
            statements_retriever=statements,  # type: ignore[arg-type]
        )
        judge = Judge(
            LLMClient(
                base_url=settings.judge_base_url,
                api_key=settings.judge_api_key_effective,  # type: ignore[arg-type]
                model=settings.judge_model,
                temperature=eval_config.judge.temperature,
            ),
            load_prompt(eval_config.judge.prompt),
            settings.twin_name,
            temperature=eval_config.judge.temperature,
            max_tokens=eval_config.judge.max_tokens,
        )
        holdout_ids = {p.pair_id for p in holdout}

        def progress(done: int, total: int) -> None:
            typer.echo(f"\r{done}/{total}", nl=False, err=True)

        run = run_eval(
            holdout,
            holdout_ids,
            backend,
            retriever,
            judge,
            eval_config,
            manifest.dataset_version,
            manifest.messages_dataset_version,
            style_profile_sha256=style_profile_digest(settings),
            progress=progress,
            statements_retriever=statements,
            dossier_sha256=dossier_digest(settings),
        )
        typer.echo("", err=True)
        path = write_run(run, settings.data_dir / "eval")
    except (
        ConfigError,
        PromptError,
        IndexMismatchError,
        LeakageError,
        LLMError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    s = run.summary
    typer.echo(f"run: {path}")
    typer.echo(
        f"{run.metadata.mode} · {run.metadata.model} · {run.metadata.prompt_version}: n={s.n}, "
        f"judged={s.judged}, silent={s.silent}, fallbacks={s.fallbacks}, errors={s.errors}, "
        f"overall={s.overall}, means={s.means}, latency p50 {s.latency_ms_p50} ms"
    )


@app.command()
def compare(
    baseline: Annotated[str, typer.Option("--baseline", help="Mode used as the baseline.")] = "rag",
) -> None:
    """Compare evaluation runs in a table (also written to data/eval/compare.md)."""
    from twin.config import load_settings
    from twin.eval.compare import compare_runs, render_markdown
    from twin.eval.harness import list_runs, read_run

    settings = load_settings()
    paths = list_runs(settings.data_dir / "eval")
    if not paths:
        typer.echo("error: no evaluation runs in data/eval; run `twin eval` first", err=True)
        raise typer.Exit(code=1)
    text = render_markdown(compare_runs([read_run(p) for p in paths], baseline))
    (settings.data_dir / "eval" / "compare.md").write_text(text, encoding="utf-8")
    typer.echo(text)


@app.command()
def duel(
    run_a: Annotated[str, typer.Argument(help="Run file, or a unique run-id prefix (A).")],
    run_b: Annotated[str, typer.Argument(help="Run file, or a unique run-id prefix (B).")],
    limit: Annotated[int | None, typer.Option("--limit", help="Judge only N pairs.")] = None,
) -> None:
    """Blind pairwise judge: which run replies more alive, still in character."""
    from twin.config import ConfigError, load_settings
    from twin.core.llm_client import LLMClient
    from twin.core.prompts import PromptError, load_prompt
    from twin.eval.duel import DUEL_PROMPT, DuelJudge, resolve_run, run_duel, write_duel
    from twin.eval.harness import read_run

    settings = load_settings()
    directory = settings.data_dir / "eval"
    try:
        settings.require_judge()
        first = read_run(resolve_run(directory, run_a))
        second = read_run(resolve_run(directory, run_b))
        judge = DuelJudge(
            LLMClient(
                base_url=settings.judge_base_url,
                api_key=settings.judge_api_key_effective,  # type: ignore[arg-type]
                model=settings.judge_model,
                temperature=0.0,
            ),
            load_prompt(DUEL_PROMPT),
            settings.twin_name or "Он",
        )

        def progress(done: int, total: int) -> None:
            typer.echo(f"\r{done}/{total}", nl=False, err=True)

        result = run_duel(first, second, judge, limit=limit, progress=progress)
        typer.echo("", err=True)
        path = write_duel(result, directory / "duels")
    except (ConfigError, PromptError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    s = result.summary
    typer.echo(f"duel: {path}")
    typer.echo(
        f"A {result.run_a} ({result.prompt_a}) vs B {result.run_b} ({result.prompt_b}): "
        f"n={s.n}, A wins {s.a_wins}, B wins {s.b_wins}, ties {s.ties}, errors {s.errors}, "
        f"sign test p={s.p_value}"
    )


@app.command()
def report(
    out: Annotated[
        Path | None, typer.Option("--out", help="Default: data/eval/report.html")
    ] = None,
    baseline: Annotated[str, typer.Option("--baseline")] = "rag",
) -> None:
    """Render the self-contained HTML evaluation report."""
    from twin.config import load_settings
    from twin.eval.harness import list_runs, read_run
    from twin.eval.report import render_report

    settings = load_settings()
    paths = list_runs(settings.data_dir / "eval")
    if not paths:
        typer.echo("error: no evaluation runs in data/eval; run `twin eval` first", err=True)
        raise typer.Exit(code=1)
    target = out or settings.data_dir / "eval" / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_report([read_run(p) for p in paths], settings.twin_name or "Он", baseline),
        encoding="utf-8",
    )
    typer.echo(f"report: {target} ({len(paths)} run(s))")


if __name__ == "__main__":
    app()
