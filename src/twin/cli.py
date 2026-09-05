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
def index(
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Drop and rebuild the index.")] = False,
) -> None:
    """Build the retrieval index from the training pairs."""
    _not_yet("index", 4)


@app.command()
def chat() -> None:
    """Talk to the twin in the terminal."""
    _not_yet("chat", 4)


@app.command()
def run(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Generate replies but never send them.")
    ] = False,
) -> None:
    """Run the Telegram Business bot (long polling)."""
    _not_yet("run", 5)


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/train/full.yaml"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Tiny model, 2 steps, CPU.")] = False,
    remote: Annotated[bool, typer.Option("--remote", help="Run as a Modal GPU job.")] = False,
) -> None:
    """Prepare the dataset and run LoRA fine-tuning."""
    _not_yet("train", 6)


@app.command("smoke-test-model")
def smoke_test_model(
    model: Annotated[
        str, typer.Option("--model", help="Adapter name served by the Modal endpoint.")
    ] = "base",
) -> None:
    """Send one Russian message through the fine-tuned endpoint."""
    _not_yet("smoke-test-model", 6)


@app.command("eval")
def eval_(
    mode: Annotated[Mode, typer.Option("--mode")] = Mode.RAG,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/eval/default.yaml"),
) -> None:
    """Generate and judge every holdout pair for one mode."""
    _not_yet("eval", 7)


@app.command()
def compare() -> None:
    """Compare evaluation runs in a table."""
    _not_yet("compare", 7)


@app.command()
def report() -> None:
    """Render the self-contained HTML evaluation report."""
    _not_yet("report", 7)


if __name__ == "__main__":
    app()
