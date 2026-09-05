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
) -> None:
    """Parse the Telegram export and build the pair dataset with a time-based split."""
    _not_yet("ingest", 1)


@app.command("analyze-data")
def analyze_data() -> None:
    """Profile the pair dataset and write the Markdown report."""
    _not_yet("analyze-data", 2)


@app.command("style-profile")
def style_profile(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing, possibly hand-edited profile.")
    ] = False,
) -> None:
    """Derive the style profile (Russian rules) from sampled replies."""
    _not_yet("style-profile", 3)


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
