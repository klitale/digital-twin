from __future__ import annotations

from typer.testing import CliRunner

from twin import __version__
from twin.cli import app

runner = CliRunner()

EXPECTED_COMMANDS = [
    "ingest",
    "analyze-data",
    "style-profile",
    "index",
    "chat",
    "run",
    "train",
    "smoke-test-model",
    "eval",
    "compare",
    "report",
]


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in EXPECTED_COMMANDS:
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_report_without_runs_fails_loudly() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 1
    assert "no evaluation runs" in result.output
