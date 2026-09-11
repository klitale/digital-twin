from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from twin.bot.state import StateStore
from twin.cli import app


def test_control_cli_changes_the_persisted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    result = runner.invoke(app, ["control", "off"])
    assert result.exit_code == 0 and "bot disabled" in result.output
    store = StateStore(tmp_path / "data" / "state")
    assert store.load_state().enabled is False
    result = runner.invoke(app, ["control", "learn", "on"])
    assert result.exit_code == 0 and "learn on" in result.output
    state = store.load_state()
    assert state.feature_on("learn") and state.enabled is False
    assert runner.invoke(app, ["control", "on"]).exit_code == 0
    assert store.load_state().enabled is True
    # sending needs the Telegram layer: poke has its own command
    assert runner.invoke(app, ["control", "poke", "1"]).exit_code == 2


def test_poke_cli_rejects_unknown_kinds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(app, ["poke", "1", "hello"]).exit_code == 2
