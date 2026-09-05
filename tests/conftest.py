from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in an empty directory so the real ``.env`` is never picked up."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def synthetic_export_path() -> Path:
    return FIXTURES / "export_synthetic.json"
