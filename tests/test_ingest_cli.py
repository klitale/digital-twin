from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from twin.cli import app
from twin.config import ConfigError, Settings
from twin.core.schemas import ChatSummary, DatasetManifest, ExportKind, MessagesManifest
from twin.ingest.pipeline import resolve_export_path

runner = CliRunner()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for key in ("TWIN_SENDER_ID", "RAW_EXPORT_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def test_ingest_without_sender_id_prints_top_senders_and_stops(
    env: Path, synthetic_export_path: Path
) -> None:
    result = runner.invoke(app, ["ingest", "--export", str(synthetic_export_path)])
    assert result.exit_code == 1
    assert "TWIN_SENDER_ID is not set" in result.output
    assert "user1001 (12 messages)" in result.output
    assert not (env / "processed").exists()


def test_ingest_rejects_sender_id_absent_from_export(
    env: Path, synthetic_export_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "999999")
    result = runner.invoke(app, ["ingest", "--export", str(synthetic_export_path)])
    assert result.exit_code == 1
    assert "matches no sender" in result.output
    assert "user1001 (12 messages)" in result.output
    assert not (env / "processed").exists()


def test_ingest_writes_messages_and_manifest(
    env: Path, synthetic_export_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    result = runner.invoke(app, ["ingest", "--export", str(synthetic_export_path)])
    assert result.exit_code == 0, result.output
    assert "kept: 17" in result.output
    assert "sticker" in result.output
    assert "<- twin" in result.output
    assert "sanity (5.4):" in result.output and "me_rows" in result.output
    lines = (env / "processed" / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 17
    manifest = MessagesManifest.model_validate_json(
        (env / "processed" / "messages_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.export_kind is ExportKind.SINGLE_CHAT
    assert manifest.source_path == str(synthetic_export_path)
    assert manifest.source_bytes == synthetic_export_path.stat().st_size
    assert len(manifest.source_sha256) == 64
    assert len(manifest.dataset_version) == 12
    assert manifest.twin_sender_id == 1001
    assert manifest.raw_messages == 21 and manifest.kept_messages == 17
    assert manifest.dropped == {"media_without_caption": 2, "service": 1, "sticker": 1}
    assert manifest.dropped_service_by_action == {"phone_call": 1}
    assert manifest.dropped_media_by_kind == {"photo": 1, "voice_message": 1}
    assert manifest.dropped_payload_by_kind == {} and manifest.skipped_chats_by_type == {}
    assert manifest.anomalies == {}
    assert manifest.chats == [
        ChatSummary(
            chat_id=1002,
            chat_type="personal_chat",
            raw_messages=21,
            kept_messages=17,
            me_messages=10,
        )
    ]
    assert manifest.sender_counts == {"user1001": 10, "user1002": 7}
    assert (manifest.ts_min, manifest.ts_max) == (1709280000, 1709370060)
    assert manifest.ts_source_counts == {"date_unixtime": 17}
    assert manifest.local_utc_offsets == {"+0": 17}


def test_dataset_version_is_deterministic(
    env: Path, synthetic_export_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    versions = []
    for name in ("a", "b"):
        monkeypatch.setenv("DATA_DIR", str(tmp_path / name))
        assert runner.invoke(app, ["ingest", "--export", str(synthetic_export_path)]).exit_code == 0
        manifest = json.loads(
            (tmp_path / name / "processed" / "messages_manifest.json").read_text()
        )
        versions.append(manifest["dataset_version"])
    assert versions[0] == versions[1]


def test_ingest_finds_the_single_export_under_data_raw(
    env: Path, synthetic_export_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    raw = env / "raw" / "some_export"
    raw.mkdir(parents=True)
    (raw / "result.json").write_bytes(synthetic_export_path.read_bytes())
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0, result.output
    assert "some_export" in result.output


def test_ingest_uses_raw_export_path(
    env: Path, synthetic_export_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    monkeypatch.setenv("RAW_EXPORT_PATH", str(synthetic_export_path))
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0, result.output
    monkeypatch.setenv("RAW_EXPORT_PATH", str(env / "nope.json"))
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "RAW_EXPORT_PATH not found" in result.output


@pytest.mark.parametrize("content", ["{}", "{not json", "[]"])
def test_ingest_reports_non_export_files(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--export", str(path)])
    assert result.exit_code == 1
    assert result.output.startswith("error:")
    result = runner.invoke(app, ["ingest", "--export", str(tmp_path / "missing.json")])
    assert result.exit_code == 1 and "not found" in result.output


@pytest.mark.parametrize(
    "data",
    [{"type": "personal_chat", "id": 1002, "messages": []}, {"chats": {"list": []}}],
)
def test_empty_export_fails_loudly(
    env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: dict[str, object]
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--export", str(path)])
    assert result.exit_code == 1
    assert "no messages in private chats" in result.output
    assert not (env / "processed").exists()


def test_resolve_export_path_branches(tmp_path: Path, synthetic_export_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    with pytest.raises(ConfigError, match="not found"):
        resolve_export_path(settings, tmp_path / "missing.json")
    assert resolve_export_path(settings, synthetic_export_path) == synthetic_export_path
    with pytest.raises(ConfigError, match=r"no result\.json"):
        resolve_export_path(settings, None)
    for name in ("one", "two"):
        target = tmp_path / "data" / "raw" / name / "result.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(synthetic_export_path.read_bytes())
    with pytest.raises(ConfigError, match=r"2 result\.json"):
        resolve_export_path(settings, None)


def test_ingest_builds_pairs_split_and_report(
    env: Path, synthetic_export_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    config = Path(__file__).parent.parent / "configs" / "data" / "default.yaml"
    result = runner.invoke(
        app, ["ingest", "--export", str(synthetic_export_path), "--config", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert "messages 17 -> turns 14 -> twin turns 7 -> pairs 6" in result.output
    assert "too short for holdout" in result.output
    processed = env / "processed"
    assert len((processed / "pairs.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    assert (processed / "holdout.jsonl").read_text(encoding="utf-8") == ""
    manifest = DatasetManifest.model_validate_json(
        (processed / "dataset_manifest.json").read_text()
    )
    assert manifest.messages == 17 and manifest.pairs_kept == 6 and manifest.train == 6
    assert manifest.pairs_dropped == {"no_context": 1}
    assert manifest.anonymized == {"mention": 1}
    assert manifest.config["split"]["eval_sample_size"] == 80
    assert manifest.config_source == str(config)
    assert manifest.chats[0].too_short_for_holdout is True
    manifest_text = (processed / "dataset_manifest.json").read_text(encoding="utf-8")
    assert "1002" not in manifest_text and "Злата" not in manifest_text  # counts only
    report = (processed / "profile_report.md").read_text(encoding="utf-8")
    assert "## Caveats" in report and "contributes no holdout" in report

    analyze = runner.invoke(app, ["analyze-data", "--config", str(config)])
    assert analyze.exit_code == 0, analyze.output
    assert "Caveats:" in analyze.output and "contributes no holdout" in analyze.output


def test_ingest_stop_after_parse(
    env: Path, synthetic_export_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWIN_SENDER_ID", "1001")
    result = runner.invoke(
        app, ["ingest", "--export", str(synthetic_export_path), "--stop-after", "parse"]
    )
    assert result.exit_code == 0, result.output
    assert not (env / "processed" / "pairs.jsonl").exists()
    bad = runner.invoke(
        app, ["ingest", "--export", str(synthetic_export_path), "--stop-after", "x"]
    )
    assert bad.exit_code == 2


def test_analyze_data_without_dataset(env: Path) -> None:
    result = runner.invoke(app, ["analyze-data"])
    assert result.exit_code == 1 and "run `twin ingest` first" in result.output
