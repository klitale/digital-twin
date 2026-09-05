"""Tests for scripts/privacy_check.py (importable thanks to pytest ``pythonpath``)."""

from __future__ import annotations

from pathlib import Path

import pytest

import privacy_check as pc

# Every literal below is synthetic. The trailing markers keep both guards quiet:
# ours (privacy-check: allow) and gitleaks (gitleaks:allow).
USER_ID = "user12345678"  # privacy-check: allow
BOT_TOKEN = "123456789:" + "x" * 35  # privacy-check: allow gitleaks:allow
PHONE = "+7 999 123-45-67"  # privacy-check: allow
SK_KEY = "sk-abcdefghijklmnopqrstuvwxyz"  # privacy-check: allow gitleaks:allow
AK_TOKEN = "ak-abcdefghijklmnopqrstuv"  # privacy-check: allow gitleaks:allow
HF_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz"  # privacy-check: allow gitleaks:allow

SAMPLES = {
    "telegram-user-id": f'from_id: "{USER_ID}"',
    "telegram-bot-token": f"token={BOT_TOKEN}",
    "phone-number": f"call me {PHONE} tonight",
    "llm-gateway-key": f"key={SK_KEY}",
    "modal-token": f"id={AK_TOKEN}",
    "hf-token": HF_TOKEN,
}


def categories(findings: list[pc.Finding]) -> set[str]:
    return {f.category for f in findings}


@pytest.mark.parametrize(("category", "text"), sorted(SAMPLES.items()))
def test_builtin_patterns_detect_each_shape(category: str, text: str) -> None:
    assert categories(pc.scan_text(text, [])) == {category}


def test_clean_text_has_no_findings() -> None:
    text = "date_unixtime: 1711234567\nid: 42\nuser1001 said hi\nas-is\n"
    assert pc.scan_text(text, []) == []


def test_digit_runs_inside_hashes_are_not_phone_numbers() -> None:
    text = 'sha256 = "c3a81234567890fe0"\nsize = 812345678901\n'
    assert pc.scan_text(text, []) == []


def test_allow_marker_skips_the_line() -> None:
    text = f"phone {PHONE}  {pc.ALLOW_MARKER}"
    assert pc.scan_text(text, []) == []


def test_blocklist_is_case_insensitive(tmp_path: Path) -> None:
    blocklist = tmp_path / "blocklist.txt"
    blocklist.write_text("# comment\n\nЗлатоуст\nab\n", encoding="utf-8")
    terms = pc.load_blocklist(blocklist)
    assert len(terms) == 1  # 'ab' is shorter than the minimum length
    assert categories(pc.scan_text("Привет, ЗЛАТОУСТ!", terms)) == {"blocklist"}
    assert pc.scan_text("Привет, мир", terms) == []


def test_env_terms_cover_ids_name_and_secret_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "TWIN_NAME=Radomir\nTWIN_SENDER_ID=55501234\nALLOWED_USER_IDS=55505678, 55509999\n"
        "LLM_API_KEY=zz-not-a-real-key-value\nMODAL_TOKEN_ID=ak-tokenid\nHF_REPO_ID=\n",
        encoding="utf-8",
    )
    terms = pc.load_env_terms(env)
    assert categories(pc.scan_text("hello radomir", terms)) == {"twin-name"}
    assert categories(pc.scan_text("chat 55505678 here", terms)) == {"telegram-id-from-env"}
    assert pc.scan_text("155505678", terms) == []  # digits must be a standalone number
    assert categories(pc.scan_text("zz-not-a-real-key-value", terms)) == {"secret-from-env"}
    assert categories(pc.scan_text("ak-tokenid", terms)) == {"secret-from-env"}


def test_scan_paths_skips_env_private_and_binary_files(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(USER_ID + "\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text(PHONE + "\n", encoding="utf-8")
    private = tmp_path / "data" / "private"
    private.mkdir(parents=True)
    (private / "blocklist.txt").write_text("secret name\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01" + USER_ID.encode())
    example = tmp_path / ".env.example"
    example.write_text("TG_BOT_TOKEN=\nSAFE=1\n", encoding="utf-8")
    dirty = tmp_path / "notes.md"
    dirty.write_text(f"id {USER_ID}\n", encoding="utf-8")
    paths = [
        tmp_path / ".env",
        tmp_path / ".env.local",
        private / "blocklist.txt",
        tmp_path / "blob.bin",
        example,
        dirty,
    ]
    findings = pc.scan_paths(paths, [])
    assert [(Path(f.path).name, f.line, f.category) for f in findings] == [
        ("notes.md", 1, "telegram-user-id")
    ]


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    none = str(tmp_path / "none")
    clean = tmp_path / "clean.py"
    clean.write_text("print('ok')\n", encoding="utf-8")
    assert pc.main([str(clean), "--blocklist", none, "--env", none]) == 0
    dirty = tmp_path / "dirty.py"
    dirty.write_text(f"x = '{HF_TOKEN}'\n", encoding="utf-8")
    assert pc.main([str(dirty), "--blocklist", none, "--env", none]) == 1
    out = capsys.readouterr()
    assert "dirty.py:1: hf-token" in out.out
    assert HF_TOKEN[:8] not in out.out  # never echo the match itself
