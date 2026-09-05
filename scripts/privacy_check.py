#!/usr/bin/env python3
"""Privacy guard for the public repository (runs in pre-commit on staged files).

Exit code 1 if any scanned file contains something that must never be published:

* Telegram user ids in export form (``user`` + digits) or ids listed in ``.env``;
* phone numbers and Telegram bot tokens;
* API keys for the LLM gateway, Modal and Hugging Face, both by shape and by the
  exact values found in ``.env``;
* every literal from ``data/private/blocklist.txt`` (names, nicknames, contacts) and
  the ``TWIN_NAME`` value from ``.env``, matched case-insensitively.

Only the category and the location are printed, never the matched text. A line that
intentionally contains synthetic data (tests) may carry the marker
``privacy-check: allow``; there is no file-level escape hatch on purpose.
Stdlib only and Python 3.9 compatible, so it runs from any system interpreter.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ALLOW_MARKER = "privacy-check: allow"
MIN_TERM_LENGTH = 3
MIN_SECRET_LENGTH = 8
SECRET_KEY_MARKERS = ("KEY", "TOKEN", "SECRET")

PATTERNS: dict[str, re.Pattern[str]] = {
    "telegram-user-id": re.compile(r"\buser\d{5,}\b"),
    "telegram-bot-token": re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"),
    # Hex look-arounds keep digit runs inside sha256 hashes from looking like phones.
    "phone-number": re.compile(
        r"(?<![0-9a-fA-F])(?:\+\d{1,3}|8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s-]*\d{2}[\s-]*\d{2}"
        r"(?![0-9a-fA-F])"
    ),
    "llm-gateway-key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "modal-token": re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{16,}\b"),
    "hf-token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
}
SKIP_DIR_PARTS = {".git", ".venv", "node_modules", "__pycache__"}


@dataclass(frozen=True)
class Term:
    category: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    category: str


def load_blocklist(path: Path) -> list[Term]:
    """One literal per line; ``#`` comments and blank lines ignored; case-insensitive."""
    if not path.is_file():
        return []
    terms: list[Term] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#") or len(entry) < MIN_TERM_LENGTH:
            continue
        terms.append(Term("blocklist", re.compile(re.escape(entry), re.IGNORECASE)))
    return terms


def load_env_terms(path: Path) -> list[Term]:
    """Turn the real ``.env`` into terms: ids, the persona name and secret values."""
    if not path.is_file():
        return []
    terms: list[Term] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip().strip("'\"")
        if not value:
            continue
        if any(marker in key for marker in SECRET_KEY_MARKERS):
            if len(value) >= MIN_SECRET_LENGTH:
                terms.append(Term("secret-from-env", re.compile(re.escape(value))))
        elif key.endswith(("_ID", "_IDS")):
            for token in re.findall(r"\d{5,}", value):
                pattern = re.compile(r"(?<!\d)" + re.escape(token) + r"(?!\d)")
                terms.append(Term("telegram-id-from-env", pattern))
        elif key == "TWIN_NAME" and len(value) >= MIN_TERM_LENGTH:
            terms.append(Term("twin-name", re.compile(re.escape(value), re.IGNORECASE)))
    return terms


def builtin_terms() -> list[Term]:
    return [Term(category, pattern) for category, pattern in PATTERNS.items()]


def scan_text(text: str, terms: list[Term], path: str = "<text>") -> list[Finding]:
    findings: list[Finding] = []
    all_terms = builtin_terms() + terms
    for number, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for term in all_terms:
            if term.pattern.search(line):
                findings.append(Finding(path, number, term.category))
    return findings


def is_skipped(path: Path) -> bool:
    """Never scan the private inputs themselves or vendored/binary trees."""
    if any(part in SKIP_DIR_PARTS for part in path.parts):
        return True
    name = path.name
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    return "data/private" in path.as_posix()


def read_text_file(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8000]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_paths(paths: list[Path], terms: list[Term]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if is_skipped(path) or not path.is_file():
            continue
        text = read_text_file(path)
        if text is None:
            continue
        findings.extend(scan_text(text, terms, path.as_posix()))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="files to scan (staged files)")
    parser.add_argument("--blocklist", type=Path, default=Path("data/private/blocklist.txt"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)

    if not args.blocklist.is_file():
        print(f"privacy-check: warning: blocklist {args.blocklist} not found", file=sys.stderr)
    terms = load_blocklist(args.blocklist) + load_env_terms(args.env)
    findings = scan_paths(args.paths, terms)
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.category}")
    if findings:
        print(
            f"privacy-check: {len(findings)} finding(s); commit refused. "
            "Remove the data or, for synthetic test data only, add the marker "
            f"'{ALLOW_MARKER}' to that line.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
