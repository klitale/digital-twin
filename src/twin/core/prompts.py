"""Versioned prompt templates stored as ``prompts/<name>.md``.

File layout::

    <!-- version: name_v1 -->
    <!-- system -->
    ...system text...
    <!-- user -->
    ...user text with ${placeholders}...

Placeholders use :class:`string.Template` syntax so braces in Russian text or JSON never
collide with formatting. The version string is recorded with every generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

_VERSION_RE = re.compile(r"<!--\s*version:\s*([\w.-]+)\s*-->")
_SYSTEM_MARK = "<!-- system -->"
_USER_MARK = "<!-- user -->"


class PromptError(ValueError):
    """A template is missing, malformed or misses a placeholder value."""


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str

    def render(self, **values: object) -> tuple[str, str]:
        try:
            return (
                Template(self.system).substitute(values),
                Template(self.user).substitute(values),
            )
        except KeyError as exc:
            raise PromptError(f"{self.version}: missing placeholder value {exc}") from exc


def resolve_prompts_dir() -> Path:
    """``prompts/`` in the working directory, else next to the package (repo checkout)."""
    local = Path("prompts")
    if local.is_dir():
        return local
    return Path(__file__).resolve().parents[3] / "prompts"


def parse_template(text: str) -> PromptTemplate:
    match = _VERSION_RE.search(text)
    if not match or _SYSTEM_MARK not in text or _USER_MARK not in text:
        raise PromptError("template needs a version comment and system/user markers")
    system_part = text.split(_SYSTEM_MARK, 1)[1]
    system, user = system_part.split(_USER_MARK, 1)
    return PromptTemplate(version=match.group(1), system=system.strip(), user=user.strip())


def load_prompt(name: str, prompts_dir: Path | None = None) -> PromptTemplate:
    path = (prompts_dir or resolve_prompts_dir()) / f"{name}.md"
    if not path.is_file():
        raise PromptError(f"prompt template not found: {path}")
    return parse_template(path.read_text(encoding="utf-8"))
