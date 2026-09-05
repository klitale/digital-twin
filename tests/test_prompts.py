from __future__ import annotations

from pathlib import Path

import pytest

from twin.core.prompts import PromptError, load_prompt, parse_template, resolve_prompts_dir

TEMPLATE = """<!-- version: demo_v1 -->
<!-- system -->
Ты — ${name}.
<!-- user -->
Ответь на: ${question} {не плейсхолдер}
"""


def test_parse_and_render() -> None:
    template = parse_template(TEMPLATE)
    assert template.version == "demo_v1"
    system, user = template.render(name="Радомир", question="как дела?")
    assert system == "Ты — Радомир."
    assert user == "Ответь на: как дела? {не плейсхолдер}"


def test_missing_placeholder_and_markers_raise() -> None:
    with pytest.raises(PromptError, match="question"):
        parse_template(TEMPLATE).render(name="x")
    with pytest.raises(PromptError):
        parse_template("<!-- system -->\nx\n<!-- user -->\ny")  # no version
    with pytest.raises(PromptError):
        parse_template("<!-- version: v1 -->\nno markers")


def test_load_prompt_from_dir(tmp_path: Path) -> None:
    (tmp_path / "demo_v1.md").write_text(TEMPLATE, encoding="utf-8")
    assert load_prompt("demo_v1", tmp_path).version == "demo_v1"
    with pytest.raises(PromptError, match="not found"):
        load_prompt("nope", tmp_path)


def test_repo_templates_are_found_from_any_cwd() -> None:
    assert (resolve_prompts_dir() / "style_profile_v1.md").is_file()
    template = load_prompt("style_profile_v1")
    system, user = template.render(name="Радомир", n_examples=3, stats="- x", examples="1. a")
    assert "Радомир" in user and "${" not in user and "${" not in system
