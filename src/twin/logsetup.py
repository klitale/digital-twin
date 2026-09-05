"""structlog configuration shared by the CLI, the bot and the evaluation harness.

Every generated reply is logged as one structured record (see CLAUDE.md). Keys that
look like credentials are redacted before rendering so a misplaced ``bind()`` can never
leak a token into journald.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_SECRET_MARKERS = ("token", "api_key", "apikey", "secret", "password", "authorization")


def _redact_secrets(_logger: object, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict):
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_MARKERS):
            event_dict[key] = "***"
    return event_dict


class _StderrLogger:
    """Writes each rendered line to the *current* ``sys.stderr``.

    structlog's PrintLogger captures the stream at construction; under pytest's
    CliRunner that stream gets closed later, so the target is resolved per call.
    """

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr)

    log = debug = info = warning = warn = error = critical = exception = fatal = msg


def _stderr_logger_factory(*_args: Any) -> _StderrLogger:
    return _StderrLogger()


def configure_logging(level: str = "INFO", json_output: bool | None = None) -> None:
    """Configure structlog. JSON when not attached to a TTY (systemd), console otherwise."""
    if json_output is None:
        json_output = not sys.stderr.isatty()
    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=_stderr_logger_factory,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name) if name else structlog.get_logger()
