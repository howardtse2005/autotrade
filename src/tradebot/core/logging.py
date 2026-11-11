"""Structured logging with secret redaction."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import structlog

from tradebot.settings import DATA_DIR

SECRET_KEYS = (
    "api_key", "apikey", "token", "secret", "password", "authorization",
    "finnhub_api_key", "marketaux_api_key", "app_api_token", "sec_user_agent",
)
_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^\s&,;\"']+)"),
    re.compile(r"\b[0-9a-f]{20,}\b"),
]


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pat in _PATTERNS:
            out = pat.sub(lambda m: f"{m.group(1)}=***" if m.groups() else "***", out)
        return out
    return value


def redaction_processor(_logger: Any, _name: str, event_dict: dict) -> dict:
    for k in list(event_dict):
        if any(s in k.lower() for s in SECRET_KEYS):
            event_dict[k] = "***"
        else:
            event_dict[k] = _redact(event_dict[k])
    return event_dict


_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    log_dir = Path(DATA_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    # noisy third parties
    # ib_async dumps every order/fill object at INFO - it swamps our own logs
    for name in ("httpx", "httpcore", "yfinance", "urllib3", "transformers",
                 "apscheduler", "ib_async", "ib_insync"):
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redaction_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "tradebot") -> structlog.stdlib.BoundLogger:
    if not _configured:
        setup_logging()
    return structlog.get_logger(name)
