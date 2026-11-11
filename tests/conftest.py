"""Test configuration: never touch the live database or the running app."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

# A unique database per test session keeps runs isolated and avoids file locks.
_TMP = Path(tempfile.gettempdir()) / "tradebot-tests"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TRADEBOT_DB"] = str(_TMP / f"test-{uuid.uuid4().hex[:8]}.db")
os.environ.setdefault("TRADING_MODE", "dry_run")
