"""Entry point: `python -m tradebot.main` or `tradebot` console script."""

from __future__ import annotations

import os
import sys
import webbrowser

import uvicorn

from tradebot.core.logging import setup_logging
from tradebot.settings import get_env, load_config


def main() -> int:
    setup_logging()
    env = get_env()
    cfg = load_config()
    # uvicorn wants an import string for reload; we build the app directly (single process)
    from tradebot.web.server import create_app

    app = create_app(cfg)
    host = env.APP_HOST or "127.0.0.1"
    port = int(env.APP_PORT or 8000)
    url = f"http://{host}:{port}"
    print(f"\n  Autotrade → {url}   (mode: {cfg.mode}, IB {env.IB_HOST}:{env.IB_PORT})\n")
    if os.getenv("AUTOTRADE_OPEN_BROWSER", "1") == "1":
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
