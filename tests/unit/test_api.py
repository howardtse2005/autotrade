"""API surface, security guards and walk-forward split properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tradebot.models.cv import walk_forward_folds


@pytest.fixture(scope="module")
def client():
    import asyncio

    from tradebot.db.session import init_db
    from tradebot.web.server import create_app

    asyncio.run(init_db())
    app = create_app()
    # NOTE: no context manager -> lifespan (IB/scheduler) does not start.
    # base_url sets the Host header to a value the localhost guard accepts.
    return TestClient(app, base_url="http://127.0.0.1:8000")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_status_shape(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("mode", "ib", "market", "account", "model", "connectors"):
        assert key in body


def test_pages_render(client):
    for path in ("/", "/pipeline", "/positions", "/orders", "/backtest", "/models",
                 "/settings", "/logs"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "<!DOCTYPE html>" in r.text


def test_settings_schema(client):
    r = client.get("/api/settings/schema")
    assert r.status_code == 200
    fields = r.json()["fields"]
    keys = {f["key"] for f in fields}
    assert "pipeline.interval_minutes" in keys
    assert "risk.risk_per_trade_pct" in keys
    assert "strategy.max_positions" in keys


def test_mutation_requires_csrf(client):
    r = client.post("/api/killswitch", json={})
    assert r.status_code == 403


def test_csrf_token_allows_mutation(client):
    token = client.app.state.csrf
    r = client.put("/api/settings", json={"values": {}}, headers={"X-CSRF-Token": token})
    assert r.status_code == 200
    # leave no overrides behind for other tests / the running app
    client.post("/api/settings/reset", json={}, headers={"X-CSRF-Token": token})


def test_invalid_host_rejected(client):
    r = client.get("/api/health", headers={"Host": "evil.example.com"})
    assert r.status_code == 403


def test_settings_validation_rejects_bad_values(client):
    token = client.app.state.csrf
    r = client.put(
        "/api/settings",
        json={"values": {"strategy.max_positions": 99}},
        headers={"X-CSRF-Token": token},
    )
    assert r.status_code == 400


def test_walk_forward_has_no_overlap():
    idx = pd.date_range("2026-08-20", "2026-11-05", freq="15min", tz="UTC")
    folds = walk_forward_folds(idx, n_splits=4, purge_bars=8, embargo_bars=4,
                               train_days=30, test_days=5)
    assert folds, "expected at least one fold"
    for f in folds:
        train_ts = set(idx[f.train_idx])
        test_ts = set(idx[f.test_idx])
        assert not (train_ts & test_ts), "train/test timestamps overlap"
        assert max(train_ts) < min(test_ts), "training must precede the test window"


def test_walk_forward_respects_purge():
    idx = pd.date_range("2026-08-20", "2026-11-05", freq="15min", tz="UTC")
    folds = walk_forward_folds(idx, n_splits=3, purge_bars=8, embargo_bars=4,
                               train_days=30, test_days=5)
    for f in folds:
        train_ts = np.sort(np.array(sorted(set(idx[f.train_idx]))))
        test_start = min(idx[f.test_idx])
        gap = (test_start - train_ts[-1]) / pd.Timedelta(minutes=15)
        assert gap >= 8, f"purge gap too small: {gap}"
