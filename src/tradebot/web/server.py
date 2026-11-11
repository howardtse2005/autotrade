"""FastAPI application: pages, JSON API, SSE, security middleware."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sse_starlette.sse import EventSourceResponse

from tradebot.app import App
from tradebot.core.clock import fmt_hkt, market_clock, now_utc, to_hkt
from tradebot.core.events import bus
from tradebot.core.logging import get_logger
from tradebot.core.pipeline import Pipeline
from tradebot.db.models import (
    AccountSnapshot,
    Alert,
    Backtest,
    ConnectorHealth,
    FillRow,
    ModelRecord,
    NewsItem,
    OrderIntentRow,
    OrderRow,
    PositionRow,
    Prediction,
    Run,
    RunEvent,
    SettingOverride,
    SettingsAudit,
    Signal,
    Trade,
    UniverseSnapshot,
)
from tradebot.db.session import session_scope, state_get, state_set
from tradebot.models import registry
from tradebot.settings import AppConfig, flatten_schema, get_path, load_yaml, set_path

log = get_logger("web")
HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.filters["hkt"] = lambda v, d=True: fmt_hkt(v) if d else fmt_hkt(v)
templates.env.filters["money"] = lambda v, c="USD": _money(v, c)
templates.env.filters["num"] = lambda v, d=2: _num(v, d)
templates.env.filters["pct"] = lambda v, d=2: _pct(v, d)
templates.env.filters["cls"] = lambda v: "pos" if (v or 0) > 0 else ("neg" if (v or 0) < 0 else "dim")
templates.env.filters["reason"] = lambda c: _reason_label(c)


def _money(v, currency="USD") -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except Exception:  # noqa: BLE001
        return "—"
    sym = {"USD": "$", "HKD": "HK$", "JPY": "¥", "EUR": "€", "GBP": "£"}.get(currency, currency + " ")
    digits = 0 if abs(v) >= 1000 else 2
    return f"{sym}{v:,.{digits}f}"


def _num(v, digits=2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:  # noqa: BLE001
        return "—"


def _pct(v, digits=2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):+,.{digits}f}%"
    except Exception:  # noqa: BLE001
        return "—"


def _reason_label(code: str) -> str:
    from tradebot.strategy.reason_codes import label

    return label(code) if code else ""


def _clean(obj):
    """Recursively replace NaN/Inf with None so JSON stays valid."""
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, allow_nan=False,
                          default=str).encode("utf-8")


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    app_state = App(cfg)
    csrf = secrets.token_urlsafe(24)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app_state.start()
        yield
        await app_state.stop()

    app = FastAPI(title="Autotrade", lifespan=lifespan, docs_url=None, redoc_url=None,
                  default_response_class=SafeJSONResponse)
    app.state.app = app_state
    app.state.csrf = csrf
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # -------------------------------------------------------------- security
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").split(":")[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1", ""):
            return JSONResponse({"detail": "invalid host"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            ok = any(o in origin for o in ("127.0.0.1", "localhost", "::1"))
            if not ok:
                return JSONResponse({"detail": "invalid origin"}, status_code=403)
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            token = request.headers.get("x-csrf-token") or request.headers.get("x-api-token")
            if token != csrf and not (app_state.env.APP_API_TOKEN and token == app_state.env.APP_API_TOKEN):
                return JSONResponse({"detail": "invalid CSRF token"}, status_code=403)
        return await call_next(request)

    # ----------------------------------------------------------------- pages
    def render(request: Request, template: str, **ctx) -> HTMLResponse:
        base = {
            "request": request,
            "csrf_token": csrf,
            "mode": app_state.cfg.mode,
            "active": ctx.pop("active", ""),
            "app": app_state,
        }
        base.update(ctx)
        return templates.TemplateResponse(request, template, base)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        status = await app_state.status()
        acct = await app_state.get_account()
        positions = await app_state.get_positions()
        async with session_scope() as s:
            runs = (await s.execute(select(Run).order_by(Run.id.desc()).limit(6))).scalars().all()
            alerts = (await s.execute(select(Alert).order_by(Alert.id.desc()).limit(8))).scalars().all()
            pending_approvals = (await s.execute(
                select(OrderIntentRow).where(OrderIntentRow.status == "awaiting_approval")
                .order_by(OrderIntentRow.id.desc()).limit(10)
            )).scalars().all()
            curve = (await s.execute(
                select(AccountSnapshot).order_by(AccountSnapshot.ts.desc()).limit(600)
            )).scalars().all()
            trades = (await s.execute(
                select(Trade).order_by(Trade.created_at.desc()).limit(8)
            )).scalars().all()
        curve = list(reversed(curve))
        longs = [p for p in positions if p.qty > 0]
        return render(
            request, "dashboard.html", active="dashboard", status=status, account=acct,
            positions=longs, runs=runs, alerts=alerts, curve=curve, trades=trades,
            last_run=runs[0] if runs else None, cfg=app_state.cfg,
            pending_approvals=pending_approvals, clock=market_clock(),
        )

    @app.get("/pipeline", response_class=HTMLResponse)
    async def pipeline_page(request: Request, run_id: int | None = None):
        async with session_scope() as s:
            runs = (await s.execute(select(Run).order_by(Run.id.desc()).limit(40))).scalars().all()
            if run_id is None and runs:
                run_id = runs[0].id
            run = await s.get(Run, run_id) if run_id else None
            events = []
            universe = []
            decisions = []
            news = []
            if run is not None:
                events = (await s.execute(
                    select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.id)
                )).scalars().all()
                universe = (await s.execute(
                    select(UniverseSnapshot).where(UniverseSnapshot.run_id == run.id)
                    .order_by(UniverseSnapshot.rank)
                )).scalars().all()
                decisions = (await s.execute(
                    select(Signal).where(Signal.run_id == run.id)
                    .order_by(Signal.rank.is_(None), Signal.rank, Signal.ticker)
                )).scalars().all()
                news = (await s.execute(
                    select(NewsItem).where(NewsItem.run_id == run.id)
                    .order_by(NewsItem.published_at.desc()).limit(150)
                )).scalars().all()
            conns = (await s.execute(select(ConnectorHealth))).scalars().all()
            preds = {}
            if run is not None:
                prows = (await s.execute(
                    select(Prediction).where(Prediction.run_id == run.id)
                )).scalars().all()
                preds = {p.ticker: p for p in prows}
        return render(
            request, "pipeline.html", active="pipeline", runs=runs, run=run, events=events,
            universe=universe, decisions=decisions, news=news, connectors=conns, preds=preds,
        )

    @app.get("/positions", response_class=HTMLResponse)
    async def positions_page(request: Request):
        positions = await app_state.get_positions()
        acct = await app_state.get_account()
        status = await app_state.status()
        return render(request, "positions.html", active="positions", positions=positions,
                      account=acct, status=status, cfg=app_state.cfg)

    @app.get("/orders", response_class=HTMLResponse)
    async def orders_page(request: Request):
        async with session_scope() as s:
            open_orders = (await s.execute(
                select(OrderRow).where(~OrderRow.status.in_(["Filled", "Cancelled", "ApiCancelled", "Inactive"]))
                .order_by(OrderRow.id.desc()).limit(100)
            )).scalars().all()
            history = (await s.execute(
                select(OrderRow).order_by(OrderRow.id.desc()).limit(200)
            )).scalars().all()
            trades = (await s.execute(
                select(Trade).order_by(Trade.created_at.desc()).limit(200)
            )).scalars().all()
            intents = (await s.execute(
                select(OrderIntentRow).order_by(OrderIntentRow.id.desc()).limit(100)
            )).scalars().all()
            fills = (await s.execute(select(FillRow).order_by(FillRow.id.desc()).limit(100))).scalars().all()
        ib_orders = await app_state.ib_client.open_orders() if app_state.ib_client.connected else []
        pending = [i for i in intents if i.status == "awaiting_approval"]
        clock = market_clock()
        # cash committed to working BUY orders vs what is available
        queued_notional = await app_state.orders.queued_buy_notional()
        try:
            acct = await app_state.get_account()
            cash_usd = acct.cash_usd
            avail_usd = acct.available_usd
        except Exception:  # noqa: BLE001
            cash_usd = avail_usd = 0.0
        return render(request, "orders.html", active="orders", open_orders=open_orders,
                      history=history, trades=trades, intents=intents, fills=fills,
                      ib_orders=ib_orders, pending=pending, clock=clock,
                      queued_notional=queued_notional, cash_usd=cash_usd,
                      avail_usd=avail_usd,
                      approval_mode=app_state.cfg.approval_mode,
                      queue_outside=app_state.cfg.pipeline.queue_entries_outside_rth,
                      cfg=app_state.cfg)

    @app.get("/backtest", response_class=HTMLResponse)
    async def backtest_page(request: Request, id: int | None = None):
        async with session_scope() as s:
            rows = (await s.execute(select(Backtest).order_by(Backtest.id.desc()).limit(30))).scalars().all()
            current = await s.get(Backtest, id) if id else (rows[0] if rows else None)
        return render(request, "backtest.html", active="backtest", backtests=rows, current=current,
                      cfg=app_state.cfg)

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        records = await registry.list_records(40)
        champion_id, champion_meta = await registry.champion()
        return render(request, "models.html", active="models", models=records,
                      champion_id=champion_id, champion_meta=champion_meta,
                      predictor=app_state.predictor.status(), cfg=app_state.cfg)

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        fields = flatten_schema(AppConfig)
        sections: dict[str, list] = {}
        for f in fields:
            section = f["key"].split(".")[0] if "." in f["key"] else "general"
            sections.setdefault(section, []).append(f)
        values = {f["key"]: get_path(app_state.cfg, f["key"]) for f in fields}
        async with session_scope() as s:
            audit = (await s.execute(
                select(SettingsAudit).order_by(SettingsAudit.id.desc()).limit(30)
            )).scalars().all()
        return render(request, "settings.html", active="settings", sections=sections,
                      values=values, audit=audit, warnings=app_state.cfg.coupled_warnings())

    @app.get("/logs", response_class=HTMLResponse)
    async def logs_page(request: Request, level: str | None = None, stage: str | None = None):
        async with session_scope() as s:
            q = select(RunEvent).order_by(RunEvent.id.desc()).limit(400)
            if level:
                q = q.where(RunEvent.level == level)
            if stage:
                q = q.where(RunEvent.stage == stage)
            events = (await s.execute(q)).scalars().all()
            alerts = (await s.execute(select(Alert).order_by(Alert.id.desc()).limit(100))).scalars().all()
            audit = (await s.execute(
                select(SettingsAudit).order_by(SettingsAudit.id.desc()).limit(50)
            )).scalars().all()
        return render(request, "logs.html", active="logs", events=events, alerts=alerts,
                      audit=audit, level=level, stage=stage,
                      ib_notices=app_state.ib_client.notices[-40:][::-1])

    # ------------------------------------------------------------------- API
    @app.get("/api/status")
    async def api_status():
        return await app_state.status()

    @app.get("/api/account")
    async def api_account():
        acct = await app_state.get_account()
        return {
            "currency": acct.currency, "net_liq": acct.net_liq, "cash": acct.cash,
            "available_funds": acct.available_funds, "buying_power": acct.buying_power,
            "excess_liquidity": acct.excess_liquidity, "excess_liq_pct": acct.excess_liq_pct,
            "gross_position_value": acct.gross_position_value, "realized_pnl": acct.realized_pnl,
            "unrealized_pnl": acct.unrealized_pnl, "daily_pnl": acct.daily_pnl,
            "daily_pnl_pct": acct.daily_pnl_pct, "fx_to_usd": acct.fx_to_usd,
            "net_liq_usd": acct.net_liq_usd, "init_margin": acct.init_margin,
            "maint_margin": acct.maint_margin,
        }

    @app.get("/api/account/history")
    async def api_account_history(range: str = "1d"):
        limits = {"1d": 96, "1w": 700, "all": 5000}
        n = limits.get(range, 600)
        async with session_scope() as s:
            rows = (await s.execute(
                select(AccountSnapshot).order_by(AccountSnapshot.ts.desc()).limit(n)
            )).scalars().all()
        rows = list(reversed(rows))
        return {"points": [{"ts": r.ts.isoformat(), "net_liq": r.net_liq, "daily_pnl": r.daily_pnl,
                            "currency": r.currency} for r in rows]}

    @app.get("/api/positions")
    async def api_positions():
        positions = await app_state.get_positions()
        return {"positions": [_pos_dict(p) for p in positions if abs(p.qty) > 0]}

    @app.post("/api/positions/{ticker}/close")
    async def api_close_position(ticker: str, request: Request):
        body = await _json(request)
        pct = int(body.get("pct", 100))
        if pct not in (25, 50, 100):
            raise HTTPException(400, "pct must be 25, 50 or 100")
        if app_state.cfg.mode in ("off", "dry_run"):
            return {"ok": True, "dry_run": True, "message": f"DRY RUN: would sell {pct}% of {ticker}"}
        try:
            intent = await app_state.orders.close_position(ticker.upper(), pct)
            return {"ok": True, "status": intent.status, "qty": intent.qty}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/positions/close-all")
    async def api_close_all():
        if app_state.cfg.mode in ("off", "dry_run"):
            return {"ok": True, "dry_run": True, "message": "DRY RUN: would close all positions"}
        results = await app_state.orders.close_all()
        return {"ok": True, "closed": [r.ticker for r in results]}

    @app.put("/api/positions/{ticker}/stops")
    async def api_edit_stops(ticker: str, request: Request):
        body = await _json(request)
        sl = body.get("sl")
        tp = body.get("tp")
        ticker = ticker.upper()
        positions = await app_state.get_positions()
        pos = next((p for p in positions if p.ticker == ticker), None)
        if pos is None:
            raise HTTPException(404, "position not found")
        last = pos.last or pos.avg_cost
        if sl is not None and float(sl) >= last:
            raise HTTPException(400, "stop-loss must be below the last price")
        if tp is not None and float(tp) <= last:
            raise HTTPException(400, "take-profit must be above the last price")
        if app_state.cfg.mode in ("off", "dry_run"):
            return {"ok": True, "dry_run": True, "message": "DRY RUN: stops updated in the UI only"}
        result = await app_state.orders.modify_stops(
            ticker, float(sl) if sl is not None else None, float(tp) if tp is not None else None
        )
        if not result.get("ok"):
            raise HTTPException(400, result.get("error", "failed"))
        return result

    @app.post("/api/positions/{ticker}/reprotect")
    async def api_reprotect(ticker: str):
        if app_state.cfg.mode in ("off", "dry_run"):
            return {"ok": True, "dry_run": True}
        result = await app_state.orders.reprotect(ticker.upper())
        if not result.get("ok"):
            raise HTTPException(400, result.get("error", "failed"))
        return result

    @app.get("/api/orders")
    async def api_orders():
        async with session_scope() as s:
            rows = (await s.execute(select(OrderRow).order_by(OrderRow.id.desc()).limit(300))).scalars().all()
            intents = (await s.execute(
                select(OrderIntentRow).order_by(OrderIntentRow.id.desc()).limit(120)
            )).scalars().all()
        return {"orders": [
            {"id": r.id, "ib_order_id": r.ib_order_id, "ticker": r.ticker, "action": r.action,
             "qty": r.qty, "type": r.order_type,
             "limit": None if (r.limit_price or 0) >= 1e30 else r.limit_price,
             "aux": None if (r.aux_price or 0) >= 1e30 else r.aux_price,
             "status": r.status, "filled": r.filled, "avg_fill": r.avg_fill_price,
             "role": r.role, "ref": r.order_ref,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ],
        "intents": [
            {"id": i.id, "run_id": i.run_id, "ticker": i.ticker, "side": i.side,
             "qty": i.qty, "type": i.order_type,
             "limit": None if (i.limit_price or 0) >= 1e30 else i.limit_price,
             "sl": i.planned_sl, "tp": i.planned_tp, "status": i.status,
             "reason": i.reason, "ref": i.order_ref,
             "created_at": i.created_at.isoformat() if i.created_at else None}
            for i in intents
        ],
        "pending": [i.id for i in intents if i.status == "awaiting_approval"]}

    @app.post("/api/orders/{order_id}/cancel")
    async def api_cancel_order(order_id: int):
        ok = await app_state.orders.cancel_order(order_id)
        if not ok:
            raise HTTPException(404, "order not found at IB")
        return {"ok": True}

    @app.get("/api/trades")
    async def api_trades(limit: int = 200):
        async with session_scope() as s:
            rows = (await s.execute(
                select(Trade).order_by(Trade.created_at.desc()).limit(min(limit, 2000))
            )).scalars().all()
        return {"trades": [
            {"id": r.id, "ticker": r.ticker, "qty": r.qty, "entry_ts": r.entry_ts.isoformat() if r.entry_ts else None,
             "entry_price": r.entry_price, "exit_ts": r.exit_ts.isoformat() if r.exit_ts else None,
             "exit_price": r.exit_price, "pnl": r.pnl, "pnl_pct": r.pnl_pct,
             "r_multiple": r.r_multiple, "exit_reason": r.exit_reason, "hold_bars": r.hold_bars,
             "commission": r.commission, "snapshot": r.entry_snapshot}
            for r in rows
        ]}

    @app.get("/api/trades/export.csv")
    async def api_trades_csv():
        async with session_scope() as s:
            rows = (await s.execute(select(Trade).order_by(Trade.created_at.desc()))).scalars().all()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "ticker", "qty", "entry_ts", "entry_price", "exit_ts", "exit_price",
                    "pnl", "pnl_pct", "r_multiple", "exit_reason", "hold_bars", "commission"])
        for r in rows:
            w.writerow([r.id, r.ticker, r.qty, r.entry_ts, r.entry_price, r.exit_ts, r.exit_price,
                        round(r.pnl, 4), round(r.pnl_pct, 4), round(r.r_multiple, 4),
                        r.exit_reason, r.hold_bars, r.commission])
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=trades.csv"},
        )

    @app.get("/api/pipeline/runs")
    async def api_runs(limit: int = 40):
        async with session_scope() as s:
            rows = (await s.execute(select(Run).order_by(Run.id.desc()).limit(limit))).scalars().all()
        return {"runs": [_run_dict(r) for r in rows]}

    @app.get("/api/pipeline/runs/{run_id}")
    async def api_run(run_id: int):
        async with session_scope() as s:
            run = await s.get(Run, run_id)
            if run is None:
                raise HTTPException(404, "run not found")
            events = (await s.execute(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.id)
            )).scalars().all()
        return {**_run_dict(run), "events": [
            {"ts": e.ts.isoformat(), "stage": e.stage, "level": e.level, "message": e.message}
            for e in events
        ]}

    @app.post("/api/pipeline/run")
    async def api_run_now(request: Request):
        body = await _json(request)
        if app_state.pipeline.running:
            raise HTTPException(409, "a run is already in progress")
        asyncio.create_task(app_state.pipeline.run(trigger=str(body.get("trigger", "manual")), force=True))
        return {"ok": True, "message": "run started"}

    @app.post("/api/pipeline/pause")
    async def api_pause():
        app_state.pause()
        return {"ok": True, "paused": True}

    @app.post("/api/pipeline/resume")
    async def api_resume():
        app_state.resume()
        return {"ok": True, "paused": False}

    @app.get("/api/universe")
    async def api_universe(run_id: int | None = None):
        async with session_scope() as s:
            if run_id is None:
                res = await s.execute(select(func.max(Run.id)))
                run_id = res.scalar()
            rows = (await s.execute(
                select(UniverseSnapshot).where(UniverseSnapshot.run_id == run_id)
                .order_by(UniverseSnapshot.rank)
            )).scalars().all()
        return {"run_id": run_id, "universe": [
            {"ticker": r.ticker, "price": r.price, "change_pct": r.change_pct,
             "rel_volume": r.rel_volume, "dollar_volume": r.dollar_volume, "sector": r.sector,
             "score": r.screen_score, "rank": r.rank, "is_held": r.is_held, "stale": r.stale}
            for r in rows
        ]}

    @app.get("/api/news")
    async def api_news(run_id: int | None = None, ticker: str | None = None, limit: int = 100):
        async with session_scope() as s:
            q = select(NewsItem).order_by(NewsItem.published_at.desc()).limit(min(limit, 500))
            if run_id:
                q = q.where(NewsItem.run_id == run_id)
            if ticker:
                q = q.where(NewsItem.ticker == ticker.upper())
            rows = (await s.execute(q)).scalars().all()
        return {"news": [
            {"id": r.id, "ticker": r.ticker, "title": r.title, "source": r.source,
             "source_type": r.source_type, "url": r.url,
             "published_at": r.published_at.isoformat(),
             "first_seen_at": r.first_seen_at.isoformat(),
             "relevance": r.relevance, "cluster_size": r.cluster_size,
             "p_pos": r.p_pos, "p_neg": r.p_neg, "p_neu": r.p_neu, "label": r.label}
            for r in rows
        ]}

    @app.get("/api/decisions")
    async def api_decisions(run_id: int | None = None):
        async with session_scope() as s:
            if run_id is None:
                res = await s.execute(select(func.max(Run.id)))
                run_id = res.scalar()
            rows = (await s.execute(select(Signal).where(Signal.run_id == run_id))).scalars().all()
            preds = (await s.execute(select(Prediction).where(Prediction.run_id == run_id))).scalars().all()
        pmap = {p.ticker: p for p in preds}
        out = []
        for r in rows:
            p = pmap.get(r.ticker)
            out.append({
                "ticker": r.ticker, "action": r.action, "reason_codes": r.reason_codes,
                "rank": r.rank, "rank_score": r.rank_score, "planned_qty": r.planned_qty,
                "planned_sl": r.planned_sl, "planned_tp": r.planned_tp, "detail": r.detail,
                "p_win": p.p_win if p else None, "expected_r": p.expected_r if p else None,
                "vol_hat": p.vol_hat if p else None, "p_entry": p.p_entry if p else None,
                "disagreement": p.disagreement if p else None, "ml_score": p.ml_score if p else None,
                "shap_top3": p.shap_top3 if p else None,
            })
        return {"run_id": run_id, "decisions": out}

    @app.get("/api/ticker/{ticker}")
    async def api_ticker(ticker: str, bars: int = 160):
        ticker = ticker.upper()
        df = app_state.bars.cached_bars(ticker)
        if df.empty:
            raise HTTPException(404, "no bars cached")
        df = df.tail(bars)
        candles = [
            {"time": int(ts.timestamp()), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close), "volume": float(r.volume)}
            for ts, r in df.iterrows()
        ]
        from tradebot.ingest.market.indicators import compute_features

        daily = app_state.bars.cached_bars(ticker, 1440, "daily")
        feats = compute_features(df, app_state.cfg.data.bar_minutes, daily=daily)
        last = feats.iloc[-1].dropna().to_dict() if not feats.empty else {}
        async with session_scope() as s:
            news = (await s.execute(
                select(NewsItem).where(NewsItem.ticker == ticker)
                .order_by(NewsItem.published_at.desc()).limit(12)
            )).scalars().all()
            pred = (await s.execute(
                select(Prediction).where(Prediction.ticker == ticker)
                .order_by(Prediction.id.desc()).limit(1)
            )).scalars().first()
        return {
            "ticker": ticker, "candles": candles,
            "features": {k: (None if v != v else round(float(v), 5)) for k, v in last.items()},
            "news": [{"title": n.title, "source": n.source, "url": n.url,
                      "published_at": n.published_at.isoformat(), "label": n.label,
                      "p_pos": n.p_pos, "p_neg": n.p_neg} for n in news],
            "prediction": None if pred is None else {
                "p_win": pred.p_win, "expected_r": pred.expected_r, "p_entry": pred.p_entry,
                "vol_hat": pred.vol_hat, "disagreement": pred.disagreement,
                "ml_score": pred.ml_score, "shap_top3": pred.shap_top3,
                "model_id": pred.model_id, "bar_ts": pred.bar_ts.isoformat(),
            },
        }

    @app.get("/api/settings")
    async def api_settings():
        return app_state.cfg.model_dump(mode="json")

    @app.get("/api/settings/schema")
    async def api_settings_schema():
        return {"fields": flatten_schema(AppConfig)}

    @app.put("/api/settings")
    async def api_settings_update(request: Request):
        body = await _json(request)
        overrides = await state_get("settings_overrides", {}) or {}
        changed = []
        for key, value in (body.get("values") or {}).items():
            old = get_path(app_state.cfg, key)
            if old == value:
                continue
            overrides[key] = value
            changed.append((key, old, value))
        # validate the full config *before* persisting anything
        from tradebot.settings import AppConfig as _Cfg, set_path as _set

        candidate = app_state.cfg.model_dump()
        for key, value in overrides.items():
            _set(candidate, key, value)
        try:
            _Cfg(**candidate)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"invalid settings: {exc}") from exc
        try:
            await app_state.reload_config(overrides)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"invalid settings: {exc}") from exc
        async with session_scope() as s:
            for key, old, value in changed:
                s.add(SettingsAudit(key=key, old_value=old, new_value=value, source="ui"))
        if body.get("reset"):
            await state_set("settings_overrides", {})
            await app_state.apply_db_overrides()
        return {"ok": True, "changed": len(changed), "values": app_state.cfg.model_dump(mode="json")}

    @app.post("/api/settings/reset")
    async def api_settings_reset():
        await state_set("settings_overrides", {})
        await app_state.apply_db_overrides()
        return {"ok": True}

    @app.get("/api/settings/audit")
    async def api_settings_audit():
        async with session_scope() as s:
            rows = (await s.execute(
                select(SettingsAudit).order_by(SettingsAudit.id.desc()).limit(100)
            )).scalars().all()
        return {"audit": [
            {"ts": r.ts.isoformat(), "key": r.key, "old": r.old_value, "new": r.new_value,
             "source": r.source} for r in rows
        ]}

    @app.post("/api/mode")
    async def api_set_mode(request: Request):
        body = await _json(request)
        try:
            return await app_state.set_mode(str(body.get("mode", "")).lower())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/killswitch")
    async def api_killswitch():
        return await app_state.kill_switch()

    @app.post("/api/breaker/rearm")
    async def api_rearm():
        return await app_state.rearm()

    @app.post("/api/approvals/{intent_id}/{action}")
    async def api_approve(intent_id: int, action: str):
        if action not in ("approve", "reject"):
            raise HTTPException(400, "bad action")
        from tradebot.execution.order_manager import as_utc
        from tradebot.core.clock import now_utc as _now

        ttl = int(app_state.cfg.approval_ttl_minutes)
        max_drift = float(app_state.cfg.execution.approval_max_drift_pct)
        snap: dict = {}
        async with session_scope() as s:
            row = await s.get(OrderIntentRow, intent_id)
            if row is None:
                raise HTTPException(404, "intent not found")
            if row.status != "awaiting_approval":
                raise HTTPException(409, f"intent is {row.status}")
            if action == "reject":
                row.status = "rejected"
                return {"ok": True, "status": "rejected"}

            # An approval is only valid while the decision is still current.
            created = as_utc(row.created_at)
            age_min = ((_now() - created).total_seconds() / 60) if created else 0.0
            if ttl > 0 and age_min > ttl:
                row.status = "expired"
                expired_msg = (
                    f"expired: this order is {age_min:.0f} min old (limit {ttl} min). "
                    "Run the pipeline again for a fresh decision."
                )
            else:
                expired_msg = ""
            snap = {
                "run_id": row.run_id, "ticker": row.ticker, "side": row.side,
                "qty": row.qty, "order_type": row.order_type,
                "limit_price": row.limit_price, "reason": row.reason,
                "order_ref": row.order_ref, "planned_sl": row.planned_sl,
                "planned_tp": row.planned_tp, "meta": row.meta or {},
            }
        if expired_msg:
            raise HTTPException(409, expired_msg)

        # The market moves while an approval sits in the queue. A *marketable*
        # limit must be priced relative to the live quote, otherwise it simply
        # never crosses; anything else is rejected once it has drifted too far.
        if snap["side"] == "BUY":
            last = None
            try:
                q = await app_state.quotes.get_quotes([snap["ticker"]])
                qq = q.get(snap["ticker"])
                last = float(qq.price) if qq and qq.price else None
            except Exception:  # noqa: BLE001
                last = None
            if last and last > 0:
                ref = float(snap["limit_price"] or last)
                drift = (ref - last) / last * 100.0
                if abs(drift) > max_drift:
                    async with session_scope() as s:
                        r = await s.get(OrderIntentRow, intent_id)
                        if r is not None:
                            r.status = "expired"
                    raise HTTPException(
                        409,
                        f"price moved {drift:+.2f}% since this order was created "
                        f"(limit {ref:.2f} vs last {last:.2f}). "
                        "Run the pipeline again.",
                    )
                if snap["order_type"] == "LMT_MARKETABLE":
                    buf = app_state.cfg.execution.slippage_buffer_bps / 10_000
                    snap["limit_price"] = round(last * (1 + buf), 4)

        from tradebot.core.types import OrderIntent

        intent = OrderIntent(
            run_id=snap["run_id"], ticker=snap["ticker"], side=snap["side"],
            qty=snap["qty"], order_type=snap["order_type"],
            limit_price=snap["limit_price"], reason=snap["reason"],
            order_ref=snap["order_ref"] + ":approved", source="manual",
            planned_sl=snap["planned_sl"], planned_tp=snap["planned_tp"],
            meta=snap["meta"],
        )
        await app_state.orders.submit(intent)

        # Record what actually happened - never claim "approved" for a rejection.
        final = intent.status or "submitted"
        stored = final if final not in ("submitted", "pending", "") else "approved"
        async with session_scope() as s:
            r2 = await s.get(OrderIntentRow, intent_id)
            if r2 is not None:
                r2.status = stored
        if final in ("rejected", "blocked_live", "duplicate", "unfilled"):
            friendly = {
                "unfilled": "the limit did not fill in time and was cancelled — "
                            "the price moved away from the decision",
                "rejected": "rejected by the order manager",
                "blocked_live": "live trading is disabled",
                "duplicate": "an identical order already exists",
            }.get(final, final)
            raise HTTPException(409, friendly)
        return {
            "ok": True,
            "status": stored,
            "queued": final == "queued",
            "order_status": final,
        }

    # ------------------------------------------------------------- backtests
    @app.post("/api/backtests")
    async def api_backtest(request: Request):
        body = await _json(request)
        from tradebot.backtest.engine import run_backtest

        try:
            bt_id = await run_backtest(app_state, body)
            return {"ok": True, "id": bt_id}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/backtests")
    async def api_backtests():
        async with session_scope() as s:
            rows = (await s.execute(select(Backtest).order_by(Backtest.id.desc()).limit(50))).scalars().all()
        return {"backtests": [{"id": r.id, "status": r.status, "metrics": r.metrics,
                               "created_at": r.created_at.isoformat(),
                               "config": r.config} for r in rows]}

    @app.get("/api/backtests/{bt_id}")
    async def api_backtest_detail(bt_id: int):
        async with session_scope() as s:
            r = await s.get(Backtest, bt_id)
            if r is None:
                raise HTTPException(404, "not found")
        return {"id": r.id, "status": r.status, "metrics": r.metrics, "config": r.config,
                "equity_curve": r.equity_curve, "trades": r.trades, "error": r.error}

    # ---------------------------------------------------------------- models
    @app.get("/api/models")
    async def api_models():
        records = await registry.list_records(50)
        return {"models": [
            {"model_id": r.model_id, "status": r.status, "created_at": r.created_at.isoformat(),
             "rows": r.rows, "features": r.features, "metrics": r.metrics, "notes": r.notes}
            for r in records
        ]}

    @app.post("/api/models/train")
    async def api_train(request: Request):
        body = await _json(request)
        if getattr(app_state, "_training", False):
            raise HTTPException(409, "training already running")
        asyncio.create_task(_train_task(app_state, bool(body.get("promote", True))))
        return {"ok": True, "message": "training started"}

    @app.post("/api/models/{model_id}/promote")
    async def api_promote(model_id: str):
        if not registry.exists(model_id):
            raise HTTPException(404, "model not found")
        await registry.promote(model_id)
        await app_state.predictor.load()
        return {"ok": True}

    # ------------------------------------------------------------------- SSE
    @app.get("/api/events")
    async def api_events(request: Request):
        async def stream():
            q = await bus.subscribe()
            try:
                yield {"event": "hello", "data": json.dumps({"ok": True})}
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                        yield {"event": ev.topic, "data": json.dumps(ev.as_dict(), default=str)}
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": "{}"}
            finally:
                await bus.unsubscribe(q)

        return EventSourceResponse(stream())

    @app.get("/api/health")
    async def api_health():
        return {"ok": True, "ts": now_utc().isoformat(), "mode": app_state.cfg.mode}

    return app


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _pos_dict(p) -> dict:
    return {
        "ticker": p.ticker, "qty": p.qty, "avg_cost": p.avg_cost, "last": p.last,
        "market_value": p.market_value, "unrealized": p.unrealized_pnl,
        "unrealized_pct": p.unrealized_pct, "unrealized_r": p.unrealized_r,
        "sl": p.sl, "tp": p.tp, "bars_held": p.bars_held, "trail_active": p.trail_active,
        "sector": p.sector, "external": p.external, "status": p.status,
        "entry_ts": p.entry_ts.isoformat() if p.entry_ts else None,
        "ib_stop_id": p.ib_stop_id, "ib_tp_id": p.ib_tp_id,
    }


def _run_dict(r: Run) -> dict:
    return {
        "id": r.id, "started_at": r.started_at.isoformat(), "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "status": r.status, "trigger": r.trigger, "mode": r.mode, "duration_ms": r.duration_ms,
        "funnel": r.funnel, "stage_timings": r.stage_timings, "error": r.error,
        "model_id": r.model_id, "summary": r.summary,
    }


async def _train_task(app_state: App, promote: bool) -> None:
    app_state._training = True
    try:
        from tradebot.features.builder import build_panel
        from tradebot.models.train import train_and_register

        cfg = app_state.cfg
        positions = await app_state.get_positions()
        held = [p.ticker for p in positions if p.qty > 0]
        rows = await app_state.universe.get(held=held, force=True)
        tickers = [r.ticker for r in rows]
        await app_state.bars.get_bars(tickers)
        await app_state.bars.get_daily(tickers + cfg.universe.benchmarks + ["^VIX"])
        bars = {t: app_state.bars.cached_bars(t) for t in tickers}
        daily = {t: app_state.bars.cached_bars(t, 1440, "daily") for t in tickers}
        spy = app_state.bars.cached_bars(cfg.universe.benchmarks[0], 1440, "daily")
        vix = app_state.bars.cached_bars("^VIX", 1440, "daily")
        panel = await asyncio.to_thread(build_panel, bars, cfg, daily, spy, vix)
        rep = await train_and_register(panel, cfg, promote_if_better=promote)
        await app_state.predictor.load()
        await app_state.alert("info", "training", f"model {rep.model_id} trained (AUC {rep.metrics.get('auc', 0):.3f})")
        await bus.publish("model.trained", model_id=rep.model_id, metrics=rep.metrics)
    except Exception as exc:  # noqa: BLE001
        log.exception("training failed")
        await app_state.alert("error", "training", f"training failed: {exc}")
    finally:
        app_state._training = False
