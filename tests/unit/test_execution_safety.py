"""Execution safety: an exit must never sell shares the account does not own."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tradebot.core.clock import now_utc
from tradebot.core.types import OrderIntent
from tradebot.execution.order_manager import OrderManager


class FakeIB:
    def __init__(self, positions: dict[str, float] | None = None) -> None:
        self.connected = True
        self.is_connected = True
        self.market_data_ok = True
        self.account = "DU"
        self._positions = dict(positions or {})
        self.placed: list = []
        self.orders: list = []
        self.ib = SimpleNamespace(
            placeOrder=lambda contract, order: self._place(contract, order),
            bracketOrder=self._bracket,
            cancelOrder=lambda order: None,
        )

    def positions(self):
        return [
            SimpleNamespace(contract=SimpleNamespace(symbol=s), position=q)
            for s, q in self._positions.items()
        ]

    def _place(self, contract, order):
        self.placed.append(order)
        return SimpleNamespace(
            order=order,
            contract=SimpleNamespace(symbol=contract.symbol),
            orderStatus=SimpleNamespace(status="Submitted", filled=0.0, avgFillPrice=0.0),
        )

    def _bracket(self, action, qty, limit_px, tp, sl):
        from tradebot.execution.ib_compat import LimitOrder, StopOrder

        parent = LimitOrder(action, qty, limit_px)
        take = LimitOrder("SELL", qty, tp)
        stop = StopOrder("SELL", qty, sl)
        parent.transmit = False
        take.transmit = False
        stop.transmit = True
        return [parent, take, stop]

    def on_order_status(self, fn):
        self._status_fn = fn

    def on_fill(self, fn):
        self._fill_fn = fn

    def on_connection(self, fn):
        pass

    async def qualify(self, ticker):
        return SimpleNamespace(symbol=ticker)

    async def open_orders(self):
        return self.orders

    def round_to_tick(self, ticker, price):
        return round(price, 2)


@pytest.fixture()
def make_manager():
    from tradebot.db.session import init_db
    from tradebot.settings import load_config

    asyncio.run(init_db())

    def _make(positions: dict[str, float]):
        cfg = load_config()
        cfg.mode = "paper"
        cfg.execution.outside_rth = True
        fake = FakeIB(positions)
        return OrderManager(cfg, SimpleNamespace(ALLOW_LIVE=0, IB_PORT=4002), fake), fake

    return _make


def _sell(ticker: str = "AAPL", qty: int = 5, ref: str = "t:AAPL:SELL") -> OrderIntent:
    return OrderIntent(
        run_id=1, ticker=ticker, side="SELL", qty=qty, order_type="MKT",
        reason="SELL_TIME", order_ref=ref, source="manual",
    )


def test_sell_is_rejected_when_the_account_is_flat(make_manager):
    """The bug that shorted the account: a stale exit into a closed position."""
    mgr, fake = make_manager({})  # flat
    intent = _sell(ref="t:AAPL:SELL:flat")
    asyncio.run(mgr.submit(intent))
    assert intent.status == "rejected"
    assert fake.placed == [], "nothing may be sent to IBKR for a flat position"


def test_sell_is_clamped_to_the_position(make_manager):
    mgr, fake = make_manager({"AAPL": 3.0})
    intent = _sell(qty=10, ref="t:AAPL:SELL:clamp")
    asyncio.run(mgr.submit(intent))
    assert intent.qty == 3, "an exit may only reduce the position, never flip it short"
    assert len(fake.placed) == 1
    assert fake.placed[0].totalQuantity == 3
    assert fake.placed[0].action == "SELL"


def test_protective_fill_closes_the_position_row(make_manager):
    """A stop fill must close the position so no later exit can sell it again."""
    mgr, _fake = make_manager({})
    from tradebot.db.session import session_scope
    from tradebot.db.models import PositionRow, Trade
    from sqlalchemy import select

    async def scenario():
        async with session_scope() as s:
            s.add(PositionRow(
                ticker="ECPG", qty=197, avg_cost=127.06, status="open",
                entry_ts=now_utc(), planned_r=2.23, sl=124.83, tp=129.64,
            ))
        trade = SimpleNamespace(
            order=SimpleNamespace(orderId=11, orderRef="1:ECPG:BUY:sl", action="SELL"),
            contract=SimpleNamespace(symbol="ECPG"),
            orderStatus=SimpleNamespace(status="Filled", filled=197.0, avgFillPrice=124.83),
        )
        await mgr._on_protective_fill(trade, "sl")
        async with session_scope() as s:
            row = (await s.execute(
                select(PositionRow).where(PositionRow.ticker == "ECPG")
            )).scalars().first()
            assert row.status == "closed"
            tr = (await s.execute(select(Trade))).scalars().first()
            assert tr is not None and tr.exit_reason == "SELL_STOP"
            assert tr.pnl == pytest.approx((124.83 - 127.06) * 197, rel=1e-6)

    asyncio.run(scenario())


def test_stale_approvals_expire(make_manager):
    """Pending approvals must not survive into a later run or past the TTL."""
    mgr, _fake = make_manager({})
    from tradebot.db.session import session_scope
    from tradebot.db.models import OrderIntentRow
    from sqlalchemy import select

    async def scenario():
        old = now_utc() - timedelta(minutes=mgr.cfg.approval_ttl_minutes + 5)
        async with session_scope() as s:
            s.add(OrderIntentRow(
                run_id=1, ticker="AAA", side="BUY", qty=1, order_type="MKT",
                order_ref="stale:1", status="awaiting_approval", created_at=old,
            ))
            s.add(OrderIntentRow(
                run_id=1, ticker="BBB", side="BUY", qty=1, order_type="MKT",
                order_ref="fresh:1", status="awaiting_approval", created_at=now_utc(),
            ))
        # a newer run supersedes the fresh one as well
        n = await mgr.expire_stale(current_run_id=2)
        assert n == 2
        async with session_scope() as s:
            rows = (await s.execute(
                select(OrderIntentRow).where(OrderIntentRow.ticker.in_(["AAA", "BBB"]))
            )).scalars().all()
            assert {r.status for r in rows} == {"expired"}

    asyncio.run(scenario())
