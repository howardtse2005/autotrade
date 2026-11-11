"""Queued (outside-RTH) entry lifecycle: dedupe, re-price, cash ceiling."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tradebot.core.types import OrderIntent
from tradebot.execution.ib_compat import LimitOrder, StopOrder
from tradebot.execution.order_manager import OrderManager


class FakeTrade:
    def __init__(self, order, symbol: str) -> None:
        self.order = order
        self.contract = SimpleNamespace(symbol=symbol)
        self.orderStatus = SimpleNamespace(status="Submitted", filled=0.0, avgFillPrice=0.0)


class FakeIB:
    def __init__(self) -> None:
        self.connected = True
        self.is_connected = True
        self.market_data_ok = True
        self.account = "DU"
        self.orders: list[FakeTrade] = []
        self.placed: list = []
        self.cancelled: list[int] = []
        self.ib = SimpleNamespace(
            placeOrder=self._place,
            cancelOrder=self._cancel,
            bracketOrder=self._bracket,
        )

    def positions(self):
        return []

    def _place(self, contract, order):
        self.placed.append(order)
        return SimpleNamespace(order=order, contract=contract,
                               orderStatus=SimpleNamespace(status="Submitted", filled=0.0,
                                                           avgFillPrice=0.0))

    def _cancel(self, order):
        self.cancelled.append(order.orderId)
        self.orders = [t for t in self.orders if t.order.orderId != order.orderId]

    def _bracket(self, action, qty, limit_px, tp, sl):
        parent = LimitOrder(action, qty, limit_px)
        take = LimitOrder("SELL", qty, tp)
        stop = StopOrder("SELL", qty, sl)
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
        return list(self.orders)

    def round_to_tick(self, ticker, price):
        return round(price, 2)

    # helpers -------------------------------------------------------------
    def add_bracket(self, oid: int, ticker: str, qty: float, limit: float,
                    sl: float, tp: float, run: int = 1) -> None:
        ref = f"{run}:{ticker}:BUY"
        parent = LimitOrder("BUY", qty, limit)
        parent.orderId, parent.orderRef = oid, ref
        take = LimitOrder("SELL", qty, tp)
        take.orderId, take.orderRef = oid + 1, f"{ref}:tp"
        stop = StopOrder("SELL", qty, sl)
        stop.orderId, stop.orderRef = oid + 2, f"{ref}:sl"
        for o in (parent, take, stop):
            self.orders.append(FakeTrade(o, ticker))


@pytest.fixture()
def make_manager():
    from tradebot.db.session import init_db
    from tradebot.settings import load_config

    asyncio.run(init_db())

    def _make():
        cfg = load_config()
        cfg.mode = "paper"
        cfg.execution.outside_rth = True
        cfg.execution.queued_reprice_band_pct = 0.5
        cfg.risk.cash_buffer_pct = 0.0
        fake = FakeIB()
        return OrderManager(cfg, SimpleNamespace(ALLOW_LIVE=0, IB_PORT=4002), fake), fake

    return _make


def _intent(ticker: str, qty: int, limit: float, sl: float, tp: float,
            run: int = 9) -> OrderIntent:
    return OrderIntent(
        run_id=run, ticker=ticker, side="BUY", qty=qty, order_type="LMT_QUEUED",
        limit_price=limit, planned_sl=sl, planned_tp=tp,
        order_ref=f"{run}:{ticker}:BUY", source="auto", meta={"queued": True},
    )


def test_duplicate_brackets_collapse_to_the_newest(make_manager):
    mgr, fake = make_manager()
    fake.add_bracket(100, "AAA", 10, 50.0, 49.0, 52.0, run=1)
    fake.add_bracket(200, "AAA", 10, 50.0, 49.0, 52.0, run=2)

    res = asyncio.run(mgr.sync_queued_entries([_intent("AAA", 10, 50.0, 49.0, 52.0)]))

    assert res["cancelled"] == 1
    live = [t.order.orderId for t in fake.orders]
    assert 200 in live and 201 in live and 202 in live, "newest bracket survives"
    assert 100 not in live, "older duplicate cancelled"


def test_bracket_is_cancelled_when_the_decision_drops_it(make_manager):
    mgr, fake = make_manager()
    fake.add_bracket(100, "AAA", 10, 50.0, 49.0, 52.0)
    fake.add_bracket(200, "BBB", 10, 20.0, 19.5, 21.0)

    res = asyncio.run(mgr.sync_queued_entries([_intent("BBB", 10, 20.0, 19.5, 21.0)]))

    assert res["cancelled"] == 1
    assert [t.contract.symbol for t in fake.orders] == ["BBB"] * 3


def test_qty_and_limit_are_modified_in_place(make_manager):
    mgr, fake = make_manager()
    fake.add_bracket(100, "AAA", 10, 50.0, 49.0, 52.0)

    res = asyncio.run(mgr.sync_queued_entries([_intent("AAA", 4, 51.0, 50.0, 53.0)]))

    assert res["modified"] == 1
    parent = next(t for t in fake.orders if t.order.orderId == 100)
    assert parent.order.totalQuantity == 4
    assert parent.order.lmtPrice == pytest.approx(51.0)
    kids = [t for t in fake.orders if t.order.orderId in (101, 102)]
    assert all(k.order.totalQuantity == 4 for k in kids), "children resized too"
    stop = next(t for t in kids if t.order.orderRef.endswith(":sl"))
    take = next(t for t in kids if t.order.orderRef.endswith(":tp"))
    assert stop.order.auxPrice == pytest.approx(50.0)
    assert take.order.lmtPrice == pytest.approx(53.0)


def test_small_price_move_is_left_alone(make_manager):
    """Inside the re-price band the order is kept - no churn every 15 minutes."""
    mgr, fake = make_manager()
    fake.add_bracket(100, "AAA", 10, 50.0, 49.0, 52.0)

    res = asyncio.run(mgr.sync_queued_entries([_intent("AAA", 10, 50.1, 49.0, 52.0)]))

    assert res["modified"] == 0 and res["kept"] == 1
    assert not fake.placed


def test_queue_is_trimmed_to_available_cash(make_manager):
    mgr, fake = make_manager()
    # 4 brackets x 1000 @ 50 = 200k of notional against 100k of cash
    for i, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
        fake.add_bracket(100 + i * 10, ticker, 1000, 50.0, 49.0, 52.0)
    desired = [_intent(t, 1000, 50.0, 49.0, 52.0) for t in ("AAA", "BBB", "CCC", "DDD")]

    res = asyncio.run(mgr.sync_queued_entries(desired, available_usd=100_000))

    assert res["trimmed_for_cash"] == 2, "must cancel until it fits"
    assert res["queued_notional"] <= 100_000
    assert len({t.contract.symbol for t in fake.orders}) == 2


def test_no_decision_keeps_the_newest_max_positions(make_manager):
    """An empty decision must not wipe a healthy queue."""
    mgr, fake = make_manager()
    for i, ticker in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        fake.add_bracket(100 + i * 10, ticker, 10, 50.0, 49.0, 52.0)

    res = asyncio.run(mgr.sync_queued_entries([], available_usd=1_000_000))

    assert res["cancelled"] == 1, "6 queued > max_positions (5)"
    assert len({t.contract.symbol for t in fake.orders}) == 5


def test_second_submit_for_the_same_name_is_suppressed(make_manager):
    mgr, fake = make_manager()
    fake.add_bracket(100, "AAA", 10, 50.0, 49.0, 52.0)

    intent = _intent("AAA", 10, 50.0, 49.0, 52.0, run=10)
    asyncio.run(mgr.submit(intent))

    assert intent.status == "duplicate"
    assert not fake.placed, "no second order may be sent"


def test_cancelled_bracket_closes_its_intent(make_manager):
    """A cancelled bracket must not leave the intent 'queued' forever."""
    mgr, fake = make_manager()
    from tradebot.db.session import session_scope
    from tradebot.db.models import OrderIntentRow
    from sqlalchemy import select

    async def scenario():
        fake.add_bracket(100, "AAA", 10, 50.0, 49.0, 52.0, run=7)
        fake.add_bracket(200, "BBB", 10, 20.0, 19.5, 21.0, run=7)
        async with session_scope() as s:
            for t in ("AAA", "BBB"):
                s.add(OrderIntentRow(
                    run_id=7, ticker=t, side="BUY", qty=10, order_type="LMT_QUEUED",
                    order_ref=f"7:{t}:BUY", status="queued",
                ))
        # the decision now only wants BBB
        await mgr.sync_queued_entries([_intent("BBB", 10, 20.0, 19.5, 21.0, run=7)])
        async with session_scope() as s:
            row = (await s.execute(
                select(OrderIntentRow).where(OrderIntentRow.order_ref == "7:AAA:BUY")
            )).scalars().first()
            assert row.status == "cancelled"
            assert (row.meta or {}).get("error")

    asyncio.run(scenario())


def test_partial_parent_fill_resizes_the_children(make_manager):
    """Filled 40 of 100: the stop/target must cover 40 shares, not 100."""
    mgr, fake = make_manager()
    from tradebot.db.session import session_scope
    from tradebot.db.models import PositionRow
    from sqlalchemy import select

    async def scenario():
        parent = LimitOrder("BUY", 100, 50.0)
        parent.orderId, parent.orderRef = 500, "9:AAA:BUY"
        stop = StopOrder("SELL", 100, 49.0)
        stop.orderId, stop.orderRef, stop.parentId = 501, "9:AAA:BUY:sl", 500
        take = LimitOrder("SELL", 100, 52.0)
        take.orderId, take.orderRef, take.parentId = 502, "9:AAA:BUY:tp", 500
        fake.orders = [FakeTrade(o, "AAA") for o in (stop, take)]
        trade = SimpleNamespace(
            order=SimpleNamespace(orderId=500, orderRef="9:AAA:BUY", action="BUY"),
            contract=SimpleNamespace(symbol="AAA"),
            orderStatus=SimpleNamespace(status="Cancelled", filled=40.0, avgFillPrice=50.1),
        )
        await mgr._on_parent_fill(trade)
        assert all(t.order.totalQuantity == 40 for t in fake.orders), \
            "protective orders must be resized to the filled quantity"
        async with session_scope() as s:
            row = (await s.execute(
                select(PositionRow).where(PositionRow.ticker == "AAA")
            )).scalars().first()
            assert row is not None and row.qty == 40

    asyncio.run(scenario())


def test_queued_parent_is_gtc(make_manager):
    """A DAY parent would be cancelled by IB's rollover before the next open."""
    mgr, fake = make_manager()
    intent = _intent("AAA", 10, 50.0, 49.0, 52.0, run=11)
    intent.order_type = "LMT_QUEUED"
    intent.meta = {"queued": True, "sl_dist": 1.0, "tp_dist": 2.0}
    asyncio.run(mgr.submit(intent))
    assert intent.status == "queued"
    assert fake.placed, "bracket must be placed"
    assert all(o.tif == "GTC" for o in fake.placed), "parent and children must be GTC"


def test_intents_without_a_working_order_become_stale(make_manager):
    mgr, fake = make_manager()
    from tradebot.db.session import session_scope
    from tradebot.db.models import OrderIntentRow
    from sqlalchemy import select

    async def scenario():
        fake.add_bracket(300, "BBB", 10, 20.0, 19.5, 21.0, run=12)
        async with session_scope() as s:
            s.add(OrderIntentRow(run_id=12, ticker="BBB", side="BUY", qty=10,
                                 order_type="LMT_QUEUED", order_ref="12:BBB:BUY",
                                 status="queued"))
            s.add(OrderIntentRow(run_id=12, ticker="AAA", side="BUY", qty=10,
                                 order_type="LMT_QUEUED", order_ref="12:AAA:BUY",
                                 status="queued"))
        await mgr.reconcile_intents()
        async with session_scope() as s:
            rows = {r.ticker: r.status for r in (await s.execute(
                select(OrderIntentRow).where(OrderIntentRow.run_id == 12)
            )).scalars().all()}
        assert rows == {"BBB": "queued", "AAA": "stale"}

    asyncio.run(scenario())


def test_a_stale_intent_does_not_block_a_new_order(make_manager):
    """The deadlock: a 'queued' row with no order must not veto every re-submit."""
    mgr, fake = make_manager()
    from tradebot.db.session import session_scope
    from tradebot.db.models import OrderIntentRow

    async def scenario():
        async with session_scope() as s:
            s.add(OrderIntentRow(run_id=13, ticker="AAA", side="BUY", qty=10,
                                 order_type="LMT_QUEUED", order_ref="13:AAA:BUY",
                                 status="queued"))
        intent = _intent("AAA", 10, 50.0, 49.0, 52.0, run=14)
        intent.order_type = "LMT_QUEUED"
        intent.meta = {"queued": True, "sl_dist": 1.0, "tp_dist": 2.0}
        await mgr.submit(intent)
        assert intent.status != "duplicate", "stale row must not suppress the order"
        assert fake.placed

    asyncio.run(scenario())
