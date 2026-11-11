"""Queued (outside-RTH) entries: bracket placement and deferred fill handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradebot.core.types import OrderIntent
from tradebot.execution.order_manager import OrderManager


class FakeIB:
    """Minimal stand-in for the IBClient surface used by the order manager."""

    def __init__(self) -> None:
        self.placed: list = []
        self.connected = True
        self.is_connected = True
        self.market_data_ok = True
        self.account = "DU"
        self.orders: list = []
        self.ib = SimpleNamespace(
            placeOrder=lambda contract, order: self._place(contract, order),
            bracketOrder=self._bracket,
            cancelOrder=lambda order: None,
        )

    def _place(self, contract, order):
        self.placed.append(order)
        return SimpleNamespace(
            order=order,
            contract=SimpleNamespace(symbol="ECPG"),
            orderStatus=SimpleNamespace(status="PreSubmitted", filled=0.0, avgFillPrice=0.0),
        )

    def _bracket(self, action, qty, limit_px, tp, sl):
        from tradebot.execution.ib_compat import LimitOrder, StopOrder

        parent = LimitOrder(action, qty, limit_px)
        take = LimitOrder("SELL", qty, tp)
        stop = StopOrder("SELL", qty, sl)
        parent.transmit = False
        take.transmit = False
        stop.transmit = True
        stop.parentId = take.parentId = 1
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
def manager():
    import asyncio

    from tradebot.db.session import init_db
    from tradebot.settings import load_config

    asyncio.run(init_db())
    cfg = load_config()
    cfg.mode = "paper"
    cfg.execution.outside_rth = True
    fake = FakeIB()
    return OrderManager(cfg, SimpleNamespace(ALLOW_LIVE=0, IB_PORT=4002), fake), fake


def test_queued_entry_places_a_bracket(manager):
    mgr, fake = manager
    intent = OrderIntent(
        run_id=1, ticker="ECPG", side="BUY", qty=197, order_type="LMT_QUEUED",
        limit_price=127.06, planned_sl=124.83, planned_tp=129.01,
        order_ref="1:ECPG:BUY", source="manual",
        meta={"queued": True, "sl_dist": 2.23, "tp_dist": 4.18},
    )
    import asyncio

    asyncio.run(mgr.submit(intent))
    assert intent.status == "queued"
    placed = fake.placed
    assert len(placed) == 3, "parent + stop + target must all be transmitted"
    assert placed[0].action == "BUY" and placed[0].orderType == "LMT"
    assert placed[0].lmtPrice == pytest.approx(127.06)
    refs = {o.orderRef for o in placed}
    assert refs == {"1:ECPG:BUY", "1:ECPG:BUY:sl", "1:ECPG:BUY:tp"}
    # children must survive the session, the parent is a day order
    assert placed[1].tif == "GTC" and placed[2].tif == "GTC"
    assert placed[0].outsideRth is True


def test_parent_fill_anchors_protection_to_the_fill_price(manager):
    mgr, fake = manager
    import asyncio

    from tradebot.db.session import init_db, session_scope
    from tradebot.db.models import OrderIntentRow, PositionRow
    from sqlalchemy import select

    asyncio.run(init_db())

    async def scenario():
        # isolate from other tests that also use ECPG
        from sqlalchemy import delete

        async with session_scope() as s:
            await s.execute(delete(PositionRow).where(PositionRow.ticker == "ECPG"))
        async with session_scope() as s:
            s.add(OrderIntentRow(
                run_id=1, ticker="ECPG", side="BUY", qty=197, order_ref="1:ECPG:BUY:fill",
                status="queued", meta={"sl_dist": 2.0, "tp_dist": 3.0},
            ))
        # children as they exist at IB after the parent filled
        fake.orders = [
            SimpleNamespace(
                order=SimpleNamespace(orderId=11, parentId=10, orderRef="1:ECPG:BUY:fill:sl",
                                      auxPrice=124.83, lmtPrice=0.0),
                contract=SimpleNamespace(symbol="ECPG"),
                orderStatus=SimpleNamespace(status="PreSubmitted"),
            ),
            SimpleNamespace(
                order=SimpleNamespace(orderId=12, parentId=10, orderRef="1:ECPG:BUY:fill:tp",
                                      auxPrice=0.0, lmtPrice=129.01),
                contract=SimpleNamespace(symbol="ECPG"),
                orderStatus=SimpleNamespace(status="PreSubmitted"),
            ),
        ]
        trade = SimpleNamespace(
            order=SimpleNamespace(orderId=10, orderRef="1:ECPG:BUY:fill", action="BUY"),
            contract=SimpleNamespace(symbol="ECPG"),
            orderStatus=SimpleNamespace(status="Filled", filled=197.0, avgFillPrice=130.0),
        )
        await mgr._on_parent_fill(trade)
        async with session_scope() as s:
            row = (await s.execute(
                select(PositionRow).where(PositionRow.ticker == "ECPG")
            )).scalars().first()
            assert row is not None
            assert row.qty == 197
            assert row.avg_cost == pytest.approx(130.0)
            # anchored to the fill (130 ± distances), not to the decision price
            assert row.sl == pytest.approx(128.0, abs=0.01)
            assert row.tp == pytest.approx(133.0, abs=0.01)
            assert row.ib_stop_id == 11 and row.ib_tp_id == 12

    asyncio.run(scenario())
