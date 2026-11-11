"""ib_async import shim (falls back to ib_insync if that is what is installed)."""

from __future__ import annotations

try:  # preferred: maintained successor
    from ib_async import (  # noqa: F401
        IB,
        Contract,
        ContractDetails,
        Execution,
        Fill,
        LimitOrder,
        MarketOrder,
        Order,
        OrderStatus,
        Stock,
        StopOrder,
        Trade,
        Ticker,
        util,
    )

    IB_LIBRARY = "ib_async"
except Exception:  # pragma: no cover - only if ib_async is absent
    from ib_insync import (  # type: ignore  # noqa: F401
        IB,
        Contract,
        ContractDetails,
        Execution,
        Fill,
        LimitOrder,
        MarketOrder,
        Order,
        OrderStatus,
        Stock,
        StopOrder,
        Trade,
        Ticker,
        util,
    )

    IB_LIBRARY = "ib_insync"


def make_stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    return Stock(symbol, exchange, currency)
