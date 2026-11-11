"""Account summary -> AccountState (with FX conversion to USD)."""

from __future__ import annotations

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.types import AccountState

log = get_logger("account")

TAGS = {
    "NetLiquidation": "net_liq",
    "TotalCashValue": "cash",
    "AvailableFunds": "available_funds",
    "BuyingPower": "buying_power",
    "ExcessLiquidity": "excess_liquidity",
    "GrossPositionValue": "gross_position_value",
    "InitMarginReq": "init_margin",
    "MaintMarginReq": "maint_margin",
    "SettledCash": "settled_cash",
    "RealizedPnL": "realized_pnl",
    "UnrealizedPnL": "unrealized_pnl",
}


def _f(value) -> float:
    import math

    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return 0.0
    return v if math.isfinite(v) else 0.0


async def read_account(ib_client, quote_service=None) -> AccountState:
    """Build an AccountState from IB tags; convert to USD using an FX rate."""
    rows = await ib_client.account_summary()
    if not rows:
        return AccountState()

    base = "USD"
    for r in rows:
        if r.tag == "NetLiquidation" and r.currency:
            base = r.currency
            break

    values: dict[str, float] = {}
    for r in rows:
        key = TAGS.get(r.tag)
        if key and r.currency in (base, "USD", ""):
            # prefer the base-currency line
            if key not in values or r.currency == base:
                values[key] = _f(r.value)

    fx = 1.0
    if base != "USD" and quote_service is not None:
        try:
            fx = await quote_service.fx_to_usd(base)
        except Exception:  # noqa: BLE001
            fx = 1.0
    if base != "USD":
        for r in rows:
            if r.tag == "ExchangeRate" and r.currency == base:
                rate = _f(r.value)
                if rate > 0:
                    fx = rate

    # Prefer account-summary tags, then the live portfolio, then reqPnL.
    realized = _f(values.get("realized_pnl"))
    unrealized = _f(values.get("unrealized_pnl"))
    try:
        portfolio = ib_client.portfolio()
        if portfolio:
            unrealized = sum(_f(getattr(p, "unrealizedPNL", 0)) for p in portfolio)
            realized = sum(_f(getattr(p, "realizedPNL", 0)) for p in portfolio) or realized
    except Exception:  # noqa: BLE001
        pass
    daily = 0.0
    try:
        pnl = ib_client.ib.reqPnL(ib_client.account) if ib_client.connected else None
        if pnl is not None:
            d = _f(getattr(pnl, "dailyPnL", 0))
            r = _f(getattr(pnl, "realizedPnL", 0))
            u = _f(getattr(pnl, "unrealizedPnL", 0))
            daily = d if d else (r + u)
            if not realized:
                realized = r
            if not unrealized:
                unrealized = u
    except Exception:  # noqa: BLE001
        pass
    if not daily:
        daily = realized + unrealized

    state = AccountState(
        currency=base,
        net_liq=values.get("net_liq", 0.0),
        cash=values.get("cash", 0.0),
        available_funds=values.get("available_funds", 0.0),
        buying_power=values.get("buying_power", 0.0),
        excess_liquidity=values.get("excess_liquidity", 0.0),
        gross_position_value=values.get("gross_position_value", 0.0),
        init_margin=values.get("init_margin", 0.0),
        maint_margin=values.get("maint_margin", 0.0),
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        daily_pnl=daily,
        fx_to_usd=fx,
    )
    if state.net_liq <= 0:
        log.warning("account summary missing NetLiquidation", tags=len(rows))
    return state


def empty_state(currency: str = "USD") -> AccountState:
    return AccountState(currency=currency)


def snapshot_row(state: AccountState) -> dict:
    return {
        "ts": now_utc(),
        "currency": state.currency,
        "net_liq": state.net_liq,
        "cash": state.cash,
        "available_funds": state.available_funds,
        "buying_power": state.buying_power,
        "excess_liquidity": state.excess_liquidity,
        "gross_position_value": state.gross_position_value,
        "realized_pnl": state.realized_pnl,
        "unrealized_pnl": state.unrealized_pnl,
        "daily_pnl": state.daily_pnl,
        "fx_to_usd": state.fx_to_usd,
    }
