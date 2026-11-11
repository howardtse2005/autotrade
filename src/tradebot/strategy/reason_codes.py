"""Reason codes shown in the UI for every decision."""

from __future__ import annotations

OK_BUY = "OK_BUY"
OK_HOLD = "OK_HOLD"
WARN_NEG_EDGE = "WARN_NEG_EDGE"
WARN_QUEUED_OPEN = "WARN_QUEUED_OPEN"

SKIP_LOW_PWIN = "SKIP_LOW_PWIN"
SKIP_NEG_EDGE = "SKIP_NEG_EDGE"
SKIP_POOR_ENTRY = "SKIP_POOR_ENTRY"
SKIP_DISAGREEMENT = "SKIP_DISAGREEMENT"
SKIP_NEG_SENTIMENT = "SKIP_NEG_SENTIMENT"
SKIP_SPREAD = "SKIP_SPREAD"
SKIP_STALE_QUOTE = "SKIP_STALE_QUOTE"
SKIP_ILLIQUID = "SKIP_ILLIQUID"
SKIP_EARNINGS = "SKIP_EARNINGS"
SKIP_HALTED = "SKIP_HALTED"
SKIP_TIME_WINDOW = "SKIP_TIME_WINDOW"
SKIP_SECTOR_CAP = "SKIP_SECTOR_CAP"
SKIP_COOLDOWN = "SKIP_COOLDOWN"
SKIP_NO_SLOT = "SKIP_NO_SLOT"
SKIP_BREAKER = "SKIP_BREAKER"
SKIP_MARGIN = "SKIP_MARGIN"
SKIP_NO_DATA = "SKIP_NO_DATA"
SKIP_MODEL_UNHEALTHY = "SKIP_MODEL_UNHEALTHY"
SKIP_ALREADY_HELD = "SKIP_ALREADY_HELD"
SKIP_RISK_TOO_SMALL = "SKIP_RISK_TOO_SMALL"
SKIP_MARKET_CLOSED = "SKIP_MARKET_CLOSED"

SELL_STOP = "SELL_STOP"
SELL_TP = "SELL_TP"
SELL_TRAIL = "SELL_TRAIL"
SELL_TIME = "SELL_TIME"
SELL_EOD = "SELL_EOD"
SELL_SIGNAL_DECAY = "SELL_SIGNAL_DECAY"
SELL_SENT_SHOCK = "SELL_SENT_SHOCK"
SELL_ROTATION = "SELL_ROTATION"
SELL_BREAKER = "SELL_BREAKER"
SELL_MANUAL = "SELL_MANUAL"
SELL_KILLSWITCH = "SELL_KILLSWITCH"

LABELS: dict[str, str] = {
    OK_BUY: "Buy signal",
    OK_HOLD: "Hold",
    WARN_NEG_EDGE: "Negative E[R] (edge gate in warn mode)",
    WARN_QUEUED_OPEN: "Queued — executes at the market open",
    SKIP_LOW_PWIN: "Below p(win) threshold",
    SKIP_NEG_EDGE: "Negative expected edge",
    SKIP_POOR_ENTRY: "Poor entry timing",
    SKIP_DISAGREEMENT: "Model lenses disagree",
    SKIP_NEG_SENTIMENT: "Negative news sentiment",
    SKIP_SPREAD: "Spread too wide",
    SKIP_STALE_QUOTE: "Stale quote",
    SKIP_ILLIQUID: "Illiquid",
    SKIP_EARNINGS: "Earnings blackout",
    SKIP_HALTED: "Halted",
    SKIP_TIME_WINDOW: "Outside entry window",
    SKIP_SECTOR_CAP: "Sector cap reached",
    SKIP_COOLDOWN: "Cooldown after stop",
    SKIP_NO_SLOT: "No free slot",
    SKIP_BREAKER: "Risk breaker tripped",
    SKIP_MARGIN: "Intraday margin tight",
    SKIP_NO_DATA: "No fresh data",
    SKIP_MODEL_UNHEALTHY: "Model not ready",
    SKIP_ALREADY_HELD: "Already held",
    SKIP_RISK_TOO_SMALL: "Position too small",
    SKIP_MARKET_CLOSED: "Market closed",
    SELL_STOP: "Stop loss",
    SELL_TP: "Take profit",
    SELL_TRAIL: "Trailing stop",
    SELL_TIME: "Time stop",
    SELL_EOD: "End of day flatten",
    SELL_SIGNAL_DECAY: "Signal decayed",
    SELL_SENT_SHOCK: "Negative news shock",
    SELL_ROTATION: "Rotated to better candidate",
    SELL_BREAKER: "Risk breaker",
    SELL_MANUAL: "Manual close",
    SELL_KILLSWITCH: "Kill switch",
}


def label(code: str) -> str:
    return LABELS.get(code, code)
