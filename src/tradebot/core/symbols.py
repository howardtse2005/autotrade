"""Ticker normalization across TradingView, IBKR and Yahoo conventions."""

from __future__ import annotations

import re

_TV_PREFIX = re.compile(r"^(NASDAQ|NYSE|AMEX|NYSEARCA|ARCA|BATS|OTC):", re.I)


def canonical(tv_symbol: str) -> str:
    """'NASDAQ:AAPL' -> 'AAPL', 'NYSE:BRK.B' -> 'BRK.B'."""
    return _TV_PREFIX.sub("", (tv_symbol or "").strip()).upper()


def ib_symbol(ticker: str) -> str:
    """IBKR uses a space for share classes: 'BRK.B' -> 'BRK B'."""
    return ticker.replace(".", " ").strip().upper()


def yf_symbol(ticker: str) -> str:
    """Yahoo uses a dash for share classes: 'BRK.B' -> 'BRK-B'."""
    return ticker.replace(".", "-").replace(" ", "-").strip().upper()


def from_ib_symbol(symbol: str) -> str:
    return symbol.replace(" ", ".").strip().upper()


def is_share_class(ticker: str) -> bool:
    return "." in ticker


# Tickers that are also common English words: require a cashtag/company match.
AMBIGUOUS = {
    "A", "ALL", "ON", "IT", "NOW", "CAT", "SO", "FOR", "AT", "BE", "BY", "CAN", "CASH",
    "DD", "DO", "GO", "GOOD", "HAS", "HE", "I", "IF", "IS", "KEY", "L", "LOVE", "M",
    "MAN", "ME", "MO", "MY", "NEW", "NEXT", "NO", "ONE", "OR", "OUT", "PLAY", "PM",
    "REAL", "SEE", "SO", "T", "TELL", "TWO", "UP", "US", "V", "WELL", "WHO", "WILL",
    "YOU", "Z", "AI", "AN", "AS", "ARE", "BIG", "BUY", "CEO", "DAY", "FIT", "HOT",
    "JOB", "LIFE", "LIVE", "LOW", "OPEN", "SAFE", "SALE", "SUN", "TOP", "TRUE", "WE",
}

# Corporate suffixes stripped when deriving company-name aliases.
_SUFFIXES = [
    "incorporated", "inc", "corporation", "corp", "company", "co", "plc", "ltd",
    "limited", "holdings", "holding", "group", "the", "technologies", "technology",
    "international", "industries", "systems", "solutions", "class a", "class b",
    "class c", "common stock", "ordinary shares", "depositary shares",
]


def company_aliases(description: str) -> list[str]:
    """Derive matchable company names from a screener description."""
    if not description:
        return []
    name = description.split(" - ")[0]
    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(r"[^A-Za-z0-9&.\s]", " ", name)
    words = name.split()
    aliases: set[str] = set()
    cleaned = " ".join(words).lower()
    for suf in _SUFFIXES:
        if cleaned.endswith(" " + suf):
            cleaned = cleaned[: -(len(suf) + 1)].strip()
    if len(cleaned) >= 4:
        aliases.add(cleaned)
    if len(words) >= 2:
        two = " ".join(words[:2]).lower()
        if len(two) >= 6:
            aliases.add(two)
    return sorted(a for a in aliases if len(a) >= 4)[:4]
