"""Time, market-calendar and session helpers.

All storage is UTC. The UI renders HKT. Sessions come from the XNYS calendar
(holidays and 13:00 ET half-days handled automatically, including DST).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

UTC = timezone.utc
HKT = ZoneInfo("Asia/Hong_Kong")
ET = ZoneInfo("America/New_York")

CAL_NAME = "XNYS"


def now_utc() -> datetime:
    return datetime.now(UTC)


def to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def to_hkt(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return to_utc(ts).astimezone(HKT)


def fmt_hkt(ts: datetime | None, with_date: bool = True) -> str:
    if ts is None:
        return "—"
    t = to_hkt(ts)
    return t.strftime("%d %b %H:%M") if with_date else t.strftime("%H:%M")


def fmt_hkt_full(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    return to_hkt(ts).strftime("%Y-%m-%d %H:%M:%S")


@lru_cache(maxsize=1)
def calendar():
    return xcals.get_calendar(CAL_NAME)


def _session_label(ts: datetime) -> pd.Timestamp:
    """Tz-naive session label (calendar date in ET) for a UTC timestamp."""
    return pd.Timestamp(to_utc(ts).astimezone(ET).date())


def is_session_day(d: date | datetime) -> bool:
    if isinstance(d, datetime):
        d = d.date()
    return bool(calendar().is_session(pd.Timestamp(d)))


def session_bounds(d: datetime | date) -> tuple[datetime, datetime]:
    """Open/close (UTC) of the session containing/starting at `d`."""
    label = pd.Timestamp(d.date() if isinstance(d, datetime) else d)
    cal = calendar()
    if not cal.is_session(label):
        label = cal.date_to_session(label, direction="next")
    return cal.session_open(label).to_pydatetime(), cal.session_close(label).to_pydatetime()


def previous_session_close(ts: datetime) -> datetime:
    """Close of the last *completed* session before `ts`."""
    return calendar().previous_close(pd.Timestamp(to_utc(ts))).to_pydatetime()


@dataclass(slots=True)
class MarketClock:
    now: datetime
    is_open: bool
    session_open: datetime | None
    session_close: datetime | None
    next_open: datetime
    next_close: datetime
    minutes_to_close: float | None
    minutes_since_open: float | None
    session_label: str

    @property
    def is_session(self) -> bool:
        return self.session_open is not None

    def phase(self, skip_open_min: int, no_entries_last_min: int) -> str:
        """closed | pre | open_skip | open | late | after"""
        if not self.is_open:
            return "closed"
        if self.minutes_since_open is not None and self.minutes_since_open < skip_open_min:
            return "open_skip"
        if self.minutes_to_close is not None and self.minutes_to_close <= no_entries_last_min:
            return "late"
        return "open"

    def allows_new_entries(self, skip_open_min: int, no_entries_last_min: int) -> tuple[bool, str]:
        ph = self.phase(skip_open_min, no_entries_last_min)
        if ph == "open":
            return True, ""
        return False, ph

    def should_flatten(self, minutes_before_close: int = 10) -> bool:
        return bool(
            self.is_open
            and self.minutes_to_close is not None
            and self.minutes_to_close <= minutes_before_close
        )


def market_clock(now: datetime | None = None) -> MarketClock:
    now = to_utc(now or now_utc())
    cal = calendar()
    label = _session_label(now)
    is_open = bool(cal.is_open_on_minute(pd.Timestamp(now)))
    try:
        sess = cal.date_to_session(label, direction="previous" if not is_open else "next")
        # date_to_session with direction="next" returns the session at/after label
        if cal.is_session(label):
            sess = label
    except Exception:  # noqa: BLE001
        sess = label

    s_open = s_close = None
    if cal.is_session(sess):
        s_open = cal.session_open(sess).to_pydatetime()
        s_close = cal.session_close(sess).to_pydatetime()

    next_open = cal.next_open(pd.Timestamp(now)).to_pydatetime()
    next_close = cal.next_close(pd.Timestamp(now)).to_pydatetime()

    mins_to_close = (s_close - now).total_seconds() / 60 if (is_open and s_close) else None
    mins_since_open = (now - s_open).total_seconds() / 60 if (is_open and s_open) else None

    return MarketClock(
        now=now,
        is_open=is_open,
        session_open=s_open,
        session_close=s_close,
        next_open=next_open,
        next_close=next_close,
        minutes_to_close=mins_to_close,
        minutes_since_open=mins_since_open,
        session_label=str(sess.date()),
    )


def trading_days_back(n: int, end: datetime | None = None) -> list[date]:
    cal = calendar()
    end = end or now_utc()
    label = cal.date_to_session(_session_label(end), direction="previous")
    sessions = cal.sessions_window(label, -n)
    return [s.date() for s in sessions]


def align_bar_ts(ts: datetime, minutes: int) -> datetime:
    """Floor a timestamp to the bar grid (UTC)."""
    ts = to_utc(ts)
    epoch = int(ts.timestamp())
    step = minutes * 60
    return datetime.fromtimestamp(epoch - epoch % step, UTC)


def bar_end(ts: datetime, minutes: int) -> datetime:
    return align_bar_ts(ts, minutes) + timedelta(minutes=minutes)


def available_at(bar_ts: datetime, delay_minutes: int, bar_minutes: int) -> datetime:
    """When a bar's data could first have been known (bar close + data delay)."""
    return bar_end(bar_ts, bar_minutes) + timedelta(minutes=delay_minutes)
