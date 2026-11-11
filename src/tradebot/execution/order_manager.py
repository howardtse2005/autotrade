"""The only code path that submits, modifies or cancels IB orders."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.core.types import OrderIntent, PositionState
from tradebot.db.models import OrderIntentRow, OrderRow, PositionRow, Trade
from tradebot.db.session import session_scope
from tradebot.execution.ib_compat import LimitOrder, MarketOrder, StopOrder
from tradebot.strategy import reason_codes as rc

log = get_logger("orders")

DONE_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class OrderManager:
    def __init__(self, cfg, env, ib_client) -> None:
        self.cfg = cfg
        self.env = env
        self.ib_client = ib_client
        self.trades: dict[str, object] = {}
        self._persist_locks: dict[int, asyncio.Lock] = {}
        self.last_error = ""
        self._lock = asyncio.Lock()
        if ib_client is not None:
            ib_client.on_order_status(self._record_order_status)
            ib_client.on_fill(self._record_fill)

    # ------------------------------------------------------------- utilities
    @property
    def can_trade(self) -> bool:
        return bool(
            self.ib_client
            and self.ib_client.connected
            and self.cfg.mode in ("paper", "live")
        )

    def _live_allowed(self) -> bool:
        if self.cfg.mode != "live":
            return True
        return bool(self.env.ALLOW_LIVE) and self.env.IB_PORT == 4001

    async def _intent_exists(self, order_ref: str) -> bool:
        async with session_scope() as s:
            res = await s.execute(
                select(OrderIntentRow).where(OrderIntentRow.order_ref == order_ref).limit(1)
            )
            return res.scalars().first() is not None

    async def reconcile_intents(self) -> int:
        """Mark intents that claim to be working but have no order at IB as stale.

        Without this the intent table drifts away from reality: the UI shows rows
        "queued" that no longer exist, and `_has_live_order` would keep blocking
        new orders for those tickers.
        """
        live_refs: set[str] = set()
        try:
            for t in await self.ib_client.open_orders():
                if t.orderStatus.status in DONE_STATUSES:
                    continue
                live_refs.add(t.order.orderRef or "")
        except Exception:  # noqa: BLE001
            return 0
        n = 0
        async with session_scope() as s:
            rows = (
                await s.execute(
                    select(OrderIntentRow).where(
                        OrderIntentRow.status.in_(("queued", "submitted", "approved"))
                    )
                )
            ).scalars().all()
            for r in rows:
                if r.order_ref and r.order_ref not in live_refs:
                    r.status = "stale"
                    r.updated_at = now_utc()
                    n += 1
        if n:
            log.info("marked stale intents", count=n)
        return n

    async def expire_stale(self, current_run_id: int | None = None) -> int:
        """Expire pending approvals that are too old or superseded by a newer run.

        A pending intent encodes a decision taken at a specific price with a
        specific position state. Once either has moved the approval is no longer
        valid, so it must never be executed later — that is how a stale exit ends
        up selling shares the account no longer owns.
        """
        ttl = int(getattr(self.cfg, "approval_ttl_minutes", 30) or 0)
        cutoff = now_utc() - timedelta(minutes=ttl) if ttl > 0 else None
        n = 0
        async with session_scope() as s:
            rows = (
                await s.execute(
                    select(OrderIntentRow).where(OrderIntentRow.status == "awaiting_approval")
                )
            ).scalars().all()
            for r in rows:
                created = as_utc(r.created_at)
                stale = bool(cutoff is not None and created is not None and created < cutoff)
                superseded = bool(
                    current_run_id and r.run_id is not None and r.run_id < current_run_id
                )
                if stale or superseded:
                    r.status = "expired"
                    r.updated_at = now_utc()
                    n += 1
        if n:
            log.info("expired pending approvals", count=n, run=current_run_id)
        return n

    async def _has_live_order(self, ticker: str, side: str) -> bool:
        """True when a working (or pending) order already exists for ticker+side.

        `order_ref` embeds the run id, so a ref-based check can never catch the
        same decision being re-issued by the next scheduled run. Identity has to
        be the instrument and direction, not the run.
        """
        want = ticker.replace(" ", ".")
        try:
            for t in await self.ib_client.open_orders():
                if t.orderStatus.status in DONE_STATUSES:
                    continue
                if t.contract.symbol.replace(" ", ".") != want:
                    continue
                if t.order.action != side:
                    continue
                ref = t.order.orderRef or ""
                if ref.endswith((":sl", ":tp")):
                    continue  # protective children are not entries/exits
                return True
        except Exception:  # noqa: BLE001
            pass
        # Only a *pending user decision* counts on the database side. A row that
        # says "queued" proves nothing - the order may have been cancelled by IB
        # (day rollover, manual cancel, margin) and would otherwise block this
        # ticker forever. Live state is authoritative, checked above.
        async with session_scope() as s:
            row = (
                await s.execute(
                    select(OrderIntentRow).where(
                        OrderIntentRow.ticker == want,
                        OrderIntentRow.side == side,
                        OrderIntentRow.status == "awaiting_approval",
                    ).limit(1)
                )
            ).scalars().first()
        return row is not None

    async def queued_buy_tickers(self) -> set[str]:
        """Tickers with a working queued BUY entry (they occupy a slot)."""
        out: set[str] = set()
        try:
            for t in await self.ib_client.open_orders():
                if t.order.action != "BUY":
                    continue
                if (t.order.orderRef or "").endswith((":sl", ":tp")):
                    continue
                if t.orderStatus.status in DONE_STATUSES:
                    continue
                out.add(t.contract.symbol.replace(" ", "."))
        except Exception:  # noqa: BLE001
            pass
        return out

    async def queued_buy_notional(self) -> float:
        """Cash already committed to working BUY orders (unfilled remainder)."""
        total = 0.0
        try:
            for t in await self.ib_client.open_orders():
                if t.order.action != "BUY":
                    continue
                if (t.order.orderRef or "").endswith((":sl", ":tp")):
                    continue
                if t.orderStatus.status in DONE_STATUSES:
                    continue
                qty = float(t.order.totalQuantity or 0) - float(t.orderStatus.filled or 0)
                px = float(t.order.lmtPrice or 0)
                if px >= 1e30 or px <= 0:
                    px = float(getattr(t.order, "auxPrice", 0) or 0)
                if qty > 0 and px > 0:
                    total += qty * px
        except Exception:  # noqa: BLE001
            pass
        return total

    async def _persist_intent(self, intent: OrderIntent, status: str) -> int:
        if intent.id:
            await self._update_intent(intent.id, status)
            return intent.id
        async with session_scope() as s:
            row = OrderIntentRow(
                run_id=intent.run_id, ticker=intent.ticker, side=intent.side, qty=intent.qty,
                order_type=intent.order_type, limit_price=intent.limit_price,
                reason=intent.reason, order_ref=intent.order_ref, source=intent.source,
                planned_sl=intent.planned_sl, planned_tp=intent.planned_tp,
                status=status, meta=intent.meta or None, updated_at=now_utc(),
            )
            s.add(row)
            await s.flush()
            intent.id = row.id
            return row.id

    async def _update_intent(self, intent_id: int, status: str, error: str = "") -> None:
        async with session_scope() as s:
            row = await s.get(OrderIntentRow, intent_id)
            if row is not None:
                row.status = status
                row.updated_at = now_utc()
                if error:
                    meta = dict(row.meta or {})
                    meta["error"] = error[:400]
                    row.meta = meta

    async def _persist_order(self, trade, role: str, intent_id: int | None = None) -> None:
        o = trade.order
        st = trade.orderStatus
        lock = self._persist_locks.setdefault(int(o.orderId), asyncio.Lock())
        async with lock:
            await self._persist_order_locked(trade, role, intent_id, o, st)

    async def _persist_order_locked(self, trade, role, intent_id, o, st) -> None:  # noqa: ANN001
        async with session_scope() as s:
            # one row per IB order id: status events arrive concurrently
            res = await s.execute(
                select(OrderRow).where(OrderRow.ib_order_id == o.orderId)
                .order_by(OrderRow.id).limit(1)
            )
            row = res.scalars().first()
            if row is None:
                row = OrderRow(
                    intent_id=intent_id, ib_order_id=o.orderId, perm_id=o.permId,
                    parent_id=o.parentId or None, ticker=trade.contract.symbol,
                    action=o.action, qty=float(o.totalQuantity or 0), order_type=o.orderType,
                    limit_price=o.lmtPrice or None, aux_price=o.auxPrice or None, tif=o.tif,
                    oca_group=o.ocaGroup or None, order_ref=o.orderRef or "", role=role,
                )
                s.add(row)
            row.status = st.status
            row.filled = float(st.filled or 0)
            row.avg_fill_price = float(st.avgFillPrice or 0)
            row.updated_at = now_utc()

    def _record_order_status(self, trade) -> None:
        ref = trade.order.orderRef or ""
        self.trades[ref] = trade
        role = _role(ref)
        asyncio.create_task(self._persist_order(trade, role))
        filled = float(trade.orderStatus.filled or 0)
        terminal = trade.orderStatus.status in DONE_STATUSES
        # a queued parent can fill partially and then be cancelled at the open -
        # whatever filled still has to become a (protected) position
        if role == "parent" and trade.order.action == "BUY" and filled > 0 and terminal:
            asyncio.create_task(self._on_parent_fill(trade))
        elif role in ("sl", "tp") and trade.orderStatus.status == "Filled":
            asyncio.create_task(self._on_protective_fill(trade, role))

    async def _on_protective_fill(self, trade, role: str) -> None:  # noqa: ANN001
        """A stop or target filled: close the position row immediately.

        Without this the position stays "open" in the database after the stop
        fires, and the next exit decision sells into a flat account — turning a
        stopped-out trade into an unintended short.
        """
        ticker = trade.contract.symbol.replace(" ", ".")
        filled = float(trade.orderStatus.filled or 0)
        price = float(trade.orderStatus.avgFillPrice or 0)
        if filled <= 0 or price <= 0:
            return
        reason = rc.SELL_STOP if role == "sl" else rc.SELL_TP
        async with session_scope() as s:
            res = await s.execute(
                select(PositionRow).where(PositionRow.ticker == ticker,
                                          PositionRow.status == "open").limit(1)
            )
            row = res.scalars().first()
            if row is None:
                log.warning("protective fill without an open position row", ticker=ticker)
            else:
                qty = min(filled, abs(float(row.qty or 0))) or filled
                pnl = (price - row.avg_cost) * qty
                risk = (row.planned_r or 0) * qty
                s.add(
                    Trade(
                        position_id=row.id, ticker=row.ticker, qty=qty,
                        entry_ts=row.entry_ts, entry_price=row.avg_cost,
                        exit_ts=now_utc(), exit_price=price, pnl=pnl,
                        pnl_pct=(price / row.avg_cost - 1) * 100 if row.avg_cost else 0.0,
                        r_multiple=pnl / risk if risk else 0.0,
                        exit_reason=reason, hold_bars=row.bars_held,
                        entry_snapshot=row.entry_snapshot,
                    )
                )
                row.status = "closed"
                row.closed_at = now_utc()
                log.info("position closed by protection", ticker=ticker, qty=qty,
                         price=price, reason=reason)
        # the OCA sibling should already be dead; make sure nothing is left working
        await asyncio.sleep(1.0)
        await self._cancel_protection(ticker)

    def _held_qty(self, ticker: str) -> float:
        """Net position at the broker right now (0 when unknown)."""
        try:
            want = ticker.replace(" ", ".")
            for p in self.ib_client.positions():
                if p.contract.symbol.replace(" ", ".") == want:
                    return float(p.position)
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _record_fill(self, trade, fill, report=None) -> None:  # noqa: ANN001
        asyncio.create_task(self._persist_fill(trade, fill, report))

    async def _persist_fill(self, trade, fill, report=None) -> None:  # noqa: ANN001
        from sqlalchemy.exc import IntegrityError

        from tradebot.db.models import FillRow

        ex = getattr(fill, "execution", None) or fill
        exec_id = getattr(ex, "execId", None) or getattr(fill, "execId", "")
        if not exec_id:
            return
        side = getattr(ex, "side", "") or getattr(fill, "side", "")
        shares = float(getattr(ex, "shares", 0) or getattr(fill, "shares", 0) or 0)
        price = float(getattr(ex, "price", 0) or getattr(fill, "price", 0) or 0)
        commission = float(getattr(report, "commission", 0) or 0) if report is not None else 0.0
        try:
            async with session_scope() as s:
                existing = await s.execute(
                    select(FillRow).where(FillRow.exec_id == exec_id).limit(1)
                )
                if existing.scalars().first() is not None:
                    return
                s.add(
                    FillRow(
                        order_ref=trade.order.orderRef or "",
                        exec_id=exec_id,
                        ticker=trade.contract.symbol,
                        side=side,
                        qty=shares,
                        price=price,
                        commission=commission,
                        ts=now_utc(),
                    )
                )
        except IntegrityError:
            # two order-status callbacks can report the same execution; the
            # unique index is the real guard, so a duplicate insert is benign
            log.debug("duplicate fill ignored", exec_id=exec_id)

    async def _wait(self, trade, timeout: float | None = None) -> str:
        timeout = timeout or self.cfg.execution.fill_timeout_seconds
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if trade.orderStatus.status in DONE_STATUSES:
                return trade.orderStatus.status
            await asyncio.sleep(0.25)
        return trade.orderStatus.status

    # ---------------------------------------------------------------- submit
    async def submit(self, intent: OrderIntent) -> OrderIntent:
        async with self._lock:
            if await self._intent_exists(intent.order_ref):
                intent.status = "duplicate"
                return intent
            # A re-issued decision must not stack a second order on the same name.
            if await self._has_live_order(intent.ticker, intent.side):
                intent.status = "duplicate"
                log.info("duplicate order suppressed", ticker=intent.ticker,
                         side=intent.side)
                return intent
            if not self._live_allowed():
                intent.status = "blocked_live"
                await self._persist_intent(intent, "blocked_live")
                return intent
            if self.cfg.approval_mode == "manual_approve" and intent.source == "auto":
                intent.status = "awaiting_approval"
                await self._persist_intent(intent, "awaiting_approval")
                return intent
            if self.cfg.mode == "dry_run" or self.cfg.mode == "off":
                intent.status = "dry_run"
                await self._persist_intent(intent, "dry_run")
                return intent
            if not self.can_trade:
                intent.status = "rejected"
                self.last_error = "IB not connected"
                await self._persist_intent(intent, "rejected")
                return intent
            try:
                if intent.side == "BUY":
                    await self._submit_buy(intent)
                else:
                    await self._submit_sell(intent)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                log.error("order submit failed", ticker=intent.ticker, error=self.last_error)
                intent.status = "rejected"
                if intent.id:
                    await self._update_intent(intent.id, "rejected", self.last_error)
            return intent

    async def _submit_buy(self, intent: OrderIntent) -> None:
        contract = await self.ib_client.qualify(intent.ticker)
        if contract is None:
            raise RuntimeError(f"cannot qualify {intent.ticker}")

        if (intent.meta or {}).get("queued"):
            await self._submit_queued_entry(intent, contract)
            return

        price = None
        if intent.order_type == "LMT_MARKETABLE":
            price = intent.limit_price
        rth = bool(self.cfg.execution.outside_rth)
        if price is None:
            order = MarketOrder("BUY", intent.qty, tif="DAY", orderRef=intent.order_ref,
                                outsideRth=rth)
        else:
            px = self.ib_client.round_to_tick(intent.ticker, price)
            order = LimitOrder("BUY", intent.qty, px, tif="DAY", orderRef=intent.order_ref,
                               outsideRth=rth)

        intent.id = await self._persist_intent(intent, "submitted")
        trade = self.ib_client.ib.placeOrder(contract, order)
        self.trades[intent.order_ref] = trade
        status = await self._wait(trade)
        await self._persist_order(trade, "parent", intent.id)

        filled = float(trade.orderStatus.filled or 0)
        avg = float(trade.orderStatus.avgFillPrice or 0)

        # Never leave an unprotected working order behind: cancel whatever has not
        # filled and protect only the shares we actually own.
        if status not in DONE_STATUSES:
            with contextlib.suppress(Exception):
                self.ib_client.ib.cancelOrder(order)
            await asyncio.sleep(1.0)
            status = trade.orderStatus.status
            filled = float(trade.orderStatus.filled or 0)
            avg = float(trade.orderStatus.avgFillPrice or 0)

        if filled <= 0:
            await self._update_intent(intent.id, "unfilled",
                                      f"order {status} without fill")
            intent.status = "unfilled"
            log.info("entry not filled", ticker=intent.ticker, status=status)
            return

        intent.status = "filled" if filled >= intent.qty else "partial"
        await self._update_intent(intent.id, intent.status)
        await self._open_position(intent, contract, int(filled), avg)
        await self._protect(intent, contract, int(filled), avg)

    async def _cancel_bracket(self, bracket: dict, note: str = "queue sync") -> None:
        parent = bracket.get("parent")
        if parent is None:
            return
        ref = parent.order.orderRef or ""
        for t in [parent] + list(bracket.get("children") or []):
            with contextlib.suppress(Exception):
                self.ib_client.ib.cancelOrder(t.order)
        # close the loop in the database: a "queued" intent with no working order
        # would otherwise block every future order for that ticker
        if ref:
            async with session_scope() as s:
                row = (
                    await s.execute(
                        select(OrderIntentRow).where(
                            OrderIntentRow.order_ref == ref,
                            OrderIntentRow.status.in_(("queued", "approved", "submitted")),
                        ).limit(1)
                    )
                ).scalars().first()
                if row is not None:
                    row.status = "cancelled"
                    meta = dict(row.meta or {})
                    meta["error"] = note[:200]
                    row.meta = meta
                    row.updated_at = now_utc()
        await asyncio.sleep(0.6)

    async def sync_queued_entries(
        self, desired: list[OrderIntent], available_usd: float | None = None,
    ) -> dict:
        """Make the working queued entries match the latest decision.

        A queued entry was priced when it was created. Between then and the open
        the price moves and the decision can change (dropped, resized, or turned
        into an exit). Rather than leaving stale brackets to fire at the open,
        this reconciles them in place:

        * ticker no longer wanted  -> cancel the whole bracket
        * qty changed              -> modify parent and both children
        * limit moved beyond band  -> modify the limit and re-anchor stop/target
        * total queued notional over available cash -> cancel the lowest-ranked
        """
        if not self.can_trade:
            return {"skipped": "not connected"}
        cfg = self.cfg
        band = float(getattr(cfg.execution, "queued_reprice_band_pct", 0.5)) / 100.0
        want = {i.ticker: i for i in desired if i.side == "BUY"}
        try:
            live = await self.ib_client.open_orders()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:160]}

        # A ticker can legitimately have several working brackets (older runs
        # stacked them before the duplicate guard existed). Group by ticker and
        # keep only the newest.
        by_ref: dict[str, dict] = {}
        groups: dict[str, list[dict]] = {}
        for t in live:
            ref = t.order.orderRef or ""
            if ref.endswith((":sl", ":tp")) or t.order.action != "BUY":
                continue
            if t.orderStatus.status in DONE_STATUSES:
                continue
            ticker = t.contract.symbol.replace(" ", ".")
            g = {"parent": t, "children": [], "ref": ref}
            groups.setdefault(ticker, []).append(g)
            by_ref[ref] = g
        for t in live:
            ref = t.order.orderRef or ""
            if not ref.endswith((":sl", ":tp")):
                continue
            g = by_ref.get(ref.rsplit(":", 1)[0])
            if g is not None:
                g["children"].append(t)

        cancelled = modified = kept = 0

        # 1) collapse duplicates: keep the newest bracket per ticker
        for ticker, glist in list(groups.items()):
            glist.sort(key=lambda g: int(g["parent"].order.orderId or 0), reverse=True)
            for extra in glist[1:]:
                await self._cancel_bracket(extra)
                cancelled += 1
                log.info("duplicate queued bracket cancelled", ticker=ticker,
                         order_id=extra["parent"].order.orderId)
            groups[ticker] = [glist[0]]

        # 2) drop what the current decision no longer wants. When a run produces
        #    no candidates at all we keep the newest `max_positions` instead of
        #    wiping the queue (an empty decision can just mean cash is already
        #    fully reserved by the queue itself).
        max_slots = int(getattr(cfg.strategy, "max_positions", 5))
        if want:
            doomed = [t for t in groups if t not in want]
        else:
            newest_first = sorted(
                groups, key=lambda t: -int(groups[t][0]["parent"].order.orderId or 0)
            )
            doomed = newest_first[max_slots:]
        for ticker in doomed:
            bucket = groups.pop(ticker, None)
            if not bucket:
                continue
            await self._cancel_bracket(bucket[0])
            cancelled += 1
            log.info("queued entry cancelled - no longer wanted", ticker=ticker)

        # 3) bring the survivors in line with the decision
        for ticker, glist in groups.items():
            order = want.get(ticker)
            if order is None:
                kept += 1
                continue
            b = glist[0]
            parent = b["parent"]
            new_qty = int(order.qty)
            new_px = float(order.limit_price or 0)
            old_qty = int(parent.order.totalQuantity or 0)
            old_px = float(parent.order.lmtPrice or 0)
            changed = False
            if new_qty > 0 and new_qty != old_qty:
                parent.order.totalQuantity = new_qty
                for c in b["children"]:
                    c.order.totalQuantity = new_qty
                changed = True
            if new_px > 0 and old_px > 0 and abs(new_px - old_px) / old_px > band:
                parent.order.lmtPrice = self.ib_client.round_to_tick(ticker, new_px)
                for c in b["children"]:
                    ref = c.order.orderRef or ""
                    if ref.endswith(":sl") and order.planned_sl:
                        c.order.auxPrice = self.ib_client.round_to_tick(
                            ticker, order.planned_sl)
                    elif ref.endswith(":tp") and order.planned_tp:
                        c.order.lmtPrice = self.ib_client.round_to_tick(
                            ticker, order.planned_tp)
                changed = True
            if changed:
                for c in [parent] + b["children"]:
                    with contextlib.suppress(Exception):
                        self.ib_client.ib.placeOrder(c.contract, c.order)
                modified += 1
                log.info("queued entry updated", ticker=ticker, qty=new_qty, limit=new_px)
            else:
                kept += 1
        if modified:
            await asyncio.sleep(0.8)

        # ---- cash ceiling: the queue may never exceed available funds ----
        trimmed = 0
        if available_usd is not None and available_usd > 0:
            buffer_pct = float(getattr(cfg.risk, "cash_buffer_pct", 0.0)) / 100.0
            limit = available_usd * (1.0 - buffer_pct)
            total = await self.queued_buy_notional()
            for intent in reversed(desired):  # lowest ranked first
                if total <= limit:
                    break
                glist = groups.get(intent.ticker) or []
                if not glist:
                    continue
                b = glist[0]
                parent = b["parent"]
                qty = float(parent.order.totalQuantity or 0) - float(
                    parent.orderStatus.filled or 0)
                px = float(parent.order.lmtPrice or 0)
                await self._cancel_bracket(b)
                groups.pop(intent.ticker, None)
                total -= max(qty, 0) * max(px, 0)
                trimmed += 1
                log.warning("queued entry cancelled - over cash", ticker=intent.ticker,
                            queued_total=round(total, 2), limit=round(limit, 2))
            if trimmed:
                cancelled += trimmed

        return {"cancelled": cancelled, "modified": modified, "kept": kept,
                "trimmed_for_cash": trimmed,
                "queued_notional": round(await self.queued_buy_notional(), 2)}

    async def _submit_queued_entry(self, intent: OrderIntent, contract) -> None:
        """Entry placed outside RTH: a bracket waits at IBKR and executes at the open.

        The parent is a limit order (market orders are not accepted outside RTH) and
        the stop/target children are transmitted with it, so the position is
        protected server-side even if this app is not running when it fills.
        """
        price = intent.limit_price
        if not price or price <= 0:
            raise RuntimeError("queued entry needs a limit price")
        ticker = intent.ticker
        limit_px = self.ib_client.round_to_tick(ticker, price)
        sl = intent.planned_sl or limit_px * 0.98
        tp = intent.planned_tp or limit_px * 1.03
        sl = self.ib_client.round_to_tick(ticker, sl)
        tp = self.ib_client.round_to_tick(ticker, tp)

        bracket = self.ib_client.ib.bracketOrder("BUY", intent.qty, limit_px, tp, sl)
        parent, take, stop = bracket
        parent.orderRef = intent.order_ref
        take.orderRef = f"{intent.order_ref}:tp"
        stop.orderRef = f"{intent.order_ref}:sl"
        for o in bracket:
            # GTC for the parent too: IB cancels DAY orders at its daily rollover,
            # so a weekend-queued entry would never reach Monday's open.
            o.tif = "GTC"
            o.outsideRth = True

        intent.id = await self._persist_intent(intent, "queued")
        intent.status = "queued"
        trades = [self.ib_client.ib.placeOrder(contract, o) for o in bracket]
        self.trades[intent.order_ref] = trades[0]
        self.trades[take.orderRef] = trades[1]
        self.trades[stop.orderRef] = trades[2]
        await asyncio.sleep(1.5)
        await self._persist_order(trades[0], "parent", intent.id)
        await self._persist_order(trades[1], "tp")
        await self._persist_order(trades[2], "sl")
        log.info("entry queued for the open", ticker=ticker, qty=intent.qty,
                 limit=limit_px, sl=sl, tp=tp)

    async def _on_parent_fill(self, trade) -> None:  # noqa: ANN001
        """A queued entry filled (e.g. at the open): record the position and
        re-anchor the protective children to the actual fill price."""
        ref = trade.order.orderRef or ""
        if _role(ref) != "parent" or trade.order.action != "BUY":
            return
        filled = float(trade.orderStatus.filled or 0)
        avg = float(trade.orderStatus.avgFillPrice or 0)
        if filled <= 0 or avg <= 0:
            return
        async with session_scope() as s:
            res = await s.execute(
                select(OrderIntentRow).where(OrderIntentRow.order_ref == ref).limit(1)
            )
            intent_row = res.scalars().first()
        meta = (intent_row.meta if intent_row else {}) or {}
        sl_dist = float(meta.get("sl_dist") or 0) or None
        tp_dist = float(meta.get("tp_dist") or 0) or None
        ticker = trade.contract.symbol.replace(" ", ".")
        sl = self.ib_client.round_to_tick(ticker, avg - sl_dist) if sl_dist else None
        tp = self.ib_client.round_to_tick(ticker, avg + tp_dist) if tp_dist else None

        # find the bracket children that belong to this parent
        children = []
        try:
            for t in await self.ib_client.open_orders():
                if t.order.parentId == trade.order.orderId:
                    children.append(t)
        except Exception:  # noqa: BLE001
            pass

        contract = await self.ib_client.qualify(ticker)
        filled_qty = int(filled)
        for child in children:
            # A partial fill must not leave a protective order for more shares
            # than we own - that would flip the account short when it triggers.
            child.order.totalQuantity = filled_qty
            if child.order.orderRef.endswith(":sl") and sl:
                child.order.auxPrice = sl
            elif child.order.orderRef.endswith(":tp") and tp:
                child.order.lmtPrice = tp
            else:
                continue
            with contextlib.suppress(Exception):
                self.ib_client.ib.placeOrder(contract, child.order)

        async with session_scope() as s:
            res = await s.execute(
                select(PositionRow).where(PositionRow.ticker == ticker,
                                          PositionRow.status == "open").limit(1)
            )
            row = res.scalars().first()
            if row is None:
                row = PositionRow(ticker=ticker, status="open", entry_ts=now_utc())
                s.add(row)
            row.qty = filled
            row.avg_cost = avg
            row.entry_ts = row.entry_ts or now_utc()
            row.sl = sl or row.sl
            row.tp = tp or row.tp
            row.planned_r = (avg - sl) if sl else row.planned_r
            row.external = False
            row.entry_snapshot = {
                "intent": "queued_entry_filled", "fill": avg, "qty": filled,
                "sl_dist": sl_dist, "tp_dist": tp_dist, **(meta.get("sizing") or {}),
            }
            for child in children:
                if child.order.orderRef.endswith(":sl"):
                    row.ib_stop_id = child.order.orderId
                elif child.order.orderRef.endswith(":tp"):
                    row.ib_tp_id = child.order.orderId
        if intent_row is not None:
            await self._update_intent(intent_row.id, "filled")
        log.info("queued entry filled", ticker=ticker, qty=filled, avg=avg, sl=sl, tp=tp)

    async def _protect(self, intent: OrderIntent, contract, qty: int, fill_price: float) -> None:
        if self.cfg.execution.protective_orders == "none" or qty <= 0:
            return
        ticker = intent.ticker
        meta = intent.meta or {}
        sl_dist = float(meta.get("sl_dist") or 0)
        tp_dist = float(meta.get("tp_dist") or 0)
        # anchor to the real fill, not the (possibly stale) decision price
        if sl_dist > 0 and fill_price > 0:
            sl = fill_price - sl_dist
            tp = fill_price + tp_dist if tp_dist > 0 else (intent.planned_tp or fill_price * 1.02)
        else:
            sl = intent.planned_sl
            tp = intent.planned_tp
        if not sl or not tp:
            return
        sl = self.ib_client.round_to_tick(ticker, sl)
        tp = self.ib_client.round_to_tick(ticker, tp)
        oca = f"{intent.order_ref}:oca"
        stop = StopOrder("SELL", qty, sl, tif="GTC", ocaGroup=oca, ocaType=1,
                         orderRef=f"{intent.order_ref}:sl", outsideRth=False)
        take = LimitOrder("SELL", qty, tp, tif="GTC", ocaGroup=oca, ocaType=1,
                          orderRef=f"{intent.order_ref}:tp", outsideRth=False)
        stop_trade = self.ib_client.ib.placeOrder(contract, stop)
        tp_trade = self.ib_client.ib.placeOrder(contract, take)
        self.trades[stop.orderRef] = stop_trade
        self.trades[take.orderRef] = tp_trade
        await asyncio.sleep(1.0)
        await self._persist_order(stop_trade, "sl")
        await self._persist_order(tp_trade, "tp")
        async with session_scope() as s:
            res = await s.execute(
                select(PositionRow).where(PositionRow.ticker == ticker,
                                          PositionRow.status == "open").limit(1)
            )
            row = res.scalars().first()
            if row is not None:
                row.ib_stop_id = stop_trade.order.orderId
                row.ib_tp_id = tp_trade.order.orderId
                row.sl = sl
                row.tp = tp
        log.info("protection placed", ticker=ticker, qty=qty, sl=sl, tp=tp,
                 stop_id=stop_trade.order.orderId, tp_id=tp_trade.order.orderId)

    async def _open_position(self, intent: OrderIntent, contract, qty: int, avg: float) -> None:
        async with session_scope() as s:
            res = await s.execute(
                select(PositionRow).where(PositionRow.ticker == intent.ticker,
                                          PositionRow.status == "open").limit(1)
            )
            row = res.scalars().first()
            if row is None:
                row = PositionRow(ticker=intent.ticker, status="open", entry_ts=now_utc())
                s.add(row)
            row.qty = float(qty)
            row.avg_cost = float(avg)
            row.entry_ts = row.entry_ts or now_utc()
            row.entry_run_id = intent.run_id
            row.sl = intent.planned_sl
            row.tp = intent.planned_tp
            row.external = False
            row.entry_snapshot = {
                "intent": intent.reason, "qty": qty, "fill": avg,
                "planned_sl": intent.planned_sl, "planned_tp": intent.planned_tp,
                **(intent.meta or {}),
            }
            if intent.planned_sl and avg:
                row.planned_r = float(avg - intent.planned_sl)

    async def _oca_group_for(self, ticker: str) -> str:
        """Existing OCA group of this position's protective orders (if any)."""
        try:
            for trade in await self.ib_client.open_orders():
                ref = trade.order.orderRef or ""
                sym = trade.contract.symbol.replace(" ", ".")
                if sym == ticker and ref.endswith((":sl", ":tp")) and trade.order.ocaGroup:
                    if trade.orderStatus.status not in DONE_STATUSES:
                        return trade.order.ocaGroup
        except Exception:  # noqa: BLE001
            pass
        return ""

    async def _submit_sell(self, intent: OrderIntent) -> None:
        # A SELL may only ever reduce an existing long. Re-check the *broker*
        # position at submit time: a stale approval, a duplicated exit or a stop
        # that already fired must never flip a flat account into a short.
        intent.id = await self._persist_intent(intent, "submitted")
        held = self._held_qty(intent.ticker)
        if held <= 0:
            intent.status = "rejected"
            await self._update_intent(
                intent.id, "rejected",
                f"no position to close ({intent.ticker} is flat at IBKR)",
            )
            log.warning("exit rejected - flat position", ticker=intent.ticker,
                        want=intent.qty)
            return
        if intent.qty > held:
            log.warning("exit qty clamped to the position", ticker=intent.ticker,
                        want=intent.qty, held=held)
            intent.qty = int(held)

        contract = await self.ib_client.qualify(intent.ticker)
        if contract is None:
            raise RuntimeError(f"cannot qualify {intent.ticker}")

        # Join the position's existing OCA group so the stop/target and this exit
        # can never both execute (no accidental short). Protection is only
        # cancelled *after* the exit actually fills.
        oca = await self._oca_group_for(intent.ticker)
        kwargs = {"outsideRth": bool(self.cfg.execution.outside_rth)}
        if oca:
            kwargs.update({"ocaGroup": oca, "ocaType": 1})
        order = MarketOrder("SELL", intent.qty, tif="DAY", orderRef=intent.order_ref, **kwargs)
        trade = self.ib_client.ib.placeOrder(contract, order)
        self.trades[intent.order_ref] = trade
        status = await self._wait(trade)
        await self._persist_order(trade, "exit", intent.id)
        filled = float(trade.orderStatus.filled or 0)
        avg = float(trade.orderStatus.avgFillPrice or 0)

        if filled > 0:
            intent.status = "filled"
            await self._update_intent(intent.id, "filled")
            await self._cancel_protection(intent.ticker)
            await self._close_position(intent, filled, avg)
            return

        # Not filled. Outside RTH a manual exit is deliberately left working so it
        # executes at the open; the order shares the position's OCA group, so the
        # stop/target can never double-fill.
        from tradebot.core.clock import market_clock

        market_closed = not market_clock().is_open
        if market_closed and status not in DONE_STATUSES:
            intent.status = "queued"
            await self._update_intent(intent.id, "queued",
                                      f"queued for the open ({trade.orderStatus.status})")
            log.info("exit queued for the open", ticker=intent.ticker)
            return
        if status not in DONE_STATUSES:
            with contextlib.suppress(Exception):
                self.ib_client.ib.cancelOrder(order)
            await asyncio.sleep(1.0)
        intent.status = "unfilled"
        await self._update_intent(intent.id, "unfilled",
                                  f"exit order {trade.orderStatus.status} without fill")
        log.info("exit not filled - protection kept", ticker=intent.ticker)

    async def _cancel_protection(self, ticker: str) -> None:
        """Cancel every live protective order for `ticker` (queried from IB, not memory)."""
        targets: list = []
        for ref, trade in list(self.trades.items()):
            if ref.endswith((":sl", ":tp")) and trade.orderStatus.status not in DONE_STATUSES:
                if trade.contract.symbol.replace(" ", ".") == ticker:
                    targets.append(trade)
        try:
            for trade in await self.ib_client.open_orders():
                ref = trade.order.orderRef or ""
                if not ref.endswith((":sl", ":tp")):
                    continue
                if trade.contract.symbol.replace(" ", ".") != ticker:
                    continue
                if trade.orderStatus.status in DONE_STATUSES:
                    continue
                if all(t.order.orderId != trade.order.orderId for t in targets):
                    targets.append(trade)
        except Exception:  # noqa: BLE001
            pass
        for trade in targets:
            with contextlib.suppress(Exception):
                self.ib_client.ib.cancelOrder(trade.order)
        if targets:
            await asyncio.sleep(1.0)

    async def _close_position(self, intent: OrderIntent, qty: float, price: float) -> None:
        async with session_scope() as s:
            res = await s.execute(
                select(PositionRow).where(PositionRow.ticker == intent.ticker,
                                          PositionRow.status == "open").limit(1)
            )
            row = res.scalars().first()
            if row is None:
                return
            pnl = (price - row.avg_cost) * qty
            risk = (row.planned_r or 0) * qty
            s.add(
                Trade(
                    position_id=row.id, ticker=row.ticker, qty=qty,
                    entry_ts=row.entry_ts, entry_price=row.avg_cost,
                    exit_ts=now_utc(), exit_price=price, pnl=pnl,
                    pnl_pct=(price / row.avg_cost - 1) * 100 if row.avg_cost else 0.0,
                    r_multiple=pnl / risk if risk else 0.0,
                    exit_reason=intent.reason or rc.SELL_MANUAL,
                    hold_bars=row.bars_held,
                    entry_snapshot=row.entry_snapshot,
                )
            )
            row.status = "closed"
            row.closed_at = now_utc()

    # ------------------------------------------------------------- operations
    async def close_position(self, ticker: str, pct: int = 100, reason: str = rc.SELL_MANUAL,
                             run_id: int | None = None) -> OrderIntent:
        qty = 0.0
        if self.ib_client is not None:
            for p in self.ib_client.positions():
                if p.contract.symbol.replace(" ", ".") == ticker.replace(" ", "."):
                    qty = float(p.position)
        if qty <= 0:
            async with session_scope() as s:
                res = await s.execute(
                    select(PositionRow).where(PositionRow.ticker == ticker,
                                              PositionRow.status == "open").limit(1)
                )
                row = res.scalars().first()
                qty = float(row.qty) if row else 0.0
        if qty <= 0:
            raise ValueError(f"no open position for {ticker}")
        sell_qty = max(1, int(qty * max(1, min(pct, 100)) / 100))
        intent = OrderIntent(
            run_id=run_id, ticker=ticker, side="SELL", qty=sell_qty, order_type="MKT",
            reason=reason, order_ref=f"manual:{ticker}:{int(now_utc().timestamp())}:{reason}",
            source="manual",
        )
        return await self.submit(intent)

    async def close_all(self, reason: str = rc.SELL_MANUAL, run_id: int | None = None) -> list[OrderIntent]:
        out: list[OrderIntent] = []
        tickers: set[str] = set()
        if self.ib_client is not None:
            for p in self.ib_client.positions():
                if p.position > 0:
                    tickers.add(p.contract.symbol.replace(" ", "."))
        async with session_scope() as s:
            res = await s.execute(select(PositionRow).where(PositionRow.status == "open"))
            for row in res.scalars().all():
                if row.qty > 0:
                    tickers.add(row.ticker)
        for t in sorted(tickers):
            try:
                out.append(await self.close_position(t, 100, reason, run_id))
            except Exception as exc:  # noqa: BLE001
                log.error("close failed", ticker=t, error=str(exc)[:160])
        return out

    async def cancel_order(self, ib_order_id: int) -> bool:
        for trade in list(self.trades.values()):
            if trade.order.orderId == ib_order_id:
                self.ib_client.ib.cancelOrder(trade.order)
                await asyncio.sleep(0.5)
                return True
        for trade in await self.ib_client.open_orders():
            if trade.order.orderId == ib_order_id:
                self.ib_client.ib.cancelOrder(trade.order)
                await asyncio.sleep(0.5)
                return True
        return False

    async def modify_stops(self, ticker: str, sl: float | None = None,
                           tp: float | None = None) -> dict:
        """Modify protective orders in place (same orderId, new price)."""
        if not self.can_trade:
            return {"ok": False, "error": "not in paper/live mode or IB disconnected"}
        contract = await self.ib_client.qualify(ticker)
        if contract is None:
            return {"ok": False, "error": "cannot qualify"}
        changed = {}
        async with session_scope() as s:
            res = await s.execute(
                select(PositionRow).where(PositionRow.ticker == ticker,
                                          PositionRow.status == "open").limit(1)
            )
            row = res.scalars().first()
            if row is None:
                return {"ok": False, "error": "no open position"}
            if sl is not None and sl > 0:
                row.sl = self.ib_client.round_to_tick(ticker, sl)
                changed["sl"] = row.sl
            if tp is not None and tp > 0:
                row.tp = self.ib_client.round_to_tick(ticker, tp)
                changed["tp"] = row.tp
        open_orders = await self.ib_client.open_orders()
        for trade in open_orders:
            ref = trade.order.orderRef or ""
            sym = trade.contract.symbol.replace(" ", ".")
            if sym != ticker:
                continue
            if ref.endswith(":sl") and "sl" in changed:
                trade.order.auxPrice = changed["sl"]
                self.ib_client.ib.placeOrder(contract, trade.order)
            elif ref.endswith(":tp") and "tp" in changed:
                trade.order.lmtPrice = changed["tp"]
                self.ib_client.ib.placeOrder(contract, trade.order)
        await asyncio.sleep(0.5)
        return {"ok": True, **changed}

    async def reprotect(self, ticker: str) -> dict:
        """Re-place missing protective orders for a position."""
        if not self.can_trade:
            return {"ok": False, "error": "not in paper/live mode"}
        pos = None
        for p in self.ib_client.positions():
            if p.contract.symbol.replace(" ", ".") == ticker:
                pos = p
                break
        if pos is None or pos.position <= 0:
            return {"ok": False, "error": "no long position"}
        async with session_scope() as s:
            res = await s.execute(
                select(PositionRow).where(PositionRow.ticker == ticker,
                                          PositionRow.status == "open").limit(1)
            )
            row = res.scalars().first()
        if row is None or not row.sl or not row.tp:
            return {"ok": False, "error": "no planned SL/TP stored"}
        contract = await self.ib_client.qualify(ticker)
        intent = OrderIntent(
            run_id=None, ticker=ticker, side="BUY", qty=int(pos.position),
            order_ref=f"protect:{ticker}:{int(now_utc().timestamp())}",
            planned_sl=row.sl, planned_tp=row.tp, source="manual",
        )
        await self._protect(intent, contract, int(pos.position), float(pos.avgCost))
        return {"ok": True}

    async def watchdog(self) -> list[dict]:
        """Re-place protection on naked long positions; flatten after retries."""
        from tradebot.execution.reconcile import positions_needing_protection

        naked = await positions_needing_protection(self.ib_client)
        fixed = []
        for item in naked:
            ticker = item["ticker"]
            result = await self.reprotect(ticker)
            fixed.append({"ticker": ticker, "missing": item["missing"], "result": result})
            log.warning("naked position detected", ticker=ticker, missing=item["missing"])
        return fixed


def _role(ref: str) -> str:
    if ref.endswith(":sl"):
        return "sl"
    if ref.endswith(":tp"):
        return "tp"
    return "parent"


def position_state_from_row(row: PositionRow, last: float | None = None) -> PositionState:
    return PositionState(
        ticker=row.ticker, qty=row.qty, avg_cost=row.avg_cost, last=last,
        sl=row.sl, tp=row.tp, entry_ts=row.entry_ts, entry_run_id=row.entry_run_id,
        bars_held=row.bars_held, trail_active=row.trail_active,
        highest_close=row.highest_close, ib_stop_id=row.ib_stop_id, ib_tp_id=row.ib_tp_id,
        status=row.status, external=row.external, sector=row.sector or "Unknown",
        entry_snapshot=row.entry_snapshot or {},
    )
