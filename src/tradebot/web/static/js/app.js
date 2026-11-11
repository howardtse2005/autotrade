/* Shared UI helpers: formatting, fetch with CSRF, toasts, confirm modal, SSE. */
(function () {
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";

  const fmt = {
    money(v, cur) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      const c = cur || "USD";
      const sym = c === "USD" ? "$" : c === "HKD" ? "HK$" : c === "JPY" ? "¥" : c + " ";
      const abs = Math.abs(v);
      const digits = abs >= 1000 ? 0 : 2;
      return sym + v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
    },
    num(v, d) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      return Number(v).toLocaleString(undefined, { minimumFractionDigits: d ?? 2, maximumFractionDigits: d ?? 2 });
    },
    pct(v, d) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      return (v >= 0 ? "+" : "") + Number(v).toFixed(d ?? 2) + "%";
    },
    signed(v, d) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      return (v >= 0 ? "+" : "") + Number(v).toFixed(d ?? 2);
    },
    cls(v) { return v > 0 ? "pos" : v < 0 ? "neg" : "dim"; },
    time(iso, withDate) {
      if (!iso) return "—";
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return "—";
      const opts = withDate === false
        ? { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Hong_Kong" }
        : { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Hong_Kong" };
      return d.toLocaleString("en-GB", opts).replace(",", "");
    },
    ago(iso) {
      if (!iso) return "—";
      const s = (Date.now() - new Date(iso).getTime()) / 1000;
      if (s < 60) return Math.round(s) + "s";
      if (s < 3600) return Math.round(s / 60) + "m";
      if (s < 86400) return Math.round(s / 3600) + "h";
      return Math.round(s / 86400) + "d";
    },
  };

  async function api(path, opts) {
    const o = Object.assign({ headers: {} }, opts || {});
    o.headers = Object.assign({ "X-CSRF-Token": csrf }, o.headers || {});
    if (o.body && typeof o.body !== "string") {
      o.body = JSON.stringify(o.body);
      o.headers["Content-Type"] = "application/json";
    }
    const res = await fetch(path, o);
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      const msg = (data && (data.detail || data.error)) || res.statusText;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  function toast(msg, kind) {
    let wrap = document.querySelector(".toast-wrap");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "toast-wrap";
      document.body.appendChild(wrap);
    }
    const el = document.createElement("div");
    el.className = "toast " + (kind || "");
    el.textContent = msg;
    wrap.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 250); }, 3800);
  }

  let confirmResolve = null;
  function confirmDialog(title, message, danger) {
    const back = document.getElementById("confirm-modal");
    if (!back) return Promise.resolve(window.confirm(message));
    document.getElementById("confirm-title").textContent = title;
    document.getElementById("confirm-msg").textContent = message;
    const btn = document.getElementById("confirm-ok");
    btn.className = "btn " + (danger ? "danger" : "primary");
    back.classList.add("show");
    return new Promise((resolve) => { confirmResolve = resolve; });
  }
  document.addEventListener("click", (e) => {
    if (e.target.id === "confirm-ok") {
      document.getElementById("confirm-modal").classList.remove("show");
      if (confirmResolve) confirmResolve(true);
      confirmResolve = null;
    } else if (e.target.id === "confirm-cancel" || e.target.id === "confirm-modal") {
      document.getElementById("confirm-modal").classList.remove("show");
      if (confirmResolve) confirmResolve(false);
      confirmResolve = null;
    }
  });

  function connectSSE(onEvent) {
    if (!window.EventSource) return null;
    const es = new EventSource("/api/events");
    const handler = (ev) => {
      try { onEvent(JSON.parse(ev.data)); } catch (e) { /* ignore */ }
    };
    ["run.start", "run.stage", "run.done", "order.update", "position.update",
     "account.update", "alert", "news.new", "decision.update"].forEach((t) => {
      es.addEventListener(t, handler);
    });
    es.onerror = () => { /* browser retries automatically */ };
    return es;
  }

  function countdown(iso, el) {
    if (!el) return;
    const tick = () => {
      if (!iso) { el.textContent = "—"; return; }
      const ms = new Date(iso).getTime() - Date.now();
      if (ms <= 0) { el.textContent = "due"; return; }
      const m = Math.floor(ms / 60000), s = Math.floor((ms % 60000) / 1000);
      el.textContent = m > 0 ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
    };
    tick();
    return setInterval(tick, 1000);
  }

  window.UI = { fmt, api, toast, confirmDialog, connectSSE, countdown, csrf };
})();
