"""FinBERT scoring with a persistent hash cache and a time budget."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np

from tradebot.core.clock import now_utc
from tradebot.core.logging import get_logger
from tradebot.db.models import SentimentScore
from tradebot.db.session import session_scope
from tradebot.settings import MODELS_DIR

log = get_logger("sentiment")

MODEL_VERSION_SUFFIX = "fp32-v1"


def text_hash(text: str) -> str:
    norm = " ".join((text or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


@dataclass
class Score:
    p_pos: float
    p_neg: float
    p_neu: float

    @property
    def label(self) -> str:
        m = max(self.p_pos, self.p_neg, self.p_neu)
        if m == self.p_pos:
            return "positive"
        if m == self.p_neg:
            return "negative"
        return "neutral"

    @property
    def value(self) -> float:
        return self.p_pos - self.p_neg


class FinBert:
    """Lazily loaded CPU FinBERT. Label order is read from the model config."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_name = cfg.sentiment.model
        self.version = f"{self.model_name}@{MODEL_VERSION_SUFFIX}"
        self._tok = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._lock = asyncio.Lock()
        self.load_seconds: float | None = None
        self.last_error: str = ""
        self._scored_total = 0

    # ------------------------------------------------------------- loading
    def _load_sync(self) -> None:
        os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        threads = self.cfg.sentiment.threads or max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(int(threads))
        t0 = time.time()
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        if set(self._id2label.values()) != {"positive", "negative", "neutral"}:
            raise RuntimeError(f"unexpected FinBERT labels: {self._id2label}")
        self.load_seconds = round(time.time() - t0, 2)
        log.info("finbert loaded", model=self.model_name, seconds=self.load_seconds,
                 labels=self._id2label, threads=threads)

    async def ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        async with self._lock:
            if self._model is not None:
                return True
            try:
                await asyncio.to_thread(self._load_sync)
                return True
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                log.error("finbert load failed", error=self.last_error)
                return False

    @property
    def ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------ inference
    def _infer_sync(self, texts: list[str]) -> np.ndarray:
        import torch

        bs = self.cfg.sentiment.batch_size
        max_len = self.cfg.sentiment.max_tokens
        out = np.zeros((len(texts), 3), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for start in range(0, len(order), bs):
                idxs = order[start : start + bs]
                batch = [texts[i][:1200] for i in idxs]
                enc = self._tok(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
                )
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row, i in enumerate(idxs):
                    for j, label in self._id2label.items():
                        if label == "positive":
                            out[i, 0] = probs[row, j]
                        elif label == "negative":
                            out[i, 1] = probs[row, j]
                        elif label == "neutral":
                            out[i, 2] = probs[row, j]
        return out

    async def score_texts(
        self,
        pairs: list[tuple[str, str]],
        budget_s: float | None = None,
    ) -> dict[str, Score]:
        """pairs = [(hash, text)] in priority order. Cache hits never re-run the model."""
        if not pairs:
            return {}
        budget_s = budget_s if budget_s is not None else self.cfg.sentiment.score_budget_seconds
        uniq: dict[str, str] = {}
        for h, text in pairs:
            if h and text and h not in uniq:
                uniq[h] = text

        cached = await self._load_cache(list(uniq))
        missing = [(h, t) for h, t in uniq.items() if h not in cached]
        if missing and self.cfg.sentiment.enabled:
            if not await self.ensure_loaded():
                return cached
            t0 = time.monotonic()
            batch: list[tuple[str, str]] = []
            for h, text in missing:
                if time.monotonic() - t0 > budget_s and batch:
                    break
                batch.append((h, text))
                if len(batch) >= 128:
                    break
            if batch:
                arr = await asyncio.to_thread(self._infer_sync, [t for _, t in batch])
                fresh: list[SentimentScore] = []
                for (h, _t), row in zip(batch, arr):
                    sc = Score(float(row[0]), float(row[1]), float(row[2]))
                    cached[h] = sc
                    fresh.append(
                        SentimentScore(
                            text_hash=h, model_version=self.version,
                            p_pos=sc.p_pos, p_neg=sc.p_neg, p_neu=sc.p_neu, label=sc.label,
                        )
                    )
                self._scored_total += len(fresh)
                await self._store_cache(fresh)
        elif missing:
            for h, _t in missing:
                cached[h] = Score(0.25, 0.25, 0.5)
        return cached

    async def _load_cache(self, hashes: list[str]) -> dict[str, Score]:
        out: dict[str, Score] = {}
        if not hashes:
            return out
        from sqlalchemy import select

        async with session_scope() as s:
            for chunk in [hashes[i : i + 500] for i in range(0, len(hashes), 500)]:
                rows = await s.execute(
                    select(SentimentScore).where(
                        SentimentScore.text_hash.in_(chunk),
                        SentimentScore.model_version == self.version,
                    )
                )
                for r in rows.scalars().all():
                    out[r.text_hash] = Score(r.p_pos, r.p_neg, r.p_neu)
        return out

    async def _store_cache(self, rows: list[SentimentScore]) -> None:
        if not rows:
            return
        async with session_scope() as s:
            for r in rows:
                existing = await s.get(SentimentScore, (r.text_hash, r.model_version))
                if existing is None:
                    s.add(r)

    def status(self) -> dict:
        return {
            "model": self.model_name,
            "version": self.version,
            "ready": self.ready,
            "load_seconds": self.load_seconds,
            "labels": self._id2label,
            "scored_total": self._scored_total,
            "error": self.last_error,
            "enabled": self.cfg.sentiment.enabled,
        }
