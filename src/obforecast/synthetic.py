"""Synthetic order books in the Kaggle schema, for tests and pipeline smoke checks ONLY.

Planted structure (so models have something to find): a market factor shared by all stocks in a
window with mild momentum from the first half into the second, and a spread regime whose
late-first-half level persists into the second half. Numbers produced from these books say
nothing about real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import InMemorySource

# dtypes of the Kaggle parquet files (float32 prices matter for exact-equality checks)
KAGGLE_BOOK_DTYPES = {"time_id": "int16", "seconds_in_bucket": "int16", "bid_price1": "float32",
                      "ask_price1": "float32", "bid_price2": "float32", "ask_price2": "float32",
                      "bid_size1": "int32", "ask_size1": "int32", "bid_size2": "int32", "ask_size2": "int32"}
KAGGLE_TRADE_DTYPES = {"time_id": "int16", "seconds_in_bucket": "int16", "price": "float32",
                       "size": "int32", "order_count": "int16"}


def make_synthetic(n_stocks: int = 8, n_windows: int = 60, seed: int = 0, update_prob: float = 0.6):
    rng = np.random.default_rng(seed)
    time_ids = np.sort(rng.choice(np.arange(5, 32_000), size=n_windows, replace=False))   # fits int16
    sec = np.arange(600)
    # market factor per window: first-half path, second-half drift tied to first-half move
    mkt = np.zeros((n_windows, 600))
    steps = rng.normal(0, 2e-5, (n_windows, 600))
    first_move = steps[:, :300].sum(1)
    steps[:, 300:] += 0.6 * first_move[:, None] / 300
    mkt = np.cumsum(steps, axis=1)
    regime_late = rng.normal(0, 0.25, n_windows)                  # market-wide spread shock late in half 1
    books, trades = {}, {}
    for s in range(n_stocks):
        sid = int(s if s < 12 else s + 1)                         # mimic gaps in Kaggle stock ids
        tick = 1e-4 * (1 + s % 3)
        base_ticks = 1 + (s % 4)
        beta = 0.6 + 0.1 * (s % 5)
        rows, trows = [], []
        for w, tid in enumerate(time_ids):
            idio = np.cumsum(rng.normal(0, 3e-5, 600))
            mid = 1.0 * np.exp(beta * mkt[w] + idio) * (1 + rng.normal(0, 2e-3))
            stock_late = rng.normal(0, 0.3)
            lvl = np.where(sec < 240, 0.0, regime_late[w] + stock_late)
            lvl = np.where(sec >= 300, 0.7 * (regime_late[w] + stock_late), lvl)
            n_ticks = np.maximum(1, np.round(base_ticks * np.exp(lvl + rng.normal(0, 0.2, 600)))).astype(int)
            half = n_ticks * tick / 2
            bid1, ask1 = mid - half, mid + half
            upd = rng.random(600) < update_prob
            upd[0] = rng.random() < 0.97                          # a few windows start late
            if not upd[:300].any():
                upd[rng.integers(0, 300)] = True
            idx = np.flatnonzero(upd)
            bs1 = rng.integers(1, 400, idx.size)
            as1 = rng.integers(1, 400, idx.size)
            rows.append(pd.DataFrame({
                "time_id": tid, "seconds_in_bucket": idx,
                "bid_price1": bid1[idx], "ask_price1": ask1[idx],
                "bid_price2": bid1[idx] - tick, "ask_price2": ask1[idx] + tick,
                "bid_size1": bs1, "ask_size1": as1,
                "bid_size2": rng.integers(1, 500, idx.size), "ask_size2": rng.integers(1, 500, idx.size)}))
            n_tr = rng.integers(0, 40)
            ts = np.sort(rng.choice(600, size=n_tr, replace=False))
            trows.append(pd.DataFrame({"time_id": tid, "seconds_in_bucket": ts,
                                       "price": mid[ts] * (1 + rng.normal(0, 5e-5, n_tr)),
                                       "size": rng.integers(1, 300, n_tr), "order_count": rng.integers(1, 10, n_tr)}))
        books[sid] = pd.concat(rows, ignore_index=True).astype(KAGGLE_BOOK_DTYPES)
        trades[sid] = pd.concat(trows, ignore_index=True).astype(KAGGLE_TRADE_DTYPES)
    return InMemorySource(books, trades), sorted(books)
