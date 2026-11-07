"""Correctness checks that run on REAL data during a run (section 8.4), and on synthetic books in
``tests/``. Each check returns True/False; ``run.py`` writes the table and fails loudly on False.

Strengthened relative to the archived notebook: the future-data perturbation now also covers the
31 upgraded book/trade inputs (``extra_stock``), with trades after the cut perturbed as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .book import BOOK_COLS, KEYS, build_stock, dense_grid
from .config import Timing
from .extra import extra_stock
from .trading import DECIDE_INPUTS, decide, net_return_bps
from .validation import assign_folds


def perturb_future(book: pd.DataFrame, trades: pd.DataFrame | None, timing: Timing):
    """Change every book and trade observation at or after the cut (prices, sizes, counts).

    Column dtypes are preserved (Kaggle stores prices as float32): casting to float64 would change
    pre-cut values at ~1e-6 and make the check fail for a reason unrelated to look-ahead.
    """
    def scale(df, mask, cols, factor=None, add=None):
        for c in cols:
            v = df.loc[mask, c].astype("float64")
            v = v * factor if factor is not None else v + add
            df.loc[mask, c] = v.astype(df[c].dtype)

    bk = book.copy()
    late = bk.seconds_in_bucket >= timing.cut
    scale(bk, late, ["bid_price1"], 0.99)
    scale(bk, late, ["ask_price1"], 1.02)
    scale(bk, late, ["bid_price2"], 0.985)
    scale(bk, late, ["ask_price2"], 1.025)
    scale(bk, late, ["bid_size1", "ask_size1", "bid_size2", "ask_size2"], add=7)
    tr = None
    if trades is not None:
        tr = trades.copy()
        late_t = tr.seconds_in_bucket >= timing.cut
        scale(tr, late_t, ["price"], 1.03)
        scale(tr, late_t, ["size", "order_count"], add=11)
        # add an extra late trade in every window
        add = tr.groupby("time_id").head(1).copy()
        add["seconds_in_bucket"] = np.asarray(timing.cut + 5).astype(tr.seconds_in_bucket.dtype)
        tr = pd.concat([tr, add], ignore_index=True)
    return bk, tr


def future_perturbation(stock_id, book, trades, timing: Timing) -> dict:
    f_a, o_a, _, _ = build_stock(stock_id, book, timing)
    bk2, tr2 = perturb_future(book, trades, timing)
    f_b, o_b, _, _ = build_stock(stock_id, bk2, timing)
    out = {"future perturbation: baseline features identical": f_a.equals(f_b),
           "future perturbation: targets/fills change": not o_a.y_price.equals(o_b.y_price)}
    if trades is not None:
        x_a = extra_stock(stock_id, book, trades, timing)
        x_b = extra_stock(stock_id, bk2, tr2, timing)
        out["future perturbation: upgraded book/trade features identical"] = x_a.equals(x_b)
    return out


def synthetic_book(rows):
    cols = ["time_id", "seconds_in_bucket"] + BOOK_COLS
    return pd.DataFrame(rows, columns=cols).astype({"time_id": int, "seconds_in_bucket": int})


def quote(tid, sec, bid, ask, size=10):
    return [tid, sec, bid, ask, bid - 0.0001, ask + 0.0001, size, size, size, size]


def forward_fill_boundaries() -> dict:
    b = synthetic_book([quote(1, 0, 1.0, 1.001), quote(2, 10, 2.0, 2.002), quote(2, 20, 2.1, 2.102)])
    _, g, _ = dense_grid(b)
    w2 = g["bid_price1"][1]
    return {"no backfill before a window's first quote": bool(np.isnan(w2[:10]).all()),
            "no carry-over between windows": bool((g["bid_price1"][0] == 1.0).all() and w2[10] == 2.0
                                                  and w2[19] == 2.0 and w2[20] == 2.1)}


def flat_round_trip(timing: Timing) -> dict:
    flat = synthetic_book([quote(1, 0, 0.999, 1.001)])
    _, o_flat, _, _ = build_stock(0, flat, timing)
    pnl = (o_flat[f"bid_{timing.exit_second}"] - o_flat[f"ask_{timing.entry_second}"]).iloc[0]
    return {"flat round trip loses exactly the spread": bool(
        np.isclose(pnl, -(1.001 - 0.999)) and np.isclose(net_return_bps(o_flat, 0.0, timing)[0], -1e4 * 0.002 / 1.001))}


def run_checks(feats, outs, dev_f, cfg, timing: Timing, sample_stock=None) -> pd.DataFrame:
    """All checks on the real tables; ``sample_stock`` = (stock_id, book, trades) for perturbation."""
    s = cfg["split"]
    checks = {"unique keys, features = outcomes row for row":
              (not feats.duplicated(KEYS).any()) and feats[KEYS].equals(outs[KEYS])}
    tids = dev_f.time_id.to_numpy()
    inner_ok = True
    for k in range(s["outer_folds"]):
        trk = (dev_f.fold != k).to_numpy()
        inner = assign_folds(tids[trk], s["inner_folds"], s["inner_seed_offset"] + k)
        for j in range(s["inner_folds"]):
            inner_ok &= np.intersect1d(tids[trk][inner != j], tids[trk][inner == j]).size == 0
        inner_ok &= np.intersect1d(tids[trk], tids[~trk]).size == 0
    checks["no time_id in two folds (outer and inner)"] = bool(dev_f.groupby("time_id").fold.nunique().max() == 1 and inner_ok)
    if sample_stock is not None:
        checks.update(future_perturbation(*sample_stock, timing))
        sid, bk, _ = sample_stock
        f_a, o_a, _, _ = build_stock(sid, bk, timing)
        scaled = bk.astype({c: float for c in BOOK_COLS})
        scaled[["bid_price1", "ask_price1", "bid_price2", "ask_price2"]] *= 37.5
        f_c, o_c, _, _ = build_stock(sid, scaled, timing)
        same = ["y_price", "y_spread"]
        checks["price scaling invariance"] = bool(np.allclose(o_a[same], o_c[same], equal_nan=True)
                                                  and np.allclose(f_a.q_299, f_c.q_299, equal_nan=True)
                                                  and np.allclose(net_return_bps(o_a, 0.5, timing),
                                                                  net_return_bps(o_c, 0.5, timing), equal_nan=True))
    checks.update(forward_fill_boundaries())
    checks.update(flat_round_trip(timing))
    head = outs.head(5000)
    nets = [net_return_bps(head, c, timing) for c in (0, 0.5, 1, 5)]
    checks["higher costs never raise profit"] = all(
        (np.nan_to_num(nets[i + 1], nan=-np.inf) <= np.nan_to_num(nets[i], nan=np.inf) + 1e-12).all() for i in range(3))
    import inspect
    checks["decide() takes no outcome inputs"] = list(inspect.signature(decide).parameters) == DECIDE_INPUTS
    return pd.Series(checks, name="passed").rename_axis("check").reset_index()
