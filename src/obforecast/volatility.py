"""Realized-volatility features in the style of the Kaggle competition notebooks.

The Optiver competition asked for the realized volatility (RV) of the weighted average price
(WAP) over the ten minutes after each window. Public notebooks converged on a common recipe:

* RV of WAP log returns over the whole window and over its later parts;
* book state (spreads, depth, imbalance) and trade activity summarised over the same parts;
* the same features averaged over every stock in the window ("market" features), because
  volatility moves together across stocks;
* models trained on RMSPE, the competition metric, through weights of 1/y^2.

The winning solutions added features from other windows that looked similar (nearest
neighbours by recovered price level or by the cross-section of volatilities). Because those
neighbours can lie later in calendar time, those features are look-ahead in a real-time setting;
``neighbour_features`` exists so that the gap can be measured, and the notebook reports it
separately.

Two framings are supported through ``cut``:

* ``cut=600``: the competition task. Features use the whole window; the target is the RV of the
  next ten minutes, given in ``train.csv``.
* ``cut=300``: this project's task. Features use seconds 0..299; the target is the RV of WAP1
  over seconds 300..599 of the same window (``rv_target``), computed from the raw snapshots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["stock_id", "time_id"]


def _group_index(time_id: np.ndarray):
    tids, idx = np.unique(time_id, return_inverse=True)
    return tids, idx


def _sum(idx, n, x, mask=None):
    x = np.where(np.isfinite(x), x, 0.0)
    if mask is not None:
        x = np.where(mask, x, 0.0)
    return np.bincount(idx, weights=x, minlength=n)


def _count(idx, n, mask):
    return np.bincount(idx, weights=mask.astype(float), minlength=n)


def _mean(idx, n, x, mask):
    ok = mask & np.isfinite(x)
    c = _count(idx, n, ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _sum(idx, n, x, ok) / c


def _std(idx, n, x, mask):
    m = _mean(idx, n, x, mask)
    ok = mask & np.isfinite(x)
    c = _count(idx, n, ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(_sum(idx, n, (x - m[idx]) ** 2, ok) / c)


def _log_returns(price: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(price)
    r = np.diff(lp, prepend=np.nan)
    first = np.r_[True, time_id[1:] != time_id[:-1]]
    r[first] = np.nan
    return r


def book_frame(book: pd.DataFrame) -> pd.DataFrame:
    """Row-level derived quantities on the raw snapshots (no forward filling)."""
    b = book.sort_values(["time_id", "seconds_in_bucket"]).reset_index(drop=True)
    bp1, ap1 = b.bid_price1.to_numpy(float), b.ask_price1.to_numpy(float)
    bp2, ap2 = b.bid_price2.to_numpy(float), b.ask_price2.to_numpy(float)
    bs1, as1 = b.bid_size1.to_numpy(float), b.ask_size1.to_numpy(float)
    bs2, as2 = b.bid_size2.to_numpy(float), b.ask_size2.to_numpy(float)
    tid = b.time_id.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        wap1 = (bp1 * as1 + ap1 * bs1) / (bs1 + as1)
        wap2 = (bp2 * as2 + ap2 * bs2) / (bs2 + as2)
        mid = (ap1 + bp1) / 2
        out = pd.DataFrame({
            "time_id": tid, "sec": b.seconds_in_bucket.to_numpy(),
            "r1": _log_returns(wap1, tid), "r2": _log_returns(wap2, tid),
            "spread": (ap1 - bp1) / mid,
            "spread2": (ap2 - bp2) / mid,
            "bid_gap": (bp1 - bp2) / mid,
            "ask_gap": (ap2 - ap1) / mid,
            "wap_bal": np.abs(wap1 - wap2) / mid,
            "depth": bs1 + as1 + bs2 + as2,
            "imb": np.abs((bs1 + bs2) - (as1 + as2)) / (bs1 + as1 + bs2 + as2),
        })
    return out


def price_level(book: pd.DataFrame) -> pd.Series:
    """Recovered price level per window, the 1st-place trick.

    Prices in the data are normalised to start near 1, but they still move in steps of one cent
    of the real price. The smallest gap between distinct price values is therefore about
    0.01 / real_price, so 0.01 / gap estimates the real price. Windows of one stock with similar
    real prices are close in calendar time.
    """
    px = book[["time_id", "bid_price1", "ask_price1", "bid_price2", "ask_price2"]]
    long = px.melt(id_vars="time_id", value_name="p")[["time_id", "p"]].drop_duplicates()
    long = long.sort_values(["time_id", "p"])
    gap = long.groupby("time_id").p.diff()
    gap = gap[gap > 0]
    tick = gap.groupby(long.loc[gap.index, "time_id"]).min()
    return (0.01 / tick).rename("price_level")


def window_features(bf: pd.DataFrame, trades: pd.DataFrame | None, cut: int,
                    starts: list[int]) -> pd.DataFrame:
    """Per-window summaries of book rows (and trades) with ``start <= sec < cut``."""
    tids, idx = _group_index(bf.time_id.to_numpy())
    n = len(tids)
    sec = bf.sec.to_numpy()
    out = {"time_id": tids}
    for s in starts:
        m = (sec >= s) & (sec < cut)
        sfx = f"_{s}"
        for r in ("r1", "r2"):
            x = bf[r].to_numpy()
            ok = m & np.isfinite(x)
            out[f"rv{r[1]}{sfx}"] = np.sqrt(_sum(idx, n, x ** 2, ok))
        out[f"n_book{sfx}"] = _count(idx, n, m)
        if s == starts[0]:
            for c in ("spread", "spread2", "bid_gap", "ask_gap", "wap_bal", "depth", "imb"):
                out[f"{c}_mean"] = _mean(idx, n, bf[c].to_numpy(), m)
            out["spread_std"] = _std(idx, n, bf["spread"].to_numpy(), m)
            out["depth_std"] = _std(idx, n, bf["depth"].to_numpy(), m)
            r1 = bf["r1"].to_numpy()
            ok = m & np.isfinite(r1)
            q = _sum(idx, n, r1 ** 4, ok)
            rv2 = _sum(idx, n, r1 ** 2, ok)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["rquart"] = np.log(np.sqrt(np.sqrt(q)) + 1e-8)    # realized quarticity, 4th root
                bp = np.abs(r1) * np.abs(np.r_[np.nan, r1[:-1]])         # bipower: jump-robust
                same = np.r_[False, bf.time_id.to_numpy()[1:] == bf.time_id.to_numpy()[:-1]]
                out["bipower_ratio"] = (np.pi / 2) * _sum(idx, n, bp, ok & same) / rv2
    f = pd.DataFrame(out)

    if trades is not None:
        t = trades[trades.seconds_in_bucket < cut].sort_values(["time_id", "seconds_in_bucket"])
        tt = t.time_id.to_numpy()
        tr = _log_returns(t.price.to_numpy(float), tt)
        tpos = np.searchsorted(tids, tt)
        valid = (tpos < n) & (tids[np.minimum(tpos, n - 1)] == tt)
        tpos, ts = tpos[valid], t.seconds_in_bucket.to_numpy()[valid]
        tr = tr[valid]
        size = t["size"].to_numpy(float)[valid]
        oc = t.order_count.to_numpy(float)[valid]
        for s in starts:
            m = ts >= s
            sfx = f"_{s}"
            f[f"trv{sfx}"] = np.sqrt(_sum(tpos, n, tr ** 2, m & np.isfinite(tr)))
            f[f"n_trade{sfx}"] = _count(tpos, n, m)
            f[f"size{sfx}"] = _sum(tpos, n, size, m)
            f[f"orders{sfx}"] = _sum(tpos, n, oc, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["size_per_order"] = f[f"size_{starts[0]}"] / f[f"orders_{starts[0]}"]
    return f


def rv_target(bf: pd.DataFrame, cut: int) -> pd.DataFrame:
    """RV of WAP1 returns that end at or after second ``cut``, per window (``cut=300`` task)."""
    tids, idx = _group_index(bf.time_id.to_numpy())
    r = bf.r1.to_numpy()
    m = (bf.sec.to_numpy() >= cut) & np.isfinite(r)
    return pd.DataFrame({"time_id": tids, "target": np.sqrt(_sum(idx, len(tids), r ** 2, m)),
                         "n_after": _count(idx, len(tids), bf.sec.to_numpy() >= cut)})


def build_stock(stock_id: int, book: pd.DataFrame, trades: pd.DataFrame | None, cut: int,
                starts: list[int]) -> pd.DataFrame:
    bf = book_frame(book)
    f = window_features(bf, trades, cut, starts)
    f = f.merge(price_level(book[book.seconds_in_bucket < cut]).reset_index(), on="time_id", how="left")
    if cut < 600:
        f = f.merge(rv_target(bf, cut), on="time_id", how="left")
    f.insert(0, "stock_id", int(stock_id))
    return f


def add_log_features(df: pd.DataFrame) -> list[str]:
    """Log-transform the scale-like columns in place; return the model feature list."""
    skip = set(KEYS) | {"target", "n_after", "fold", "bipower_ratio", "rquart", "imb_mean",
                        "price_level"}
    feats = []
    for c in df.columns:
        if c in skip or c.startswith("y_"):
            continue
        if (df[c].dropna() >= 0).all():
            df[c] = np.log(df[c] + 1e-8)
        feats.append(c)
    return feats + ["bipower_ratio", "rquart", "imb_mean"]


def add_market_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Cross-sectional mean and std over stocks in the same window, plus each stock's deviation.

    Uses only features observed at the cut, so it is available in real time.
    """
    g = df.groupby("time_id")[cols]
    mean, std = g.transform("mean"), g.transform("std")
    new = []
    for c in cols:
        df[f"mkt_{c}"] = mean[c]
        df[f"mktsd_{c}"] = std[c]
        df[f"dev_{c}"] = df[c] - mean[c]
        new += [f"mkt_{c}", f"mktsd_{c}", f"dev_{c}"]
    return new


def add_stock_features(df: pd.DataFrame, cols: list[str], train_mask: np.ndarray) -> list[str]:
    """Each stock's average of ``cols`` over training windows only."""
    s = df[train_mask].groupby("stock_id")[cols].mean().add_prefix("stk_")
    for c in s.columns:
        df[c] = df.stock_id.map(s[c])
    return list(s.columns)


def _window_matrix(df: pd.DataFrame, pivot_col: str, metric: str):
    """One row per window: its cross-sectional vector of ``pivot_col``, scaled for distances."""
    piv = df.pivot(index="time_id", columns="stock_id", values=pivot_col)
    piv = piv.fillna(piv.mean())
    X = piv.to_numpy()
    if metric == "corr":
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    else:                       # min-max per stock, as in the 1st-place price-level neighbours
        X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    return piv.index.to_numpy(), X


def neighbour_features(df: pd.DataFrame, pivot_col: str, value_cols: list[str],
                       ks: list[int], name: str, metric: str = "corr") -> list[str]:
    """Average of ``value_cols`` over each window's k most similar other windows.

    Similarity is between windows' cross-sectional vectors of ``pivot_col`` (one value per
    stock). Uses features only, never targets, but neighbours may be later in calendar time.
    Small k (1-3) with price-level neighbours picks out the windows just before and after.
    """
    from sklearn.neighbors import NearestNeighbors

    tids, X = _window_matrix(df, pivot_col, metric)
    nn = NearestNeighbors(n_neighbors=max(ks) + 1).fit(X)
    _, ind = nn.kneighbors(X)
    ind = ind[:, 1:]                                  # drop the window itself
    new = []
    pos = pd.Series(np.arange(len(tids)), index=tids)
    row_t = pos.loc[df.time_id].to_numpy()
    for c in value_cols:
        vp = df.pivot(index="time_id", columns="stock_id", values=c).reindex(tids)
        V = vp.to_numpy()
        col_s = vp.columns.get_indexer(df.stock_id)
        for k in ks:
            nb = np.nanmean(V[ind[:, :k]], axis=1)    # (window, stock)
            df[f"nn{name}{k}_{c}"] = nb[row_t, col_s]
            new.append(f"nn{name}{k}_{c}")
    return new


def rmspe(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))
