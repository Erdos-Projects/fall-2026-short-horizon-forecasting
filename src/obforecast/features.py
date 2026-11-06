"""Fold-level model inputs: baseline (47 columns) and upgraded (131 columns incl. stock_id).

Ported from the merged team notebook sections 3.1 and 5.2b. Everything learned from data
(stock means, stock target statistics) is computed from the TRAINING frame only. Market
features are leave-one-out means over the other stocks in the same window; folds are grouped
by ``time_id``, so each frame always holds whole windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .extra import XCOLS

FAMILIES = {
    "spread": ["q_299", "log_S1", "sp_last60", "sp_last30", "sp_last10", "sp_now", "sp_first60",
               "sp_logstd", "sp2_rel"],
    "min_spread": ["sp_in_min", "frac_at_min", "log_rel_min_sp"],
    "volatility": ["log_rv", "log_rv_last100", "rv_ratio"],
    "price": ["ret_0_299", "ret_last60", "ret_last10", "abs_ret", "wap_tilt_now", "wap_tilt_last30"],
    "activity": ["log_nupd", "upd_last60", "quote_age_299"],
    "depth": ["log_depth1", "log_depth2", "depth1_last60"],
    "imbalance": ["imbalance_abs", "imb_now", "imb_last30"],
}
BASE = sum(FAMILIES.values(), [])
VS_STOCK = ["log_S1", "log_rv", "log_nupd", "log_depth1"]          # minus the stock's training-fold mean
MKT_SRC = ["sp_last60", "sp_now", "log_S1_vs_stock", "log_rv_vs_stock", "rv_ratio", "ret_0_299",
           "ret_last60", "log_nupd_vs_stock", "upd_last60", "log_depth1_vs_stock", "imb_now"]
REL = ["sp_last60", "ret_0_299"]                                    # own value minus market value
FEATURES = (BASE + [c + "_vs_stock" for c in VS_STOCK] + ["mkt_" + c for c in MKT_SRC]
            + ["rel_" + c for c in REL])
FAMILY_OF = {c: fam for fam, cols in FAMILIES.items() for c in cols}

# upgraded-model additions (section 5.2b)
EXTRA_MKT = [c for c in BASE if c not in MKT_SRC]
UP_REL = ["sp_last30", "sp_last10", "sp_now", "log_rv", "ret_last60", "ret_last10", "imb_now", "log_nupd"]
UP_XSD = ["ret_last60", "ret_0_299", "sp_last60"]
UP_EXTRA_MKT_VS = ["x_lq_p50", "x_n_spchg", "t_ln", "t_lvol", "t_lrv", "x_frac_wide", "t_since"]
UP_TARGETS = {"spread": "y_spread", "price": "y_price"}


def family(col: str) -> str:
    base = col.removeprefix("mkt_").removeprefix("rel_").removesuffix("_vs_stock")
    return FAMILY_OF[base]


def leave_one_out_mean(d: pd.DataFrame, col: str) -> pd.Series:
    """Mean of ``col`` over the OTHER stocks in the same window (focal stock excluded)."""
    g = d.groupby("time_id")[col]
    s, n = g.transform("sum"), g.transform("count")
    x = d[col]
    return ((s - x.fillna(0)) / (n - x.notna())).where(n - x.notna() > 0)


def fold_features(tr: pd.DataFrame, te: pd.DataFrame):
    """Baseline model inputs (``FEATURES``) for a training and an evaluation frame."""
    means = tr.groupby("stock_id")[VS_STOCK].mean()
    out = []
    for d in (tr, te):
        d = d.copy()
        for c in VS_STOCK:
            d[c + "_vs_stock"] = d[c] - d.stock_id.map(means[c]).fillna(tr[c].mean())
        for c in MKT_SRC:
            d["mkt_" + c] = leave_one_out_mean(d, c)
        for c in REL:
            d["rel_" + c] = d[c] - d["mkt_" + c]
        out.append(d[FEATURES])
    return out


def upgraded_features(tr: pd.DataFrame, te: pd.DataFrame, y_tr: dict):
    """Section 5.2b inputs: baseline + market/relative/dispersion/stock-level/extra inputs.

    ``tr``/``te`` must contain the baseline features and the 31 ``XCOLS``. ``y_tr`` maps
    target name -> training-fold target array (used for the stock target mean/std ``te_*``,
    ``tsd_*``). Note: these stock statistics include each training row's own target, so they
    are mildly optimistic IN-SAMPLE (a training-fit effect), never computed from ``te``.
    """
    st_mean, st_std = {}, {}
    for t, y in y_tr.items():
        ok = np.isfinite(y)
        grp = pd.Series(y[ok]).groupby(tr.stock_id.to_numpy()[ok])
        st_mean[t], st_std[t] = grp.mean(), grp.std()
    out = []
    for d, X in zip((tr, te), fold_features(tr, te)):
        X = X.copy()
        for c in EXTRA_MKT:
            X["mkt_" + c] = leave_one_out_mean(d, c).to_numpy()
        for c in UP_REL:
            X["rel_" + c] = X[c] - X["mkt_" + c]
        for c in UP_XSD:
            X["xsd_" + c] = d.groupby("time_id")[c].transform("std").to_numpy()
        for t in y_tr:
            X[f"te_{t}"] = d.stock_id.map(st_mean[t]).to_numpy()
            X[f"tsd_{t}"] = d.stock_id.map(st_std[t]).to_numpy()
        for c in XCOLS:
            X[c] = d[c].to_numpy()
        for c in UP_EXTRA_MKT_VS:
            X["mkt_" + c] = leave_one_out_mean(d, c).to_numpy()
            X[c + "_vs_stock"] = d[c].to_numpy() - d.stock_id.map(tr.groupby("stock_id")[c].mean()).to_numpy()
        cats = sorted(int(v) for v in tr.stock_id.unique())
        seen = d.stock_id.isin(cats).to_numpy()            # stocks absent from training -> missing
        X["stock_id"] = pd.Categorical.from_codes(
            np.where(seen, np.searchsorted(cats, d.stock_id.to_numpy()), -1), categories=cats)
        out.append(X)
    return out


def is_stock_specific(col: str) -> bool:
    return col == "stock_id" or col.startswith(("te_", "tsd_")) or col.endswith("_vs_stock")


def upgraded_feature_names() -> list[str]:
    """Upgraded input names in construction order (derived from the same rules as above)."""
    names = list(FEATURES)
    names += ["mkt_" + c for c in EXTRA_MKT]
    names += ["rel_" + c for c in UP_REL]
    names += ["xsd_" + c for c in UP_XSD]
    for t in UP_TARGETS:
        names += [f"te_{t}", f"tsd_{t}"]
    names += list(XCOLS)
    for c in UP_EXTRA_MKT_VS:
        names += ["mkt_" + c, c + "_vs_stock"]
    names.append("stock_id")
    return names


def feature_inventory() -> pd.DataFrame:
    """One row per model input with its set, group and provenance; exported to results/metrics."""
    rows = []
    for c in FEATURES:
        kind = ("market (leave-one-out)" if c.startswith("mkt_") else "relative to market"
                if c.startswith("rel_") else "stock-relative (train-fold mean)" if c.endswith("_vs_stock")
                else "raw first-half")
        rows.append({"feature": c, "baseline_47": True, "upgraded_131": True, "family": family(c), "kind": kind})
    base_set = set(FEATURES)
    for c in upgraded_feature_names():
        if c in base_set:
            continue
        if c == "stock_id":
            fam, kind = "identity", "categorical stock id"
        elif c.startswith(("te_", "tsd_")):
            fam, kind = "target statistic", "stock target mean/std (train fold)"
        elif c.startswith("xsd_"):
            fam, kind = family(c.removeprefix("xsd_")), "cross-stock dispersion in window"
        elif c in XCOLS:
            fam = "trade prints" if c.startswith("t_") else "spread path / depth"
            kind = "raw first-half (extra)"
        else:
            src = c.removeprefix("mkt_").removeprefix("rel_").removesuffix("_vs_stock")
            fam = ("trade prints" if src.startswith("t_") else "spread path / depth" if src.startswith("x_")
                   else family(src))
            kind = ("market (leave-one-out)" if c.startswith("mkt_") else "relative to market"
                    if c.startswith("rel_") else "stock-relative (train-fold mean)")
        rows.append({"feature": c, "baseline_47": False, "upgraded_131": True, "family": fam, "kind": kind})
    return pd.DataFrame(rows)
