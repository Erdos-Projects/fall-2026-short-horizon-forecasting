"""Extra first-half inputs for the upgraded models (section 5.2a) and path points (section 8.2).

Ported from the merged team notebook. ``extra_stock`` produces 20 spread-path/depth inputs
(``x_*``) and 11 trade-print inputs (``t_*``): 31 columns in total. All use book seconds
``0 .. cut-1`` and trades with ``seconds_in_bucket < cut``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .book import dense_grid
from .config import Timing

EXTRA_BOOK_COLS = ["x_lq_p10", "x_lq_p50", "x_lq_p90", "x_lq_max60", "x_lq_min60", "x_lq_120",
                   "x_lq_200_240", "x_sp_slope", "x_n_spchg", "x_n_spchg60", "x_since_spchg",
                   "x_frac_wide", "x_frac_wide60", "x_lq_now_vs_med", "x_lq2_60", "x_l2gap",
                   "x_ldep60", "x_ldep_trend", "x_rv30", "x_rv_first150"]
EXTRA_TRADE_COLS = ["t_ln", "t_lvol", "t_loc", "t_ln60", "t_lvol60", "t_since", "t_lrv",
                    "t_avgsize", "t_vwap_vs_mid", "t_vwap60_vs_mid", "t_per_upd"]
XCOLS = EXTRA_BOOK_COLS + EXTRA_TRADE_COLS


def extra_stock(stock_id: int, book: pd.DataFrame, trades: pd.DataFrame,
                timing: Timing = Timing()) -> pd.DataFrame:
    CUT = timing.cut
    SEC = np.arange(CUT)
    tids, g, _ = dense_grid(book)
    bp, ap = g["bid_price1"][:, :CUT], g["ask_price1"][:, :CUT]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mid = (ap + bp) / 2
        q = (ap - bp) / mid
        lq = np.log(q)
        med = np.nanmedian(q, axis=1, keepdims=True)
        dq = np.abs(np.diff(q, axis=1)) > 1e-12
        chg_sec = np.where(np.concatenate([np.zeros((len(tids), 1), bool), dq], axis=1), SEC, -1)
        xc = SEC - SEC.mean()
        ok = np.isfinite(lq)
        lq0 = np.where(ok, lq, 0.0)
        xm = (ok * xc).sum(1) / ok.sum(1)
        slope = (((lq0 - np.nanmean(lq, 1, keepdims=True)) * (xc - xm[:, None]) * ok).sum(1)
                 / (((xc - xm[:, None]) ** 2) * ok).sum(1))
        q2 = (g["ask_price2"][:, :CUT] - g["bid_price2"][:, :CUT]) / mid
        dep = (g["bid_size1"][:, :CUT] + g["ask_size1"][:, :CUT]
               + np.nan_to_num(g["bid_size2"][:, :CUT]) + np.nan_to_num(g["ask_size2"][:, :CUT]))
        wap = ((bp * g["ask_size1"][:, :CUT] + ap * g["bid_size1"][:, :CUT])
               / (g["bid_size1"][:, :CUT] + g["ask_size1"][:, :CUT]))
        lr = np.diff(np.log(wap), axis=1)
        f = pd.DataFrame({
            "stock_id": int(stock_id), "time_id": tids,
            "x_lq_p10": np.nanpercentile(lq, 10, axis=1), "x_lq_p50": np.nanpercentile(lq, 50, axis=1),
            "x_lq_p90": np.nanpercentile(lq, 90, axis=1), "x_lq_max60": np.nanmax(lq[:, -60:], axis=1),
            "x_lq_min60": np.nanmin(lq[:, -60:], axis=1), "x_lq_120": np.log(np.nanmean(q[:, -120:], axis=1)),
            "x_lq_200_240": np.log(np.nanmean(q[:, 200:240], axis=1)),
            "x_sp_slope": slope * 100, "x_n_spchg": np.log1p(dq.sum(1)),
            "x_n_spchg60": np.log1p(dq[:, -60:].sum(1)),
            "x_since_spchg": CUT - 1 - chg_sec.max(1),
            "x_frac_wide": np.nanmean(q > 1.5 * med, axis=1),
            "x_frac_wide60": np.nanmean(q[:, -60:] > 1.5 * med, axis=1),
            "x_lq_now_vs_med": lq[:, -1] - np.log(med[:, 0]),
            "x_lq2_60": np.log(np.nanmean(q2[:, -60:], axis=1)),
            "x_l2gap": np.log(np.nanmean(q2[:, -60:], axis=1)) - np.log(np.nanmean(q[:, -60:], axis=1)),
            "x_ldep60": np.log(np.nanmean(dep[:, -60:], axis=1)),
            "x_ldep_trend": np.log(np.nanmean(dep[:, -60:], axis=1)) - np.log(np.nanmean(dep[:, :240], axis=1)),
            "x_rv30": np.log(np.sqrt(np.nansum(lr[:, -30:] ** 2, axis=1)) + 1e-5),
            "x_rv_first150": np.log(np.sqrt(np.nansum(lr[:, :150] ** 2, axis=1)) + 1e-5),
        })
    tr = trades[trades.seconds_in_bucket < CUT].sort_values(["time_id", "seconds_in_bucket"]).copy()
    tr["lr"] = np.log(tr.price).groupby(tr.time_id).diff()
    tr["notional"] = tr.price * tr["size"]
    a = tr.groupby("time_id").agg(n=("size", "size"), vol=("size", "sum"), oc=("order_count", "sum"),
                                  last=("seconds_in_bucket", "max"),
                                  rvt=("lr", lambda s: np.sqrt((s ** 2).sum())),
                                  notional=("notional", "sum"))
    b = tr[tr.seconds_in_bucket >= CUT - 60].groupby("time_id").agg(
        n60=("size", "size"), vol60=("size", "sum"), notional60=("notional", "sum"))
    a = a.join(b, how="left").reindex(tids)
    mid_last = pd.Series(mid[:, -1], index=tids)
    with np.errstate(all="ignore"):
        f["t_ln"] = np.log1p(a.n.fillna(0).to_numpy())
        f["t_lvol"] = np.log1p(a.vol.fillna(0).to_numpy())
        f["t_loc"] = np.log1p(a.oc.fillna(0).to_numpy())
        f["t_ln60"] = np.log1p(a.n60.fillna(0).to_numpy())
        f["t_lvol60"] = np.log1p(a.vol60.fillna(0).to_numpy())
        f["t_since"] = (CUT - 1 - a["last"]).fillna(CUT).to_numpy()
        f["t_lrv"] = np.log(a.rvt.fillna(0).to_numpy() + 1e-5)
        f["t_avgsize"] = np.log1p((a.vol / a.n).to_numpy())
        f["t_vwap_vs_mid"] = ((a.notional / a.vol / mid_last - 1) * 1e4).to_numpy()
        f["t_vwap60_vs_mid"] = ((a.notional60 / a.vol60 / mid_last - 1) * 1e4).to_numpy()
        f["t_per_upd"] = f["t_ln"] - np.log1p(g["upd"][:, :CUT].sum(1))
    assert list(f.columns[2:]) == XCOLS
    return f


def path_points(stock_id: int, book: pd.DataFrame, path_seconds, timing: Timing = Timing()):
    """Mid at chosen seconds and the exit-quote age, for the section 8.2 stress tests.

    These are OUTCOME-side quantities except ``mid_<s>`` for s < cut, which are used only to
    build lagged (seconds <= 269) placebo inputs.
    """
    CUT = timing.cut
    tids, g, _ = dense_grid(book)
    with np.errstate(all="ignore"):
        mid = (g["ask_price1"] + g["bid_price1"]) / 2
    d = {"stock_id": int(stock_id), "time_id": tids, "age_exit_quote": g["age"][:, timing.exit_second],
         "mid_first": mid[np.arange(len(tids)), np.argmax(~np.isnan(mid[:, :CUT]), axis=1)]}
    d.update({f"mid_{s}": mid[:, s] for s in sorted({CUT - 1, *path_seconds})})
    return pd.DataFrame(d)
