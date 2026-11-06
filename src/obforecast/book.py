"""One-second book reconstruction, baseline features, outcomes and data-quality rows.

Ported from the merged team notebook (section 2.2/2.3). Definitions are unchanged; the only
difference is that timing is passed in as a ``Timing`` object instead of read from globals.

Invariants (tested in ``tests/test_book.py``):
* invalid level-1 snapshots (non-positive price or size, crossed or locked quotes) are dropped
  before forward-filling;
* a second carries forward the latest valid quote of the SAME window only, never backfills;
* features use seconds ``0 .. timing.cut - 1`` only; outcomes use later seconds and are never
  model inputs.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from .config import Timing

GRID = 600
BOOK_COLS = ["bid_price1", "ask_price1", "bid_price2", "ask_price2",
             "bid_size1", "ask_size1", "bid_size2", "ask_size2"]
KEYS = ["stock_id", "time_id"]


def snapshot_errors(book: pd.DataFrame) -> np.ndarray:
    """Reason a level-1 snapshot is a data error (empty string = valid)."""
    reason = np.full(len(book), "", dtype=object)
    reason[(book.ask_price1 <= book.bid_price1).to_numpy()] = "crossed or locked"
    reason[((book.bid_price1 <= 0) | (book.ask_price1 <= 0)).to_numpy()] = "nonpositive price"
    reason[((book.bid_size1 <= 0) | (book.ask_size1 <= 0)).to_numpy()] = "nonpositive size"
    return reason


def dense_grid(book: pd.DataFrame, tids=None):
    """Forward-filled one-second book state per window (rows) and second 0..599 (columns).

    Returns ``(tids, g, reason)`` where ``g`` maps each book column, ``"upd"`` (a snapshot
    arrived that second) and ``"age"`` (seconds since the standing quote arrived) to arrays.
    """
    reason = snapshot_errors(book)
    good = book[reason == ""]
    if tids is None:
        tids = np.sort(book.time_id.unique())
    row = np.searchsorted(tids, good.time_id.to_numpy())
    col = good.seconds_in_bucket.to_numpy()
    upd = np.zeros((len(tids), GRID), bool)
    upd[row, col] = True
    sec = np.arange(GRID)
    last = np.maximum.accumulate(np.where(upd, sec, -1), axis=1)   # last update at or before t
    has = last >= 0
    take = np.where(has, last, 0)
    rows = np.arange(len(tids))[:, None]
    g = {}
    for c in BOOK_COLS:
        a = np.full((len(tids), GRID), np.nan)
        a[row, col] = good[c].to_numpy(dtype=float)
        g[c] = np.where(has, a[rows, take], np.nan)
    g["upd"] = upd
    g["age"] = np.where(has, sec - last, np.nan)
    # level 2 is only used for features; an inconsistent level 2 is treated as missing
    with np.errstate(invalid="ignore"):
        bad2 = ((g["bid_price2"] <= 0) | (g["ask_price2"] <= 0)
                | (g["bid_price2"] > g["bid_price1"]) | (g["ask_price2"] < g["ask_price1"]))
    for c in ["bid_price2", "ask_price2", "bid_size2", "ask_size2"]:
        g[c] = np.where(bad2, np.nan, g[c])
    return tids, g, reason


def nanlogmean(a, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.nanmean(a[:, lo:hi], axis=1))


def build_stock(stock_id: int, book: pd.DataFrame, timing: Timing = Timing()):
    """Features (seconds 0..cut-1 only), outcomes (targets + simulation inputs) and quality rows."""
    import warnings
    CUT, ENTRY, EXIT = timing.cut, timing.entry_second, timing.exit_second
    tids, g, reason = dense_grid(book)
    bp1, ap1, bp2, ap2 = g["bid_price1"], g["ask_price1"], g["bid_price2"], g["ask_price2"]
    bs1, as1, bs2, as2 = g["bid_size1"], g["ask_size1"], g["bid_size2"], g["ask_size2"]
    mid = (ap1 + bp1) / 2
    q = (ap1 - bp1) / mid                           # relative quoted spread
    wap = (bp1 * as1 + ap1 * bs1) / (bs1 + as1)
    keys = pd.DataFrame({"stock_id": int(stock_id), "time_id": tids})

    # ---- data quality: invalid snapshots and windows without an initial state
    bad = pd.Series(reason[reason != ""]).value_counts()
    t = CUT - 1
    no_state = np.isnan(mid[:, t])
    quality = keys.assign(
        invalid_snapshots=pd.Series(reason != "").groupby(book.time_id.to_numpy()).sum()
        .reindex(tids).to_numpy(),
        no_initial_state=no_state,
        no_update_after_cut=~g["upd"][:, CUT:].any(axis=1),
        age_last_feature=g["age"][:, t], age_entry=g["age"][:, ENTRY], age_exit=g["age"][:, EXIT],
    )

    f = keys.copy()
    o = keys.copy()
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN slices -> NaN, intended
        # ---- features: seconds 0..cut-1 only
        S1 = np.nanmean(q[:, :CUT], axis=1)
        ls1 = np.log(S1)
        r = np.diff(np.log(wap), axis=1, prepend=np.nan)            # 1-second WAP log returns
        f["q_299"] = q[:, t]                                         # current relative spread
        f["log_S1"] = ls1
        f["sp_last60"] = nanlogmean(q, CUT - 60, CUT) - ls1
        f["sp_last30"] = nanlogmean(q, CUT - 30, CUT) - ls1
        f["sp_last10"] = nanlogmean(q, CUT - 10, CUT) - ls1
        f["sp_now"] = np.log(q[:, t]) - ls1
        f["sp_first60"] = nanlogmean(q, 0, 60) - ls1
        f["sp_logstd"] = np.nanstd(np.log(q[:, :CUT]), axis=1)
        abs_sp = (ap1 - bp1)[:, :CUT]
        min_sp = np.nanmin(np.where(abs_sp > 0, abs_sp, np.nan), axis=1)
        f["sp_in_min"] = np.log(np.nanmean(abs_sp, axis=1) / min_sp)
        f["frac_at_min"] = np.nanmean(np.where(np.isnan(abs_sp), np.nan,
                                               abs_sp <= min_sp[:, None] * 1.001), axis=1)
        f["log_rel_min_sp"] = np.log(min_sp / mid[:, t])
        f["log_rv"] = np.log(np.sqrt(np.nansum(r[:, 1:CUT] ** 2, axis=1)) + 1e-5)
        f["log_rv_last100"] = np.log(np.sqrt(np.nansum(r[:, CUT - 100:CUT] ** 2, axis=1)) + 1e-5)
        f["rv_ratio"] = f["log_rv_last100"] - f["log_rv"]
        first_mid = mid[np.arange(len(tids)), np.argmax(~np.isnan(mid[:, :CUT]), axis=1)]
        f["ret_0_299"] = np.log(mid[:, t] / first_mid) * 1e4                     # bps
        f["ret_last60"] = np.log(mid[:, t] / mid[:, CUT - 61]) * 1e4
        f["ret_last10"] = np.log(mid[:, t] / mid[:, CUT - 11]) * 1e4
        f["abs_ret"] = np.abs(f["ret_0_299"])
        f["wap_tilt_now"] = (wap[:, t] - mid[:, t]) / (ap1[:, t] - bp1[:, t])    # in [-0.5, 0.5]
        f["wap_tilt_last30"] = np.nanmean(((wap - mid) / (ap1 - bp1))[:, CUT - 30:CUT], axis=1)
        n_upd = g["upd"][:, :CUT].sum(axis=1)
        f["log_nupd"] = np.log(np.maximum(n_upd, 1))
        f["upd_last60"] = np.log(g["upd"][:, CUT - 60:CUT].sum(axis=1) + 1) - np.log(n_upd / 5 + 1)
        f["quote_age_299"] = g["age"][:, t]
        dep1 = bs1 + as1
        f["log_depth1"] = nanlogmean(dep1, 0, CUT)
        f["log_depth2"] = nanlogmean(bs2 + as2, 0, CUT)
        f["depth1_last60"] = nanlogmean(dep1, CUT - 60, CUT) - f["log_depth1"]
        f["imbalance_abs"] = np.nanmean((np.abs(bs1 - as1) / dep1)[:, :CUT], axis=1)
        f["imb_now"] = ((bs1 - as1) / dep1)[:, t]
        f["imb_last30"] = np.nanmean(((bs1 - as1) / dep1)[:, CUT - 30:CUT], axis=1)
        f["sp2_rel"] = nanlogmean((ap2 - bp2) / mid, 0, CUT) - ls1

        # ---- outcomes: targets and simulation inputs (seconds >= cut). Never model inputs.
        o["y_price"] = (mid[:, EXIT] / mid[:, t] - 1) * 1e4                      # bps
        o["S2"] = np.nanmean(q[:, CUT:], axis=1)
        o["y_spread"] = np.log(o["S2"] / S1)                                     # log(S2/S1)
        o["q_max_2nd_over_S1"] = np.nanmax(q[:, CUT:], axis=1) / S1
        o["frac_above_1p5_S1"] = np.nanmean(q[:, CUT:] >= 1.5 * S1[:, None], axis=1)
    for d in timing.entry_delays:
        o[f"ask_{ENTRY + d}"] = ap1[:, ENTRY + d]
        o[f"ask_size_{ENTRY + d}"] = as1[:, ENTRY + d]
        o[f"age_{ENTRY + d}"] = g["age"][:, ENTRY + d]
    o[f"bid_{EXIT}"] = bp1[:, EXIT]
    o[f"bid_size_{EXIT}"] = bs1[:, EXIT]
    o[f"age_{EXIT}"] = g["age"][:, EXIT]

    keep = ~no_state
    return (f[keep].reset_index(drop=True), o[keep].reset_index(drop=True), quality,
            bad.rename(int(stock_id)))


def build_tables(source, stock_ids, timing: Timing = Timing(), log=print):
    """Features, outcomes, data-quality rows and invalid-snapshot counts for every stock."""
    started = perf_counter()
    F, O, Q, B = [], [], [], []
    for i, sid in enumerate(stock_ids):
        f, o, qual, bad = build_stock(sid, source.load("book", sid), timing)
        F.append(f); O.append(o); Q.append(qual); B.append(bad)
        if log and ((i + 1) % 20 == 0 or i + 1 == len(stock_ids)):
            log(f"  built {i + 1}/{len(stock_ids)} stocks ({(perf_counter() - started) / 60:.1f} min)")
    feats, outs, qual = (pd.concat(x, ignore_index=True) for x in (F, O, Q))
    invalid = pd.DataFrame(B).fillna(0).astype(int).rename_axis("stock_id").reset_index()
    assert not feats.duplicated(KEYS).any() and not outs.duplicated(KEYS).any()
    assert feats[KEYS].equals(outs[KEYS]), "features and outcomes must share keys row for row"
    return feats, outs, qual, invalid
