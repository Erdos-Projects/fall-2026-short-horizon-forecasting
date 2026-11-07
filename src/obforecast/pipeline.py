"""Analysis stages of the merged notebook as functions with explicit inputs (no hidden state).

Section numbers refer to ``notebooks/04_forecasting_and_trading.ipynb`` (same numbering as the
archived merged notebook). Every function returns DataFrames/dicts; ``run.py`` writes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .book import KEYS
from .config import Timing, taus
from .features import (FAMILIES, FEATURES, MKT_SRC, REL, UP_TARGETS, VS_STOCK, family,
                       fold_features, is_stock_specific, leave_one_out_mean, upgraded_features)
from .models import (TARGETS, baselines, fit_predict, heads, lgb_params, mae, make_head, r2, rmse,
                     upgraded_predict)
from .trading import (decide, fillable, net_return_bps, paired_bootstrap, per_opportunity,
                      policy_summary, select_rule, window_t)
from .validation import assign_folds, benchmark_split, check_grouped

POLICIES = ["price only", "price + spread"]


# ------------------------------------------------------------------------------------ setup
@dataclass
class Dev:
    """Development and benchmark frames plus descriptive stock groups."""
    dev_f: pd.DataFrame
    dev_o: pd.DataFrame
    bench_f: pd.DataFrame
    bench_o: pd.DataFrame
    split_info: dict
    activity: np.ndarray = field(repr=False, default=None)
    min_bound: np.ndarray = field(repr=False, default=None)


def prepare(feats, outs, cfg, split_path=None) -> Dev:
    s = cfg["split"]
    bench_tids, info = benchmark_split(feats.time_id.unique(), s["benchmark_seed"], s["benchmark_frac"], split_path)
    is_bench = feats.time_id.isin(bench_tids).to_numpy()
    dev_f, dev_o = feats[~is_bench].reset_index(drop=True), outs[~is_bench].reset_index(drop=True)
    bench_f, bench_o = feats[is_bench].reset_index(drop=True), outs[is_bench].reset_index(drop=True)
    dev_f["fold"] = assign_folds(dev_f.time_id.to_numpy(), s["outer_folds"], seed=s["outer_fold_seed"])
    assert check_grouped(dev_f.time_id, dev_f.fold), "a time_id spans two folds"
    # descriptive stock groups for reporting only (from development rows; never model inputs)
    stats = dev_f.groupby("stock_id").agg(nupd=("log_nupd", "mean"), at_min=("frac_at_min", "mean"))
    n_groups = min(3, stats.nupd.nunique())
    labels = ["low", "mid", "high"][:n_groups] if n_groups == 3 else [f"g{i}" for i in range(n_groups)]
    activity = dev_f.stock_id.map(pd.qcut(stats.nupd, n_groups, labels=labels)).astype(str).to_numpy()
    min_bound = dev_f.stock_id.map(stats.at_min > 0.5).to_numpy().astype(bool)
    info.update(dev_windows=int(dev_f.time_id.nunique()), dev_rows=int(len(dev_f)),
                bench_windows=int(bench_f.time_id.nunique()), bench_rows=int(len(bench_f)),
                outer_fold_windows=dev_f.groupby("fold").time_id.nunique().to_dict())
    return Dev(dev_f, dev_o, bench_f, bench_o, info, activity, min_bound)


# ------------------------------------------------------------------- section 5: nested CV
def inner_forecasts(f_tr, o_tr, seed, cfg):
    """Inner grouped out-of-fold forecasts of every head, for both targets."""
    k_in = cfg["split"]["inner_folds"]
    inner = assign_folds(f_tr.time_id.to_numpy(), k_in, seed=seed)
    H = heads(cfg)
    preds = {t: {h: np.full(len(f_tr), np.nan) for h in H} for t in TARGETS}
    for j in range(k_in):
        a, b = inner != j, inner == j
        X_a, X_b = fold_features(f_tr[a], f_tr[b])
        for t, col in TARGETS.items():
            y_a = o_tr[col].to_numpy()[a]
            for h in H:
                preds[t][h][b] = fit_predict(h, X_a, y_a, X_b, cfg)
    return preds


@dataclass
class CVResult:
    oof: pd.DataFrame
    selections: pd.DataFrame
    rule_tables: pd.DataFrame
    inner_store: dict
    minutes: float


def nested_cv(d: Dev, cfg, timing: Timing, log=print) -> CVResult:
    dev_f, dev_o = d.dev_f, d.dev_o
    s, tr_cfg = cfg["split"], cfg["trading"]
    REF, H = tr_cfg["reference_cost_bps"], heads(cfg)
    margins, tau_list, min_tr, Q = tr_cfg["margins_bps"], taus(cfg), tr_cfg["min_inner_trades"], tr_cfg["quantity"]
    oof = dev_f[KEYS + ["fold"]].copy()
    selections, rule_tables, inner_store = [], [], {}
    t_all = perf_counter()
    for k in range(s["outer_folds"]):
        t0 = perf_counter()
        tr, va = (dev_f.fold != k).to_numpy(), (dev_f.fold == k).to_numpy()
        f_tr, o_tr = dev_f[tr].reset_index(drop=True), dev_o[tr].reset_index(drop=True)
        f_va, o_va = dev_f[va].reset_index(drop=True), dev_o[va].reset_index(drop=True)

        # 1) inner folds: choose one head per target by inner RMSE
        ip = inner_forecasts(f_tr, o_tr, seed=s["inner_seed_offset"] + k, cfg=cfg)
        chosen = {t: min(H, key=lambda h: rmse(o_tr[col].to_numpy(), ip[t][h])) for t, col in TARGETS.items()}
        ridge_best = {t: min([h for h in H if h.startswith("ridge")],
                             key=lambda h: rmse(o_tr[col].to_numpy(), ip[t][h])) for t, col in TARGETS.items()}

        # 2) inner folds: choose trading rules on inner forecasts and realised training outcomes
        net_ref = np.where(fillable(o_tr, timing, 0, Q), net_return_bps(o_tr, REF, timing, 0, Q), 0.0)
        rules = {}
        for pol in POLICIES:
            best, tab = select_rule(pol, ip["price"][chosen["price"]], ip["spread"][chosen["spread"]],
                                    f_tr.q_299.to_numpy(), net_ref, REF, margins, tau_list, min_tr)
            rules[pol] = best
            rule_tables.append(tab.assign(fold=k, policy=pol))
        inner_store[k] = {"price": ip["price"][chosen["price"]], "spread": ip["spread"][chosen["spread"]],
                          "rows": dev_f.index[tr].to_numpy()}

        # 3) refit on the whole outer-training fold, predict the outer validation fold
        X_tr, X_va = fold_features(f_tr, f_va)
        idx = dev_f.index[va]
        for t, col in TARGETS.items():
            y_tr = o_tr[col].to_numpy()
            for name, p in baselines(t, f_tr, o_tr, f_va).items():
                oof.loc[idx, f"{t}: {name}"] = p
            fitted = {h: fit_predict(h, X_tr, y_tr, X_va, cfg) for h in {ridge_best[t], "HGB", chosen[t]}}
            oof.loc[idx, f"{t}: ridge"] = fitted[ridge_best[t]]
            oof.loc[idx, f"{t}: HGB"] = fitted["HGB"]
            oof.loc[idx, f"{t}: selected"] = fitted[chosen[t]]

        # 4) apply the frozen rules to the outer validation fold
        pp, ps = oof.loc[idx, "price: selected"].to_numpy(), oof.loc[idx, "spread: selected"].to_numpy()
        for pol in POLICIES:
            r = rules[pol]
            oof.loc[idx, f"enter: {pol}"] = (decide(pp, ps, f_va.q_299.to_numpy(), r[0], r[1], REF)
                                             if r is not None else False)
        selections.append({"fold": k, "price head": chosen["price"], "spread head": chosen["spread"],
                           "price ridge alpha": ridge_best["price"], "spread ridge alpha": ridge_best["spread"],
                           **{f"{pol}: margin_bps": (rules[pol] or (None,))[0] for pol in POLICIES},
                           **{f"{pol}: tau": (rules[pol] or (None, None))[1] for pol in POLICIES},
                           **{f"{pol}: inner trades": (rules[pol] or (None, None, None, None))[3] for pol in POLICIES},
                           "minutes": (perf_counter() - t0) / 60})
        if log:
            log(f"outer fold {k + 1}/{s['outer_folds']}: heads {chosen} | rules "
                f"{ {p: (rules[p][:2] if rules[p] else None) for p in POLICIES} } | {(perf_counter() - t0) / 60:.1f} min")
    for pol in POLICIES:
        oof[f"enter: {pol}"] = oof[f"enter: {pol}"].astype(bool)
    return CVResult(oof, pd.DataFrame(selections), pd.concat(rule_tables, ignore_index=True), inner_store,
                    (perf_counter() - t_all) / 60)


def forecast_table(t: str, d: Dev, oof: pd.DataFrame) -> pd.DataFrame:
    """Section 5.1: MAE, RMSE, R^2 (overall and by stock group) and RMSE gain over baselines."""
    y = d.dev_o[TARGETS[t]].to_numpy()
    models = [c.split(": ")[1] for c in oof.columns if c.startswith(f"{t}: ")]
    groups = {g: d.activity == g for g in pd.unique(d.activity)}
    rows = {}
    for m in models:
        p = oof[f"{t}: {m}"].to_numpy()
        row = {"MAE": mae(y, p), "RMSE": rmse(y, p), "R2": r2(y, p)}
        for g, mask in groups.items():
            row[f"R2 {g}-activity stocks"] = r2(y[mask], p[mask])
        if d.min_bound.any():
            row["R2 min-spread-bound"] = r2(y[d.min_bound], p[d.min_bound])
        row["median per-stock R2"] = float(pd.Series({s_: r2(y[g], p[g]) for s_, g in
                                                      d.dev_f.groupby("stock_id").indices.items()}).median())
        rows[m] = row
    tab = pd.DataFrame(rows).T
    for b in ["zero", "stock mean", "persistence"]:
        tab[f"RMSE gain vs {b}"] = 1 - tab.RMSE / tab.loc[b, "RMSE"]
    return tab.rename_axis("model").reset_index().assign(target=t)


def calibration(t: str, d: Dev, oof: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    y, p = d.dev_o[TARGETS[t]].to_numpy(), oof[f"{t}: selected"].to_numpy()
    b = pd.qcut(p, bins, labels=False, duplicates="drop")
    return (pd.DataFrame({"p": p, "y": y}).groupby(b)
            .agg(pred=("p", "mean"), actual=("y", "mean"), se=("y", "sem"), n=("y", "size"))
            .rename_axis("decile").reset_index().assign(target=t))


# --------------------------------------------------------------------- section 6: trading
def trading(d: Dev, cv: CVResult, cfg, timing: Timing) -> dict:
    tr_cfg, bs = cfg["trading"], cfg["bootstrap"]
    REF, Q = tr_cfg["reference_cost_bps"], tr_cfg["quantity"]
    dev_o, tid = d.dev_o, d.dev_f.time_id.to_numpy()
    ENTER = {"cash": np.zeros(len(cv.oof), bool), **{p: cv.oof[f"enter: {p}"].to_numpy() for p in POLICIES}}
    summary = pd.DataFrame({p: policy_summary(e, dev_o, REF, timing, quantity=Q) for p, e in ENTER.items()}).T
    comps = []
    po = {p: per_opportunity(ENTER[p], dev_o, REF, timing, quantity=Q) for p in POLICIES}
    m, lo, hi = paired_bootstrap(po["price + spread"], po["price only"], tid, bs["resamples"], bs["seed"])
    comps.append({"comparison": "price + spread minus price only", "mean_bps": m, "ci_low": lo, "ci_high": hi})
    for p in POLICIES:
        m, lo, hi = paired_bootstrap(po[p], np.zeros(len(tid)), tid, bs["resamples"], bs["seed"])
        comps.append({"comparison": f"{p} minus cash", "mean_bps": m, "ci_low": lo, "ci_high": hi})
    done = ENTER["price only"] & fillable(dev_o, timing, 0, Q)
    t_stat, n_w = window_t(net_return_bps(dev_o, REF, timing, 0, Q), done, tid)
    comps = pd.DataFrame(comps).assign(resamples=bs["resamples"], cluster="time_id", cost_bps_per_side=REF)
    return {"summary": summary.rename_axis("policy").reset_index(), "bootstrap": comps,
            "window_t": {"price only window-level t": t_stat, "traded windows": n_w}, "ENTER": ENTER}


def sensitivity(d: Dev, cv: CVResult, ENTER: dict, cfg, timing: Timing) -> dict:
    """Section 6.1: fixed decisions under costs/delays/stale quotes; retuned rules; stock groups."""
    tr_cfg = cfg["trading"]
    REF, Q = tr_cfg["reference_cost_bps"], tr_cfg["quantity"]
    dev_f, dev_o = d.dev_f, d.dev_o
    rows = []
    for p in POLICIES:
        for c in tr_cfg["cost_scenarios_bps"]:
            rows.append({"policy": p, "check": f"cost {c} bp/side", **policy_summary(ENTER[p], dev_o, c, timing, quantity=Q)})
        for dl in timing.entry_delays[1:]:
            rows.append({"policy": p, "check": f"entry delay +{dl} s",
                         **policy_summary(ENTER[p], dev_o, REF, timing, delay=dl, quantity=Q)})
        for max_age in [timing.stale_quote_seconds, 2]:
            fresh = (dev_o[f"age_{timing.entry_second}"] <= max_age).to_numpy()
            rows.append({"policy": p, "check": f"skip entries on asks older than {max_age} s",
                         **policy_summary(ENTER[p], dev_o, REF, timing, extra_mask=fresh, quantity=Q)})
    fixed = pd.DataFrame(rows)

    retuned = []
    for c in tr_cfg["cost_scenarios_bps"]:
        enter = {p: np.zeros(len(dev_f), bool) for p in POLICIES}
        for k, st in cv.inner_store.items():
            f_tr, o_tr = dev_f.loc[st["rows"]], dev_o.loc[st["rows"]]
            net_c = np.where(fillable(o_tr, timing, 0, Q), net_return_bps(o_tr, c, timing, 0, Q), 0.0)
            va = (dev_f.fold == k).to_numpy()
            for p in POLICIES:
                best, _ = select_rule(p, st["price"], st["spread"], f_tr.q_299.to_numpy(), net_c, c,
                                      tr_cfg["margins_bps"], taus(cfg), tr_cfg["min_inner_trades"])
                if best is not None:
                    enter[p][va] = decide(cv.oof.loc[va, "price: selected"].to_numpy(),
                                          cv.oof.loc[va, "spread: selected"].to_numpy(),
                                          dev_f.q_299.to_numpy()[va], best[0], best[1], c)
        for p in POLICIES:
            retuned.append({"cost bp/side": c, "policy": p, **policy_summary(enter[p], dev_o, c, timing, quantity=Q)})

    grp = []
    group_masks = [(f"{a} activity", d.activity == a) for a in pd.unique(d.activity)]
    group_masks += [("min-spread-bound", d.min_bound), ("not min-spread-bound", ~d.min_bound)]
    for p in POLICIES:
        for name, mask in group_masks:
            if not mask.any():
                continue
            e = ENTER[p] & mask
            s_ = policy_summary(e, dev_o, REF, timing, quantity=Q)
            s_["net bps per opportunity"] = float(per_opportunity(e, dev_o, REF, timing, quantity=Q)[mask].mean())
            s_["opportunities"] = int(mask.sum())
            grp.append({"policy": p, "group": name, **s_})
    return {"fixed": fixed, "retuned": pd.DataFrame(retuned), "groups": pd.DataFrame(grp)}


# ----------------------------------------------------- section 7: why the spread does not help
MKT_W = ["sp_last60", "sp_now", "log_S1_vs_stock", "log_rv_vs_stock", "rv_ratio", "ret_0_299",
         "ret_last60", "log_nupd_vs_stock", "upd_last60", "log_depth1_vs_stock", "imb_now"]


def market_two_stage(X_a, f_a, y_a, X_b, f_b, cfg):
    """Window-mean return forecast from window-averaged inputs, times each stock's train beta."""
    ta, tb = f_a.time_id.to_numpy(), f_b.time_id.to_numpy()
    Wa, Wb = X_a[MKT_W].groupby(ta).mean(), X_b[MKT_W].groupby(tb).mean()
    ya = pd.Series(y_a).groupby(ta).mean()
    window_pred = pd.Series(make_head("ridge a=10", cfg).fit(Wa, ya.loc[Wa.index]).predict(Wb), index=Wb.index)
    dd = pd.DataFrame({"s": f_a.stock_id.to_numpy(), "y": y_a, "m": pd.Series(ta).map(ya).to_numpy()})
    beta = dd.groupby("s").apply(lambda g: np.cov(g.y, g.m)[0, 1] / g.m.var(), include_groups=False)
    return f_b.stock_id.map(beta).fillna(1.0).to_numpy() * pd.Series(tb).map(window_pred).to_numpy()


def why_spread(d: Dev, cv: CVResult, ENTER: dict, cfg, timing: Timing) -> dict:
    tr_cfg, bs, s = cfg["trading"], cfg["bootstrap"], cfg["split"]
    REF, Q = tr_cfg["reference_cost_bps"], tr_cfg["quantity"]
    dev_f, dev_o, oof, tid = d.dev_f, d.dev_o, cv.oof, d.dev_f.time_id.to_numpy()
    Y_COST = (dev_o.y_price - net_return_bps(dev_o, 0.0, timing, 0, Q)).to_numpy()
    q_bps = dev_f.q_299.to_numpy() * 1e4
    cost_fc, mkt_fc = np.full(len(dev_f), np.nan), np.full(len(dev_f), np.nan)
    x_enter = {"oracle: exact realised cost": np.zeros(len(dev_f), bool),
               "market-only price model": np.zeros(len(dev_f), bool)}
    for k, st in cv.inner_store.items():
        tr, va = (dev_f.fold != k).to_numpy(), (dev_f.fold == k).to_numpy()
        f_tr, o_tr, f_va = dev_f[tr].reset_index(drop=True), dev_o[tr].reset_index(drop=True), dev_f[va].reset_index(drop=True)
        X_tr, X_va = fold_features(f_tr, f_va)
        cost_fc[va] = fit_predict("ridge a=10", X_tr, Y_COST[tr], X_va, cfg)
        mkt_fc[va] = market_two_stage(X_tr, f_tr, o_tr.y_price.to_numpy(), X_va, f_va, cfg)
        inner = assign_folds(f_tr.time_id.to_numpy(), s["inner_folds"], seed=s["inner_seed_offset"] + k)
        mk_in = np.full(len(f_tr), np.nan)
        for j in range(s["inner_folds"]):
            a, b = inner != j, inner == j
            X_a, X_b = fold_features(f_tr[a], f_tr[b])
            mk_in[b] = market_two_stage(X_a, f_tr[a], o_tr.y_price.to_numpy()[a], X_b, f_tr[b], cfg)
        net_in = np.where(fillable(o_tr, timing, 0, Q), net_return_bps(o_tr, REF, timing, 0, Q), 0.0)
        p_va = oof.loc[va, "price: selected"].to_numpy()
        scores = {"oracle: exact realised cost": (st["price"] - Y_COST[tr], p_va - Y_COST[va]),
                  "market-only price model": (mk_in - q_bps[tr], mkt_fc[va] - q_bps[va])}
        for p, (s_in, s_va) in scores.items():      # margin chosen on inner forecasts, as in section 6
            ok = [m for m in tr_cfg["margins_bps"] if (s_in > 2 * REF + m).sum() >= tr_cfg["min_inner_trades"]]
            if not ok:
                continue
            m = max(ok, key=lambda m: np.where(s_in > 2 * REF + m, net_in, 0.0).mean())
            x_enter[p][va] = s_va > 2 * REF + m
    y_p = dev_o.y_price.to_numpy()
    r2_all, r2_mkt = r2(y_p, oof["price: selected"].to_numpy()), r2(y_p, mkt_fc)
    diag = {"cost R2 using q_299": r2(Y_COST, q_bps), "cost R2 ridge forecast": r2(Y_COST, cost_fc),
            "cost RMSE ridge (bps)": rmse(Y_COST, cost_fc),
            "price-forecast RMSE (bps)": rmse(y_p, oof["price: selected"].to_numpy()),
            "price R2 selected (47 inputs)": r2_all, "price R2 market-only": r2_mkt,
            "market-only share of price R2": r2_mkt / r2_all if r2_all else np.nan}
    rows, base = [], per_opportunity(ENTER["price only"], dev_o, REF, timing, quantity=Q)
    for p, e in {"price only (section 6)": ENTER["price only"], **x_enter}.items():
        m, lo, hi = paired_bootstrap(per_opportunity(e, dev_o, REF, timing, quantity=Q), base, tid, bs["resamples"], bs["seed"])
        s_ = policy_summary(e, dev_o, REF, timing, quantity=Q)
        rows.append({"policy": p, "entries": s_["entries"], "net bps per trade": s_["net bps per trade"],
                     "net bps per opportunity": s_["net bps per opportunity"], "minus price only": m,
                     "ci_low": lo, "ci_high": hi})
    dec = pd.DataFrame({"forecast": oof["price: selected"].to_numpy(), "realised": y_p,
                        "net if bought": net_return_bps(dev_o, REF, timing, 0, Q)})
    dec = dec.groupby(pd.qcut(dec.forecast, 10, labels=False, duplicates="drop"), observed=True).mean()
    return {"diagnostics": pd.Series(diag, name="value").rename_axis("quantity").reset_index(),
            "policies": pd.DataFrame(rows), "deciles": dec.rename_axis("decile").reset_index()}


# ---------------------------------------------------------------- section 8: robustness
def ablation(d: Dev, cfg) -> pd.DataFrame:
    """Section 8.1: drop one family (with all derived versions) at a time; fast HGB, 3 grouped folds."""
    fast = {**cfg["models"]["hgb"], **cfg["robustness"]["fast_hgb_overrides"]}
    dev_f, dev_o = d.dev_f, d.dev_o
    folds = dev_f.fold.to_numpy() % 3
    cache = {}
    for j in range(3):
        a, b = folds != j, folds == j
        cache[j] = (a, b, *fold_features(dev_f[a], dev_f[b]))

    def fast_r2(cols):
        out = {}
        for t, col in TARGETS.items():
            y = dev_o[col].to_numpy()
            p = np.full(len(y), np.nan)
            for j, (a, b, X_a, X_b) in cache.items():
                ok = np.isfinite(y[a])
                p[b] = HistGradientBoostingRegressor(**fast).fit(X_a[cols][ok], y[a][ok]).predict(X_b[cols])
            out[t] = r2(y, p)
        return out

    abl = {"all features": {**fast_r2(FEATURES), "n_cols": len(FEATURES)}}
    for fam in FAMILIES:
        keep = [c for c in FEATURES if family(c) != fam]
        abl[f"without {fam}"] = {**fast_r2(keep), "n_cols": len(keep)}
    own = [c for c in FEATURES if not c.startswith(("mkt_", "rel_"))]
    abl["only own-stock inputs (no mkt_/rel_)"] = {**fast_r2(own), "n_cols": len(own)}
    mk = [c for c in FEATURES if c.startswith("mkt_")]
    abl["only market inputs (mkt_*)"] = {**fast_r2(mk), "n_cols": len(mk)}
    abl = pd.DataFrame(abl).T
    for t in TARGETS:
        abl[f"{t}: R2 loss"] = abl.loc["all features", t] - abl[t]
    return abl.rename_axis("feature set").reset_index()


def features_fullframe(dev_f, train_mask):
    """Baseline inputs for ALL rows with stock means learned on ``train_mask`` rows only.
    Market (leave-one-out) inputs use only first-half features of the same window, so they are
    computed on the full frame and are identical for every split."""
    means = dev_f[train_mask].groupby("stock_id")[VS_STOCK].mean()
    dd = dev_f.copy()
    for c in VS_STOCK:
        dd[c + "_vs_stock"] = dd[c] - dd.stock_id.map(means[c]).fillna(dev_f.loc[train_mask, c].mean())
    for c in MKT_SRC:
        dd["mkt_" + c] = leave_one_out_mean(dd, c)
    for c in REL:
        dd["rel_" + c] = dd[c] - dd["mkt_" + c]
    return dd[FEATURES]


def leakage_demo(d: Dev, cfg) -> pd.DataFrame:
    """Section 8.3: the same HGB scored with folds grouped by window vs random rows.

    Change from the archived notebook: stock means are now learned on each training fold
    (the notebook learned them once on all development rows)."""
    dev_f, dev_o = d.dev_f, d.dev_o
    k = cfg["split"]["outer_folds"]
    random_fold = np.empty(len(dev_f), int)
    for j, (_, b) in enumerate(KFold(k, shuffle=True, random_state=0).split(dev_f)):
        random_fold[b] = j
    out = {}
    for split, folds in [("grouped by window", dev_f.fold.to_numpy()), ("random rows", random_fold)]:
        for t, col in TARGETS.items():
            y, p = dev_o[col].to_numpy(), np.full(len(dev_f), np.nan)
            for j in range(k):
                a, b = folds != j, folds == j
                X = features_fullframe(dev_f, a)
                p[b] = fit_predict("HGB", X[a], y[a], X[b], cfg)
            out.setdefault(split, {})[f"{t} R2"] = r2(y, p)
    tab = pd.DataFrame(out).T
    tab.loc["random / grouped ratio"] = tab.loc["random rows"] / tab.loc["grouped by window"]
    return tab.rename_axis("split").reset_index()


def stress(d: Dev, cv: CVResult, ENTER: dict, paths: pd.DataFrame, cfg, timing: Timing) -> dict:
    """Section 8.2: accrual path, fresh-quote split, lagged/shuffled market placebos, concentration,
    neighbouring margins and break-even fee. ``paths`` = path points aligned to dev rows."""
    tr_cfg, bs, rb = cfg["trading"], cfg["bootstrap"], cfg["robustness"]
    REF, Q = tr_cfg["reference_cost_bps"], tr_cfg["quantity"]
    dev_f, dev_o, tid = d.dev_f.copy(), d.dev_o, d.dev_f.time_id.to_numpy()
    assert paths[KEYS].equals(dev_f[KEYS]), "path points must align row for row with development rows"
    last = timing.cut - 1
    E, CAN = ENTER["price only"], fillable(dev_o, timing, 0, Q)
    TR, NET = E & CAN, net_return_bps(dev_o, REF, timing, 0, Q)
    secs = [s_ for s_ in rb["path_seconds"] if s_ >= timing.cut]
    accrual = pd.DataFrame({"second": secs, "mean mid move since decision (bps)":
                            [float(((paths[f"mid_{s_}"] / paths[f"mid_{last}"] - 1) * 1e4)[TR].mean()) for s_ in secs]})
    fresh = {"own quote updated at decision second": dev_f.quote_age_299.to_numpy() == 0,
             "own quote 1+ s old at decision second": dev_f.quote_age_299.to_numpy() > 0,
             "both legs on fresh quotes": (dev_o[f"age_{timing.entry_second}"].to_numpy() == 0)
             & (paths.age_exit_quote.to_numpy() == 0)}
    fresh_tab = pd.DataFrame([{"subset": k_, "trades": int((TR & m).sum()),
                               "net bps per trade": float(NET[TR & m].mean()) if (TR & m).any() else np.nan}
                              for k_, m in fresh.items()])

    # placebo / lag inputs: seconds 209..269 only (strictly earlier than the decision)
    dev_f["ret_last60_270"] = np.log(paths.mid_269 / paths.mid_209).to_numpy() * 1e4
    dev_f["ret_0_270"] = np.log(paths.mid_269 / paths.mid_first).to_numpy() * 1e4

    def stress_inputs(tr, te, kind, rng):
        out = []
        for dd, X in zip((tr, te), fold_features(tr, te)):
            X = X.copy()
            if kind == "market lagged 30 s":
                X["mkt_ret_last60"] = leave_one_out_mean(dd, "ret_last60_270").to_numpy()
                X["mkt_ret_0_299"] = leave_one_out_mean(dd, "ret_0_270").to_numpy()
            elif kind == "market shuffled across windows":
                cols = [c for c in X if c.startswith("mkt_")]
                u = np.unique(dd.time_id)
                per_window = X[cols].groupby(dd.time_id.to_numpy()).first()
                X[cols] = per_window.loc[pd.Series(dd.time_id).map(dict(zip(u, rng.permutation(u))))].to_numpy()
            elif kind == "own-stock inputs only":
                X = X[[c for c in X if not c.startswith(("mkt_", "rel_"))]]
            if kind != "own-stock inputs only":
                for c in REL:
                    X["rel_" + c] = X[c] - X["mkt_" + c]
            out.append(X)
        return out

    rng, y_p, rows = np.random.default_rng(0), dev_o.y_price.to_numpy(), []
    for kind in ["as built", "market lagged 30 s", "market shuffled across windows", "own-stock inputs only"]:
        p = np.full(len(dev_f), np.nan)
        for k in range(cfg["split"]["outer_folds"]):
            a, b = (dev_f.fold != k).to_numpy(), (dev_f.fold == k).to_numpy()
            X_a, X_b = stress_inputs(dev_f[a], dev_f[b], kind, rng)
            p[b] = fit_predict(rb["stress_ridge"], X_a, y_p[a], X_b, cfg)
        e = decide(p, 0, dev_f.q_299.to_numpy(), rb["stress_margin_bps"], np.inf, REF)
        m, lo, hi = paired_bootstrap(np.where(e & CAN, NET, 0.0), np.zeros(len(e)), tid, bs["resamples"], bs["seed"])
        rows.append({"inputs": kind, "price R2": r2(y_p, p), "trades": int(e.sum()),
                     "net bps per trade": float(NET[e & CAN].mean()) if (e & CAN).any() else np.nan,
                     "per opportunity": m, "ci_low": lo, "ci_high": hi})
    placebo = pd.DataFrame(rows)

    PNL, LONG = np.where(TR, NET, 0.0), dev_o.y_price.to_numpy()
    w_pnl, w_long = pd.Series(PNL).groupby(tid).sum(), pd.Series(LONG).groupby(tid).sum()
    w_move = pd.Series(LONG).groupby(tid).mean().abs()
    k1, k5 = max(1, round(0.01 * len(w_pnl))), max(1, round(0.05 * len(w_pnl)))

    def trimmed(x, drop):
        return float(np.nanmean(x[~np.isin(tid, drop)]))

    conc = pd.DataFrame({
        "all windows": (float(np.nanmean(PNL)), float(np.nanmean(LONG))),
        "drop own best 5% of windows": (trimmed(PNL, w_pnl.nlargest(k5).index), trimmed(LONG, w_long.nlargest(k5).index)),
        "drop 5% largest |market move|": (trimmed(PNL, w_move.nlargest(k5).index), trimmed(LONG, w_move.nlargest(k5).index)),
    }, index=["rule, net bps per opportunity", "always long at the mid, no costs"]).T
    total = w_pnl.sum()
    conc_info = {"trades": int(TR.sum()), "traded windows": int(pd.Series(tid[TR]).nunique()),
                 "windows": int(len(w_pnl)),
                 "share of rule P&L in best 1% of windows": float(w_pnl.nlargest(k1).sum() / total) if total else np.nan}
    pp = cv.oof["price: selected"].to_numpy()
    neigh = pd.DataFrame([{"margin_bps": m_, "net bps per opportunity":
                           float(np.where(decide(pp, 0, dev_f.q_299.to_numpy(), m_, np.inf, REF) & CAN, NET, 0.0).mean())}
                          for m_ in rb["neighbour_margins_bps"]])
    gross = float(net_return_bps(dev_o, 0.0, timing, 0, Q)[TR].mean()) if TR.any() else np.nan
    conc_info.update({"gross bps per trade (fee 0)": gross, "break-even fee bp per side (decisions fixed)": gross / 2})
    return {"accrual": accrual, "fresh": fresh_tab, "placebo": placebo,
            "concentration": conc.rename_axis("trim").reset_index(), "concentration_info": conc_info,
            "margins": neigh}


# ---------------------------------------------------- section 9: legacy benchmark, scored once
def benchmark_run(d: Dev, cfg, timing: Timing) -> dict:
    """Frozen pipeline fit on all development rows and scored once on the benchmark windows.

    Change from the archived notebook: BOTH policies are now selected and scored (the archived
    cell selected only "price only", so its "spread filter chosen: never" row was not a test)."""
    tr_cfg, bs, s = cfg["trading"], cfg["bootstrap"], cfg["split"]
    REF, Q, H = tr_cfg["reference_cost_bps"], tr_cfg["quantity"], heads(cfg)
    dev_f, dev_o, bench_f, bench_o = d.dev_f, d.dev_o, d.bench_f, d.bench_o
    ip = inner_forecasts(dev_f, dev_o, seed=s["holdout_inner_seed"], cfg=cfg)
    chosen = {t: min(H, key=lambda h: rmse(dev_o[col].to_numpy(), ip[t][h])) for t, col in TARGETS.items()}
    net_ref = np.where(fillable(dev_o, timing, 0, Q), net_return_bps(dev_o, REF, timing, 0, Q), 0.0)
    rules = {p: select_rule(p, ip["price"][chosen["price"]], ip["spread"][chosen["spread"]], dev_f.q_299.to_numpy(),
                            net_ref, REF, tr_cfg["margins_bps"], taus(cfg), tr_cfg["min_inner_trades"])[0]
             for p in POLICIES}
    X_dev, X_h = fold_features(dev_f, bench_f)
    pred = {t: fit_predict(chosen[t], X_dev, dev_o[col].to_numpy(), X_h, cfg) for t, col in TARGETS.items()}
    skill = []
    for t, col in TARGETS.items():
        y = bench_o[col].to_numpy()
        for name, p in {**baselines(t, dev_f, dev_o, bench_f), f"selected ({chosen[t]})": pred[t]}.items():
            skill.append({"target": t, "model": name, "RMSE": rmse(y, p), "R2": r2(y, p)})
    net_h, can_h, tid_h = net_return_bps(bench_o, REF, timing, 0, Q), fillable(bench_o, timing, 0, Q), bench_f.time_id.to_numpy()
    rows = []
    for p in POLICIES:
        r = rules[p]
        enter_h = (decide(pred["price"], pred["spread"], bench_f.q_299.to_numpy(), r[0], r[1], REF)
                   if r is not None else np.zeros(len(bench_f), bool))
        done = enter_h & can_h
        per_opp = np.where(done, net_h, 0.0)
        m, lo, hi = paired_bootstrap(per_opp, np.zeros(len(per_opp)), tid_h, bs["resamples"], bs["seed"])
        t_stat, n_w = window_t(net_h, done, tid_h)
        row = {"policy": p, "margin_bps": r[0] if r else None, "tau": r[1] if r else None,
               "windows": int(len(np.unique(tid_h))), "opportunities": int(len(done)), "trades": int(done.sum()),
               "traded windows": n_w, "net bps per trade": float(net_h[done].mean()) if done.any() else np.nan,
               "net bps per opportunity": m, "ci_low": lo, "ci_high": hi,
               "win rate": float((net_h[done] > 0).mean()) if done.any() else np.nan, "window-level t": t_stat}
        for c in list(tr_cfg["cost_scenarios_bps"]) + [2.0]:
            row[f"per opportunity at {c} bp/side"] = float(np.where(done, net_return_bps(bench_o, c, timing, 0, Q), 0.0).mean())
        rows.append(row)
    return {"skill": pd.DataFrame(skill), "trading": pd.DataFrame(rows),
            "frozen": {"heads": chosen, "rules": {p: (rules[p][:2] if rules[p] else None) for p in POLICIES}},
            "pred": pred}


# ------------------------------------------------- sections 5.2 / 5.3 / 9.1: upgraded models
def upgraded_cv(d: Dev, extra: pd.DataFrame, cv: CVResult, cfg, log=print) -> dict:
    dev_x = d.dev_f.merge(extra, on=KEYS, how="left")
    assert dev_x[KEYS].equals(d.dev_f[KEYS])
    up_oof = d.dev_f[KEYS].copy()
    n_inputs = None
    for k in range(cfg["split"]["outer_folds"]):
        t0 = perf_counter()
        a, b = (dev_x.fold != k).to_numpy(), (dev_x.fold == k).to_numpy()
        y_a = {t: d.dev_o[c].to_numpy()[a] for t, c in UP_TARGETS.items()}
        Xa, Xb = upgraded_features(dev_x[a].reset_index(drop=True), dev_x[b].reset_index(drop=True), y_a)
        n_inputs = Xa.shape[1]
        sp, pr = upgraded_predict(Xa, y_a, Xb, cfg)
        up_oof.loc[b, "spread: upgraded"], up_oof.loc[b, "price: upgraded"] = sp, pr
        if log:
            log(f"upgraded outer fold {k + 1}: {Xa.shape[1]} inputs | {(perf_counter() - t0) / 60:.1f} min")
    assert up_oof[KEYS].equals(cv.oof[KEYS])
    tab = pd.DataFrame([{"target": t,
                         "section 5 selected head R2": r2(d.dev_o[c].to_numpy(), cv.oof[f"{t}: selected"].to_numpy()),
                         "upgraded model R2": r2(d.dev_o[c].to_numpy(), up_oof[f"{t}: upgraded"].to_numpy())}
                        for t, c in UP_TARGETS.items()])
    tab["relative gain"] = tab["upgraded model R2"] / tab["section 5 selected head R2"] - 1
    return {"table": tab.assign(inputs=n_inputs), "oof": up_oof, "dev_x": dev_x}


def upgraded_checks(d: Dev, dev_x: pd.DataFrame, cfg, log=print) -> dict:
    """Section 5.3: LightGBM contribution (SHAP-style) drivers and spread skill on unseen stocks."""
    from .models import _lgb
    lgb = _lgb()
    small = lgb_params(cfg, "small")
    a, b = (dev_x.fold != 0).to_numpy(), (dev_x.fold == 0).to_numpy()
    y_a = {t: d.dev_o[c].to_numpy()[a] for t, c in UP_TARGETS.items()}
    Xa, Xb = upgraded_features(dev_x[a].reset_index(drop=True), dev_x[b].reset_index(drop=True), y_a)
    Xs = Xb.sample(min(cfg["upgraded"]["shap_sample"], len(Xb)), random_state=0)
    drivers = []
    for t, params in [("spread", small), ("price", {**small, **cfg["upgraded"]["price_overrides"]})]:
        ok = np.isfinite(y_a[t])
        m = lgb.LGBMRegressor(**params, random_state=0).fit(Xa[ok], y_a[t][ok])
        contrib = np.abs(m.predict(Xs, pred_contrib=True)[:, :-1]).mean(0)
        drivers += [{"target": t, "feature": c, "mean_abs_contribution": float(v)} for c, v in zip(Xa.columns, contrib)]
    k_out = cfg["split"]["outer_folds"]
    stocks = np.random.default_rng(cfg["upgraded"]["unseen_stock_seed"]).permutation(sorted(dev_x.stock_id.unique()))
    sfold = dev_x.stock_id.map(dict(zip(stocks, np.arange(len(stocks)) % k_out))).to_numpy()
    unseen = []
    for k in range(k_out):
        test = (dev_x.fold == k).to_numpy() & (sfold == k)
        for setting, train in [("seen stocks", (dev_x.fold != k).to_numpy()),
                               ("unseen stocks", (dev_x.fold != k).to_numpy() & (sfold != k))]:
            y_tr = {t: d.dev_o[c].to_numpy()[train] for t, c in UP_TARGETS.items()}
            Xa, Xb = upgraded_features(dev_x[train].reset_index(drop=True), dev_x[test].reset_index(drop=True), y_tr)
            cols = [c for c in Xa.columns if not (setting == "unseen stocks" and is_stock_specific(c))]
            ok = np.isfinite(y_tr["spread"])
            p = lgb.LGBMRegressor(**small, random_state=0).fit(Xa.loc[ok, cols], y_tr["spread"][ok]).predict(Xb[cols])
            unseen.append({"fold": k, "setting": setting, "spread R2": r2(d.dev_o.y_spread.to_numpy()[test], p),
                           "rows": int(test.sum())})
        if log:
            log(f"unseen-stock check fold {k + 1}/{k_out}")
    return {"drivers": pd.DataFrame(drivers), "unseen": pd.DataFrame(unseen)}


def upgraded_benchmark(d: Dev, extra: pd.DataFrame, bench: dict, cfg) -> pd.DataFrame:
    """Section 9.1: upgraded models fit on all development rows, scored once on the benchmark."""
    dev_x = d.dev_f.merge(extra, on=KEYS, how="left")
    bench_x = d.bench_f.merge(extra, on=KEYS, how="left")
    y_dev = {t: d.dev_o[c].to_numpy() for t, c in UP_TARGETS.items()}
    Xd, Xh = upgraded_features(dev_x, bench_x, y_dev)
    up_sp, up_pr = upgraded_predict(Xd, y_dev, Xh, cfg)
    rows = []
    for (t, c), new in zip(UP_TARGETS.items(), [up_sp, up_pr]):
        y = d.bench_o[c].to_numpy()
        rows.append({"target": t, "section 9 frozen head R2": r2(y, bench["pred"][t]), "upgraded model R2": r2(y, new)})
    tab = pd.DataFrame(rows)
    tab["relative gain"] = tab["upgraded model R2"] / tab["section 9 frozen head R2"] - 1
    return tab
