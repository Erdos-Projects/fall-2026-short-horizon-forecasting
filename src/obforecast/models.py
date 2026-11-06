"""Forecast heads, baselines, metrics and the upgraded (LightGBM) models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {"price": "y_price", "spread": "y_spread"}     # y_price in bps; y_spread = log(S2/S1)


def heads(cfg: dict) -> list[str]:
    return [f"ridge a={a:g}" for a in cfg["models"]["ridge_alphas"]] + ["HGB"]


def make_head(name: str, cfg: dict):
    if name.startswith("ridge"):
        alpha = float(name.split("=")[1])
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
    if name == "HGB":
        return HistGradientBoostingRegressor(**cfg["models"]["hgb"])
    raise ValueError(name)


def fit_predict(name: str, X_tr, y_tr, X_te, cfg: dict):
    """Fit on rows with a finite target; imputer/scaler are fitted on the training rows only."""
    ok = np.isfinite(y_tr)
    return make_head(name, cfg).fit(X_tr[ok], y_tr[ok]).predict(X_te)


def baselines(target: str, f_tr, o_tr, f_te) -> dict:
    """Zero change, training-fold stock mean, and persistence (in the target's own units)."""
    y = o_tr[TARGETS[target]]
    stock_mean = (f_te.stock_id.map(y.groupby(f_tr.stock_id.to_numpy()).mean())
                  .fillna(y.mean()).to_numpy())
    if target == "price":
        persistence = np.zeros(len(f_te))          # keep the current midpoint = zero return
    else:
        persistence = f_te.sp_last60.to_numpy()    # last-60 s mean spread persists: log(q_last60 / S1)
    return {"zero": np.zeros(len(f_te)), "stock mean": stock_mean, "persistence": persistence}


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[ok] - p[ok])))


def r2(y, p) -> float:
    """R^2 against the evaluation-set mean (out-of-sample convention used throughout)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


# ---------------------------------------------------------------- upgraded models (section 5.2)
def _lgb():
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("The upgraded models need lightgbm: pip install -r requirements.txt") from e
    return lgb


def lgb_params(cfg: dict, which: str) -> dict:
    u = cfg["upgraded"]
    base = dict(u["lgb_base"])
    if which == "spread":
        return {**base, **u["spread_overrides"]}
    if which == "price":
        return {**base, **u["price_overrides"]}
    if which == "small":
        return {**base, **u["small_overrides"]}
    raise ValueError(which)


def upgraded_predict(Xa: pd.DataFrame, ya: dict, Xb: pd.DataFrame, cfg: dict):
    """Spread: LightGBM averaged over seeds. Price: blend of Huber LightGBM and ridge."""
    lgb = _lgb()
    u = cfg["upgraded"]
    num = [c for c in Xa.columns if c != "stock_id"]
    ys, yp = ya["spread"], ya["price"]
    ok_s, ok_p = np.isfinite(ys), np.isfinite(yp)
    sp = np.mean([lgb.LGBMRegressor(**lgb_params(cfg, "spread"), random_state=s)
                  .fit(Xa[ok_s], ys[ok_s]).predict(Xb) for s in u["seeds"]], axis=0)
    p_h = lgb.LGBMRegressor(**lgb_params(cfg, "price")).fit(Xa[ok_p], yp[ok_p]).predict(Xb)
    p_r = (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=u["price_ridge_alpha"]))
           .fit(Xa.loc[ok_p, num], yp[ok_p]).predict(Xb[num]))
    w = u["price_lgb_weight"]
    return sp, w * p_h + (1 - w) * p_r
