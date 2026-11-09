"""Write kaggle/optiver_vol_submission.ipynb: a self-contained notebook for a late Kaggle submission.

The competition is a code competition without internet, so the notebook carries the feature
code from src/obforecast/volatility.py inline. Regenerate after changing that module:

    python scripts/make_kaggle_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
module_src = (ROOT / "src/obforecast/volatility.py").read_text()

cells = [
    nbf.v4.new_markdown_cell("""# Optiver realized volatility (late submission)

This is the best model from our notebook 05 (Erdős Institute project, Fall 2026), put into one notebook so it runs on Kaggle. It uses features from the whole ten-minute window to predict RV for the next ten minutes, with XGBoost weighted by 1/y² so it trains on RMSPE.

`USE_NEIGHBOURS` switches the neighbour features on or off:

* `False`: only book, trade, stock and market features. CV 0.222, private LB 0.2455.
* `True`: also averages features over the most similar windows (the trick the top solutions used), including the 1–3 closest windows by recovered price. CV 0.203. Neighbours are searched over train and test windows together, like the top solutions did.

An earlier version of the neighbour features (CV 0.206) got 0.2186 private."""),
    nbf.v4.new_code_cell("""USE_NEIGHBOURS = True
N_ESTIMATORS = 700          # same settings as notebook 05
N_JOBS = 4                  # Kaggle notebooks have 4 CPUs"""),
    nbf.v4.new_markdown_cell("Feature code (copied from `src/obforecast/volatility.py` in our repo)."),
    nbf.v4.new_code_cell(module_src),
    nbf.v4.new_code_cell(r'''import os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)
def find_data():
    # The competition folder's location under /kaggle/input varies, so look for it.
    if os.environ.get("OBF_DATA_DIR"):
        return Path(os.environ["OBF_DATA_DIR"])
    root = Path("/kaggle/input")
    for csv in sorted(root.rglob("train.csv")):
        if (csv.parent / "book_train.parquet").is_dir():
            return csv.parent
    mounted = "\n  ".join(str(p) for p in sorted(root.glob("*/*"))[:50]) or "(nothing)"
    raise FileNotFoundError("Competition data not found under /kaggle/input. Add it with 'Add Input' -> "
                            f"Competitions -> Optiver Realized Volatility Prediction. Mounted now:\n  {mounted}")


DATA = find_data()
print("data:", DATA)
STARTS = [0, 150, 300, 450]
TRADE_COLS = ["time_id", "seconds_in_bucket", "price", "size", "order_count"]


def load(kind, split, sid):
    paths = sorted((DATA / f"{kind}_{split}.parquet" / f"stock_id={sid}").glob("*.parquet"))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def one(split, sid):
    book = load("book", split, sid)
    if book is None:
        return None
    trades = load("trade", split, sid)
    if trades is None:
        trades = pd.DataFrame({c: pd.Series(dtype=float) for c in TRADE_COLS})
    return build_stock(sid, book, trades, 600, STARTS)


train_y = pd.read_csv(DATA / "train.csv")
test_ids = pd.read_csv(DATA / "test.csv")
parts = {}
for split, ids in [("train", train_y), ("test", test_ids)]:
    out = Parallel(n_jobs=N_JOBS)(delayed(one)(split, s) for s in sorted(ids.stock_id.unique()))
    parts[split] = pd.concat([o for o in out if o is not None], ignore_index=True)

train = parts["train"].merge(train_y, on=KEYS, how="inner")
test = test_ids[KEYS].merge(parts["test"], on=KEYS, how="left")   # keep every requested row
test["time_id"] = test.time_id + 1_000_000                      # keep test windows distinct from train
df = pd.concat([train, test], ignore_index=True)
is_test = np.r_[np.zeros(len(train), bool), np.ones(len(test), bool)]
print(f"train rows {len(train):,}, test rows {len(test):,}")'''),
    nbf.v4.new_code_cell(r'''base = add_log_features(df)
is_trade = lambda c: c.startswith(("trv", "n_trade", "size", "orders"))
cols = [c for c in base if c != "price_level"]
cols += add_market_features(df, ["rv1_0", "rv1_300", "rv1_450", "rv2_0", "trv_0", "n_trade_0",
                                 "spread_mean", "depth_mean", "n_book_0", "size_0"])
if USE_NEIGHBOURS:
    # k = 1-3 by recovered price picks out the windows just before and after each window
    nn_src = ["rv1_0", "rv1_150", "rv1_300", "rv1_450", "rv2_0", "trv_0", "n_trade_0", "spread_mean", "depth_mean"]
    cols += neighbour_features(df, "rv1_0", nn_src, [3, 10, 50], "vol", metric="corr")
    df["log_price_level"] = np.log(df.price_level)
    cols += neighbour_features(df, "log_price_level", nn_src, [1, 2, 3, 5, 10, 20, 50], "px", metric="minmax")

# stock averages from training windows only
cols += add_stock_features(df, ["rv1_0", "trv_0", "n_trade_0"], ~is_test)
y = df.target.to_numpy()
df["stk_log_target"] = df.stock_id.map(pd.Series(np.log(y[~is_test])).groupby(df.stock_id[~is_test].to_numpy()).mean())
cols.append("stk_log_target")
print(f"{len(cols)} model inputs")'''),
    nbf.v4.new_code_cell(r'''params = dict(n_estimators=N_ESTIMATORS, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.6,
              min_child_weight=50, tree_method="hist", n_jobs=N_JOBS, random_state=0)
w = 1 / y[~is_test] ** 2
model = xgb.XGBRegressor(**params).fit(df.loc[~is_test, cols], y[~is_test], sample_weight=w / w.mean())

pred = model.predict(df.loc[is_test, cols])
fallback = np.exp(df.loc[is_test, "rv1_0"].to_numpy()) - 1e-8     # last-window RV if a prediction is unusable
pred = np.where(np.isfinite(pred) & (pred > 0), pred, fallback)
pred = np.where(np.isfinite(pred) & (pred > 0), pred, np.nanmedian(y[~is_test]))
sub = pd.DataFrame({"row_id": test_ids.stock_id.astype(str) + "-" + test_ids.time_id.astype(str), "target": pred})
sub.to_csv("submission.csv", index=False)
sub.head()'''),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "kaggle/optiver_vol_submission.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
