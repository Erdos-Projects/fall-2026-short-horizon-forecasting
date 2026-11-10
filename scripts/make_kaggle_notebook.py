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

This is the best model from our notebook 05 (Erdős Institute project, Fall 2026), put into one notebook so it runs on Kaggle. It uses features from the whole ten-minute window to predict RV for the next ten minutes.

The model is 80% neural net (averaged over 3 seeds) + 20% LightGBM, both trained on RMSPE. CV 0.1954.

`USE_NEIGHBOURS` switches the neighbour features on or off (the trick the top solutions used, averaging features over the most similar windows, including the 1–3 closest by recovered price). Neighbours are searched over train and test windows together.

Earlier submissions (single XGBoost): 0.2455 private without neighbours, 0.2186 with the first neighbour version, 0.2141 with the closest 1–3 windows added."""),
    nbf.v4.new_code_cell("""USE_NEIGHBOURS = True
W_NN = 0.8                  # weight on the neural net, rest goes to LightGBM
N_SEEDS = 3                 # neural nets averaged over this many random seeds
NN_EPOCHS = 30
N_JOBS = 4                  # Kaggle notebooks have 4 CPUs (GPU is used if switched on)"""),
    nbf.v4.new_markdown_cell("Feature code (copied from `src/obforecast/volatility.py` in our repo)."),
    nbf.v4.new_code_cell(module_src),
    nbf.v4.new_code_cell(r'''import os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    nbf.v4.new_markdown_cell("Two models: LightGBM and a small neural net, both trained on RMSPE. The final forecast is 80% neural net + 20% LightGBM (weights picked on our CV: 0.1954 vs 0.1958 for the neural net alone)."),
    nbf.v4.new_code_cell(r'''import lightgbm as lgb
import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer

Xtr, Xte, ytr = df.loc[~is_test, cols], df.loc[is_test, cols], y[~is_test]
w = 1 / ytr ** 2
w = w / w.mean()

# LightGBM
lgbm = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=127, min_child_samples=300,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                         n_jobs=N_JOBS, random_state=0, verbose=-1)
pred_lgb = lgbm.fit(Xtr, ytr, sample_weight=w).predict(Xte)
print("lightgbm done")'''),
    nbf.v4.new_code_cell(r'''# Neural net: quantile-scaled inputs + a learned vector per stock, loss = RMSPE
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(N_JOBS)
A_np, B_np = Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32)
med = np.nan_to_num(np.nanmedian(A_np, axis=0))
fill = lambda X: np.where(np.isfinite(X), X, med)
qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal", subsample=200_000, random_state=0)
A = torch.tensor(qt.fit_transform(fill(A_np)), dtype=torch.float32)
B = torch.tensor(qt.transform(fill(B_np)), dtype=torch.float32)
yt = torch.tensor(ytr, dtype=torch.float32)
st = torch.tensor(df.stock_id.to_numpy()[~is_test], dtype=torch.long)
sv = torch.tensor(df.stock_id.to_numpy()[is_test], dtype=torch.long)
scale = float(np.exp(np.mean(np.log(ytr))))


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emb = nn.Embedding(128, 16)
        self.body = nn.Sequential(nn.Linear(d + 16, 384), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(384, 192), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 1))

    def forward(self, x, s):
        return scale * torch.exp(self.body(torch.cat([x, self.emb(s)], 1)).squeeze(1))


def train_nn(seed, epochs=NN_EPOCHS, bs=2048):
    torch.manual_seed(seed)
    net = Net(A.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * (len(A) // bs + 1))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(A))
        for i in range(0, len(A), bs):
            idx = perm[i:i + bs]
            p, yy = net(A[idx].to(dev), st[idx].to(dev)), yt[idx].to(dev)
            loss = torch.sqrt(torch.mean(((yy - p) / yy) ** 2))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(B[i:i + 16384].to(dev), sv[i:i + 16384].to(dev)).cpu().numpy()
                               for i in range(0, len(B), 16384)])


pred_nn = np.mean([train_nn(seed) for seed in range(N_SEEDS)], axis=0)
print("neural net done")'''),
    nbf.v4.new_code_cell(r'''pred = W_NN * pred_nn + (1 - W_NN) * pred_lgb
fallback = np.exp(df.loc[is_test, "rv1_0"].to_numpy()) - 1e-8     # last-window RV if a prediction is unusable
pred = np.where(np.isfinite(pred) & (pred > 0), pred, fallback)
pred = np.where(np.isfinite(pred) & (pred > 0), pred, np.nanmedian(ytr))
sub = pd.DataFrame({"row_id": test_ids.stock_id.astype(str) + "-" + test_ids.time_id.astype(str), "target": pred})
sub.to_csv("submission.csv", index=False)
sub.head()'''),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "kaggle/optiver_vol_submission.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
