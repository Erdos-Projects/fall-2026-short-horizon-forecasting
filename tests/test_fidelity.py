"""Port fidelity: run the ARCHIVED merged notebook's own definitions on synthetic books and compare
with the package. Guards against drift between `src/obforecast` and the published notebook code.

Only definition cells are executed (CONFIG, book reconstruction, feature families, heads/trading,
nested CV, extra inputs, upgraded features); data loading is replaced by the synthetic source.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obforecast import pipeline
from obforecast.book import build_stock
from obforecast.config import Timing, load_config
from obforecast.extra import extra_stock
from obforecast.features import fold_features, upgraded_features

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "archive" / "Optiver_Team_Merged.ipynb"


def _cell(nb, i):
    return "".join(nb["cells"][i]["source"])


@pytest.fixture(scope="module")
def legacy(synth):
    """Namespace holding the notebook's functions, bound to the synthetic data."""
    source, stocks = synth
    nb = json.loads(NB.read_text())
    ns = {}
    exec("from pathlib import Path\nfrom time import perf_counter\nimport os, inspect\nimport datetime as dt\n"
         "import numpy as np\nimport pandas as pd\n"
         "from sklearn.ensemble import HistGradientBoostingRegressor\nfrom sklearn.impute import SimpleImputer\n"
         "from sklearn.linear_model import LinearRegression, Ridge\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\n", ns)
    ns["display"] = lambda *a, **k: None
    ns["load_stock"] = lambda kind, sid: source.load(kind, sid)
    ns["stock_list"] = stocks
    ns["KEYS"] = ["stock_id", "time_id"]          # defined in a data cell (14) of the notebook
    # assign_folds is defined inside the split cell (17), between data-dependent statements
    src17 = _cell(nb, 17)
    exec(src17[src17.index("def assign_folds"):src17.index("# stock groups used")], ns)
    for i in (11, 12, 19, 35):
        assert "def " in _cell(nb, i) or "CONFIG" in _cell(nb, i)
        exec(_cell(nb, i), ns)
    # cell 36 = inner_forecasts + nested_cv definitions followed by the call; keep definitions only
    exec(_cell(nb, 36).split("\nstarted = perf_counter()")[0], ns)
    # cell 41 = extra_stock definition followed by the build loop
    exec(_cell(nb, 41).split("\nt0 = perf_counter()")[0].replace("import lightgbm as lgb", ""), ns)
    ns["XCOLS"] = None
    # cell 42 = upgraded_features etc. followed by the CV loop
    exec(_cell(nb, 42).split("\nup_oof = dev_f[KEYS].copy()")[0], ns)
    return ns


def test_book_features_and_outcomes_match_notebook(legacy, synth):
    source, stocks = synth
    for sid in stocks[:4]:
        f0, o0, _, _ = legacy["build_stock"](sid, source.load("book", sid))
        f1, o1, _, _ = build_stock(sid, source.load("book", sid), Timing())
        pd.testing.assert_frame_equal(f0, f1)
        pd.testing.assert_frame_equal(o0, o1)


def test_extra_inputs_match_notebook(legacy, synth):
    source, stocks = synth
    for sid in stocks[:3]:
        x0 = legacy["extra_stock"](sid)
        x1 = extra_stock(sid, source.load("book", sid), source.load("trade", sid), Timing())
        pd.testing.assert_frame_equal(x0, x1)


def test_fold_and_upgraded_features_match_notebook(legacy, tables, synth):
    source, stocks = synth
    feats, outs = tables[0], tables[1]
    folds = legacy["assign_folds"](feats.time_id.to_numpy(), 3, 1)
    tr, te = feats[folds != 0].reset_index(drop=True), feats[folds == 0].reset_index(drop=True)
    for a, b in zip(legacy["fold_features"](tr, te), fold_features(tr, te)):
        pd.testing.assert_frame_equal(a, b)
    extra = pd.concat([extra_stock(s, source.load("book", s), source.load("trade", s), Timing()) for s in stocks])
    legacy["XCOLS"] = [c for c in extra.columns if c not in ("stock_id", "time_id")]
    fx = feats.merge(extra, on=["stock_id", "time_id"], how="left")
    trx, tex = fx[folds != 0].reset_index(drop=True), fx[folds == 0].reset_index(drop=True)
    y = {"spread": outs.y_spread.to_numpy()[folds != 0], "price": outs.y_price.to_numpy()[folds != 0]}
    for a, b in zip(legacy["upgraded_features"](trx, tex, y), upgraded_features(trx, tex, y)):
        pd.testing.assert_frame_equal(a.drop(columns="stock_id"), b.drop(columns="stock_id"))
        assert (a.stock_id.astype(float).fillna(-1).to_numpy() == b.stock_id.astype(float).fillna(-1).to_numpy()).all()


def test_nested_cv_matches_notebook(legacy, tables):
    """Same CONFIG values (configs/default.yaml mirrors notebook CONFIG): identical forecasts,
    selected heads, rules and trading decisions."""
    feats, outs = tables[0], tables[1]
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["trading"]["min_inner_trades"] = legacy["CONFIG"]["min_inner_trades"] = 20   # small synthetic sample
    cfg["models"]["hgb"]["max_iter"] = legacy["CONFIG"]["hgb"]["max_iter"] = 40     # speed; same in both
    d = pipeline.prepare(feats, outs, cfg)
    ours = pipeline.nested_cv(d, cfg, Timing(), log=None)
    legacy["print"] = lambda *a, **k: None
    oof0, sel0, _, _ = legacy["nested_cv"](d.dev_f, d.dev_o)
    cols = [c for c in oof0.columns if c not in ("stock_id", "time_id", "fold")]
    pd.testing.assert_frame_equal(oof0[cols].reset_index(drop=True), ours.oof[cols].reset_index(drop=True))
    assert sel0["price head"].tolist() == ours.selections["price head"].tolist()
    assert sel0["price only: margin_bps"].tolist() == ours.selections["price only: margin_bps"].tolist()
