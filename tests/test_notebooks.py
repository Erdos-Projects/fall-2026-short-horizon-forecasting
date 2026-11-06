"""Execute the final study and runnable research notebooks on synthetic books.

IPython magics and shell lines are skipped, the Kaggle loader is swapped for the synthetic source,
and LightGBM is replaced by the test stub. This checks that every cell runs in order with only
what earlier cells define; it says nothing about results on real data.
"""
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def notebook_env(monkeypatch, tmp_path, fake_lightgbm):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from obforecast import data as data_mod
    from obforecast.synthetic import make_synthetic

    source, stocks = make_synthetic(n_stocks=8, n_windows=90, seed=2)
    # The original spread experiment assumes a quote at the start of each bucket.
    # Supply that supported input condition for these notebook execution checks.
    for book in source.books.values():
        book.loc[book.groupby("time_id").head(1).index, "seconds_in_bucket"] = 0
    monkeypatch.setattr(data_mod, "check_layout", lambda d: {"train_stocks": len(stocks), "book_stocks": stocks})
    monkeypatch.setattr(data_mod, "ParquetSource", lambda d: source)
    ipy = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")
    disp.display = lambda *a, **k: None
    disp.Image = lambda *a, **k: None
    ipy.display = disp
    ipy.get_ipython = lambda: None          # attributes matplotlib probes
    ipy.version_info = (8, 0, 0)
    monkeypatch.setitem(sys.modules, "IPython", ipy)
    monkeypatch.setitem(sys.modules, "IPython.display", disp)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    for f in (ROOT / "configs").glob("*.yaml"):
        (tmp_path / "configs" / f.name).write_text(f.read_text())
    data_dir = tmp_path / "data" / "optiver-realized-volatility-prediction"
    data_dir.mkdir(parents=True)
    labels = []
    for stock in stocks:
        book = source.load("book", stock)
        mid = (book.bid_price1 + book.ask_price1) / 2
        rv = np.log(mid).groupby(book.time_id).diff().pow(2).groupby(book.time_id).sum().pow(0.5)
        labels.extend({"stock_id": stock, "time_id": tid, "target": value}
                      for tid, value in rv.items())
        for kind in ("book", "trade"):
            partition = data_dir / f"{kind}_train.parquet" / f"stock_id={stock}"
            partition.mkdir(parents=True)
            (partition / "synthetic.parquet").write_text("mocked by ParquetSource")
    pd.DataFrame(labels).to_csv(data_dir / "train.csv", index=False)
    # Published snapshot cells read real committed tables; synthetic refits remain separate.
    shutil.copytree(ROOT / "results" / "metrics", tmp_path / "results" / "metrics")
    shutil.copyfile(ROOT / "results" / "run_manifest.json", tmp_path / "results" / "run_manifest.json")
    return tmp_path


def run_notebook(path, overrides):
    nb = json.loads(Path(path).read_text())
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = [ln[: len(ln) - len(ln.lstrip())] + "pass" if ln.lstrip().startswith(("!", "%")) else ln
                 for ln in "".join(cell["source"]).splitlines()]
        src = "\n".join(lines)
        for old, new in overrides.items():
            src = src.replace(old, new)
        if "IN_COLAB =" in src:                      # setup cell: keep cwd, src is installed/on path
            src = src.replace("os.chdir(root)", "pass")
        try:
            exec(compile(src, f"{Path(path).name}[cell {i}]", "exec"), ns)
        except Exception as e:  # pragma: no cover - failure message
            raise AssertionError(f"{Path(path).name} cell {i} failed: {type(e).__name__}: {e}") from e
    return ns


def test_exploration_notebook_runs(notebook_env):
    ns = run_notebook(ROOT / "notebooks" / "00_data_exploration.ipynb", {})
    assert "curves" in ns and "corr" in ns


def test_forecasting_notebook_runs_including_benchmark_and_upgraded(notebook_env):
    ns = run_notebook(ROOT / "notebooks" / "01_forecasting_and_trading.ipynb",
                      {"SCORE_BENCHMARK = False": "SCORE_BENCHMARK = True"})
    assert len(ns["verdicts"]) == 4 and ns["checks"].passed.all()
    assert (notebook_env / "results" / "smoke" / "figures" / "figure1_accuracy_vs_value.png").exists()


def test_original_spread_progression_runs(notebook_env):
    ns = run_notebook(ROOT / "notebooks" / "03_spread_model_development.ipynb",
                      {"max_iter=500": "max_iter=20", "min_samples_leaf=200": "min_samples_leaf=5",
                       "n_splits=10": "n_splits=3"})
    assert len(ns["results"]) == 8
    assert ns["oof"].notna().all().all()


def test_event_notebook_runs_and_keeps_future_outcomes_out_of_inputs(notebook_env):
    import pandas as pd

    ns = run_notebook(ROOT / "notebooks" / "04_spread_widening_events.ipynb", {})
    assert set(ns["feature_cols"]).isdisjoint(ns["leakage_cols"])
    groups = ns["groups"]
    assert set(groups.iloc[ns["train_idx"]]).isdisjoint(groups.iloc[ns["test_idx"]])
    b = ns["book"]
    sample = b[b.time_id == b.time_id.iloc[0]].copy()
    before = ns["summarize_window"](ns["add_book_features"](sample))
    future = sample.seconds_in_bucket >= 300
    sample.loc[future, "ask_price1"] *= 1.01
    after = ns["summarize_window"](ns["add_book_features"](sample))
    first = [c for c in before.index if c.startswith("first_")]
    pd.testing.assert_series_equal(before[first], after[first])
    assert before.widening_ratio != after.widening_ratio

