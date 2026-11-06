"""Execute both research notebooks top to bottom on synthetic books (no hidden state).

IPython magics and shell lines are skipped, the Kaggle loader is swapped for the synthetic source,
and LightGBM is replaced by the test stub. This checks that every cell runs in order with only
what earlier cells define; it says nothing about results on real data.
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def notebook_env(monkeypatch, tmp_path, fake_lightgbm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from obforecast import data as data_mod
    from obforecast.synthetic import make_synthetic

    source, stocks = make_synthetic(n_stocks=8, n_windows=90, seed=2)
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
