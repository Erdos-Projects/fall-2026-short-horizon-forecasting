import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obforecast.book import build_tables  # noqa: E402
from obforecast.config import Timing, load_config  # noqa: E402
from obforecast.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def timing():
    return Timing()


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "smoke.yaml")


@pytest.fixture(scope="session")
def synth():
    source, stocks = make_synthetic(n_stocks=8, n_windows=90, seed=1)
    return source, stocks


@pytest.fixture(scope="session")
def tables(synth, timing):
    source, stocks = synth
    return build_tables(source, stocks, timing, log=None)


@pytest.fixture
def fake_lightgbm(monkeypatch):
    """Stand-in for lightgbm (not installable here) so the upgraded-model plumbing is exercised.
    It is NOT LightGBM: results from it are meaningless; only shapes/flow are tested."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    class LGBMRegressor:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            self.cols = list(X.columns)
            self.m = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(X, y)
            return self

        def predict(self, X, pred_contrib=False):
            p = self.m.predict(X[self.cols])
            if pred_contrib:
                return np.column_stack([np.zeros((len(X), len(self.cols))), p])
            return p

    mod = types.ModuleType("lightgbm")
    mod.LGBMRegressor = LGBMRegressor
    monkeypatch.setitem(sys.modules, "lightgbm", mod)
    return mod
