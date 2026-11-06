"""Row alignment, nested-CV leakage and end-to-end execution on synthetic books."""
import json

import numpy as np
import pandas as pd
import pytest

from obforecast import pipeline
from obforecast.book import KEYS
from obforecast.extra import extra_stock
from obforecast.run import main


@pytest.fixture(scope="module")
def dev(tables, cfg):
    feats, outs, _, _ = tables
    return pipeline.prepare(feats, outs, cfg)


@pytest.fixture(scope="module")
def cv(dev, cfg, timing):
    return pipeline.nested_cv(dev, cfg, timing, log=None)


def test_prepare_separates_benchmark_windows(dev):
    assert not set(dev.dev_f.time_id) & set(dev.bench_f.time_id)
    assert dev.dev_f[KEYS].equals(dev.dev_o[KEYS]) and dev.bench_f[KEYS].equals(dev.bench_o[KEYS])


def test_every_development_row_gets_one_out_of_fold_forecast(dev, cv):
    assert cv.oof[KEYS].equals(dev.dev_f[KEYS])
    for t in ("price", "spread"):
        for m in ("zero", "stock mean", "persistence", "ridge", "HGB", "selected"):
            assert cv.oof[f"{t}: {m}"].notna().all()


def test_outer_fold_forecasts_ignore_that_folds_outcomes(dev, cv, cfg, timing):
    """Change every outcome of outer fold 0's validation rows: fold-0 forecasts and decisions must
    not move (they may only depend on the other folds' outcomes)."""
    k = 0
    va = (dev.dev_f.fold == k).to_numpy()
    o2 = dev.dev_o.copy()
    rng = np.random.default_rng(5)
    for c in [c for c in o2.columns if c not in KEYS]:
        o2.loc[va, c] = o2.loc[va, c].to_numpy() * rng.uniform(0.5, 1.5, va.sum()) + rng.normal(0, 1, va.sum())
    d2 = pipeline.Dev(dev.dev_f, o2, dev.bench_f, dev.bench_o, dev.split_info, dev.activity, dev.min_bound)
    cv2 = pipeline.nested_cv(d2, cfg, timing, log=None)
    cols = [c for c in cv.oof.columns if c.startswith(("price:", "spread:", "enter:"))]
    pd.testing.assert_frame_equal(cv.oof.loc[va, cols], cv2.oof.loc[va, cols])
    assert not cv.oof.loc[~va, cols].equals(cv2.oof.loc[~va, cols])   # sanity: other folds do use them


def test_trading_and_downstream_stages(dev, cv, cfg, timing):
    tr = pipeline.trading(dev, cv, cfg, timing)
    s = tr["summary"].set_index("policy")
    assert s.loc["cash", "entries"] == 0 and s.loc["cash", "net bps per opportunity"] == 0
    sens = pipeline.sensitivity(dev, cv, tr["ENTER"], cfg, timing)
    assert set(sens) == {"fixed", "retuned", "groups"}
    why = pipeline.why_spread(dev, cv, tr["ENTER"], cfg, timing)
    assert "cost R2 using q_299" in set(why["diagnostics"].quantity)
    leak = pipeline.leakage_demo(dev, cfg)
    assert list(leak.split)[:2] == ["grouped by window", "random rows"]


def test_benchmark_run_scores_both_policies(dev, cfg, timing):
    b = pipeline.benchmark_run(dev, cfg, timing)
    assert set(b["trading"].policy) == {"price only", "price + spread"}
    assert len(b["pred"]["price"]) == len(dev.bench_f)


def test_upgraded_plumbing_with_stub(dev, cv, cfg, synth, timing, fake_lightgbm):
    source, stocks = synth
    extra = pd.concat([extra_stock(s, source.load("book", s), source.load("trade", s), timing) for s in stocks],
                      ignore_index=True)
    up = pipeline.upgraded_cv(dev, extra, cv, cfg, log=None)
    assert (up["table"].inputs == 131).all() and up["oof"][KEYS].equals(dev.dev_f[KEYS])
    chk = pipeline.upgraded_checks(dev, up["dev_x"], cfg, log=None)
    assert set(chk["unseen"].setting) == {"seen stocks", "unseen stocks"}
    b = pipeline.benchmark_run(dev, cfg, timing)
    assert len(pipeline.upgraded_benchmark(dev, extra, b, cfg)) == 2


def test_cli_synthetic_run_records_every_stage(tmp_path, monkeypatch, fake_lightgbm):
    monkeypatch.chdir(tmp_path)
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    rc = main(["--config", str(cfg_path), "--synthetic", "--output-dir", str(tmp_path / "out"), "--score-benchmark"])
    m = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert rc == 0 and m["label"] == "synthetic"
    assert all(v["status"] == "executed" for v in m["stages"].values()), m["stages"]
    assert (tmp_path / "out" / "README.md").read_text().startswith("# SYNTHETIC RUN")
    checks = pd.read_csv(tmp_path / "out" / "metrics" / "correctness_checks.csv")
    assert checks.passed.all() and len(checks) == 11
    assert (tmp_path / "out" / "splits" / "benchmark_time_ids.csv").exists()


def test_cli_without_data_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OBF_DATA_DIR", raising=False)
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        main(["--config", str(cfg_path), "--data-dir", str(tmp_path / "nowhere")])
