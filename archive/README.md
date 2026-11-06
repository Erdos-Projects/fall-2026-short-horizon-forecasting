# Original notebooks

These are the team's original Colab notebooks as of Nov 5, 2026. The cleaned-up versions are in [`notebooks/`](../notebooks/RESEARCH_GUIDE.md).

| File | Original | Notes |
|---|---|---|
| `Optiver_Team_Merged.ipynb` | [Colab](https://colab.research.google.com/drive/1OkEJZsLMAazzFa5uchzfx-BPNfYL55-B) | Main analysis, became `01_forecasting_and_trading.ipynb` |
| `Optiver_Team_Data_EDA.ipynb` | [Colab](https://colab.research.google.com/drive/1r8q1dEfwH2jVYDXWP8V1o-zmXgOwvaXU) | Team exploration (Swarnendu, Francis, Nhi) |

Code, outputs and saved figures are kept as they were. Nothing was re-run.

## A few notes

- Some saved outputs in the exploration notebook came from older versions of the code, so they don't always match the code now in the cells.
- The early R² ≈ 0.94 price result measures the price *level*, not the return. It mostly reflects the current price. When the target is the return, the models don't beat the mean baseline.
- Later sections of the merged notebook lost their outputs after a runtime restart, so some numbers in its text weren't re-checked here.
- The benchmark windows aren't a fully untouched test set. Some of them were used in the early price experiments, and an earlier classifier was scored on them once. Treat results on them as a consistency check.
