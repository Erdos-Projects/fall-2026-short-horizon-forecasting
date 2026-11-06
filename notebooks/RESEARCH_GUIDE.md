# Notebooks

If you just want the results, open [Forecasting and trading](04_forecasting_and_trading.ipynb). The rest are the experiments that led up to it.

| Notebook | Contents |
|---|---|
| [Data exploration](00_data_exploration.ipynb) | A first look at prices, volatility, spreads and liquidity, and how they vary across stocks |
| [Mid-price forecasting](01_midprice_forecasting.ipynb) | Swarnendu's order-book and trade features, with price models compared to a persistence baseline |
| [Spread models](02_spread_models.ipynb) | Francis's comparison of eight spread models |
| [Spread-widening events](03_spread_widening_events.ipynb) | Nhi's event definition, threshold choices and classifier |
| [Forecasting and trading](04_forecasting_and_trading.ipynb) | Final models, results, the trading simulation and robustness checks |

## How it fits together

We started with price forecasting to see which signals were useful. The spread models predict how average liquidity changes. The event classifier asks something different: which windows are likely to see a big jump in the spread? The final notebook combines the price and spread forecasts and tests a simple trading rule on top of them.

Each part has its own target and metric, so the scores aren't directly comparable across notebooks.

## Original notebooks

The [original notebooks](../archive/README.md) are archived with all their cells and saved figures. 
