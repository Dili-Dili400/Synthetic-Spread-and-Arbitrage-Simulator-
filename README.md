# Synthetic Spread Arbitrage Simulator

A Python simulation framework for testing a mean-reversion trading strategy on **synthetic** silver (XAGUSD-style) price data. It models realistic execution effects and runs Monte Carlo batches to see how the strategy behaves across many randomised market scenarios.

> **Disclaimer:** This is an educational and research project. It uses randomly generated data and does not make trading recommendations or performance claims.

## Features

- **Execution realism:** latency, limit-order fill probability, slippage and adverse-selection models
- **Signal logic:** VWAP z-score entries, regime detection with hysteresis, Hurst exponent and variance-ratio checks
- **Breathing-anchor enhancements:** drift detection, VWAP migration detection and adaptive ("breathing") stops
- **Risk management:** position sizing, stop-loss, time limits, profit target and a volatility circuit breaker
- **Monte Carlo batches:** vary parameters across many runs and summarise return, Sharpe, Sortino, drawdown, win rate and profit factor
- **Visualisation:** matplotlib/seaborn plots of each simulation
- **Optional PostgreSQL telemetry:** log run metrics to a database, with an automatic in-memory fallback if no database is available

## Tech stack

Python, NumPy, pandas, matplotlib, seaborn, PostgreSQL (psycopg2)

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
python run_demo.py
```

`run_demo.py` runs a small batch (20 simulations of 2,000 ticks) in in-memory mode, prints an analysis report and saves `monte_carlo_results.csv`. Short runs on synthetic data may produce few trades, so the metrics will vary a lot.

### Use it in your own code

```python
from synthetic_spread_arbitrage_simulator import run_monte_carlo_batch, analyze_monte_carlo_results

results = run_monte_carlo_batch(n_simulations=100, n_ticks=5000, db=None, show_plots=False)
analyze_monte_carlo_results(results)
```

### Optional: PostgreSQL logging

Copy `.env.example` to `.env`, fill in your database details, and load them as environment variables before running. If no database is reachable, the simulator falls back to in-memory mode.

## Project structure

```
synthetic_spread_arbitrage_simulator.py   # simulator, models, risk manager, Monte Carlo runner
run_demo.py                               # quick demo runner
requirements.txt
.env.example                              # database settings template
```

## Author

Dilizile Narayana Mji, Mechatronic Systems student and aspiring developer, Johannesburg, South Africa.
