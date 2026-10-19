"""Quick demo: run a small Monte Carlo batch without a database or pop-up plots."""
import matplotlib
matplotlib.use("Agg")  # no GUI window needed

from synthetic_spread_arbitrage_simulator import (
    run_monte_carlo_batch,
    analyze_monte_carlo_results,
)

if __name__ == "__main__":
    results = run_monte_carlo_batch(
        n_simulations=20,
        n_ticks=2000,
        verbose_simulations=0,
        db=None,          # in-memory mode (PostgreSQL is optional)
        show_plots=False,
    )
    analyze_monte_carlo_results(results)
    results.to_csv("monte_carlo_results.csv", index=False)
    print("Saved results to monte_carlo_results.csv")
