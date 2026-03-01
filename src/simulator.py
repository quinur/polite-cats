"""Monte Carlo simulation for WHL matchups."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_matchups(
    predictions_df: pd.DataFrame, num_simulations: int = 10000, random_seed: int = 42
) -> pd.DataFrame:
    """Simulate matchups using Monte Carlo draws.

    Args:
        predictions_df: DataFrame with home_team, away_team, home_win_probability.
        num_simulations: Number of simulations to run.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with simulated win rates.
    """
    rng = np.random.default_rng(random_seed)
    results = []
    for _, row in predictions_df.iterrows():
        wins = rng.binomial(num_simulations, row["home_win_probability"])
        results.append(
            {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_rate_simulated": wins / num_simulations,
                "away_win_rate_simulated": 1 - (wins / num_simulations),
            }
        )
    return pd.DataFrame(results)