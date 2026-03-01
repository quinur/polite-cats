"""Ranking engine for WHL teams."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RankingWeights:
    """Weights for ranking metrics."""

    goals_for: float = 0.4
    avg_xg: float = 0.4
    shot_efficiency: float = 0.2


def rank_teams(team_df: pd.DataFrame, weights: RankingWeights | None = None) -> pd.DataFrame:
    """Rank teams based on weighted metrics.

    Args:
        team_df: Team-level DataFrame.
        weights: Optional weights for metrics.

    Returns:
        Ranked DataFrame with score and rank columns.
    """
    weights = weights or RankingWeights()
    df = team_df.copy()
    df["score"] = (
        df["goals_for"] * weights.goals_for
        + df["avg_xg"] * weights.avg_xg
        + df["shot_efficiency"] * weights.shot_efficiency
    )
    df = df.sort_values(["score", "win_pct"], ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df