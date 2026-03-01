"""Tests for ranking engine."""

from __future__ import annotations

import pandas as pd

from src.ranker_engine import rank_teams


def test_ranker_handles_ties():
    team_df = pd.DataFrame(
        {
            "team": ["A", "B"],
            "goals_for": [10, 10],
            "avg_xg": [5.0, 5.0],
            "shot_efficiency": [0.2, 0.2],
            "win_pct": [0.5, 0.6],
        }
    )
    ranked = rank_teams(team_df)
    assert ranked.loc[0, "team"] == "B"