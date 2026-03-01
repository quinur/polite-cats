"""Pytest fixtures for WHL pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def shift_data() -> pd.DataFrame:
    """Return mock shift-level data for testing.

    Columns cover every field touched by aggregate_game_level so that tests
    importing the aggregator don't fail on KeyError.

    Game 1: A (home) beats C 2-1  — no tie, no OT requirement.
    Game 2: B (home) beats D 2-0  — clear home win.
    """
    return pd.DataFrame(
        {
            # game_id as plain integers – aggregate_game_level handles both
            # integer IDs and 'game_X' string IDs robustly now.
            "game_id": [1, 1, 2, 2],
            "home_team": ["A", "A", "B", "B"],
            "away_team": ["C", "C", "D", "D"],
            # Game 1: A scores 1+1=2, C scores 0+1=1 → home win
            # Game 2: B scores 1+1=2, D scores 0+0=0 → home win
            "home_goals": [1, 1, 1, 1],
            "away_goals": [0, 1, 0, 0],
            "home_xg": [0.6, 0.4, 0.3, 0.5],
            "away_xg": [0.2, 0.5, 0.4, 0.1],
            "home_shots": [5, 4, 3, 4],
            "away_shots": [3, 5, 3, 2],
            "home_max_xg": [0.4, 0.3, 0.2, 0.3],
            "away_max_xg": [0.2, 0.4, 0.3, 0.1],
            "home_penalties_committed": [1, 0, 2, 1],
            "away_penalties_committed": [0, 1, 0, 0],
            "home_toi": [10.0, 8.0, 9.0, 11.0],
            "away_toi": [9.0, 9.0, 10.0, 8.0],
            "went_ot": [0, 0, 0, 0],
            # Categorical lineup / goalie columns
            "home_goalie": ["G1", "G1", "G2", "G2"],
            "away_goalie": ["G3", "G3", "G4", "G4"],
            "home_off_line": ["5v5", "5v5", "5v5", "5v5"],
            "away_off_line": ["5v5", "5v5", "5v5", "5v5"],
            "home_def_pairing": ["D1-D2", "D1-D2", "D3-D4", "D3-D4"],
            "away_def_pairing": ["D5-D6", "D5-D6", "D7-D8", "D7-D8"],
        }
    )