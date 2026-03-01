"""Tests for aggregator module."""

from __future__ import annotations

from src.aggregator import aggregate_game_level, aggregate_team_level


def test_aggregate_game_level_sums_goals(shift_data):
    game_df = aggregate_game_level(shift_data)
    game1 = game_df.loc[game_df["game_id"] == 1].iloc[0]
    # Fixture: shift 1 = 1 home goal, shift 2 = 1 home goal → 2 total
    #          shift 1 = 0 away goals, shift 2 = 1 away goal → 1 total
    assert game1["home_goals"] == 2
    assert game1["away_goals"] == 1


def test_aggregate_game_level_resolves_winner(shift_data):
    """home_win must never be 0 and away_win 0 for the same game."""
    game_df = aggregate_game_level(shift_data)
    assert ((game_df["home_win"] == 0) & (game_df["away_win"] == 0)).sum() == 0


def test_aggregate_team_level_win_pct(shift_data):
    game_df = aggregate_game_level(shift_data)
    team_df = aggregate_team_level(game_df)
    team_a = team_df.loc[team_df["team"] == "A"].iloc[0]
    assert team_a["games_played"] == 1