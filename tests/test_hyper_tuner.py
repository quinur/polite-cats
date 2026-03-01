"""Tests for hyper tuner."""

from __future__ import annotations

import pandas as pd

import src.hyper_tuner as hyper_tuner


class DummyGrid:
    def __init__(self, model, param_grid, cv, scoring, **kwargs):
        self.best_estimator_ = model
        self.best_params_ = {"depth": 4}
        self.best_score_ = 0.5

    def fit(self, x, y):
        return self


class DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_tune_catboost_returns_result(monkeypatch, shift_data):
    game_df = (
        shift_data.groupby(["game_id", "home_team", "away_team", "went_ot"], as_index=False)
        .agg(
            home_goals=("home_goals", "sum"),
            away_goals=("away_goals", "sum"),
            home_xg=("home_xg", "sum"),
            away_xg=("away_xg", "sum"),
            home_shots=("home_shots", "sum"),
            away_shots=("away_shots", "sum"),
            home_penalties_committed=("home_penalties_committed", "sum"),
            away_penalties_committed=("away_penalties_committed", "sum"),
            home_toi=("home_toi", "sum"),
            away_toi=("away_toi", "sum"),
            home_goalie=("home_goalie", "first"),
            away_goalie=("away_goalie", "first"),
            home_off_line=("home_off_line", "first"),
            away_off_line=("away_off_line", "first"),
            home_def_pairing=("home_def_pairing", "first"),
            away_def_pairing=("away_def_pairing", "first"),
        )
    )
    game_df["home_win"] = (game_df["home_goals"] > game_df["away_goals"]).astype(int)
    # hyper_tuner.tune_catboost sorts by game_num; provide it here
    game_df["game_num"] = game_df["game_id"].astype(int)

    monkeypatch.setattr(hyper_tuner, "GridSearchCV", DummyGrid)
    monkeypatch.setattr(hyper_tuner, "CatBoostClassifier", DummyModel)
    result = hyper_tuner.tune_catboost(game_df)
    assert result.best_params["depth"] == 4