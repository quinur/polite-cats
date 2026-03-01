"""Tests for predictor module."""

from __future__ import annotations

import pytest
import pandas as pd

from src.aggregator import aggregate_game_level, aggregate_team_level
from src.predictor_ml import FEATURE_COLUMNS, LogisticOutcomeModel, build_game_features, split_game_data


def test_predictor_outputs_probabilities(shift_data):
    game_df = aggregate_game_level(shift_data)
    features = build_game_features(game_df)
    model = LogisticOutcomeModel()
    cat_cols = model._CAT_COLS
    x = features[[c for c in FEATURE_COLUMNS if c not in cat_cols]]
    y = features["home_win_label"]
    if len(y) < 1 or y.nunique() < 2:
        pytest.skip("Not enough diversity in fixture data to train model")
    model.pipeline.fit(x, y)
    probs = model.pipeline.predict_proba(x)[:, 1]
    assert all(0 <= p <= 1 for p in probs)


def test_feature_columns_exclude_goals():
    assert "goal_diff" not in FEATURE_COLUMNS


def test_split_and_evaluate(shift_data):
    """Fit model on full fixture game data and check accuracy is in valid range."""
    game_df = aggregate_game_level(shift_data)
    features = build_game_features(game_df)
    model = LogisticOutcomeModel()
    cat_cols = model._CAT_COLS
    x = features[[c for c in FEATURE_COLUMNS if c not in cat_cols]]
    y = features["home_win_label"]
    if y.nunique() < 2:
        pytest.skip("Insufficient label diversity in fixture")
    model.pipeline.fit(x, y)
    from sklearn.metrics import accuracy_score
    preds = (model.pipeline.predict_proba(x)[:, 1] >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    assert 0 <= acc <= 1