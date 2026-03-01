"""Hyperparameter tuning with CatBoost and GridSearchCV."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

from src.predictor_ml import FEATURE_COLUMNS, build_game_features


@dataclass
class CatBoostTuningResult:
    """Container for tuning results."""

    best_estimator: CatBoostClassifier
    best_params: dict
    best_score: float


def tune_catboost(game_df: pd.DataFrame) -> CatBoostTuningResult:
    """Run GridSearchCV with TimeSeriesSplit to tune CatBoostClassifier.

    Args:
        game_df: Game-level data (chronologically sorted).

    Returns:
        CatBoostTuningResult with best estimator and params.
    """
    # Ensure chronological order for walk-forward validation
    game_df = game_df.sort_values("game_num").reset_index(drop=True)
    features = build_game_features(game_df)
    x = features[FEATURE_COLUMNS]
    y = features["home_win_label"]
    
    if CatBoostClassifier is None:
        raise ImportError("catboost is required to tune the CatBoost model.")
        
    model = CatBoostClassifier(
        verbose=False,
        loss_function="Logloss",
        random_seed=42,
    )
    
    # Expanded grid for Phase 5 tuning
    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "iterations": [400, 800],
        "l2_leaf_reg": [1, 3, 5]
    }
    
    # TimeSeriesSplit for Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid = GridSearchCV(
        model, 
        param_grid, 
        cv=tscv, 
        scoring="accuracy", # User wants >= 65% accuracy
        n_jobs=-1
    )
    
    grid.fit(x, y)
    
    return CatBoostTuningResult(
        best_estimator=grid.best_estimator_,
        best_params=grid.best_params_,
        best_score=grid.best_score_,
    )
