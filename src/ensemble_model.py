"""Ensemble model combining Logistic Regression and CatBoost."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.config import config
from src.predictor_ml import ModelMetrics, build_game_features


class EnsembleOutcomeModel:
    """Ensemble of Logistic Regression and CatBoost."""

    def __init__(self, use_catboost: bool = True) -> None:
        params = config.model_params
        self.use_catboost = use_catboost and CatBoostClassifier is not None
        self.logistic_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=params.logistic_c, max_iter=1000)),
            ]
        )
        self.catboost_model = None
        if self.use_catboost:
            self.catboost_model = CatBoostClassifier(
                verbose=False,
                loss_function="Logloss",
                random_seed=params.random_seed,
                iterations=params.catboost_iterations,
                depth=params.catboost_depth,
                learning_rate=params.catboost_learning_rate,
                cat_features=["home_goalie", "away_goalie", "home_off_line", "away_off_line", "home_def_pairing", "away_def_pairing"]
            )

    def fit(self, game_df: pd.DataFrame) -> "EnsembleOutcomeModel":
        """Fit ensemble models using game-level data."""
        features = build_game_features(game_df)
        x = features[config.feature_columns]
        y = features["home_win_label"]
        
        # Logistic cannot handle strings
        cat_cols = ["home_goalie", "away_goalie", "home_off_line", "away_off_line", "home_def_pairing", "away_def_pairing"]
        x_num = x.drop(columns=[c for c in cat_cols if c in x.columns])
        
        self.logistic_pipeline.fit(x_num, y)
        if self.use_catboost:
            self.catboost_model.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability estimates."""
        log_weight = config.model_params.ensemble_logistic_weight
        cat_weight = config.model_params.ensemble_catboost_weight
        
        cat_cols = ["home_goalie", "away_goalie", "home_off_line", "away_off_line", "home_def_pairing", "away_def_pairing"]
        x_num = x.drop(columns=[c for c in cat_cols if c in x.columns])
        
        log_probs = self.logistic_pipeline.predict_proba(x_num)[:, 1]
        
        if self.use_catboost:
            cat_probs = self.catboost_model.predict_proba(x)[:, 1]
            return log_weight * log_probs + cat_weight * cat_probs
        
        return log_probs

    def evaluate(self, game_df: pd.DataFrame) -> ModelMetrics:
        """Evaluate ensemble on a game-level dataset."""
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        features = build_game_features(game_df)
        x = features[config.feature_columns]
        y_true = features["home_win_label"]
        
        probs = self.predict_proba(x)
        preds = (probs >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, preds)
        roc_auc = None
        if y_true.nunique() > 1:
            roc_auc = roc_auc_score(y_true, probs)
            
        return ModelMetrics(accuracy=accuracy, roc_auc=roc_auc)

    def predict_matchups(self, matchup_df: pd.DataFrame) -> pd.DataFrame:
        """Predict home win probability for matchups."""
        x = matchup_df[config.feature_columns]
        probs = self.predict_proba(x)
        
        result = matchup_df[["home_team", "away_team"]].copy()
        result["home_win_probability"] = probs
        return result

    def save(self, path: str | Path) -> None:
        """Save the ensemble model to disk."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(
            {
                "logistic": self.logistic_pipeline,
                "catboost": self.catboost_model,
                "use_catboost": self.use_catboost,
            },
            path
        )

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleOutcomeModel":
        """Load an ensemble model from disk."""
        import joblib
        data = joblib.load(path)
        model = cls(use_catboost=data["use_catboost"])
        model.logistic_pipeline = data["logistic"]
        model.catboost_model = data["catboost"]
        return model
