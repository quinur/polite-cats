"""Machine learning predictor for WHL outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from src.config import config


@dataclass
class MatchupPrediction:
    """Container for matchup predictions."""

    home_team: str
    away_team: str
    home_win_probability: float


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    accuracy: float
    roc_auc: Optional[float]


FEATURE_COLUMNS = config.feature_columns

def build_game_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature set for model training from rolling game-level data."""
    features = game_df.copy()
    
    # Calculate Diffs based on rolling team metrics in aggregate_pipeline
    feature_map = {
        "xg_diff": ("home_xg_per_game", "away_xg_per_game"),
        "win_pct_diff": ("home_win_pct", "away_win_pct"),
        "gsax_diff": ("home_gsax", "away_gsax"),
        "goalie_gsax_diff": ("home_goalie_gsax_avg", "away_goalie_gsax_avg"),
        "goalie_xga_diff": ("home_goalie_xg_against_avg", "away_goalie_xg_against_avg"),
        "pp_strength_diff": ("home_pp_strength", "away_pp_strength"),
        "pk_strength_diff": ("home_pk_strength", "away_pk_strength"),
        "finishing_skill_diff": ("home_finishing", "away_finishing"),
        "elo_prob_val": ("elo_prob", None),
        "xg_close_diff": ("home_xg_close", "away_xg_close"),
        "home_away_split_diff": ("home_home_away_split", "away_home_away_split"),
        "sos_diff": ("home_sos", "away_sos"),
        "shot_diff": ("home_shot_rate", "away_shot_rate"),
        "penalty_diff": ("home_penalty_rate", "away_penalty_rate"),
        "toi_diff": ("home_toi_avg", "away_toi_avg"),
        "xg_per_shot_diff": ("home_xg_per_shot", "away_xg_per_shot"),
        "high_danger_diff": ("home_high_danger", "away_high_danger")
    }
    
    for feat, (home_col, away_col) in feature_map.items():
        if away_col is None:
            features[feat] = features[home_col] if home_col in features.columns else 0.5
        elif home_col in features.columns and away_col in features.columns:
            features[feat] = features[home_col] - features[away_col]
        else:
            features[feat] = 0.0

    # Pass through categorical features
    cat_cols = ["home_goalie", "away_goalie", "home_off_line", "away_off_line", "home_def_pairing", "away_def_pairing"]
    for col in cat_cols:
        if col in game_df.columns:
            features[col] = game_df[col]
            
    features["home_win_label"] = features["home_win"]
    return features


def split_game_data(
    game_df: pd.DataFrame, test_size: Optional[float] = None, random_state: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split game-level data into train/test sets using Time-Series split (No Leakage)."""
    # Sort by game_num to ensure temporal split
    df = game_df.sort_values("game_num").reset_index(drop=True)
    
    if test_size is None:
        test_size = config.model_params.test_size
        
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Warm-up filtering: Remove early season noise from TRAIN set
    # Only train on games after game 50 and teams with at least 3 games
    if "home_games_played" in train_df.columns:
        train_df = train_df[
            (train_df["game_num"] > 50) &
            (train_df["home_games_played"] >= 3) & 
            (train_df["away_games_played"] >= 3)
        ]
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_matchup_features(team_df: pd.DataFrame, matchups: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Create matchup features from latest team-level stats."""
    team_stats = team_df.set_index("team")
    rows = []
    for home_team, away_team in matchups:
        try:
            home = team_stats.loc[home_team]
            away = team_stats.loc[away_team]
        except KeyError:
            # Handle unknown teams by adding them to the index with default values if necessary
            # For now, just skip or use default mean
            continue
        
        # Calculate Elo prob for matchup
        ea = 1 / (1 + 10 ** ((away["elo"] - home["elo"]) / 400))
        
        row = {
            "home_team": home_team,
            "away_team": away_team,
            "xg_diff": home["xg_per_game"] - away["xg_per_game"],
            "win_pct_diff": home["win_pct"] - away["win_pct"],
            "gsax_diff": home["gsax"] - away["gsax"],
            "goalie_gsax_diff": 0.0,
            "goalie_xga_diff": 0.0,
            "pp_strength_diff": home["pp_strength"] - away["pp_strength"],
            "pk_strength_diff": home["pk_strength"] - away["pk_strength"],
            "finishing_skill_diff": home["finishing"] - away["finishing"],
            "elo_prob_val": ea,
            "xg_close_diff": home["xg_close"] - away["xg_close"],
            "home_away_split_diff": home["home_away_split"] - away["home_away_split"],
            "sos_diff": home["sos"] - away["sos"],
            "shot_diff": home["shot_rate"] - away["shot_rate"],
            "penalty_diff": home["penalty_rate"] - away["penalty_rate"],
            "toi_diff": home["toi_avg"] - away["toi_avg"],
            "xg_per_shot_diff": home["xg_per_shot"] - away["xg_per_shot"],
            "high_danger_diff": home["high_danger"] - away["high_danger"],
            "home_goalie": home["home_goalie"],
            "away_goalie": away["home_goalie"], # Use away team's 'home' goalie default
            "home_off_line": home["home_off_line"],
            "away_off_line": away["home_off_line"],
            "home_def_pairing": home["home_def_pairing"],
            "away_def_pairing": away["home_def_pairing"]
        }
        rows.append(row)
    return pd.DataFrame(rows)


try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None  # type: ignore[assignment,misc]

class CatBoostOutcomeModel:
    """CatBoost model for game outcomes."""

    def __init__(self) -> None:
        self.model = CatBoostClassifier(
            iterations=config.model_params.catboost_iterations,
            depth=config.model_params.catboost_depth,
            learning_rate=config.model_params.catboost_learning_rate,
            verbose=False,
            random_seed=config.model_params.random_seed,
            loss_function="Logloss"
        )

    def fit(self, game_df: pd.DataFrame) -> "CatBoostOutcomeModel":
        features = build_game_features(game_df)
        x = features[FEATURE_COLUMNS]
        y = features["home_win_label"]
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, game_df: pd.DataFrame) -> ModelMetrics:
        features = build_game_features(game_df)
        x = features[FEATURE_COLUMNS]
        y_true = features["home_win_label"]
        probs = self.predict_proba(x)
        preds = (probs >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, preds)
        roc_auc = None
        if y_true.nunique() > 1:
            roc_auc = roc_auc_score(y_true, probs)
        return ModelMetrics(accuracy=accuracy, roc_auc=roc_auc)


class LogisticOutcomeModel:
    """Logistic regression pipeline for game outcomes."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=config.model_params.logistic_c)),
            ]
        )

    _CAT_COLS = [
        "home_goalie", "away_goalie",
        "home_off_line", "away_off_line",
        "home_def_pairing", "away_def_pairing",
    ]

    def _numeric_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Drop categorical columns that StandardScaler cannot handle."""
        return x.drop(columns=[c for c in self._CAT_COLS if c in x.columns])

    def fit(self, game_df: pd.DataFrame) -> "LogisticOutcomeModel":
        features = build_game_features(game_df)
        x = self._numeric_features(features[FEATURE_COLUMNS])
        y = features["home_win_label"]
        self.pipeline.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        return self.pipeline.predict_proba(self._numeric_features(x))[:, 1]

    def predict_matchups(self, matchup_df: pd.DataFrame) -> pd.DataFrame:
        """Predict home win probability for matchups."""
        x = matchup_df[FEATURE_COLUMNS]
        probs = self.predict_proba(x)
        result = matchup_df[["home_team", "away_team"]].copy()
        result["home_win_probability"] = probs
        return result

    def evaluate(self, game_df: pd.DataFrame) -> ModelMetrics:
        features = build_game_features(game_df)
        x = features[FEATURE_COLUMNS]
        y_true = features["home_win_label"]
        probs = self.predict_proba(x)
        preds = (probs >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, preds)
        roc_auc = None
        if y_true.nunique() > 1:
            roc_auc = roc_auc_score(y_true, probs)
        return ModelMetrics(accuracy=accuracy, roc_auc=roc_auc)