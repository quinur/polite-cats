"""Configuration management for the WHL Prediction Engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class RankingWeights:
    """Weights for the power ranking formula."""
    win_pct: float = 0.4
    avg_xg: float = 0.4
    toi_efficiency: float = 0.2


@dataclass(frozen=True)
class ModelParams:
    """Hyperparameters for the machine learning models."""
    test_size: float = 0.2
    random_seed: int = 42
    # Model Hyperparameters
    logistic_c: float = 1.0
    catboost_iterations: int = 1500
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.03
    ensemble_logistic_weight: float = 0.2
    ensemble_catboost_weight: float = 0.8


@dataclass(frozen=True)
class Config:
    """Global configuration for the pipeline."""
    # Paths
    data_path: Path = Path("data/whl.csv")
    matchups_path: Path = Path("data/matchups.csv")
    output_dir: Path = Path("outputs")
    
    # Simulation
    num_simulations: int = 10000
    
    # Ranking
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)
    
    # Model
    model_params: ModelParams = field(default_factory=ModelParams)
    
    # Comprehensive Features - Phase 5 Performance
    feature_columns: List[str] = field(default_factory=lambda: [
        "xg_diff",
        "win_pct_diff",
        "gsax_diff",
        "goalie_gsax_diff",
        "goalie_xga_diff",
        "pp_strength_diff",
        "pk_strength_diff",
        "finishing_skill_diff",
        "elo_prob_val",
        "xg_close_diff",
        "home_away_split_diff",
        "sos_diff",
        "shot_diff",
        "penalty_diff",
        "toi_diff",
        "xg_per_shot_diff",
        "high_danger_diff",
        "home_goalie",
        "away_goalie",
        "home_off_line",
        "away_off_line",
        "home_def_pairing",
        "away_def_pairing"
    ])


# Default configuration instance
config = Config()
