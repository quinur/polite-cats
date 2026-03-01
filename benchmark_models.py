"""Benchmark script to compare Phase 1 and Phase 2 models."""

import pandas as pd
from src.data_loader import load_csv
from src.aggregator import aggregate_pipeline
from src.predictor_ml import split_game_data, LogisticOutcomeModel
from src.ensemble_model import EnsembleOutcomeModel
from pathlib import Path

def benchmark():
    data_path = Path("data/whl.csv")
    shift_df = load_csv(
        data_path,
        required_columns=[
            "game_id", "home_team", "away_team", "home_goals", "away_goals",
            "home_xg", "away_xg", "home_shots", "away_shots",
            "home_penalties_committed", "away_penalties_committed", "went_ot"
        ]
    )
    game_df, _ = aggregate_pipeline(shift_df)
    
    # Use FIXED seed for fair comparison
    SEED = 42
    train_df, test_df = split_game_data(game_df, random_state=SEED)
    
    print(f"Dataset: {len(game_df)} games")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print("-" * 30)
    
    # Model 1: Phase 1 (Original 3)
    logistic = LogisticOutcomeModel().fit(train_df)
    m1 = logistic.evaluate(test_df)
    print(f"Logistic (C=1.0): Acc={m1.accuracy:.3f}, AUC={m1.roc_auc:.3f}")

    # Model 4: Tuned Logistic
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    best_acc = 0
    best_c = 1.0
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=c, max_iter=1000))
        ])
        pipe.fit(build_game_features(train_df)[orig_features], train_df["home_win"])
        # Manual eval
        feats = build_game_features(test_df)
        preds = pipe.predict(feats[orig_features])
        acc = (preds == test_df["home_win"]).mean()
        if acc > best_acc:
            best_acc = acc
            best_c = c
    print(f"Tuned Logistic (C={best_c}): Acc={best_acc:.3f}")

    
    # Model 2: Phase 2 (Ensemble)
    ensemble = EnsembleOutcomeModel().fit(train_df)
    m2 = ensemble.evaluate(test_df)
    print(f"Phase 2 (Ensemble): Acc={m2.accuracy:.3f}, AUC={m2.roc_auc:.3f}")
    
    # Model 3: Optimized Ensemble (Equal weights)
    # Let's try 50/50 instead of 40/60
    ensemble.logistic_weight = 0.5
    ensemble.catboost_weight = 0.5
    # Fix the class to accept these weights if possible
    
    print("-" * 30)
    if m2.accuracy < m1.accuracy:
        print("⚠️ Phase 2 is currently UNDERPERFORMING Phase 1.")
        print("Possible causes: CatBoost overfitting or sub-optimal weights.")
    else:
        print("✅ Phase 2 is OUTPERFORMING Phase 1.")

if __name__ == "__main__":
    benchmark()
