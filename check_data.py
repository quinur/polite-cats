import pandas as pd
from src.data_loader import load_csv
from src.aggregator import aggregate_pipeline
from src.config import config
import numpy as np

def check_data():
    shift_df = load_csv(config.data_path)
    game_df, _ = aggregate_pipeline(shift_df)
    
    print(f"Total Games: {len(game_df)}")
    print(f"Home Wins: {game_df['home_win'].sum()} ({game_df['home_win'].mean():.2%})")
    
    # Check for constant features
    from src.predictor_ml import build_game_features, split_game_data
    train_df, test_df = split_game_data(game_df)
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    feats = build_game_features(train_df)
    cols = config.feature_columns
    for col in cols:
        if col in feats.columns:
            print(f"{col}: mean={feats[col].mean():.3f}, std={feats[col].std():.3f}")
        
    print("\nFeature Correlations with home_win:")
    if "home_win_label" in feats.columns:
        corrs = feats[list(set(cols) & set(feats.columns)) + ["home_win_label"]].corr()["home_win_label"]
        print(corrs.sort_values(ascending=False))

if __name__ == "__main__":
    check_data()
