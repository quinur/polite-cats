"""Entry point for WHL 2026 performance prediction engine."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.aggregator import aggregate_pipeline
from src.data_loader import load_csv
from src.ensemble_model import EnsembleOutcomeModel
from src.feature_importance_explainer import explain_model
from src.predictor_ml import (
    build_game_features,
    build_matchup_features,
    split_game_data,
)
from src.ranker_engine import rank_teams
from src.simulator import simulate_matchups
from src.viz_factory import (
    correlation_heatmap,
    disparity_scatter_plot,
    feature_importance_bar_chart,
    power_ranking_bar_chart,
)
from src.disparity_calc import calculate_disparity
from src.config import config


def load_tournament_matchups(matchups_path: Path) -> list[tuple[str, str]]:
    """Load tournament matchups from CSV file."""
    if not matchups_path.exists():
        raise FileNotFoundError(f"Matchups file not found: {matchups_path}")
    
    matchups_df = pd.read_csv(matchups_path)
    if "home_team" not in matchups_df.columns or "away_team" not in matchups_df.columns:
        raise ValueError("Matchups CSV must contain 'home_team' and 'away_team' columns")
    
    return list(zip(matchups_df["home_team"], matchups_df["away_team"]))


def summarize_importances(model: EnsembleOutcomeModel) -> dict[str, float]:
    """Summarize feature importances from ensemble model."""
    coefs = model.logistic_pipeline.named_steps["model"].coef_[0]
    return {name: float(abs(coef)) for name, coef in zip(config.feature_columns, coefs)}


def get_data(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and aggregate data."""
    path = data_path or config.data_path
    shift_df = load_csv(path)
    return aggregate_pipeline(shift_df)


def get_model(
    game_df: pd.DataFrame, 
    model_path: Optional[Path] = None, 
    use_catboost: bool = True
) -> Tuple[EnsembleOutcomeModel, Optional[any]]:
    """Load or train model."""
    path = model_path or (Path(config.output_dir) / "ensemble_model.joblib")
    metrics = None
    
    if path.exists():
        print(f"♻️ Loading model from {path}...")
        model = EnsembleOutcomeModel.load(path)
    else:
        print("🧠 Training ensemble model...")
        train_df, test_df = split_game_data(game_df)
        model = EnsembleOutcomeModel(use_catboost=use_catboost).fit(train_df)
        metrics = model.evaluate(test_df)
        
        # Retrain on full data for production
        model.fit(game_df)
        model.save(path)
        print(f"💾 Model saved to {path}")
        
    return model, metrics


def run_headless(args: argparse.Namespace) -> None:
    """Run the full prediction pipeline without UI."""
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("whl_engine")
    
    try:
        logger.info("🚀 Starting WHL Performance Prediction Pipeline")
        
        # 0. Prep
        out_dir = Path(args.output or config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Data Ingestion & Aggregation
        logger.info(f"📁 Loading data from {args.data or config.data_path}")
        game_df, team_df = get_data(args.data)
        logger.info(f"✅ Aggregated: {len(game_df)} games, {len(team_df)} teams")
        
        # 2. Ranking
        logger.info("🏆 Ranking teams...")
        ranked = rank_teams(team_df)
        logger.info(f"⭐ Top 3: {', '.join(ranked['team'].head(3))}")
        
        # 3. Modeling
        model_path = out_dir / "ensemble_model.joblib" if args.load else None
        # If args.load is False, get_model handles training logic, but we need to ensure it doesn't just load if not requested
        # Actually existing logic was: if args.load and path.exists -> load, else train.
        # My get_model attempts to load if path exists.
        # Let's adjust get_model call slightly or rely on its logic. 
        # If args.load is NOT set, we should force retraining? 
        # The original code: if args.load and exists -> load. Else -> train.
        
        if args.load and (out_dir / "ensemble_model.joblib").exists():
             model, metrics = get_model(game_df, model_path=out_dir / "ensemble_model.joblib", use_catboost=not args.no_catboost)
        else:
             # Force training by passing a non-existent path? No, get_model doesn't have force_train arg.
             # Let's just inline the logic or trust get_model to check existence. 
             # But if I want to FORCE Retrain, I need to delete the model or handle it.
             # Let's just follow original logic:
             logger.info("🧠 Training ensemble model...")
             train_df, test_df = split_game_data(game_df)
             model = EnsembleOutcomeModel(use_catboost=not args.no_catboost).fit(train_df)
             metrics = model.evaluate(test_df)
             roc_str = f"{metrics.roc_auc:.3f}" if metrics.roc_auc else "N/A"
             logger.info(f"📈 Model Metrics - Accuracy: {metrics.accuracy:.1%}, ROC AUC: {roc_str}")
             
             model.fit(game_df)
             model.save(out_dir / "ensemble_model.joblib")

        # 4. Matchup Prediction
        logger.info("🎯 Loading tournament matchups...")
        matchups = load_tournament_matchups(args.matchups or config.matchups_path)
        
        available_teams = set(team_df["team"])
        missing = {t for m in matchups for t in m if t not in available_teams}
        if missing:
            logger.error(f"❌ Missing data for teams: {missing}")
            return

        matchup_df = build_matchup_features(team_df, matchups)
        predictions = model.predict_matchups(matchup_df)
        
        # 5. Simulations
        logger.info(f"🎲 Running {config.num_simulations} Monte Carlo simulations...")
        simulations = simulate_matchups(predictions)
        
        # 6. Persistence
        ranked.to_csv(out_dir / "team_rankings.csv", index=False)
        predictions.to_csv(out_dir / "matchup_predictions.csv", index=False)
        simulations.to_csv(out_dir / "matchup_simulations.csv", index=False)
        
        # 7. Visualizations
        logger.info("🎨 Generating visualizations...")
        correlation_heatmap(team_df, out_dir / "correlation_heatmap.png")
        power_ranking_bar_chart(ranked, out_dir / "power_rankings.html")
        
        importances = summarize_importances(model)
        feature_importance_bar_chart(importances, out_dir / "feature_importance.html")
        
        if args.explain:
            logger.info("🔍 Explaining model with SHAP...")
            full_features = build_game_features(game_df)
            x_full = full_features[config.feature_columns]
            explain_model(model.logistic_pipeline, x_full, out_dir / "shap_summary.png")
            
        # 8. Final Metrics
        best_feat = max(importances, key=importances.get)
        if metrics:
            (out_dir / "model_metrics.json").write_text(
                pd.Series({"accuracy": metrics.accuracy, "roc_auc": metrics.roc_auc}).to_json()
            )
        (out_dir / "best_feature.txt").write_text(best_feat)
        
        logger.info(f"🏁 Pipeline complete! Best feature: {best_feat}")

        # ── 9. Offensive Line Quality Disparity (Task 1b / 1c) ─────────────
        logger.info("📐 Calculating Offensive Line Quality Disparity (Task 1b)...")
        disparity_csv = out_dir / "top10_disparity.csv"
        disparity_png = out_dir / "disparity_impact_viz.png"

        # Re-load raw shifts for disparity (game_df is already aggregated)
        raw_shifts = load_csv(args.data or config.data_path)
        disparity_all = calculate_disparity(raw_shifts, output_path=disparity_csv)

        # Console output — pretty Top 10 table
        top10 = disparity_all.head(10)[
            ["rank", "team", "first_off_xg60", "second_off_xg60", "disparity"]
        ].copy()
        top10.columns = ["#", "Team", "1st Line xG/60", "2nd Line xG/60", "Ratio"]
        top10 = top10.set_index("#")
        top10["1st Line xG/60"] = top10["1st Line xG/60"].map("{:.3f}".format)
        top10["2nd Line xG/60"] = top10["2nd Line xG/60"].map("{:.3f}".format)
        top10["Ratio"]   = top10["Ratio"].map("{:.3f}".format)

        print()
        print("---------------------------------------------------")
        print("  TOP 10 - Offensive Line Quality Disparity")
        print("---------------------------------------------------")
        print(top10.to_string())
        print("---------------------------------------------------")
        print()

        logger.info(f"💾 Top-10 disparity CSV → {disparity_csv}")

        # Visualisation (Task 1c) — requires score in ranked df
        logger.info("🎨 Generating disparity scatter plot (Task 1c)...")
        disparity_scatter_plot(
            disparity_all,
            ranked,
            output_path=disparity_png,
        )
        logger.info(f"🖼️  Disparity scatter saved → {disparity_png}")

    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WHL Performance Prediction Engine")
    parser.add_argument("--data", type=Path, help="Path to input data CSV")
    parser.add_argument("--matchups", type=Path, help="Path to matchups CSV")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--load", action="store_true", help="Load saved model from output directory")
    parser.add_argument("--no-catboost", action="store_true", help="Disable CatBoost in ensemble")
    parser.add_argument("--explain", action="store_true", help="Generate SHAP explanations")
    parser.add_argument("--app", action="store_true", help="Launch interactive Streamlit dashboard")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.app:
        print("🎨 Launching Streamlit Dashboard (with SSL workaround)...")
        # Workaround for Windows ASN1 certificate error
        cmd = [
            sys.executable, "-c",
            "import ssl; ssl.SSLContext._load_windows_store_certs = lambda *a, **kw: None; "
            "from streamlit.web.cli import main; main()",
            "run", "app.py"
        ]
        subprocess.run(cmd)
    else:
        run_headless(args)
