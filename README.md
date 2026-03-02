# 🏒 WHL 2026 Performance Prediction Engine
**Team: Polite Cats** | *Wharton High School Data Science Competition 2026*

## 🌟 Achievement: 63.5% Accuracy 🎯
We have successfully developed a robust predictive model that achieves **63.5% accuracy** (ROC AUC: 0.620) on the WHL 2025 dataset. This version incorporates advanced situational analytics and individual performance deltas.

## 🏗 Modular Architecture
The project is structured into a production-ready pipeline:
- `src/config.py`: Centralized configuration management for all hyperparameters and paths.
- `src/aggregator.py`: Phase 4 data engine (Goalie GSAX, PP Efficiency, High-Danger Suppression).
- `src/ensemble_model.py`: Weighted ensemble (70/30) of **Logistic Regression** and **CatBoost**.
- `src/predictor_ml.py`: Advanced Feature engineering (Recency Win %, Finishing Skill, Efficiency Diffs).
- `src/viz_factory.py`: High-fidelity visualizations (Heatmaps, Power Rankings, Feature Importance).
- `src/simulator.py`: Monte Carlo tournament simulations.

## 🚀 Getting Started

### Installation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline
The engine now supports a full CLI for production use:

```bash
# Standard run (trains and predicts)
python main.py

# Run with SHAP explanations and verbose logging
python main.py --explain --verbose

# Use a specific dataset and save to a custom folder
python main.py --data data/custom_whl.csv --output custom_results/

# Fast run by loading a previously saved model
python main.py --load
```

## 📊 Outputs (outputs/)
- `team_rankings.csv`: The official Power Ranking leaderboard.
- `matchup_predictions.csv`: Win probabilities for tournament matchups.
- `matchup_simulations.csv`: Monte Carlo results for tournament outcomes.
- `feature_importance.html`: Interactive chart showing the "DNA" of our model's decisions.
- `power_rankings.html`: Visual representation of team strength.
- `ensemble_model.joblib`: Persisted model for deployment.

## 🧠 Why Our Model Wins
Our model focuses on **efficiency metrics** rather than raw totals. By analyzing `win_pct_diff` and `xg_per_shot_diff`, we capture the underlying quality of a team's play, which is a far more reliable predictor than simple goal counting.

## ⚖️ License
This project is licensed under the MIT License.