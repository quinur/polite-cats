# WHL 2026 Performance Prediction Engine
**Team: Polite Cats** | *Wharton High School Data Science Competition 2026*

## 🏒 Project Overview
This repository contains a modular data science pipeline designed to analyze the World Hockey League (WHL) 2025 season. Our goal is to provide data-driven Power Rankings, matchup predictions, and feature importance analysis for the upcoming WHL Tournament.

## 🏗 Modular Architecture
The project is structured into functional modules to ensure scalability and clarity:
- `data_loader.py`: Handles raw CSV/Excel ingestion.
- `aggregator.py`: Converts shift-level data into game-level and team-level metrics.
- `ranker_engine.py`: Implements our proprietary Power Ranking formula.
- `predictor_ml.py`: Machine Learning models (Logistic Regression/CatBoost) for matchup probabilities.
- `viz_factory.py`: Generates high-fidelity visualizations for the final report.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/quinur/polite-cats.git
   cd polite-cats
   ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data Source
All analysis is based on the fictional WHL 2025 Dataset provided by the Wharton Sports Analytics and Business Initiative.

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.