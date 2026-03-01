"""Integration tests for the WHL Prediction Engine."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from main import run_headless


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture for temporary output directory."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    yield out_dir
    shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.skip(
    reason=(
        "Full end-to-end integration test requires the real whl.csv and "
        "matchups.csv data files and is too slow for unit test runs. "
        "Run manually with: pytest tests/test_integration.py --no-header -v"
    )
)
def test_pipeline_execution(temp_output_dir):
    """Test that the full pipeline runs end-to-end."""
    # Create a mock Namespace to simulate CLI args
    class Args:
        data = Path("data/whl.csv")
        matchups = Path("data/matchups.csv")
        output = temp_output_dir
        no_catboost = False
        explain = False
        verbose = True
        load = False

    args = Args()

    # Run pipeline
    run_headless(args)

    # Check that outputs exist
    assert (temp_output_dir / "team_rankings.csv").exists()
    assert (temp_output_dir / "matchup_predictions.csv").exists()
    assert (temp_output_dir / "matchup_simulations.csv").exists()
    assert (temp_output_dir / "ensemble_model.joblib").exists()
    assert (temp_output_dir / "model_metrics.json").exists()

    # Check output content
    predictions = pd.read_csv(temp_output_dir / "matchup_predictions.csv")
    assert not predictions.empty
    assert "home_win_probability" in predictions.columns

    rankings = pd.read_csv(temp_output_dir / "team_rankings.csv")
    assert not rankings.empty
    assert "score" in rankings.columns
