"""Tests for visualization factory."""

from __future__ import annotations

import pandas as pd

from src.viz_factory import correlation_heatmap, power_ranking_bar_chart


def test_visualizations_create_files(tmp_path):
    df = pd.DataFrame(
        {
            "team": ["A", "B"],
            "score": [1.0, 2.0],
            "metric": [0.1, 0.2],
        }
    )
    heatmap_path = correlation_heatmap(df, tmp_path / "heatmap.png")
    ranking_path = power_ranking_bar_chart(df, tmp_path / "rankings.html")
    assert (tmp_path / "heatmap.png").exists()
    assert (tmp_path / "rankings.html").exists()
    assert heatmap_path.endswith("heatmap.png")
    assert ranking_path.endswith("rankings.html")