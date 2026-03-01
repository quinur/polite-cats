"""Visualization utilities for WHL analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


def correlation_heatmap(df: pd.DataFrame, output_path: str | Path) -> str:
    """Generate and save a correlation heatmap.

    Args:
        df: DataFrame to compute correlations.
        output_path: Path to save the image.

    Returns:
        Path to saved image.
    """
    # Filter only the columns we use in the model
    from src.config import config
    import numpy as np
    
    cols = [c for c in df.columns if any(f.split('_')[0] in c for f in config.feature_columns)]
    if not cols:
        cols = df.select_dtypes(include=[np.number]).columns
    
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", square=True)
    plt.title("WHL Feature Correlation Heatmap")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)


def feature_importance_bar_chart(importances: dict[str, float], output_path: str | Path) -> str:
    """Generate and save a feature importance bar chart.
    
    Args:
        importances: Dictionary mapping feature names to importance scores.
        output_path: Path to save the image.
        
    Returns:
        Path to saved image.
    """
    df = pd.DataFrame(list(importances.items()), columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=True)
    
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Model Feature Importance",
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=500, margin=dict(l=150))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return str(output_path)


def power_ranking_bar_chart(rank_df: pd.DataFrame, output_path: str | Path) -> str:
    """Generate and save a power ranking bar chart.

    Args:
        rank_df: Ranked team DataFrame.
        output_path: Path to save HTML.

    Returns:
        Path to saved HTML.
    """
    fig = px.bar(
        rank_df,
        x="team",
        y="score",
        title="WHL Power Rankings",
        color="score",
        color_continuous_scale="Blues",
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return str(output_path)


def disparity_scatter_plot(
    disparity_df: pd.DataFrame,
    team_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 10,
) -> str:
    """Plot Offensive Line Disparity vs Win Percentage with a regression trendline.

    Args:
        disparity_df: Full disparity DataFrame from ``calculate_disparity``.
        team_df: Team-level DataFrame (must contain ``team`` and ``win_pct``).
        output_path: Path to save the PNG.
        top_n: Number of top-disparity teams to highlight and label.

    Returns:
        Path to saved image as string.
    """
    import numpy as np
    import matplotlib.patches as mpatches

    # ── Merge disparity with win_pct ──────────────────────────────────────
    merged = disparity_df.merge(
        team_df[["team", "win_pct"]].rename(columns={"win_pct": "win_pct"}),
        on="team",
        how="inner",
    ).dropna(subset=["disparity", "win_pct"])

    if merged.empty:
        raise ValueError("No overlapping teams between disparity_df and team_df.")

    top_mask = merged["rank"] <= top_n

    # ── Figure setup — dark premium theme ────────────────────────────────
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 9), dpi=160)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Subtle grid
    ax.grid(color="#2a2d35", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # ── Regression line (OLS) ─────────────────────────────────────────────
    x = merged["disparity"].values
    y = merged["win_pct"].values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
    ax.plot(
        x_line,
        m * x_line + b,
        color="#ff6b35",
        linewidth=2.2,
        linestyle="--",
        alpha=0.85,
        zorder=2,
        label=f"Trend  (slope={m:+.3f})",
    )

    # ── Background (non-top) points ───────────────────────────────────────
    bg = merged[~top_mask]
    ax.scatter(
        bg["disparity"],
        bg["win_pct"],
        color="#4a9eff",
        alpha=0.55,
        s=70,
        edgecolors="#2a7dd4",
        linewidths=0.8,
        zorder=3,
        label="All teams",
    )

    # ── Top-N highlighted points ──────────────────────────────────────────
    top = merged[top_mask]
    scatter_top = ax.scatter(
        top["disparity"],
        top["win_pct"],
        c=top["disparity"],
        cmap="plasma",
        vmin=merged["disparity"].min(),
        vmax=merged["disparity"].max(),
        s=200,
        edgecolors="white",
        linewidths=1.4,
        zorder=5,
        label=f"Top-{top_n} highest disparity",
    )

    # Add a colorbar for the top-N gradient
    cbar = fig.colorbar(scatter_top, ax=ax, pad=0.02, shrink=0.75)
    cbar.set_label("Disparity (xG60 gap)", color="#c9d1d9", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

    # ── Team label annotations ────────────────────────────────────────────
    for _, row in top.iterrows():
        ax.annotate(
            row["team"].capitalize(),
            xy=(row["disparity"], row["win_pct"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8.5,
            color="#e6edf3",
            fontweight="bold",
            va="bottom",
        )

    # ── Correlation annotation in corner ─────────────────────────────────
    r = float(np.corrcoef(x, y)[0, 1])
    ax.text(
        0.02, 0.97,
        f"Pearson r = {r:+.3f}",
        transform=ax.transAxes,
        fontsize=12,
        color="#ff6b35",
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#1a1f2b", ec="#ff6b35", alpha=0.85),
    )

    # ── Quadrant reference lines (median) ────────────────────────────────
    ax.axvline(np.median(x), color="#c9d1d9", linewidth=0.7, alpha=0.35, linestyle=":")
    ax.axhline(np.median(y), color="#c9d1d9", linewidth=0.7, alpha=0.35, linestyle=":")

    # ── Labels, title, legend ─────────────────────────────────────────────
    ax.set_xlabel("Offensive Line Quality Disparity  (xG/60 gap: best − worst line)",
                  color="#c9d1d9", fontsize=13, labelpad=12)
    ax.set_ylabel("Team Win Percentage", color="#c9d1d9", fontsize=13, labelpad=12)
    ax.set_title(
        "Offensive Line Disparity vs Team Success\n"
        "WHL 2026 · Wharton High School Data Science Competition",
        color="#e6edf3", fontsize=15, fontweight="bold", pad=18,
    )

    ax.tick_params(colors="#8b949e", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    legend = ax.legend(
        fontsize=10,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
        markerscale=1.2,
        loc="lower right",
    )

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    plt.style.use("default")   # reset style so other plots are unaffected

    return str(output_path)
