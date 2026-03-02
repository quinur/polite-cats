"""Offensive Line Quality Disparity calculations (Task 1b).

Methodology
-----------
We work from the raw shift-level DataFrame and measure how "top-heavy" each
team's offensive production is across its 5v5 lines.

Steps:
  1. Melt the wide-format data into a long "team / off_line" view, keeping
     only the two genuine 5-on-5 offensive line slots (``first_off`` and
     ``second_off``).  Special-situation rows (PP, PK, empty-net) are
     excluded because they are not comparable units of offensive quality.

  2. For each (team, off_line) combination aggregate:
       - total xG generated  (``xg_sum``)
       - total TOI in seconds (``toi_sum``)
       - xG per 60 minutes:  ``xg60 = xg_sum / (toi_sum / 3600) * 60``

  3. Compute the **Disparity Metric** per team:
       ``disparity = first_off_xg60 / second_off_xg60``

     A higher value means the first line vastly outperforms the second line —
     i.e. the team relies heavily on its top line for offence.
     We handle division by zero by safely falling back to NaN or computing normally.

  4. Output ``outputs/top10_disparity.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Only these labels represent comparable 5v5 offensive line shifts.
_EVEN_STRENGTH_LINES = {"first_off", "second_off"}


def calculate_disparity(
    shift_df: pd.DataFrame,
    output_path: str | Path | None = "outputs/top10_disparity.csv",
) -> pd.DataFrame:
    """Calculate Offensive Line Quality Disparity for every team.

    Parameters
    ----------
    shift_df:
        Raw shift-level DataFrame (as loaded by ``load_csv``).
    output_path:
        Where to write the Top-10 CSV.  Pass ``None`` to skip writing.

    Returns
    -------
    pd.DataFrame
        Full disparity table for all 32 teams, sorted descending by
        ``disparity``.  Columns: team, first_off_xg60, second_off_xg60,
        disparity, disparity_std.
    """
    df = shift_df.copy()

    # ── 1. Melt into long form ────────────────────────────────────────────
    home_cols = {
        "home_team": "team",
        "home_off_line": "off_line",
        "home_xg": "xg",
        "toi": "toi",
    }
    away_cols = {
        "away_team": "team",
        "away_off_line": "off_line",
        "away_xg": "xg",
        "toi": "toi",
    }

    home = df[list(home_cols)].rename(columns=home_cols)
    away = df[list(away_cols)].rename(columns=away_cols)
    long = pd.concat([home, away], ignore_index=True)

    # Keep only genuine even-strength offensive lines
    long = long[long["off_line"].isin(_EVEN_STRENGTH_LINES)]
    logger.debug(
        "Disparity calc: %d shift-rows after filtering to %s",
        len(long),
        _EVEN_STRENGTH_LINES,
    )

    # ── 2. Aggregate xG and TOI per (team, off_line) ──────────────────────
    by_line = (
        long.groupby(["team", "off_line"], as_index=False)
        .agg(xg_sum=("xg", "sum"), toi_sum=("toi", "sum"))
    )
    # xG per 60 minutes  (TOI is in seconds in this dataset)
    by_line["xg60"] = by_line["xg_sum"] / (by_line["toi_sum"] / 3600)

    logger.debug("Line xG60 sample:\n%s", by_line.head(8).to_string(index=False))

    # ── 3. Pivot and compute disparity metrics ────────────────────────────
    pivot = by_line.pivot_table(
        index="team", columns="off_line", values="xg60", aggfunc="first"
    ).rename_axis(None, axis=1).reset_index()

    # Guarantee both columns exist even if a team lacks one line
    for col in ("first_off", "second_off"):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot = pivot.rename(
        columns={"first_off": "first_off_xg60", "second_off": "second_off_xg60"}
    )

    # Disparity = Ratio (First Line / Second Line xG/60)
    pivot["disparity"] = np.where(
        pivot["second_off_xg60"] == 0,
        np.nan,
        pivot["first_off_xg60"] / pivot["second_off_xg60"]
    )
    # Standard deviation as secondary metric (useful for > 2 lines)
    pivot["disparity_std"] = pivot[["first_off_xg60", "second_off_xg60"]].std(
        axis=1, ddof=1
    )

    disparity_all = pivot.sort_values("disparity", ascending=False).reset_index(drop=True)
    disparity_all["rank"] = disparity_all.index + 1

    # ── 4. Top-10 slice and CSV output ────────────────────────────────────
    top10 = disparity_all.head(10).copy()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df = top10[["rank", "team", "disparity"]].rename(
            columns={"rank": "Rank", "team": "Team", "disparity": "Ratio"}
        )
        out_df.to_csv(out, index=False)
        logger.info("💾 Saved top-10 disparity table → %s", out)

    return disparity_all
