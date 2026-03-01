"""Aggregation utilities for WHL 2025 shift-level data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregationConfig:
    """Configuration for aggregation behavior."""

    home_prefix: str = "home_"
    away_prefix: str = "away_"


def aggregate_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shift-level rows into game-level totals with advanced metrics."""
    df = df.copy()

    # Robust game_num parsing: support both "game_1" string IDs and raw integers
    if pd.api.types.is_integer_dtype(df["game_id"]):
        df["game_num"] = df["game_id"].astype(int)
    else:
        df["game_num"] = (
            df["game_id"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )

    # Robust record_num parsing: optional column, fall back to row position
    if "record_id" in df.columns:
        if pd.api.types.is_integer_dtype(df["record_id"]):
            df["record_num"] = df["record_id"].astype(int)
        else:
            df["record_num"] = (
                df["record_id"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(int)
            )
    else:
        df["record_num"] = range(len(df))

    df = df.sort_values(["game_num", "record_num"])

    if "toi" in df.columns:
        df["home_toi"] = df["toi"]
        df["away_toi"] = df["toi"]

    # Score-Close XG Calculation (Stats when score diff <= 1 goal)
    df["home_cum_goals"] = df.groupby("game_id")["home_goals"].cumsum()
    df["away_cum_goals"] = df.groupby("game_id")["away_goals"].cumsum()
    df["prev_home_score"] = df.groupby("game_id")["home_cum_goals"].shift(1).fillna(0)
    df["prev_away_score"] = df.groupby("game_id")["away_cum_goals"].shift(1).fillna(0)
    df["score_diff"] = (df["prev_home_score"] - df["prev_away_score"]).abs()
    
    df["home_xg_close"] = df["home_xg"].where(df["score_diff"] <= 1, 0)
    df["away_xg_close"] = df["away_xg"].where(df["score_diff"] <= 1, 0)

    # Situational XG
    df["home_is_pp"] = (df["home_off_line"] == "PP_up").astype(int)
    df["away_is_pp"] = (df["away_off_line"] == "PP_up").astype(int)
    df["home_pp_xg"] = df["home_xg"].where(df["home_is_pp"] == 1, 0)
    df["away_pp_xg"] = df["away_xg"].where(df["away_is_pp"] == 1, 0)
    df["home_pk_xg"] = df["home_xg"].where(df["away_is_pp"] == 1, 0)
    df["away_pk_xg"] = df["away_xg"].where(df["home_is_pp"] == 1, 0)
    
    # Goalie and Lineup detection
    def get_main_features(group):
        return pd.Series({
            "home_goalie": group["home_goalie"].mode().iloc[0] if not group["home_goalie"].empty else "unknown",
            "away_goalie": group["away_goalie"].mode().iloc[0] if not group["away_goalie"].empty else "unknown",
            "home_off_line": group["home_off_line"].mode().iloc[0] if not group["home_off_line"].empty else "unknown",
            "away_off_line": group["away_off_line"].mode().iloc[0] if not group["away_off_line"].empty else "unknown",
            "home_def_pairing": group["home_def_pairing"].mode().iloc[0] if not group["home_def_pairing"].empty else "unknown",
            "away_def_pairing": group["away_def_pairing"].mode().iloc[0] if not group["away_def_pairing"].empty else "unknown"
        })

    lineups = df.groupby("game_id").apply(get_main_features).reset_index()

    agg_dict = {
        "home_goals": "sum",
        "away_goals": "sum",
        "home_xg": "sum",
        "away_xg": "sum",
        "home_shots": "sum",
        "away_shots": "sum",
        "home_max_xg": "mean",
        "away_max_xg": "mean",
        "home_penalties_committed": "sum",
        "away_penalties_committed": "sum",
        "home_is_pp": "sum",
        "away_is_pp": "sum",
        "home_xg_close": "sum",
        "away_xg_close": "sum",
        "home_pp_xg": "sum",
        "away_pp_xg": "sum",
        "home_pk_xg": "sum",
        "away_pk_xg": "sum",
    }
    
    if "home_toi" in df.columns:
        agg_dict["home_toi"] = "sum"
        agg_dict["away_toi"] = "sum"

    game_totals = (
        df.groupby(["game_num", "game_id", "home_team", "away_team", "went_ot"], as_index=False)
        .agg(agg_dict)
    )
    
    game_totals = game_totals.merge(lineups, on="game_id")
    
    # Goalie GSAX calculations
    home_goalie_gsax = df.groupby(["game_id"]).apply(
        lambda x: (x["away_xg"] - x["away_goals"]).sum()
    ).reset_index(name="home_goalie_gsax")
    
    away_goalie_gsax = df.groupby(["game_id"]).apply(
        lambda x: (x["home_xg"] - x["home_goals"]).sum()
    ).reset_index(name="away_goalie_gsax")

    game_totals = game_totals.merge(home_goalie_gsax, on="game_id")
    game_totals = game_totals.merge(away_goalie_gsax, on="game_id")

    # ── Tie / Shootout Resolution ────────────────────────────────────────────────
    # Shootout goals are excluded from shift-level data (no TOI entry), so a
    # goals==goals tie does NOT mean the real game ended in a draw.  We use a
    # hierarchy of fallbacks:
    #   1. Direct comparison works fine when one team genuinely scored more.
    #   2. For tied-goal games that went to OT/SO we award the win to the team
    #      with higher xg_close (best proxy for sustained pressure).
    #      If xg_close is also equal we fall back to total xg, then home advantage.
    #   3. Any game still tied after that is logged as CRITICAL and dropped so
    #      corrupt 0-0 labels never reach the ML model.

    tied_mask = game_totals["home_goals"] == game_totals["away_goals"]
    n_tied = tied_mask.sum()
    if n_tied > 0:
        logger.warning(
            "⚠️  %d game(s) have equal goal totals after aggregation — applying "
            "xg_close tiebreaker for OT/SO games.",
            n_tied,
        )

    # Default: goals winner
    game_totals["home_win"] = (game_totals["home_goals"] > game_totals["away_goals"]).astype(int)
    game_totals["away_win"] = (game_totals["away_goals"] > game_totals["home_goals"]).astype(int)

    # Apply tiebreaker only to OT/SO rows that ended tied on goals
    ot_tied = tied_mask & (game_totals["went_ot"].astype(int) == 1)
    if ot_tied.any():
        xg_close_home = game_totals.loc[ot_tied, "home_xg_close"]
        xg_close_away = game_totals.loc[ot_tied, "away_xg_close"]
        xg_home       = game_totals.loc[ot_tied, "home_xg"]
        xg_away       = game_totals.loc[ot_tied, "away_xg"]

        # Primary: xg_close; Secondary: total xg; Tertiary: home advantage (=1)
        home_wins_ot = (
            (xg_close_home > xg_close_away)
            | ((xg_close_home == xg_close_away) & (xg_home > xg_away))
            | ((xg_close_home == xg_close_away) & (xg_home == xg_away))
        ).astype(int)

        game_totals.loc[ot_tied, "home_win"] = home_wins_ot
        game_totals.loc[ot_tied, "away_win"] = (1 - home_wins_ot)
        logger.info(
            "🏒 Resolved %d OT/SO tie(s) via xg_close tiebreaker.", ot_tied.sum()
        )

    # Drop any remaining unresolvable ties (non-OT with equal goals — shouldn't
    # happen in a real dataset, but guards against corrupt rows).
    irresolvable = tied_mask & ~ot_tied
    if irresolvable.any():
        logger.critical(
            "🚨 %d non-OT game(s) still tied after aggregation — dropping to avoid "
            "corrupting ML labels.  Check raw data for missing goal events.",
            irresolvable.sum(),
        )
        game_totals = game_totals[~irresolvable]
    # ────────────────────────────────────────────────────────────────────────────

    return game_totals.sort_values("game_num")


def calculate_elo(game_df: pd.DataFrame, k: int = 32) -> tuple[pd.DataFrame, dict]:
    """Implement Bayesian Team Ratings (Elo)."""
    teams = pd.concat([game_df["home_team"], game_df["away_team"]]).unique()
    elos = {team: 1500.0 for team in teams}
    
    home_elo_before = []
    away_elo_before = []
    elo_prob = []
    
    for _, row in game_df.iterrows():
        h, a = row["home_team"], row["away_team"]
        home_elo_before.append(elos[h])
        away_elo_before.append(elos[a])
        
        ea = 1 / (1 + 10 ** ((elos[a] - elos[h]) / 400))
        elo_prob.append(ea)
        
        sa = 1.0 if row["home_win"] == 1 else 0.0
        elos[h] += k * (sa - ea)
        elos[a] += k * (1.0 - sa - (1.0 - ea))
        
    game_df["home_elo"] = home_elo_before
    game_df["away_elo"] = away_elo_before
    game_df["elo_prob"] = elo_prob
    return game_df, elos


def _calculate_player_features(history: list, decay_lambda: float = 0.05) -> dict:
    """Calculate rolling stats for a specific player (e.g. Goalie)."""
    if not history:
        return {"gsax_avg": 0.0, "xg_against_avg": 2.5}
    
    n = len(history)
    weights = [np.exp(-decay_lambda * (n - 1 - i)) for i in range(n)]
    sum_w = sum(weights)
    
    gsax = [h["gsax"] for h in history]
    xga = [h["xg_against"] for h in history]
    
    return {
        "gsax_avg": sum(g * w for g, w in zip(gsax, weights)) / sum_w,
        "xg_against_avg": sum(x * w for x, w in zip(xga, weights)) / sum_w
    }


def _calculate_team_features(games: list, is_home: bool, decay_lambda: float = 0.04) -> dict:
    """Calculate time-decayed metrics for a team from its history."""
    if not games:
        return {
            "win_pct": 0.5, "xg_per_game": 2.5, "avg_xg": 2.5, "goals_for": 2.5,
            "shot_efficiency": 0.09, "gsax": 0.0, "high_danger": 0.5, 
            "pp_strength": 0.15, "pk_strength": 0.15, "finishing": 0.0, 
            "xg_close": 1.5, "home_away_split": 0.0, "sos": 1500.0,
            "shot_rate": 30.0, "penalty_rate": 4.0, "toi_avg": 3600.0, "xg_per_shot": 0.08
        }
    
    games = sorted(games, key=lambda x: x['game_num'])
    n = len(games)
    weights = [np.exp(-decay_lambda * (n - 1 - i)) for i in range(n)]
    sum_weights = sum(weights)
    
    def weighted_avg(values):
        return sum(v * w for v, w in zip(values, weights)) / sum_weights

    wins = [1 if (g['home_team'] == g['team'] and g['home_win'] == 1) or 
              (g['away_team'] == g['team'] and g['away_win'] == 1) else 0 for g in games]
    
    home_games = [g for g in games if g['home_team'] == g['team']]
    away_games = [g for g in games if g['away_team'] == g['team']]
    
    h_win_rate = sum(1 for g in home_games if g['home_win'] == 1) / (len(home_games) + 1e-6)
    a_win_rate = sum(1 for g in away_games if g['away_win'] == 1) / (len(away_games) + 1e-6)
    
    xg_for = [g['home_xg'] if g['home_team'] == g['team'] else g['away_xg'] for g in games]
    shots_for = [g['home_shots'] if g['home_team'] == g['team'] else g['away_shots'] for g in games]
    goals_for = [g['home_goals'] if g['home_team'] == g['team'] else g['away_goals'] for g in games]
    toi_for = [g['home_toi'] if g['home_team'] == g['team'] else g['away_toi'] for g in games]
    penalties = [g['home_penalties_committed'] if g['home_team'] == g['team'] else g['away_penalties_committed'] for g in games]
    gsax = [g['home_goalie_gsax'] if g['home_team'] == g['team'] else g['away_goalie_gsax'] for g in games]
    xg_close = [g['home_xg_close'] if g['home_team'] == g['team'] else g['away_xg_close'] for g in games]
    pp_xg = [g['home_pp_xg'] if g['home_team'] == g['team'] else g['away_pp_xg'] for g in games]
    pp_shifts = [g['home_is_pp'] if g['home_team'] == g['team'] else g['away_is_pp'] for g in games]
    pk_xg_against = [g['home_pk_xg'] if g['home_team'] == g['team'] else g['away_pk_xg'] for g in games]
    pk_opp_pp_shifts = [g['away_is_pp'] if g['home_team'] == g['team'] else g['home_is_pp'] for g in games]
    high_danger = [g['home_max_xg'] if g['away_team'] == g['team'] else g['away_max_xg'] for g in games]
    opp_elos = [g['away_elo'] if g['home_team'] == g['team'] else g['home_elo'] for g in games]
    
    return {
        "win_pct": weighted_avg(wins),
        "xg_per_game": weighted_avg(xg_for),
        "avg_xg": weighted_avg(xg_for),
        "goals_for": weighted_avg(goals_for),
        "shot_efficiency": weighted_avg([g / (s + 1e-6) for g, s in zip(goals_for, shots_for)]),
        "gsax": weighted_avg(gsax),
        "high_danger": weighted_avg(high_danger),
        "pp_strength": sum(pp_xg) / (sum(pp_shifts) + 1e-6),
        "pk_strength": sum(pk_xg_against) / (sum(pk_opp_pp_shifts) + 1e-6),
        "finishing": weighted_avg([g - x for g, x in zip(goals_for, xg_for)]),
        "xg_close": weighted_avg(xg_close),
        "home_away_split": h_win_rate - a_win_rate,
        "sos": weighted_avg(opp_elos),
        "shot_rate": weighted_avg(shots_for),
        "penalty_rate": weighted_avg(penalties),
        "toi_avg": weighted_avg(toi_for),
        "xg_per_shot": weighted_avg([x / (s + 1e-6) for x, s in zip(xg_for, shots_for)])
    }


def aggregate_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full aggregation pipeline with strict leakage prevention."""
    import numpy as np
    
    game_level = aggregate_game_level(df)
    game_level, elos = calculate_elo(game_level)
    
    team_histories = {team: [] for team in pd.concat([game_level["home_team"], game_level["away_team"]]).unique()}
    goalie_histories = {g: [] for g in pd.concat([game_level["home_goalie"], game_level["away_goalie"]]).unique()}
    
    rows = []
    for _, row in game_level.iterrows():
        h_team, a_team = row["home_team"], row["away_team"]
        h_goalie, a_goalie = row["home_goalie"], row["away_goalie"]
        
        # 1. Calculate features BEFORE this game (No Leakage)
        h_team_stats = _calculate_team_features(team_histories[h_team], is_home=True)
        a_team_stats = _calculate_team_features(team_histories[a_team], is_home=False)
        h_goalie_stats = _calculate_player_features(goalie_histories[h_goalie])
        a_goalie_stats = _calculate_player_features(goalie_histories[a_goalie])
        
        # 2. Enrich row
        enriched_row = row.to_dict()
        enriched_row["home_games_played"] = len(team_histories[h_team])
        enriched_row["away_games_played"] = len(team_histories[a_team])
        
        for k, v in h_team_stats.items(): enriched_row[f"home_{k}"] = v
        for k, v in a_team_stats.items(): enriched_row[f"away_{k}"] = v
        for k, v in h_goalie_stats.items(): enriched_row[f"home_goalie_{k}"] = v
        for k, v in a_goalie_stats.items(): enriched_row[f"away_goalie_{k}"] = v
            
        rows.append(enriched_row)
        
        # 3. Update histories for NEXT games
        game_data = row.to_dict()
        
        h_team_data = game_data.copy(); h_team_data['team'] = h_team
        a_team_data = game_data.copy(); a_team_data['team'] = a_team
        team_histories[h_team].append(h_team_data)
        team_histories[a_team].append(a_team_data)
        
        goalie_histories[h_goalie].append({"gsax": row["home_goalie_gsax"], "xg_against": row["away_xg"]})
        goalie_histories[a_goalie].append({"gsax": row["away_goalie_gsax"], "xg_against": row["home_xg"]})
        
    game_level = pd.DataFrame(rows)
    
    # Latest stats for future predictions
    final_teams = []
    for team, history in team_histories.items():
        stats = _calculate_team_features(history, is_home=True)
        stats['team'] = team
        stats['elo'] = elos[team]
        
        # Get most recent/frequent lineup for matchups
        if history:
            last_game = history[-1]
            if last_game['home_team'] == team:
                stats['home_goalie'] = last_game['home_goalie']
                stats['home_off_line'] = last_game['home_off_line']
                stats['home_def_pairing'] = last_game['home_def_pairing']
            else:
                stats['home_goalie'] = last_game['away_goalie']
                stats['home_off_line'] = last_game['away_off_line']
                stats['home_def_pairing'] = last_game['away_def_pairing']
        else:
            stats['home_goalie'] = 'unknown'
            stats['home_off_line'] = 'unknown'
            stats['home_def_pairing'] = 'unknown'
            
        final_teams.append(stats)
    team_level = pd.DataFrame(final_teams)
    
    # Goalie metadata for matchups
    final_goalies = []
    for g, history in goalie_histories.items():
        stats = _calculate_player_features(history)
        stats['goalie'] = g
        final_goalies.append(stats)
    game_level._goalie_stats = pd.DataFrame(final_goalies) # Ad-hoc attachment for predictor
    
    return game_level, team_level


# ── Public compatibility shim ────────────────────────────────────────────────
def aggregate_team_level(game_df: pd.DataFrame) -> pd.DataFrame:
    """Return a simple team-level summary from game-level data.

    Used by legacy tests and quick-check scripts.  For the full production
    pipeline use ``aggregate_pipeline`` which includes rolling/decay features.
    """
    records = []
    all_teams = pd.concat([game_df["home_team"], game_df["away_team"]]).unique()
    for team in all_teams:
        home_g = game_df[game_df["home_team"] == team]
        away_g = game_df[game_df["away_team"] == team]
        games_played = len(home_g) + len(away_g)
        wins = home_g["home_win"].sum() + away_g["away_win"].sum()
        records.append(
            {
                "team": team,
                "games_played": int(games_played),
                "wins": int(wins),
                "win_pct": wins / games_played if games_played else 0.0,
            }
        )
    return pd.DataFrame(records)
