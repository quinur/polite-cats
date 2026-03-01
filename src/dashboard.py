"""Streamlit dashboard for WHL performance prediction engine."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import shap
from streamlit_shap import st_shap
from fpdf import FPDF
import numpy as np
from src.config import config
from src.predictor_ml import build_matchup_features, build_game_features
from src.simulator import simulate_matchups
import tempfile
import json

def run_dashboard(game_df, team_df, model):
    # Custom CSS for premium look
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@500;700;800&display=swap');
        
        :root {
            --primary: #2563EB;
            --accent: #3B82F6;
            --bg-top: #0F172A;
            --bg-bottom: #1E293B;
            --card-bg: rgba(255, 255, 255, 0.05);
            --sidebar-bg: #FFFFFF;
            --border: rgba(0, 0, 0, 0.05);
            --text-main: #F1F5F9;
            --text-sidebar: #1E293B;
        }

        .main {
            background: linear-gradient(180deg, #0F172A 0%, #172554 100%) !important;
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #0F172A 0%, #172554 100%) !important;
        }

        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            color: var(--text-sidebar) !important;
            border-right: 1px solid #E2E8F0;
        }
        
        [data-testid="stSidebar"] * {
            color: var(--text-sidebar) !important;
        }

        .stMetric {
            background: rgba(255, 255, 255, 0.03) !important;
            padding: 24px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        div.stButton > button:first-child {
            background: #2563EB !important;
            border: none !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        
        div.stButton > button:hover {
            transform: scale(1.02);
            filter: brightness(1.1);
            box-shadow: 0 0 20px rgba(79, 172, 254, 0.4);
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
            color: white !important;
        }

        /* Improved Sidebar Navigation Style */
        .stRadio [data-testid="stWidgetLabel"] {
            display: none;
        }
        
        .stRadio div [role="radiogroup"] label {
            background: rgba(255,255,255,0.03);
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 5px 10px;
            margin-bottom: 5px;
            transition: all 0.2s ease;
        }
        
        .stRadio div [role="radiogroup"] label:hover {
            border-color: var(--primary);
            background: rgba(79, 172, 254, 0.1);
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .element-container, .stPlotlyChart {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Header
    with st.sidebar:
        st.markdown(f"<h1 style='text-align: center; color: var(--primary); font-size: 2rem;'>🏒 POLITE CATS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.6;'>WHL Analytics Engine v2.0</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        menu_options = {
            "🏠 Overview": "Overview",
            "🎯 Matchup Sim": "Live Matchup Simulator",
            "📈 Trends": "Trend Charts",
            "🏆 Finals Bracket": "Tournament Bracket",
            "📄 Scout Reports": "Scouting Reports",
            "🔍 Model Logic": "SHAP Explanations"
        }
        
        choice_label = st.radio("", list(menu_options.keys()))
        choice = menu_options[choice_label]
        
        st.markdown("---")
        st.markdown("### 📊 Engine Status")
        st.success("Model: Online")
        st.info(f"Loaded: {len(game_df)} Games")

    if choice == "Overview":
        render_overview(team_df, game_df)
    elif choice == "Live Matchup Simulator":
        render_matchup_simulator(team_df, model)
    elif choice == "Trend Charts":
        render_trend_charts(game_df, team_df)
    elif choice == "Tournament Bracket":
        render_bracket_generator(team_df, model)
    elif choice == "Scouting Reports":
        render_scouting_reports(team_df, game_df, model)
    elif choice == "SHAP Explanations":
        render_shap_dashboard(game_df, team_df, model)

def render_overview(team_df, game_df):
    st.header("🏠 System Overview")
    
    # Hero metric row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Teams", len(team_df))
    with col2:
        st.metric("Games Tracked", len(game_df))
    with col3:
        top_team = team_df.sort_values("elo", ascending=False).iloc[0]["team"]
        st.metric("Top Ranked", top_team)
    with col4:
        # Calculate recent goal scoring average
        avg_g = (game_df["home_goals"].mean() + game_df["away_goals"].mean())
        st.metric("Avg G/Game", f"{avg_g:.2f}")

    st.markdown("### 🏒 Power Rankings (Top 10)")
    top_10 = team_df.sort_values("elo", ascending=False).head(10)
    
    # Ensure xg_diff exists for the scatter plots later
    if "xg_diff" not in team_df.columns and "xg_per_game" in team_df.columns:
        # If we have xg_per_game, use that as a proxy or calculate if we have against stats
        # For now, let's just create it if missing to prevent the crash
        team_df["xg_diff"] = team_df.get("xg_per_game", 0) - team_df.get("xg_against_avg", team_df["xg_per_game"].mean())

    fig = px.bar(top_10, x="elo", y="team", orientation='h', 
                color="elo", color_continuous_scale="Blues",
                labels={"elo": "ELO Rating", "team": "Team"})
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Scoring Efficiency")
        # Example chart: Shot rate vs Finishing
        if "shot_rate" in team_df.columns and "finishing" in team_df.columns:
            fig_scatter = px.scatter(team_df, x="shot_rate", y="finishing", text="team",
                                    color="xg_diff", size="win_pct",
                                    title="Volume vs. Quality")
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_b:
        st.markdown("### 🛡️ Defensive Prowess")
        if "pk_strength" in team_df.columns and "gsax" in team_df.columns:
            fig_def = px.scatter(team_df, x="pk_strength", y="gsax", text="team",
                                color="elo",
                                title="Special Teams vs. Goaltending")
            fig_def.update_traces(textposition='top center')
            fig_def.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_def, use_container_width=True)

def render_matchup_simulator(team_df, model):
    st.header("🎯 Matchup Predictor")
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1, vs_col, col2 = st.columns([4, 1, 4])
        teams = sorted(team_df["team"].unique())
        
        with col1:
            home_team = st.selectbox("Select Home Team", teams, index=0, key="home_sim")
        with vs_col:
            st.markdown("<h2 style='text-align: center; margin-top: 1.5rem;'>VS</h2>", unsafe_allow_html=True)
        with col2:
            away_team = st.selectbox("Select Away Team", teams, index=1 if len(teams) > 1 else 0, key="away_sim")
        
        if st.button("🔥 GENERATE WIN PROBABILITY"):
            with st.spinner("Analyzing matchup dynamics..."):
                matchup_df = build_matchup_features(team_df, [(home_team, away_team)])
                predictions = model.predict_matchups(matchup_df)
                sim_results = simulate_matchups(predictions, num_simulations=10000)
                
                home_prob = predictions.iloc[0]["home_win_probability"]
                away_prob = 1 - home_prob
                sim_home_win = sim_results.iloc[0]["home_win_rate_simulated"]
                
                st.markdown("---")
                
                m_col1, m_col2, m_col3 = st.columns([2, 3, 2])
                
                with m_col1:
                    st.metric(home_team, f"{home_prob:.1%}", delta=f"Sim: {sim_home_win:.1%}")
                
                with m_col2:
                    # Gauge chart with better colors
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = home_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#00D2FF"},
                            'bgcolor': "rgba(255,255,255,0.05)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(255, 0, 0, 0.1)'},
                                {'range': [40, 60], 'color': 'rgba(255, 255, 0, 0.1)'},
                                {'range': [60, 100], 'color': 'rgba(0, 255, 0, 0.1)'}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        font={'color': "white", 'family': "Outfit"},
                        height=250,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with m_col3:
                    st.metric(away_team, f"{away_prob:.1%}", delta=f"Sim: {1-sim_home_win:.1%}", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)

def render_trend_charts(game_df, team_df):
    st.header("📈 Performance Trends")
    
    tab1, tab2 = st.tabs(["Team Trends", "League Correlations"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            team = st.selectbox("Select Team", sorted(team_df["team"].unique()), key="trend_team")
        with col2:
            metric = st.selectbox("Select Metric", ["gsax", "xg_diff", "win_pct"], key="trend_metric")
        
        team_games = game_df[(game_df["home_team"] == team) | (game_df["away_team"] == team)].copy()
        
        if team_games.empty:
            st.warning("No game data found for this team.")
        else:
            # Calculate metrics
            team_games["gsax"] = team_games.apply(
                lambda x: x["home_goalie_gsax"] if x["home_team"] == team else x.get("away_goalie_gsax", 0), axis=1
            )
            team_games["team_xg"] = team_games.apply(
                lambda x: x["home_xg"] if x["home_team"] == team else x.get("away_xg", 0), axis=1
            )
            team_games["opp_xg"] = team_games.apply(
                lambda x: x["away_xg"] if x["home_team"] == team else x.get("home_xg", 0), axis=1
            )
            team_games["xg_diff"] = team_games["team_xg"] - team_games["opp_xg"]
            
            # Simple Win/Loss for win_pct trend
            team_games["is_win"] = team_games.apply(
                lambda x: (x["home_goals"] > x["away_goals"] if x["home_team"] == team else x["away_goals"] > x["home_goals"]), axis=1
            )
            team_games["win_pct"] = team_games["is_win"].rolling(window=10, min_periods=1).mean()
            
            team_games["rolling_metric"] = team_games[metric].rolling(window=5, min_periods=1).mean()
            
            fig = px.line(team_games, x=team_games.index, y="rolling_metric",
                         title=f"{team} {metric.upper()} Trend (Rolling 5-Game Avg)",
                         labels={"index": "Game Index", "rolling_metric": metric.upper()},
                         markers=True)
            
            fig.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation Analysis")
        st.write("Understand how different team statistics correlate with each other.")
        
        # We can dynamically generate the heatmap or show a plotly version
        corr = team_df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

def render_bracket_generator(team_df, model):
    st.header("🏆 Championship Path")
    st.write("Simulate 1,000 tournament scenarios to find the most likely champion.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        num_teams = st.selectbox("Tournament Format", [4, 8, 16], index=1)
        sim_count = st.slider("Simulation Count", 100, 5000, 1000)
        
        if st.button("🚀 SIMULATE BRACKET"):
            with st.spinner("Crunching tournament numbers..."):
                top_teams = team_df.sort_values("elo", ascending=False).head(num_teams)["team"].tolist()
                tourney_wins = {t: 0 for t in top_teams}
                
                for _ in range(sim_count):
                    current_round = top_teams.copy()
                    while len(current_round) > 1:
                        next_round = []
                        for i in range(0, len(current_round), 2):
                            if i + 1 >= len(current_round):
                                next_round.append(current_round[i])
                                break
                            t1, t2 = current_round[i], current_round[i+1]
                            matchup_df = build_matchup_features(team_df, [(t1, t2)])
                            prob = model.predict_matchups(matchup_df).iloc[0]["home_win_probability"]
                            winner = t1 if np.random.random() < prob else t2
                            next_round.append(winner)
                        current_round = next_round
                    tourney_wins[current_round[0]] += 1
                
                results = pd.DataFrame([
                    {"Team": t, "Prob": w / sim_count} 
                    for t, w in tourney_wins.items()
                ]).sort_values("Prob", ascending=False)
                
                st.session_state['tourney_results'] = results

    with col2:
        if 'tourney_results' in st.session_state:
            results = st.session_state['tourney_results']
            st.subheader("Champion Probability")
            fig = px.bar(results, x="Prob", y="Team", orientation='h',
                        color="Prob", color_continuous_scale="Viridis",
                        labels={"Prob": "Championship Probability", "Team": "Team"})
            fig.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top contender card
            best = results.iloc[0]
            st.success(f"**Favorite**: {best['Team']} with {best['Prob']:.1%} chance of winning.")
        else:
            st.info("Click the button to run the tournament simulation.")

def render_scouting_reports(team_df, game_df, model):
    st.header("📄 Scouting Engine")
    st.write("Generate detailed PDF scouting reports with model-driven insights.")
    
    team = st.selectbox("Select Team to Profile", sorted(team_df["team"].unique()), key="scout_team")
    
    if team:
        stats = team_df[team_df["team"] == team].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Core Attributes")
            # Display some key stats in a clean way
            display_names = {
                "gsax": "Goaltending (GSAx)",
                "finishing": "Finishing Skill",
                "high_danger": "High Danger",
                "pp_strength": "Power Play",
                "pk_strength": "Penalty Kill",
                "shot_rate": "Shot Volume"
            }
            
            for key, label in display_names.items():
                if key in stats:
                    val = stats[key]
                    avg = team_df[key].mean()
                    delta = val - avg
                    st.metric(label, f"{val:.3f}", delta=f"{delta:.3f} (vs Avg)")

        with col2:
            st.markdown("### 🏹 Skill Profile")
            # Radar-ish chart or just a bar chart of relative strengths
            metrics = [m for m in display_names.keys() if m in team_df.columns]
            rel_stats = [(stats[m] - team_df[m].mean()) / team_df[m].std() for m in metrics]
            
            profile_df = pd.DataFrame({
                "Metric": [display_names[m] for m in metrics],
                "Z-Score": rel_stats
            })
            
            fig = px.bar(profile_df, x="Z-Score", y="Metric", orientation='h',
                        color="Z-Score", color_continuous_scale="RdBu",
                        range_x=[-3, 3], title="Relative Strength (Z-Score)")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

    if st.button("📥 GENERATE FULL PDF REPORT"):
        with st.spinner("Compiling tactical data..."):
            # (Rest of the PDF generation logic remains the same)
            stats = team_df[team_df["team"] == team].iloc[0]
            display_names = {
                "gsax": "GSAx (Goalie Saved Above Expected)",
                "finishing": "Finishing Skill",
                "high_danger": "High Danger Chance Generation",
                "pp_strength": "Power Play Strength",
                "pk_strength": "Penalty Kill Strength",
                "shot_rate": "Shot Volume (Corsi)"
            }
            available_metrics = [m for m in display_names.keys() if m in team_df.columns]
            mean_stats = team_df[available_metrics].mean()
            rel_stats_dict = {m: stats[m] - mean_stats[m] for m in available_metrics}
            best = max(rel_stats_dict, key=rel_stats_dict.get)
            worst = min(rel_stats_dict, key=rel_stats_dict.get)
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(200, 10, txt=f"Scouting Report: {team}", ln=True, align='C')
            pdf.ln(10)
            
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(200, 10, txt="Core Metrics:", ln=True)
            pdf.set_font("Helvetica", '', 11)
            for m in available_metrics:
                pdf.cell(200, 8, txt=f"- {display_names[m]}: {stats[m]:.3f} (League Avg: {mean_stats[m]:.3f})", ln=True)
            
            pdf.ln(5)
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(200, 10, txt="Model Insights:", ln=True)
            pdf.set_font("Helvetica", '', 11)
            pdf.cell(200, 8, txt=f"Best Feature: {display_names[best]}", ln=True)
            pdf.cell(200, 8, txt=f"Model Weakness: {display_names[worst]}", ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf.output(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("Click here to Download PDF", f, file_name=f"{team}_scouting_report.pdf")

def render_shap_dashboard(game_df, team_df, model):
    st.header("🔍 Model Logic (SHAP)")
    st.write("Understand why the model predicts specific outcomes.")
    
    col1, col2 = st.columns(2)
    teams = sorted(team_df["team"].unique())
    with col1:
        home_team = st.selectbox("Home Team", teams, index=0, key="home_shap")
    with col2:
        away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0, key="away_shap")

    matchup_df = build_matchup_features(team_df, [(home_team, away_team)])
    explainer_pipeline = model.logistic_pipeline
    cat_cols = ["home_goalie", "away_goalie", "home_off_line", "away_off_line", "home_def_pairing", "away_def_pairing"]
    features_full = matchup_df[config.feature_columns]
    numeric_cols = [c for c in config.feature_columns if c not in cat_cols and c in features_full.columns]
    X_numeric = features_full[numeric_cols]
    
    scaler = explainer_pipeline.named_steps["scaler"]
    logit_model = explainer_pipeline.named_steps["model"]
    X_scaled = scaler.transform(X_numeric)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
    
    explainer = shap.LinearExplainer(logit_model, X_scaled_df)
    shap_values = explainer(X_scaled_df)
    
    st.subheader(f"Impact Analysis: {home_team} vs {away_team}")
    
    with st.expander("How to read this chart", expanded=False):
        st.write("""
            - **Red bars**: Feature pushes prediction towards a **Home Win**.
            - **Blue bars**: Feature pushes prediction towards an **Away Win**.
            - The length of the bar indicates the **strength** of the influence.
        """)

    # Wrap SHAP plot to prevent cropping
    st.markdown('<div style="background: rgba(255,255,255,0.02); border-radius: 15px; padding: 20px; overflow-x: auto;">', unsafe_allow_html=True)
    st_shap(shap.plots.waterfall(shap_values[0]), height=450)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.info("The model uses weighted combinations of the features above to calculate final probabilities.")
