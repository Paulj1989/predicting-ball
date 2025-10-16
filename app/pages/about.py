# app/pages/about.py

import streamlit as st


def render():
    """Display information about the model"""
    st.markdown(
        '<h2 style="font-size: 1.8rem; text-align: center;">About the Model</h2>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    ### Methodology
    This Bundesliga prediction model uses a hybrid Poisson regression approach combined with Monte Carlo simulation to generate probabilistic forecasts for the 2025/26 season.

    #### Model Components
    The prediction system consists of several integrated components that work together to produce reliable forecasts:

    **Team Strength Estimation**: We fit a Poisson regression model that estimates each team's attacking and defensive capabilities based on their weighted performance metrics. The model incorporates non-penalty expected goals (npxG), non-penalty goals, shots, and touches in the attacking penalty area. This weighted approach provides a more stable signal for team quality that is less susceptible to short-term variance.

    **Feature Engineering**: The model uses multiple data sources to capture team quality. Squad market values from Transfermarkt serve as a proxy for overall team quality and are particularly useful for newly promoted teams with limited match history. Betting odds from multiple bookmakers provide market consensus on match outcomes.

    **Calibrated Uncertainty**: We employ parametric bootstrapping with residual resampling to quantify parameter uncertainty. The model fits on 500 bootstrap samples to generate distributions for all team strength parameters. A dispersion factor calibrates the goal-scoring distributions to match empirical variance in actual match outcomes. This ensures prediction intervals have proper coverage rather than being overconfident.

    **Monte Carlo Simulation**: Season projections come from 100,000 simulations of all remaining fixtures. Each simulation samples team parameters from the bootstrap distribution and generates match outcomes using calibrated Poisson distributions. Final league positions account for the full uncertainty in both team strengths and match outcomes, producing realistic probability distributions for the Meisterschale, European qualification, and relegation.

    #### Performance

    The model has been validated on historical Bundesliga seasons using proper time-series cross-validation:

    - **Ranked Probability Score (RPS)**: Measures accuracy of ordered outcome predictions (home/draw/away)
    - **Brier Score**: Quantifies calibration of probabilistic forecasts
    - **Accuracy**: Percentage of correct outcome predictions

    #### Data Sources

    - **FBRef**: Match results, expected goals, shots, possession, and advanced statistics
    - **Transfermarkt**: Squad market values and team rosters
    - **Football-Data.co.uk**: Historical betting odds from multiple bookmakers
    - **The Odds API**: Current betting odds for upcoming fixtures

    #### Technical Details
    The codebase uses Python with scientific computing libraries including NumPy, pandas, and SciPy. The Poisson regression incorporates home advantage, betting market information, and non-penalty expected goals. Time-weighted maximum likelihood estimation gives recent matches more influence on parameter estimates. Temperature scaling provides post-hoc calibration of predicted probabilities through a single-parameter transformation that adjusts confidence levels while preserving outcome rankings. The simulation engine uses NumPy's optimised random number generation for efficient Monte Carlo sampling.

    ---

    **Last Updated**: Generated from the latest match data and model training

    **Model Version**: Production v1.0
    """)
