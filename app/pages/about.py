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

    Predicting Ball uses a bivariate Poisson model with a Dixon-Coles correction, combined with Monte Carlo simulation. This generates a probabilistic forecast for individual matches and the entire season.

    #### Model Components

    Several integrated components make up the Predicting Ball model:

    **Team Strength Estimation**: The model uses a two-stage process to separate long-term team ability and local match-specific factors.

    - **Stage 1 (Team Strengths)** - Estimates each team's attack and defense ratings, along with home advantage, and uses a Dixon-Coles correction to reduce systematic undervaluing of low-scoring matches

    - **Stage 2 (Match Features)** - Fixes team strengths and estimates coefficients for match-specific features, including a rolling 5-game form residual and an odds blend weight that combines model probabilities with bookmaker-implied probabilities.

    **Informed Priors**: The model uses informed priors to handle data limitations. A blend of Transfermarkt's squad market values and Club Elo's team ratings serves as a ratings prior and a home advantage prior is estimated from historical Bundesliga data, accounting for season-to-season variance.

    **Calibrated Uncertainty**: The model uses several mechanisms for quantifying prediction uncertainty:

    - Team ratings drawn from the maximum likelihood estimation posterior (during simulations).
    - Hot simulations update team ratings dynamically using a natural gradient step.
    - Temperature scaling adds post-hoc probability calibration.

    **Monte Carlo Simulation**: Season projections are based on 10,000 simulations of all remaining fixtures. Final league positions account for the uncertainty in both team strengths and match outcomes, producing (hopefully) realistic probability distributions for all league outcomes.

    #### Performance

    The model has been validated on previous Bundesliga seasons using walk-forward backtesting and weekly monitoring:

    - **Ranked Probability Score (RPS)** - Measures accuracy of ordered outcome predictions (home/draw/away).
    - **Brier Score** - Quantifies the accuracy of probabilistic predictions, similar to mean-squared error for probabilities.
    - **Log Loss** - Penalises incorrect predictions based on confidence, where a higher certainty in a wrong outcome results in a steeper penalty.

    #### Data Sources

    - **Opta** - Match results, expected goals, shots, possession, and advanced statistics.
    - **Transfermarkt** - Squad market values and other squad features.
    - **Club Elo** - Club Elo ratings at the beginning of each season.
    - **Football-Data.co.uk** - Historical betting odds from multiple bookmakers.
    - **The Odds API** - Current betting odds for upcoming fixtures.

    ---

    **Last Updated**: Generated from the latest match data and model training

    **Model Version**: Production v3.0
    """)
