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

    - **Stage 1 (Team Strengths)** - Estimates each team's attack and defense ratings, along with home advantage, and uses a Dixon-Coles correction to reduce systematic undervaluing of low-scoring matches.
    - **Stage 2 (Match Features)** - Fixes team strengths and estimates coefficients for match-specific features, including betting odds ratios and rolling 5-game npxGD.

    **Informed Priors**: The model uses informed priors to handle data limitations. Transfermarkt's squad market values serve as a proxy for team quality, translated to ratings priors (using a blend of squad values and the previous season's ratings for returning teams, and only squad values for promoted teams). A home advantage prior is estimated from historical Bundesliga data, accounting for season-to-season variance.

    **Calibrated Uncertainty**: The model uses several mechanisms for quantifying prediction uncertainty:

    - Residual bootstrapping (500 iterations) quantifies parameter uncertainty.
    - Dispersion calibration adjusts the distribution of goals to match the variance in outcomes.
    - Prediction intervals are empirically calibrated to achieve target coverage.
    - Temperature scaling adds post-hoc probability calibration.

    **Monte Carlo Simulation**: Season projections are based on 10,000 simulations of all remaining fixtures. Final league positions account for the uncertainty in both team strengths and match outcomes, producing (hopefully) realistic probability distributions for all league outcomes.

    #### Performance

    The model has been validated on previous Bundesliga seasons using time-series cross-validation:

    - **Ranked Probability Score (RPS)** - Measures accuracy of ordered outcome predictions (home/draw/away).
    - **Brier Score** - Quantifies the accuracy of probabilistic predictions, similar to mean-squared error for probabilities.
    - **Accuracy** - Percentage of correct outcome predictions.

    #### Data Sources

    - **FBRef** - Match results, expected goals, shots, possession, and advanced statistics.
    - **Transfermarkt** - Squad market values and other squad features.
    - **Football-Data.co.uk** - Historical betting odds from multiple bookmakers.
    - **The Odds API** - Current betting odds for upcoming fixtures.

    ---

    **Last Updated**: Generated from the latest match data and model training

    **Model Version**: Production v2.0
    """)
