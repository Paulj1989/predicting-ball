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

    This Bundesliga prediction model uses a Dixon-Coles corrected Poisson regression combined with Monte Carlo simulation to generate probabilistic forecasts for the 2025/26 season.

    #### Model Components

    The prediction system consists of several integrated components:

    **Two-Stage Team Strength Estimation**: The model uses a two-stage fitting process to cleanly separate stable team ability from transient match-specific factors:

    - **Stage 1 (Baseline Strengths)**: Estimates each team's attacking and defensive capabilities along with home advantage, and uses a Dixon-Coles correction to reduce systematic undervaluing of low-scoring matches.
    - **Stage 2 (Feature Coefficients)**: Fixes baseline team strengths and estimates coefficients for match-specific features including betting odds ratios and rolling 5-game npxGD.

    **Informed Priors**: The model uses informed priors to handle data limitations. Transfermarkt's squad market values provide a reasonable proxy for team quality, translated into expected attack and defense ratings (returning teams using a blend of squad values and the previous season's ratings and promoted teams using entirely squad values). A home advantage prior is estimated from historical Bundesliga data with season-to-season variance.

    **Calibrated Uncertainty Quantification**: The model quantifies prediction uncertainty through several mechanisms:

    - Parametric bootstrap with residual resampling (500 iterations) quantifies parameter uncertainty.
    - Dispersion calibration adjusts goal distributions to match empirical variance in actual outcomes.
    - Prediction intervals empirically calibrated to achieve target coverage (68%, 80%, 95%).
    - Temperature scaling provides post-hoc probability calibration when needed.

    **Monte Carlo Simulation**: Season projections aggregate 10,000 simulations of remaining fixtures. Each simulation samples team parameters from the bootstrap distribution and generates match outcomes using Dixon-Coles corrected probabilities. Final league positions account for the full uncertainty in both team strengths and match outcomes, producing realistic probability distributions for the Meisterschale, European qualification, and relegation.

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

    The codebase uses Python with scientific computing libraries including NumPy, pandas, SciPy, and Optuna for optimization. The Dixon-Coles correction is applied consistently during both parameter estimation and prediction to ensure accurate draw probabilities. We use maximum likelihood estimation with time-weighted observations, giving recent matches more influence on parameter estimates while still incorporating historical information. The simulation engine uses NumPy's optimized random number generation for efficient Monte Carlo sampling.

    ---

    **Last Updated**: Generated from the latest match data and model training

    **Model Version**: Production v2.0
    """)
