# src/features/odds_imputation.py

"""
Impute missing Pinnacle odds using bet365 and market average odds.

Uses linear regression models fitted on complete data to predict missing
Pinnacle odds. Falls back to single-predictor models when only one source
is available.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LinearRegression


def impute_missing_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing Pinnacle odds using bet365 and market average odds.

    Creates home_odds, draw_odds, away_odds columns from pinnacle_* columns,
    filling missing values using linear models fitted on complete cases.
    """
    logger = logging.getLogger(__name__)
    df = df.copy()

    # check for required source columns
    pinnacle_cols = ["pinnacle_home_odds", "pinnacle_draw_odds", "pinnacle_away_odds"]
    bet365_cols = ["bet365_home_odds", "bet365_draw_odds", "bet365_away_odds"]
    market_avg_cols = ["market_avg_home_odds", "market_avg_draw_odds", "market_avg_away_odds"]

    has_pinnacle = all(col in df.columns for col in pinnacle_cols)
    has_bet365 = all(col in df.columns for col in bet365_cols)
    has_market_avg = all(col in df.columns for col in market_avg_cols)

    # graceful degradation: if source columns don't exist, can't impute
    if not has_pinnacle:
        logger.warning("Pinnacle odds columns not found, skipping imputation")
        return df

    if not has_bet365 and not has_market_avg:
        logger.warning("No predictor columns (bet365/market_avg) found, copying Pinnacle directly")
        df["home_odds"] = df["pinnacle_home_odds"]
        df["draw_odds"] = df["pinnacle_draw_odds"]
        df["away_odds"] = df["pinnacle_away_odds"]
        return df

    # fit imputation models
    models = fit_imputation_models(df)

    # apply imputation
    df = apply_imputation(df, models)

    return df


def fit_imputation_models(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Fit linear models for each odds type.

    Returns dict mapping odds type to fitted models:
    - 'full': model using both bet365 and market_avg
    - 'bet365_only': fallback model using only bet365
    - 'market_avg_only': fallback model using only market_avg
    """
    logger = logging.getLogger(__name__)

    odds_types = [
        ("home", "pinnacle_home_odds", "bet365_home_odds", "market_avg_home_odds"),
        ("draw", "pinnacle_draw_odds", "bet365_draw_odds", "market_avg_draw_odds"),
        ("away", "pinnacle_away_odds", "bet365_away_odds", "market_avg_away_odds"),
    ]

    models = {}

    for odds_name, pinnacle_col, bet365_col, market_avg_col in odds_types:
        models[odds_name] = {}

        # check which predictor columns are available
        has_bet365 = bet365_col in df.columns
        has_market_avg = market_avg_col in df.columns

        # get complete cases for training (where pinnacle exists)
        mask_pinnacle = df[pinnacle_col].notna()

        if has_bet365 and has_market_avg:
            # full model with both predictors
            mask_full = mask_pinnacle & df[bet365_col].notna() & df[market_avg_col].notna()
            if mask_full.sum() >= 10:
                X = df.loc[mask_full, [bet365_col, market_avg_col]].values
                y = df.loc[mask_full, pinnacle_col].values
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                models[odds_name]["full"] = model
                logger.debug(
                    f"{odds_name}_odds full model: R²={r2:.3f}, "
                    f"coef=[{model.coef_[0]:.3f}, {model.coef_[1]:.3f}], "
                    f"intercept={model.intercept_:.3f}"
                )

        if has_bet365:
            # bet365-only fallback
            mask_bet365 = mask_pinnacle & df[bet365_col].notna()
            if mask_bet365.sum() >= 10:
                X = df.loc[mask_bet365, [bet365_col]].values
                y = df.loc[mask_bet365, pinnacle_col].values
                model = LinearRegression()
                model.fit(X, y)
                models[odds_name]["bet365_only"] = model

        if has_market_avg:
            # market_avg-only fallback
            mask_market_avg = mask_pinnacle & df[market_avg_col].notna()
            if mask_market_avg.sum() >= 10:
                X = df.loc[mask_market_avg, [market_avg_col]].values
                y = df.loc[mask_market_avg, pinnacle_col].values
                model = LinearRegression()
                model.fit(X, y)
                models[odds_name]["market_avg_only"] = model

    # log training summary
    training_count = df["pinnacle_home_odds"].notna().sum()
    logger.info(f"Odds imputation: {training_count} matches with complete Pinnacle odds (training set)")

    return models


def apply_imputation(df: pd.DataFrame, models: Dict[str, Dict]) -> pd.DataFrame:
    """Apply fitted models to impute missing values"""
    logger = logging.getLogger(__name__)

    odds_types = [
        ("home", "pinnacle_home_odds", "bet365_home_odds", "market_avg_home_odds", "home_odds"),
        ("draw", "pinnacle_draw_odds", "bet365_draw_odds", "market_avg_draw_odds", "draw_odds"),
        ("away", "pinnacle_away_odds", "bet365_away_odds", "market_avg_away_odds", "away_odds"),
    ]

    # track imputation counts
    counts = {"full": 0, "bet365_only": 0, "market_avg_only": 0, "no_predictors": 0}

    for odds_name, pinnacle_col, bet365_col, market_avg_col, output_col in odds_types:
        # start with pinnacle values
        df[output_col] = df[pinnacle_col].copy()

        # find rows that need imputation
        mask_missing = df[pinnacle_col].isna()

        if not mask_missing.any():
            continue

        type_models = models.get(odds_name, {})

        # check which predictor columns are available
        has_bet365 = bet365_col in df.columns
        has_market_avg = market_avg_col in df.columns

        for idx in df[mask_missing].index:
            bet365_val = df.loc[idx, bet365_col] if has_bet365 else np.nan
            market_avg_val = df.loc[idx, market_avg_col] if has_market_avg else np.nan

            bet365_available = pd.notna(bet365_val)
            market_avg_available = pd.notna(market_avg_val)

            imputed = False

            # try full model first
            if bet365_available and market_avg_available and "full" in type_models:
                X = np.array([[bet365_val, market_avg_val]])
                df.loc[idx, output_col] = type_models["full"].predict(X)[0]
                if odds_name == "home":
                    counts["full"] += 1
                imputed = True

            # fallback to market_avg only
            elif market_avg_available and "market_avg_only" in type_models:
                X = np.array([[market_avg_val]])
                df.loc[idx, output_col] = type_models["market_avg_only"].predict(X)[0]
                if odds_name == "home":
                    counts["market_avg_only"] += 1
                imputed = True

            # fallback to bet365 only
            elif bet365_available and "bet365_only" in type_models:
                X = np.array([[bet365_val]])
                df.loc[idx, output_col] = type_models["bet365_only"].predict(X)[0]
                if odds_name == "home":
                    counts["bet365_only"] += 1
                imputed = True

            # no predictors available
            if not imputed and odds_name == "home":
                counts["no_predictors"] += 1

    # log imputation summary
    logger.info(f"Odds imputation: {counts['full']} matches imputed using full model (bet365 + market_avg)")
    logger.info(f"Odds imputation: {counts['market_avg_only']} matches imputed using market_avg only")
    logger.info(f"Odds imputation: {counts['bet365_only']} matches imputed using bet365 only")
    if counts["no_predictors"] > 0:
        logger.info(f"Odds imputation: {counts['no_predictors']} matches with no predictors (left as NaN)")

    return df
