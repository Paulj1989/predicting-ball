# src/features/feature_builder.py

import pandas as pd
import logging
from typing import Optional

from .xg_features import add_rolling_npxgd, add_venue_npxgd
from .squad_features import add_squad_value_features
from .odds_features import add_odds_features
from .weighted_performance import calculate_weighted_performance


def prepare_model_features(
    df: pd.DataFrame,
    windows: list = [5, 10],
    include_squad_values: bool = True,
    include_odds: bool = True,
    include_weighted_performance: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build all features for model training/prediction.

    Main pipeline that orchestrates all feature engineering steps:
    1. Squad value features (ratios, logs, percentiles)
    2. Betting odds features (ratios, margins)
    3. Rolling npxGD statistics (5 and 10 game windows)
    4. Venue-specific npxGD (home at home, away away)
    5. Red card indicators
    6. Weighted performance composite (for model fitting)
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)

    df = df.copy()

    # step 1: squad value features
    if include_squad_values:
        if verbose:
            logger.info("1. Processing squad values")

        if "home_value" in df.columns and "away_value" in df.columns:
            df = add_squad_value_features(df)
        else:
            if verbose:
                logger.warning("   No squad value data found, skipping")
    else:
        if verbose:
            logger.info("1. Skipping squad values (not requested)")

    # step 2: odds features
    if include_odds:
        if verbose:
            logger.info("2. Processing betting odds")

        if "home_odds" in df.columns and "away_odds" in df.columns:
            df = add_odds_features(df)
        else:
            if verbose:
                logger.warning("   No odds data found, skipping")
    else:
        if verbose:
            logger.info("2. Skipping betting odds (not requested)")

    # step 3: rolling npxGD features
    if verbose:
        logger.info(f"3. Calculating rolling npxGD (windows: {windows})")

    df = add_rolling_npxgd(df, windows=windows)

    # step 4: venue-specific npxGD
    if verbose:
        logger.info("4. Calculating venue-specific npxGD")

    df = add_venue_npxgd(df)

    # step 5: card features (inline - simple enough)
    if verbose:
        logger.info("5. Adding red card indicators")

    mask_played = df["is_played"]
    if "home_cards_red" in df.columns:
        df.loc[mask_played, "home_had_red"] = (
            df.loc[mask_played, "home_cards_red"] > 0
        ).astype(int)
        df.loc[mask_played, "away_had_red"] = (
            df.loc[mask_played, "away_cards_red"] > 0
        ).astype(int)

    # step 6: weighted performance composite
    if include_weighted_performance:
        if verbose:
            logger.info("6. Calculating weighted performance composite")

        df = calculate_weighted_performance(df)

    if verbose:
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total matches: {len(df)}")
        logger.info(f"Total features: {len(df.columns)}")

    return df


def validate_features(
    df: pd.DataFrame, required_columns: Optional[list] = None
) -> dict:
    """Validate all model features"""
    if required_columns is None:
        required_columns = [
            "home_team",
            "away_team",
            "home_npxgd_w5",
            "home_npxgd_w10",
            "away_npxgd_w5",
            "away_npxgd_w10",
            "home_venue_npxgd_per_game",
            "away_venue_npxgd_per_game",
            "home_goals_weighted",
            "away_goals_weighted",
            "value_ratio",
            "odds_home_away_ratio",
        ]

    # check for missing columns
    missing = [col for col in required_columns if col not in df.columns]

    # check for nulls in required columns
    present_cols = [col for col in required_columns if col in df.columns]
    columns_with_nulls = {
        col: df[col].isna().sum() for col in present_cols if df[col].isna().any()
    }

    # summary
    all_present = len(missing) == 0

    if all_present and len(columns_with_nulls) == 0:
        summary = "✓ All required features present and complete"
    elif all_present:
        summary = f"⚠ All columns present but {len(columns_with_nulls)} have nulls"
    else:
        summary = f"✗ Missing {len(missing)} required columns"

    return {
        "all_present": all_present,
        "missing_columns": missing,
        "columns_with_nulls": columns_with_nulls,
        "summary": summary,
    }
