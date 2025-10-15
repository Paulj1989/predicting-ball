# src/features/weighted_performance.py

import pandas as pd
import numpy as np


def calculate_weighted_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite performance metric from match data.

    Uses npxG/npG (non-penalty versions) for stable parameter estimation.
    Formula:
    - 80% of (80% npxG + 20% npG)
    - 10% normalised touches in attacking penalty area
    - 10% normalised shots

    This composite is used as the target variable for Poisson model fitting,
    providing a more stable signal than raw goals alone.
    """
    df = df.copy()

    # calculate non-penalty goals (npG)
    df["home_npg"] = df["home_goals"].fillna(0) - df["home_pens_made"].fillna(0)
    df["away_npg"] = df["away_goals"].fillna(0) - df["away_pens_made"].fillna(0)

    # npxG already in data
    df["home_npxg"] = df["home_npxg"].fillna(0)
    df["away_npxg"] = df["away_npxg"].fillna(0)

    # normalise shots and touches to same scale as goals
    df["home_shots_norm"] = df["home_shots"].fillna(0) / np.mean(
        df["home_shots"].fillna(0).clip(lower=1)
    )
    df["away_shots_norm"] = df["away_shots"].fillna(0) / np.mean(
        df["away_shots"].fillna(0).clip(lower=1)
    )

    df["home_touches_norm"] = df["home_touches_att_pen_area"].fillna(0) / np.mean(
        df["home_touches_att_pen_area"].fillna(0).clip(lower=1)
    )
    df["away_touches_norm"] = df["away_touches_att_pen_area"].fillna(0) / np.mean(
        df["away_touches_att_pen_area"].fillna(0).clip(lower=1)
    )

    # create npxG/npG composite
    # 80% of (80% npxG + 20% npG) + 10% touches + 10% shots
    df["home_weighted_performance"] = (
        0.8 * (0.8 * df["home_npxg"] + 0.2 * df["home_npg"])
        + 0.1 * df["home_touches_norm"]
        + 0.1 * df["home_shots_norm"]
    )

    df["away_weighted_performance"] = (
        0.8 * (0.8 * df["away_npxg"] + 0.2 * df["away_npg"])
        + 0.1 * df["away_touches_norm"]
        + 0.1 * df["away_shots_norm"]
    )

    # round for Poisson model (which expects integer-like values)
    df["home_goals_weighted"] = df["home_weighted_performance"].round(2)
    df["away_goals_weighted"] = df["away_weighted_performance"].round(2)

    # fallback for missing data: use actual goals
    mask_missing = df["home_weighted_performance"].isna()
    df.loc[mask_missing, "home_goals_weighted"] = df.loc[
        mask_missing, "home_goals"
    ].fillna(0)
    df.loc[mask_missing, "away_goals_weighted"] = df.loc[
        mask_missing, "away_goals"
    ].fillna(0)

    return df


def validate_weighted_performance(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Validate weighted performance calculation"""
    metrics = {
        "mean_home": df["home_weighted_performance"].mean(),
        "mean_away": df["away_weighted_performance"].mean(),
        "correlation_home": df["home_weighted_performance"].corr(df["home_goals"]),
        "correlation_away": df["away_weighted_performance"].corr(df["away_goals"]),
    }

    if verbose:
        print("\nWeighted Performance Validation:")
        print(f"  Average home weighted performance: {metrics['mean_home']:.2f}")
        print(f"  Average away weighted performance: {metrics['mean_away']:.2f}")
        print(
            f"  Correlation with actual goals: {metrics['correlation_home']:.2f} (home), "
            f"{metrics['correlation_away']:.2f} (away)"
        )

    return metrics
