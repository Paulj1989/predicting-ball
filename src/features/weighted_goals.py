# src/features/weighted_goals.py

import pandas as pd


def calculate_weighted_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Create weighted goals (70% npxG + 30% npG) composite metric"""
    df = df.copy()

    # calculate non-penalty goals (npG)
    df["home_npg"] = df["home_goals"].fillna(0) - df["home_pens_made"].fillna(0)
    df["away_npg"] = df["away_goals"].fillna(0) - df["away_pens_made"].fillna(0)

    # npxG already in data
    df["home_npxg"] = df["home_npxg"].fillna(0)
    df["away_npxg"] = df["away_npxg"].fillna(0)

    # create npxG/npG composite
    df["home_goals_weighted"] = 0.7 * df["home_npxg"] + 0.3 * df["home_npg"]

    df["away_goals_weighted"] = 0.7 * df["away_npxg"] + 0.3 * df["away_npg"]

    # fallback for missing data: use actual goals
    mask_missing = df["home_goals_weighted"].isna()
    df.loc[mask_missing, "home_goals_weighted"] = df.loc[
        mask_missing, "home_goals"
    ].fillna(0)
    df.loc[mask_missing, "away_goals_weighted"] = df.loc[
        mask_missing, "away_goals"
    ].fillna(0)

    return df


def validate_weighted_goals(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Validate weighted goals calculation"""
    metrics = {
        "mean_home": df["home_goals_weighted"].mean(),
        "mean_away": df["away_goals_weighted"].mean(),
        "correlation_home": df["home_goals_weighted"].corr(df["home_goals"]),
        "correlation_away": df["away_goals_weighted"].corr(df["away_goals"]),
    }

    if verbose:
        print("\nWeighted Goals Validation:")
        print(f"  Average home weighted goals: {metrics['mean_home']:.2f}")
        print(f"  Average away weighted goals: {metrics['mean_away']:.2f}")
        print(
            f"  Correlation with actual goals: {metrics['correlation_home']:.2f} (home), "
            f"{metrics['correlation_away']:.2f} (away)"
        )

    return metrics
