# src/features/squad_features.py

import pandas as pd
import numpy as np


def add_squad_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add squad value derived features.

    Creates:
    - Percentile features (relative to season average)
    - Log-transformed features
    - Ratio and difference features
    """
    df = df.copy()

    # fill missing values
    for col in ["home_value", "away_value"]:
        if col in df.columns:
            # fill with season median first
            df[col] = df.groupby("season_end_year")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # then overall median
            df[col] = df[col].fillna(df[col].median())
            # fallback to 50M euros
            df[col] = df[col].fillna(50_000_000)

    # calculate season average for normalisation
    season_avg_value = (
        df.groupby("season_end_year")[["home_value", "away_value"]]
        .transform(lambda x: x.mean())
        .mean(axis=1)
    )

    # percentile features (relative to season average)
    df["home_value_pct"] = (
        (df["home_value"] - season_avg_value) / season_avg_value
    ) * 100
    df["away_value_pct"] = (
        (df["away_value"] - season_avg_value) / season_avg_value
    ) * 100
    df["value_pct_diff"] = df["home_value_pct"] - df["away_value_pct"]

    # log-transformed features
    df["home_value_log"] = np.log(df["home_value"].clip(lower=1))
    df["away_value_log"] = np.log(df["away_value"].clip(lower=1))
    df["value_diff_log"] = df["home_value_log"] - df["away_value_log"]

    # ratio and difference features
    df["value_ratio"] = df["home_value"] / df["away_value"].clip(lower=1)
    df["value_diff"] = df["home_value"] - df["away_value"]

    return df
