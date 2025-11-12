# src/features/schedule_features.py

import pandas as pd
import numpy as np


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule-based features from rest days and consecutive away games.

    Creates:
    - Rest day differential (home advantage from more rest)
    - Rest advantage indicators (significant differences)
    """
    df = df.copy()

    # fill missing rest_days with median (first match of season or missing data)
    for col in ["home_rest_days", "away_rest_days"]:
        if col in df.columns:
            # use median rest days as fallback (typically ~7 days between matches)
            median_rest = df[col].median()
            if pd.isna(median_rest):
                median_rest = 7.0  # fallback to weekly schedule
            df[col] = df[col].fillna(median_rest)

    # fill consecutive away games with 0 (start of season or home games)
    for col in ["home_consecutive_away_games", "away_consecutive_away_games"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # rest day features
    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        # difference: positive means home team had more rest
        df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

        # rest advantage indicators (3+ days difference is significant)
        df["home_rest_advantage"] = (df["rest_days_diff"] >= 3).astype(int)
        df["away_rest_advantage"] = (df["rest_days_diff"] <= -3).astype(int)

        # short rest indicators (< 4 days between matches)
        df["home_short_rest"] = (df["home_rest_days"] < 4).astype(int)
        df["away_short_rest"] = (df["away_rest_days"] < 4).astype(int)

        # long rest indicators (> 10 days, possible rust)
        df["home_long_rest"] = (df["home_rest_days"] > 10).astype(int)
        df["away_long_rest"] = (df["away_rest_days"] > 10).astype(int)

    return df
