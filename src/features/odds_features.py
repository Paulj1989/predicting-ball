# src/features/odds_features.py

import pandas as pd
import numpy as np


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add betting odds derived features"""
    df = df.copy()

    if "home_odds" in df.columns and "away_odds" in df.columns:
        # odds ratio (how much more likely is home vs away)
        df["odds_home_away_ratio"] = df["home_odds"] / df["away_odds"]

        # win margin (difference in implied probabilities)
        if "odds_home_prob" in df.columns and "odds_away_prob" in df.columns:
            df["odds_home_win_margin"] = df["odds_home_prob"] - df["odds_away_prob"]

    return df


def convert_odds_to_probabilities(
    home_odds: pd.Series,
    draw_odds: pd.Series,
    away_odds: pd.Series,
    remove_margin: bool = True,
) -> tuple:
    """Convert decimal odds to implied probabilities"""
    # convert odds to raw probabilities
    raw_home_prob = 1 / home_odds
    raw_draw_prob = 1 / draw_odds
    raw_away_prob = 1 / away_odds

    if remove_margin:
        # normalise to remove bookmaker margin (overround)
        total_prob = raw_home_prob + raw_draw_prob + raw_away_prob

        home_prob = raw_home_prob / total_prob
        draw_prob = raw_draw_prob / total_prob
        away_prob = raw_away_prob / total_prob
    else:
        home_prob = raw_home_prob
        draw_prob = raw_draw_prob
        away_prob = raw_away_prob

    return home_prob, draw_prob, away_prob


def calculate_log_odds_ratio(
    home_prob: pd.Series, away_prob: pd.Series, epsilon: float = 1e-10
) -> pd.Series:
    """
    Calculate log odds ratio for home vs away.

    Log odds ratio is a symmetric measure of relative strength:
        log_odds = log(P_home / P_away)

    Positive values favour home team, negative favour away team.
    """
    # add epsilon to avoid log(0)
    home_prob_safe = home_prob + epsilon
    away_prob_safe = away_prob + epsilon

    log_odds = np.log(home_prob_safe / away_prob_safe)

    return log_odds
