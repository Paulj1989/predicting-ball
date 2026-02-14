# src/features/odds_features.py

import numpy as np
import pandas as pd


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add betting odds derived features.

    If 'draw_odds' is present, log odds ratio is calculated using implied probabilities.
    If 'draw_odds' is missing, log odds ratio is calculated directly from decimal odds as log(away_odds / home_odds).
    """
    if "home_odds" in df.columns and "away_odds" in df.columns:
        df = df.copy()
        # convert odds to probabilities
        if "draw_odds" in df.columns:
            home_prob, draw_prob, away_prob = convert_odds_to_probabilities(
                df["home_odds"], df["draw_odds"], df["away_odds"], remove_margin=True
            )
            df["odds_home_prob"] = home_prob
            df["odds_draw_prob"] = draw_prob
            df["odds_away_prob"] = away_prob

            # log odds ratio: log(P_home / P_away)
            # positive = home favoured, negative = away favoured
            df["home_log_odds_ratio"] = calculate_log_odds_ratio(home_prob, away_prob)

            # win margin (difference in implied probabilities)
            df["odds_home_win_margin"] = home_prob - away_prob
        else:
            # fallback: calculate log odds ratio directly from decimal odds
            # log(p_home / p_away) = log((1/home_odds) / (1/away_odds)) = log(away_odds / home_odds)
            df["home_log_odds_ratio"] = np.log(df["away_odds"] / df["home_odds"])

        # odds ratio (multiplicative scale)
        df["odds_home_away_ratio"] = df["home_odds"] / df["away_odds"]

    return df


def convert_odds_to_probabilities(
    home_odds: pd.Series,
    draw_odds: pd.Series,
    away_odds: pd.Series,
    remove_margin: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
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
        log_odds_ratio = log(P_home / P_away)

    Positive values favour home team, negative favour away team.

    Parameters:
        home_prob (pd.Series): Implied probability for home team.
        away_prob (pd.Series): Implied probability for away team.
        epsilon (float): Small value added to probabilities to avoid division by zero or log(0).
    """
    # add epsilon to avoid log(0)
    home_prob_safe = home_prob + epsilon
    away_prob_safe = away_prob + epsilon

    log_odds_ratio = np.log(home_prob_safe / away_prob_safe)

    return log_odds_ratio
