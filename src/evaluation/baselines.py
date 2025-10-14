# src/evaluation/baselines.py

import numpy as np
import pandas as pd
from typing import Dict, Optional

from .metrics import (
    calculate_rps,
    calculate_brier_score,
    calculate_log_loss,
    calculate_accuracy,
)


def evaluate_implied_odds_baseline(test_data: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate implied odds as baseline.

    Uses betting market probabilities (with margin removed) as predictions.
    """
    if "odds_home_prob" not in test_data.columns:
        print("Warning: No odds data available for baseline")
        return None

    # extract implied probabilities
    predictions = test_data[
        ["odds_home_prob", "odds_draw_prob", "odds_away_prob"]
    ].values

    # get actual outcomes
    home_goals = test_data["home_goals"].astype(int).values
    away_goals = test_data["away_goals"].astype(int).values

    actuals = np.where(
        home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2)
    )

    # calculate metrics
    metrics = {
        "rps": calculate_rps(predictions, actuals),
        "brier_score": calculate_brier_score(predictions, actuals),
        "log_loss": calculate_log_loss(predictions, actuals),
        "accuracy": calculate_accuracy(predictions, actuals),
    }

    return metrics


def evaluate_odds_only_model(
    test_data: pd.DataFrame, params: Dict[str, any]
) -> Optional[Dict[str, float]]:
    """
    Evaluate model that uses only betting odds (no team strengths).

    This baseline uses odds as the only feature, testing whether
    team strength parameters add value beyond market information.
    """
    if params is None or not params.get("success", False):
        return None

    # import here to avoid circular dependency
    from .metrics import evaluate_model_comprehensive

    metrics, _, _ = evaluate_model_comprehensive(params, test_data)

    return metrics


def evaluate_historical_average_baseline(test_data: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate historical average baseline.

    Uses overall historical frequencies as predictions:
    - Home win: ~45%
    - Draw: ~25%
    - Away win: ~30%

    This is the simplest possible baseline.
    """
    # historical frequencies for buli
    home_prob = 0.45
    draw_prob = 0.25
    away_prob = 0.30

    # create predictions (same for all matches)
    n_matches = len(test_data)
    predictions = np.tile([home_prob, draw_prob, away_prob], (n_matches, 1))

    # get actual outcomes
    home_goals = test_data["home_goals"].astype(int).values
    away_goals = test_data["away_goals"].astype(int).values

    actuals = np.where(
        home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2)
    )

    # calculate metrics
    metrics = {
        "rps": calculate_rps(predictions, actuals),
        "brier_score": calculate_brier_score(predictions, actuals),
        "log_loss": calculate_log_loss(predictions, actuals),
        "accuracy": calculate_accuracy(predictions, actuals),
    }

    return metrics


def create_baseline_comparison_table(
    model_metrics: Dict[str, float], test_data: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """Create comprehensive baseline comparison table"""
    results = {
        "Your Model": model_metrics,
        "Implied Odds": evaluate_implied_odds_baseline(test_data),
        "Historical Average": evaluate_historical_average_baseline(test_data),
    }

    # remove none results
    results = {k: v for k, v in results.items() if v is not None}

    # create dataframe
    df = pd.DataFrame(results).T

    # calculate improvement vs implied odds
    if "Implied Odds" in results:
        for metric in ["rps", "brier_score", "log_loss"]:
            if metric in df.columns:
                baseline_value = results["Implied Odds"][metric]
                df[f"{metric}_improvement"] = (
                    (baseline_value - df[metric]) / baseline_value * 100
                )

    if verbose:
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON")
        print("=" * 70)
        print(df.to_string())
        print("\nNote: For RPS, Brier, and Log Loss, lower is better")
        print("      Improvement % shows how much better than baseline")

    return df
