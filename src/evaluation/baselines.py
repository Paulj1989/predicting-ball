# src/evaluation/baselines.py

import numpy as np
import pandas as pd
from typing import Dict, Optional

from .metrics import (
    calculate_rps,
    calculate_brier_score,
    calculate_log_loss,
)


def evaluate_implied_odds_baseline(
    data: pd.DataFrame, verbose: bool = False
) -> Optional[Dict[str, float]]:
    """
    Evaluate implied odds as a baseline model.

    Converts bookmaker odds to probabilities (removing margin).
    """
    # check for required columns
    required_cols = ["home_odds", "draw_odds", "away_odds", "result"]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        if verbose:
            print(f"  ⚠ Missing columns for baseline: {missing_cols}")
        return None

    # filter to matches with complete odds
    valid_mask = (
        data["home_odds"].notna()
        & data["draw_odds"].notna()
        & data["away_odds"].notna()
        & data["result"].notna()
    )

    n_valid = valid_mask.sum()
    n_total = len(data)

    if verbose:
        print(
            f"  Matches with complete odds: {n_valid}/{n_total} ({n_valid / n_total * 100:.1f}%)"
        )

    # require at least 80% coverage
    if n_valid < 0.8 * n_total:
        if verbose:
            print(f"  ⚠ Insufficient odds coverage: {n_valid / n_total * 100:.1f}%")
        return None

    if n_valid == 0:
        if verbose:
            print("  ⚠ No matches with complete odds")
        return None

    # get valid data - reset indices to avoid misalignment
    valid_data = data[valid_mask].copy().reset_index(drop=True)

    # calculate implied probabilities
    home_implied = 1 / valid_data["home_odds"]
    draw_implied = 1 / valid_data["draw_odds"]
    away_implied = 1 / valid_data["away_odds"]

    # check for any invalid values
    if (
        home_implied.isna().any()
        or draw_implied.isna().any()
        or away_implied.isna().any()
    ):
        if verbose:
            print("  ⚠ NaN values in implied probabilities")
        return None

    if (
        (home_implied <= 0).any()
        or (draw_implied <= 0).any()
        or (away_implied <= 0).any()
    ):
        if verbose:
            print("  ⚠ Non-positive implied probabilities")
        return None

    # normalise to remove bookmaker margin
    total_implied = home_implied + draw_implied + away_implied

    if (total_implied == 0).any():
        if verbose:
            print("  ⚠ Zero total implied probability")
        return None

    baseline_probs = pd.DataFrame(
        {
            "home_win": home_implied / total_implied,
            "draw": draw_implied / total_implied,
            "away_win": away_implied / total_implied,
        }
    )

    # verify probabilities sum to 1
    prob_sums = baseline_probs.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-6):
        if verbose:
            print(f"  ⚠ Probabilities don't sum to 1: {prob_sums.describe()}")
        return None

    # get actual outcomes - reset index to match baseline_probs
    actual_outcomes = valid_data["result"].reset_index(drop=True)

    # verify alignment
    if len(baseline_probs) != len(actual_outcomes):
        if verbose:
            print(
                f"  ⚠ Length mismatch: {len(baseline_probs)} probs vs {len(actual_outcomes)} outcomes"
            )
        return None

    # calculate metrics
    try:
        rps = calculate_rps(baseline_probs, actual_outcomes)
        brier = calculate_brier_score(baseline_probs, actual_outcomes)
        log_loss_val = calculate_log_loss(baseline_probs, actual_outcomes)

        # validate metrics
        if np.isnan(rps) or np.isnan(brier) or np.isnan(log_loss_val):
            if verbose:
                print("  ⚠ Baseline metrics contain NaN")
                print(f"     RPS: {rps}, Brier: {brier}, LogLoss: {log_loss_val}")
            return None

        if verbose:
            print(f"  ✓ Baseline RPS: {rps:.4f}")
            print(f"  ✓ Baseline Brier: {brier:.4f}")

        return {
            "rps": float(rps),
            "brier_score": float(brier),
            "log_loss": float(log_loss_val),
            "n_matches": int(n_valid),
            "coverage": float(n_valid / n_total),
        }

    except Exception as e:
        if verbose:
            print(f"  ✗ Error calculating baseline metrics: {e}")
            import traceback

            traceback.print_exc()
        return None


def evaluate_odds_only_model(
    test_data: pd.DataFrame, params: Dict[str, any]
) -> Optional[Dict[str, float]]:
    """Evaluate model that uses only betting odds (no team strengths)"""
    if params is None or not params.get("success", False):
        return None

    # import here to avoid circular dependency
    from .metrics import evaluate_model_comprehensive

    metrics, _, _ = evaluate_model_comprehensive(params, test_data)

    return metrics




def create_baseline_comparison_table(
    model_metrics: Dict[str, float], test_data: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """Create comprehensive baseline comparison table"""
    results = {
        "Fitted Model": model_metrics,
        "Implied Odds": evaluate_implied_odds_baseline(test_data),
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
