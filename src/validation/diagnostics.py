# src/validation/diagnostics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any


# define colour palette
COLORS = {
    "primary": "#026E99",  # blue
    "secondary": "#D93649",  # red
    "accent": "#FFA600",  # orange
}

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def analyse_prediction_errors(
    predictions: np.ndarray,
    actuals: np.ndarray,
    test_data: pd.DataFrame,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Analyse prediction errors by outcome type"""
    # get predicted outcomes
    predicted_outcomes = np.argmax(predictions, axis=1)

    # calculate confusion matrix
    n_outcomes = 3
    confusion = np.zeros((n_outcomes, n_outcomes), dtype=int)

    for actual, predicted in zip(actuals, predicted_outcomes):
        confusion[actual, predicted] += 1

    # calculate accuracy by outcome
    outcome_names = ["Home Win", "Draw", "Away Win"]
    accuracies = {}

    for i, name in enumerate(outcome_names):
        n_actual = (actuals == i).sum()
        n_correct = confusion[i, i]
        accuracy = n_correct / n_actual if n_actual > 0 else 0
        accuracies[name] = accuracy

    # calculate average confidence by outcome
    avg_confidence = {}
    for i, name in enumerate(outcome_names):
        mask = actuals == i
        if mask.sum() > 0:
            avg_confidence[name] = predictions[mask, i].mean()
        else:
            avg_confidence[name] = 0

    if verbose:
        print("\n" + "=" * 60)
        print("PREDICTION ERROR ANALYSIS")
        print("=" * 60)

        print("\nConfusion Matrix:")
        print("                Predicted")
        print("           Home  Draw  Away")
        for i, name in enumerate(outcome_names):
            row_str = f"{name:10s}"
            for j in range(n_outcomes):
                row_str += f"{confusion[i, j]:6d}"
            print(row_str)

        print("\nAccuracy by Outcome:")
        for name, acc in accuracies.items():
            print(f"  {name}: {acc:.1%}")

        print("\nAverage Confidence (for correct outcome):")
        for name, conf in avg_confidence.items():
            print(f"  {name}: {conf:.1%}")

    return {
        "confusion_matrix": confusion,
        "accuracies": accuracies,
        "avg_confidence": avg_confidence,
    }


def analyse_performance_by_team(
    predictions: np.ndarray,
    actuals: np.ndarray,
    test_data: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyse model performance by team"""
    from ..evaluation.metrics import calculate_rps, calculate_brier_score

    # get all teams
    all_teams = sorted(
        set(test_data["home_team"].unique()) | set(test_data["away_team"].unique())
    )

    team_stats = []

    for team in all_teams:
        # get matches involving this team
        team_mask = (test_data["home_team"] == team) | (test_data["away_team"] == team)

        if team_mask.sum() == 0:
            continue

        team_preds = predictions[team_mask]
        team_actuals = actuals[team_mask]

        # calculate metrics
        rps = calculate_rps(team_preds, team_actuals)
        brier = calculate_brier_score(team_preds, team_actuals)

        team_stats.append(
            {
                "team": team,
                "n_matches": team_mask.sum(),
                "rps": rps,
                "brier_score": brier,
            }
        )

    df = pd.DataFrame(team_stats).sort_values("rps")

    if verbose:
        print("\n" + "=" * 60)
        print("PERFORMANCE BY TEAM")
        print("=" * 60)
        print("\nTop 5 Best Predicted Teams:")
        print(
            df.head()[["team", "n_matches", "rps", "brier_score"]].to_string(
                index=False
            )
        )

        print("\nTop 5 Worst Predicted Teams:")
        print(
            df.tail()[["team", "n_matches", "rps", "brier_score"]].to_string(
                index=False
            )
        )

    return df


def analyse_performance_by_odds(
    predictions: np.ndarray,
    actuals: np.ndarray,
    test_data: pd.DataFrame,
    n_bins: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Analyse performance by odds-implied probability.

    Tests whether model performs differently on favourites vs. underdogs.
    """
    from ..evaluation.metrics import (
        calculate_rps,
        calculate_brier_score,
        calculate_accuracy,
    )

    if "odds_home_prob" not in test_data.columns:
        print("✗ No odds data available for analysis")
        return None

    # get favourite probability (max of home/draw/away)
    favorite_probs = test_data[
        ["odds_home_prob", "odds_draw_prob", "odds_away_prob"]
    ].max(axis=1)

    # create bins
    bins = pd.qcut(favorite_probs, q=n_bins, duplicates="drop")

    results = []

    for bin_label in bins.cat.categories:
        bin_mask = bins == bin_label

        if bin_mask.sum() == 0:
            continue

        bin_preds = predictions[bin_mask]
        bin_actuals = actuals[bin_mask]

        results.append(
            {
                "odds_bin": str(bin_label),
                "n_matches": bin_mask.sum(),
                "rps": calculate_rps(bin_preds, bin_actuals),
                "brier_score": calculate_brier_score(bin_preds, bin_actuals),
                "accuracy": calculate_accuracy(bin_preds, bin_actuals),
            }
        )

    df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 60)
        print("PERFORMANCE BY ODDS (Favourite Probability)")
        print("=" * 60)
        print(df.to_string(index=False))

        print("\nInterpretation:")
        print("  - Lower odds = stronger favourite")
        print("  - Model should perform better on favourites (more predictable)")

    return df


def create_validation_report(
    validation_results: List[Dict[str, Any]], save_path: Optional[str] = None
) -> pd.DataFrame:
    """Create comprehensive validation report"""
    rows = []

    for result in validation_results:
        row = {
            "season": result["season"],
            "n_matches": result["n_matches"],
            "rps": result["metrics"]["rps"],
            "brier_score": result["metrics"]["brier_score"],
            "log_loss": result["metrics"]["log_loss"],
            "accuracy": result["metrics"]["accuracy"],
        }

        if result.get("baseline_metrics"):
            row["baseline_rps"] = result["baseline_metrics"]["rps"]
            row["rps_improvement"] = (
                (result["baseline_metrics"]["rps"] - result["metrics"]["rps"])
                / result["baseline_metrics"]["rps"]
                * 100
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    # add summary row
    summary_row = {
        "season": "AVERAGE",
        "n_matches": df["n_matches"].sum(),
        "rps": df["rps"].mean(),
        "brier_score": df["brier_score"].mean(),
        "log_loss": df["log_loss"].mean(),
        "accuracy": df["accuracy"].mean(),
    }

    if "baseline_rps" in df.columns:
        summary_row["baseline_rps"] = df["baseline_rps"].mean()
        summary_row["rps_improvement"] = df["rps_improvement"].mean()

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    print(df.to_string(index=False))

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n✓ Report saved: {save_path}")

    return df
