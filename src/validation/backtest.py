# src/validation/backtest.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from .splits import create_train_test_split, validate_split_quality


def backtest_single_season(
    fitted_model: Dict[str, Any],
    all_data: pd.DataFrame,
    test_season: int,
    calibrators: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Backtest fitted model on a single holdout season"""
    # import here to avoid circular dependency
    from ..evaluation.metrics import evaluate_model_comprehensive
    from ..evaluation.baselines import evaluate_implied_odds_baseline

    if verbose:
        print("\n" + "=" * 60)
        print(f"BACKTESTING SEASON {test_season}")
        print("=" * 60)

    # split data
    test_data = all_data[all_data["season_end_year"] == test_season].copy()

    if len(test_data) == 0:
        print(f"✗ No data for season {test_season}")
        return None

    if verbose:
        print(f"Test set: {len(test_data)} matches")

    # evaluate model
    params = fitted_model["params"]

    if calibrators is not None:
        # check for temperature scaling
        temperature = None
        if isinstance(calibrators, dict):
            temperature = calibrators.get("temperature")
            if temperature is None:
                if verbose:
                    print("⚠ No temperature calibrator found, using uncalibrated")
        elif isinstance(calibrators, (int, float)):
            temperature = calibrators

        if temperature is not None:
            # use temperature-scaled predictions
            from ..models.calibration import apply_temperature_scaling

            metrics_uncal, predictions_uncal, actuals = evaluate_model_comprehensive(
                params, test_data
            )

            predictions_cal = apply_temperature_scaling(predictions_uncal, temperature)

            # recalculate metrics with calibrated predictions
            from ..evaluation.metrics import (
                calculate_rps,
                calculate_brier_score,
                calculate_log_loss,
                calculate_accuracy,
            )

            metrics_cal = {
                "rps": calculate_rps(predictions_cal, actuals),
                "brier_score": calculate_brier_score(predictions_cal, actuals),
                "log_loss": calculate_log_loss(predictions_cal, actuals),
                "accuracy": calculate_accuracy(predictions_cal, actuals),
            }

            predictions = predictions_cal
            metrics = metrics_cal

            if verbose:
                print(f"\nCalibrated Results (T={temperature:.3f}):")
                print(f"  RPS: {metrics['rps']:.4f}")
                print(f"  Brier: {metrics['brier_score']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.1%}")
        else:
            # no calibrators, use uncalibrated
            metrics, predictions, actuals = evaluate_model_comprehensive(
                params, test_data
            )

            if verbose:
                print("\nResults (uncalibrated):")
                print(f"  RPS: {metrics['rps']:.4f}")
                print(f"  Brier: {metrics['brier_score']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.1%}")
    else:
        # use uncalibrated predictions
        metrics, predictions, actuals = evaluate_model_comprehensive(params, test_data)

        if verbose:
            print("\nResults:")
            print(f"  RPS: {metrics['rps']:.4f}")
            print(f"  Brier: {metrics['brier_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.1%}")

    # compare to baseline
    baseline_metrics = evaluate_implied_odds_baseline(test_data)

    if baseline_metrics and verbose:
        print("\nVs. Implied Odds:")
        rps_improvement = (
            (baseline_metrics["rps"] - metrics["rps"]) / baseline_metrics["rps"] * 100
        )
        brier_improvement = (
            (baseline_metrics["brier_score"] - metrics["brier_score"])
            / baseline_metrics["brier_score"]
            * 100
        )
        print(f"  RPS: {rps_improvement:+.1f}%")
        print(f"  Brier: {brier_improvement:+.1f}%")

    return {
        "season": test_season,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "predictions": predictions,
        "actuals": actuals,
        "n_matches": len(test_data),
    }


def backtest_multiple_seasons(
    fitted_model: Dict[str, Any],
    all_data: pd.DataFrame,
    test_seasons: List[int],
    calibrators: Optional[Any] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Backtest fitted model on multiple seasons"""
    if verbose:
        print("\n" + "=" * 60)
        print(f"MULTI-SEASON BACKTESTING")
        print("=" * 60)
        print(f"Testing {len(test_seasons)} seasons: {test_seasons}")

    results = []

    for season in test_seasons:
        result = backtest_single_season(
            fitted_model, all_data, season, calibrators=calibrators, verbose=verbose
        )

        if result:
            results.append(result)

    # summary statistics
    if verbose and len(results) > 0:
        print("\n" + "=" * 60)
        print("SUMMARY ACROSS SEASONS")
        print("=" * 60)

        avg_rps = np.mean([r["metrics"]["rps"] for r in results])
        std_rps = np.std([r["metrics"]["rps"] for r in results])
        avg_brier = np.mean([r["metrics"]["brier_score"] for r in results])
        std_brier = np.std([r["metrics"]["brier_score"] for r in results])
        avg_acc = np.mean([r["metrics"]["accuracy"] for r in results])

        print(f"Average RPS: {avg_rps:.4f} ± {std_rps:.4f}")
        print(f"Average Brier: {avg_brier:.4f} ± {std_brier:.4f}")
        print(f"Average Accuracy: {avg_acc:.1%}")

        # compare to baseline
        if all(r.get("baseline_metrics") for r in results):
            avg_baseline_rps = np.mean([r["baseline_metrics"]["rps"] for r in results])
            improvement = (avg_baseline_rps - avg_rps) / avg_baseline_rps * 100
            print(f"\nVs. Implied Odds: {improvement:+.1f}% RPS improvement")

    return results


def run_rolling_validation(
    fitted_model: Dict[str, Any],
    all_data: pd.DataFrame,
    n_seasons: int = 3,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run rolling validation on most recent N seasons"""
    # get most recent N seasons
    seasons = sorted(all_data["season_end_year"].unique())
    test_seasons = seasons[-n_seasons:]

    if verbose:
        print("\n" + "=" * 60)
        print("ROLLING VALIDATION")
        print("=" * 60)
        print(f"Validating on {n_seasons} most recent seasons")

    return backtest_multiple_seasons(
        fitted_model, all_data, test_seasons, verbose=verbose
    )


def cross_validate_hyperparameters(
    all_data: pd.DataFrame,
    hyperparam_grid: Dict[str, List[float]],
    n_splits: int = 5,
    verbose: bool = True,
) -> Dict[str, float]:
    """Cross-validate hyperparameters using time-series splits"""
    from itertools import product
    from .splits import TimeSeriesSplit
    from ..models.poisson import fit_poisson_model
    from ..evaluation.metrics import evaluate_model_comprehensive

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER CROSS-VALIDATION")
        print("=" * 60)

    # generate all combinations
    param_names = list(hyperparam_grid.keys())
    param_values = list(hyperparam_grid.values())
    combinations = list(product(*param_values))

    if verbose:
        print(f"Testing {len(combinations)} parameter combinations")
        print(f"Using {n_splits}-fold time-series CV")

    splitter = TimeSeriesSplit(n_splits=n_splits)

    best_score = float("inf")
    best_params = None

    for combo in combinations:
        hyperparams = dict(zip(param_names, combo))

        if verbose:
            print(f"\nTesting: {hyperparams}")

        cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(all_data)):
            train_fold = all_data.iloc[train_idx]
            test_fold = all_data.iloc[test_idx]

            # fit model
            params = fit_poisson_model(train_fold, hyperparams, verbose=False)

            if params and params["success"]:
                # evaluate
                metrics, _, _ = evaluate_model_comprehensive(params, test_fold)
                cv_scores.append(metrics["rps"])

        if len(cv_scores) > 0:
            avg_score = np.mean(cv_scores)

            if verbose:
                print(f"  CV RPS: {avg_score:.4f} ± {np.std(cv_scores):.4f}")

            if avg_score < best_score:
                best_score = avg_score
                best_params = hyperparams

    if verbose:
        print("\n" + "=" * 60)
        print("BEST PARAMETERS")
        print("=" * 60)
        print(f"CV RPS: {best_score:.4f}")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

    return best_params
