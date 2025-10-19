# src/validation/backtest.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


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

    # prepare test data
    test_data = all_data[all_data["season_end_year"] == test_season].copy()

    if len(test_data) == 0:
        print(f"✗ No data for season {test_season}")
        return None

    if verbose:
        print(f"Test set: {len(test_data)} matches")

    # evaluate model predictions
    params = fitted_model["params"]

    # determine if using calibration
    use_calibration = False
    temperature = None

    if calibrators is not None:
        # check for temperature scaling
        if isinstance(calibrators, dict):
            temperature = calibrators.get("temperature")
            if temperature is None:
                if verbose:
                    print("⚠ No temperature calibrator found, using uncalibrated")
            else:
                use_calibration = True
        elif isinstance(calibrators, (int, float)):
            temperature = calibrators
            use_calibration = True

    # get base predictions and actuals
    metrics_base, predictions_base, actuals = evaluate_model_comprehensive(
        params, test_data
    )

    # apply calibration if available
    if use_calibration and temperature is not None:
        from ..models.calibration import apply_temperature_scaling
        from ..evaluation.metrics import (
            calculate_rps,
            calculate_brier_score,
            calculate_log_loss,
            calculate_accuracy,
        )

        predictions = apply_temperature_scaling(predictions_base, temperature)

        # recalculate metrics with calibrated predictions
        metrics = {
            "rps": calculate_rps(predictions, actuals),
            "brier_score": calculate_brier_score(predictions, actuals),
            "log_loss": calculate_log_loss(predictions, actuals),
            "accuracy": calculate_accuracy(predictions, actuals),
        }

        if verbose:
            print(f"\nCalibrated Results (T={temperature:.3f}):")
            print(f"  RPS: {metrics['rps']:.4f}")
            print(f"  Brier: {metrics['brier_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.1%}")
    else:
        # use uncalibrated predictions
        metrics = metrics_base
        predictions = predictions_base

        if verbose:
            calibration_status = " (uncalibrated)" if calibrators is None else ""
            print(f"\nResults{calibration_status}:")
            print(f"  RPS: {metrics['rps']:.4f}")
            print(f"  Brier: {metrics['brier_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.1%}")

    # evaluate baseline (implied odds)
    if verbose:
        print("\nEvaluating baseline (implied odds)...")

    baseline_metrics = evaluate_implied_odds_baseline(test_data, verbose=verbose)

    # validate baseline metrics
    baseline_valid = _validate_baseline_metrics(baseline_metrics, test_season, verbose)

    # compare to baseline if valid
    if baseline_valid and verbose:
        print("\nVs. Implied Odds:")

        rps_diff = baseline_metrics["rps"] - metrics["rps"]
        rps_improvement = (rps_diff / baseline_metrics["rps"]) * 100

        brier_diff = baseline_metrics["brier_score"] - metrics["brier_score"]
        brier_improvement = (brier_diff / baseline_metrics["brier_score"]) * 100

        print(f"  RPS: {rps_improvement:+.1f}%")
        print(f"  Brier: {brier_improvement:+.1f}%")

    # package results
    return {
        "season": test_season,
        "n_matches": len(test_data),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics if baseline_valid else None,
        "predictions": predictions,
        "actuals": actuals,
        "calibrated": use_calibration,
        "temperature": temperature if use_calibration else None,
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
        print("MULTI-SEASON BACKTESTING")
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

        # calculate average metrics
        avg_rps = np.mean([r["metrics"]["rps"] for r in results])
        std_rps = np.std([r["metrics"]["rps"] for r in results])
        avg_brier = np.mean([r["metrics"]["brier_score"] for r in results])
        std_brier = np.std([r["metrics"]["brier_score"] for r in results])
        avg_acc = np.mean([r["metrics"]["accuracy"] for r in results])

        print(f"Average RPS: {avg_rps:.4f} ± {std_rps:.4f}")
        print(f"Average Brier: {avg_brier:.4f} ± {std_brier:.4f}")
        print(f"Average Accuracy: {avg_acc:.1%}")

        # compare to baseline (only for seasons with valid baselines)
        results_with_baseline = [r for r in results if r.get("baseline_metrics")]

        if len(results_with_baseline) > 0:
            avg_baseline_rps = np.mean(
                [r["baseline_metrics"]["rps"] for r in results_with_baseline]
            )
            avg_model_rps = np.mean(
                [r["metrics"]["rps"] for r in results_with_baseline]
            )
            improvement = (avg_baseline_rps - avg_model_rps) / avg_baseline_rps * 100

            n_with_baseline = len(results_with_baseline)
            n_total = len(results)

            if n_with_baseline < n_total:
                print(
                    f"\nVs. Implied Odds: {improvement:+.1f}% RPS improvement "
                    f"({n_with_baseline}/{n_total} seasons)"
                )
            else:
                print(f"\nVs. Implied Odds: {improvement:+.1f}% RPS improvement")
        else:
            print("\n⚠ No baseline comparisons available (insufficient odds data)")

    return results


def run_rolling_validation(
    fitted_model: Dict[str, Any],
    all_data: pd.DataFrame,
    n_seasons: int = 3,
    calibrators: Optional[Any] = None,
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
        print(f"Validating on {n_seasons} most recent seasons: {test_seasons}")

    return backtest_multiple_seasons(
        fitted_model, all_data, test_seasons, calibrators=calibrators, verbose=verbose
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


def _validate_baseline_metrics(
    baseline_metrics: Optional[Dict[str, float]], season: int, verbose: bool = True
) -> bool:
    """Validate baseline metrics for quality and completeness"""
    # check if baseline metrics exist
    if baseline_metrics is None:
        if verbose:
            print(f"  ⚠ No baseline metrics available for season {season}")
        return False

    # check if it's a dictionary
    if not isinstance(baseline_metrics, dict):
        if verbose:
            print(f"  ⚠ Invalid baseline metrics type: {type(baseline_metrics)}")
        return False

    # check for required keys
    required_keys = ["rps", "brier_score"]
    missing_keys = [key for key in required_keys if key not in baseline_metrics]

    if missing_keys:
        if verbose:
            print(f"  ⚠ Baseline metrics missing keys: {missing_keys}")
        return False

    # check for NaN or invalid values
    for key in required_keys:
        value = baseline_metrics[key]

        # check if value is numeric
        if not isinstance(value, (int, float, np.number)):
            if verbose:
                print(f"  ⚠ Baseline {key} is not numeric: {type(value)}")
            return False

        # check for NaN
        if np.isnan(value):
            if verbose:
                print(f"  ⚠ Baseline {key} is NaN")
            return False

        # check for infinite values
        if np.isinf(value):
            if verbose:
                print(f"  ⚠ Baseline {key} is infinite")
            return False

        # check for reasonable range (RPS and Brier should be between 0 and 1)
        if value < 0 or value > 1:
            if verbose:
                print(f"  ⚠ Baseline {key} out of valid range [0,1]: {value:.4f}")
            return False

    # all checks passed
    return True


def diagnose_baseline_calculation(
    test_data: pd.DataFrame, season: int, verbose: bool = True
) -> Dict[str, Any]:
    """Diagnose issues with baseline calculation for a specific season"""
    from ..evaluation.baselines import evaluate_implied_odds_baseline

    if verbose:
        print("\n" + "=" * 60)
        print(f"BASELINE DIAGNOSTIC - SEASON {season}")
        print("=" * 60)

    diagnostics = {
        "season": season,
        "n_matches": len(test_data),
        "has_odds_columns": False,
        "odds_coverage": {},
        "baseline_metrics": None,
        "issues": [],
    }

    # check for odds columns
    required_odds_cols = ["home_odds", "draw_odds", "away_odds"]
    has_all_odds = all(col in test_data.columns for col in required_odds_cols)
    diagnostics["has_odds_columns"] = has_all_odds

    if verbose:
        print(f"\n1. Odds Columns Present: {has_all_odds}")

    if not has_all_odds:
        missing = [col for col in required_odds_cols if col not in test_data.columns]
        diagnostics["issues"].append(f"Missing odds columns: {missing}")
        if verbose:
            print(f"   ✗ Missing: {missing}")
        return diagnostics

    # check odds coverage
    if verbose:
        print("\n2. Odds Coverage:")

    for col in required_odds_cols:
        n_valid = test_data[col].notna().sum()
        coverage = n_valid / len(test_data) if len(test_data) > 0 else 0
        diagnostics["odds_coverage"][col] = {
            "n_valid": int(n_valid),
            "coverage_pct": coverage * 100,
        }

        if verbose:
            print(f"   {col}: {n_valid}/{len(test_data)} ({coverage * 100:.1f}%)")

        if coverage < 0.5:
            diagnostics["issues"].append(f"Low {col} coverage: {coverage * 100:.1f}%")

    # check complete odds matches
    complete_odds_mask = (
        test_data["home_odds"].notna()
        & test_data["draw_odds"].notna()
        & test_data["away_odds"].notna()
    )
    n_complete = complete_odds_mask.sum()
    complete_coverage = n_complete / len(test_data) if len(test_data) > 0 else 0

    diagnostics["n_complete_odds"] = int(n_complete)
    diagnostics["complete_coverage_pct"] = complete_coverage * 100

    if verbose:
        print(
            f"\n3. Complete Odds: {n_complete}/{len(test_data)} ({complete_coverage * 100:.1f}%)"
        )

    if complete_coverage < 0.8:
        diagnostics["issues"].append(
            f"Insufficient complete odds coverage: {complete_coverage * 100:.1f}%"
        )

    # try calculating baseline
    if verbose:
        print("\n4. Calculating Baseline Metrics...")

    try:
        baseline_metrics = evaluate_implied_odds_baseline(test_data, verbose=False)
        diagnostics["baseline_metrics"] = baseline_metrics

        if baseline_metrics is None:
            diagnostics["issues"].append("evaluate_implied_odds_baseline returned None")
            if verbose:
                print("   ✗ Baseline calculation returned None")
        else:
            # validate metrics
            is_valid = _validate_baseline_metrics(
                baseline_metrics, season, verbose=False
            )
            diagnostics["baseline_valid"] = is_valid

            if is_valid:
                if verbose:
                    print(f"   ✓ Baseline RPS: {baseline_metrics['rps']:.4f}")
                    print(f"   ✓ Baseline Brier: {baseline_metrics['brier_score']:.4f}")
            else:
                diagnostics["issues"].append("Baseline metrics failed validation")
                if verbose:
                    print("   ✗ Baseline metrics invalid")
                    print(f"      RPS: {baseline_metrics.get('rps', 'MISSING')}")
                    print(
                        f"      Brier: {baseline_metrics.get('brier_score', 'MISSING')}"
                    )

    except Exception as e:
        diagnostics["issues"].append(f"Exception during baseline calculation: {str(e)}")
        if verbose:
            print(f"   ✗ Exception: {e}")
            import traceback

            traceback.print_exc()

    # summary
    if verbose:
        print("\n5. Summary:")
        if len(diagnostics["issues"]) == 0:
            print("   ✓ No issues detected")
        else:
            print(f"   ✗ {len(diagnostics['issues'])} issue(s) found:")
            for issue in diagnostics["issues"]:
                print(f"      - {issue}")
        print("=" * 60)

    return diagnostics
