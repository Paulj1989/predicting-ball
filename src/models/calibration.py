# src/models/calibration.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from typing import Dict, Callable, Tuple, Any, Union
import warnings


def fit_temperature_scaler(
    predictions: Union[pd.DataFrame, np.ndarray],
    actuals: Union[pd.Series, np.ndarray],
    verbose: bool = True,
) -> float:
    """
    Fit temperature scaling parameter via NLL minimisation.

    Temperature scaling adjusts prediction confidence without changing
    the relative ordering of outcomes.
    """
    # convert inputs to numpy arrays
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns {required_cols}")
        predictions = predictions[required_cols].values
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    if isinstance(actuals, pd.Series):
        actuals = actuals.values
    elif not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # handle string vs integer outcomes
    if actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        try:
            actuals = np.array([outcome_map[a] for a in actuals])
        except KeyError as e:
            raise ValueError(f"Unknown outcome: {e}")
    else:
        actuals = actuals.astype(int)
        if not all(a in [0, 1, 2] for a in actuals):
            raise ValueError("Integer outcomes must be 0 (H), 1 (D), or 2 (A)")

    # clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

    # convert to logits
    logits = np.log(predictions)

    def negative_log_likelihood(temperature: float) -> float:
        """Compute NLL with given temperature"""
        # scale logits by temperature
        scaled_logits = logits / temperature

        # apply softmax to get calibrated probabilities
        # subtract max for numerical stability
        scaled_logits_stable = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits_stable)
        calibrated_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # compute negative log-likelihood
        nll = -np.log(calibrated_probs[np.arange(len(actuals)), actuals] + 1e-10).sum()

        return nll

    # optimise temperature (search range: 0.1 to 5.0)
    result = minimize_scalar(
        negative_log_likelihood, bounds=(0.1, 5.0), method="bounded"
    )

    optimal_temperature = result.x

    if verbose:
        uncalibrated_nll = negative_log_likelihood(1.0)
        calibrated_nll = result.fun

        print("   Temperature optimisation:")
        print(f"     Optimal T: {optimal_temperature:.3f}")
        print(f"     Uncalibrated NLL: {uncalibrated_nll:.2f}")
        print(f"     Calibrated NLL: {calibrated_nll:.2f}")
        print(f"     Improvement: {uncalibrated_nll - calibrated_nll:.2f}")

        if optimal_temperature > 1.5:
            print("     Model is overconfident (T > 1.5)")
        elif optimal_temperature < 0.8:
            print("     Model is underconfident (T < 0.8)")
        else:
            print("     Model confidence is reasonable")

    return optimal_temperature


def apply_temperature_scaling(
    predictions: Union[pd.DataFrame, np.ndarray], temperature: float
) -> np.ndarray:
    """
    Apply temperature scaling to predictions.

    Transforms predicted probabilities by:
    1. Converting to logits
    2. Scaling by temperature
    3. Applying softmax to renormalise
    """
    # convert to numpy array
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns {required_cols}")
        predictions = predictions[required_cols].values
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

    # convert to logits
    logits = np.log(predictions)

    # scale by temperature
    scaled_logits = logits / temperature

    # apply softmax (with numerical stability)
    scaled_logits_stable = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits_stable)
    calibrated_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return calibrated_probs


def calibrate_dispersion_for_coverage(
    base_params: Dict[str, Any],
    calibration_data: pd.DataFrame,
    target_coverage: float = 0.80,
    tolerance: float = 0.02,
    max_iterations: int = 20,
    verbose: bool = True,
) -> float:
    """
    Iteratively adjust dispersion factor to achieve target coverage.

    Uses binary search to find the dispersion factor that produces
    prediction intervals with the desired empirical coverage on a
    calibration set.
    """
    # import here to avoid circular dependency
    from .poisson import calculate_lambdas
    from ..simulation.sampling import sample_goals_calibrated

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EMPIRICAL CALIBRATION FOR {target_coverage:.0%} COVERAGE")
        print(f"{'=' * 60}")
        print(f"Calibration set: {len(calibration_data)} matches")

    def test_coverage_with_dispersion(dispersion: float) -> float:
        """Test empirical coverage with given dispersion factor"""
        n_matches = len(calibration_data)
        covered = 0

        for i in range(n_matches):
            match = calibration_data.iloc[i]
            lambda_h, lambda_a = calculate_lambdas(
                calibration_data.iloc[[i]], base_params
            )

            # simulate with this dispersion
            n_sims = 5000
            simulated_h = sample_goals_calibrated(lambda_h[0], dispersion, size=n_sims)
            simulated_a = sample_goals_calibrated(lambda_a[0], dispersion, size=n_sims)

            # get prediction intervals
            alpha = (1 - target_coverage) / 2
            h_lower, h_upper = np.percentile(
                simulated_h, [alpha * 100, (1 - alpha) * 100]
            )
            a_lower, a_upper = np.percentile(
                simulated_a, [alpha * 100, (1 - alpha) * 100]
            )

            # check actual outcome
            actual_h = int(match["home_goals"])
            actual_a = int(match["away_goals"])

            if (h_lower <= actual_h <= h_upper) and (a_lower <= actual_a <= a_upper):
                covered += 1

        return covered / n_matches

    # binary search for optimal dispersion
    dispersion_low = 0.5
    dispersion_high = 3.0
    best_dispersion = 1.0

    for iteration in range(max_iterations):
        dispersion_mid = (dispersion_low + dispersion_high) / 2
        coverage = test_coverage_with_dispersion(dispersion_mid)

        if verbose:
            print(
                f"  Iteration {iteration + 1}: dispersion={dispersion_mid:.3f}, coverage={coverage:.1%}"
            )

        # check if we've achieved target
        if abs(coverage - target_coverage) <= tolerance:
            best_dispersion = dispersion_mid
            if verbose:
                print(f"\n Converged to dispersion={best_dispersion:.3f}")
                print(f"  Empirical coverage: {coverage:.1%}")
            break

        # adjust search bounds
        if coverage < target_coverage:
            # need wider intervals -> increase dispersion
            dispersion_low = dispersion_mid
        else:
            # intervals too wide -> decrease dispersion
            dispersion_high = dispersion_mid

        best_dispersion = dispersion_mid
    else:
        # didn't converge, use best found
        coverage = test_coverage_with_dispersion(best_dispersion)
        if verbose:
            print(f"\n  Did not fully converge after {max_iterations} iterations")
            print(f"  Best dispersion: {best_dispersion:.3f}")
            print(f"  Final coverage: {coverage:.1%}")

    return best_dispersion


def calibrate_model_comprehensively(
    base_params: Dict[str, Any], calibration_data: pd.DataFrame, verbose: bool = True
) -> Tuple[Dict[float, float], None]:
    """Calibrate dispersion factors for multiple coverage levels"""
    if verbose:
        print(f"\n{'=' * 60}")
        print("COMPREHENSIVE CALIBRATION")
        print(f"{'=' * 60}")

    calibrated_dispersions = {}

    for target in [0.68, 0.80, 0.95]:
        dispersion = calibrate_dispersion_for_coverage(
            base_params, calibration_data, target_coverage=target, verbose=verbose
        )
        calibrated_dispersions[target] = dispersion

    if verbose:
        print(f"\n{'=' * 60}")
        print("CALIBRATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"68% intervals: dispersion = {calibrated_dispersions[0.68]:.3f}")
        print(f"80% intervals: dispersion = {calibrated_dispersions[0.80]:.3f}")
        print(f"95% intervals: dispersion = {calibrated_dispersions[0.95]:.3f}")

    # return none for function - it will be recreated when needed
    return calibrated_dispersions, None


def create_dispersion_interpolator(calibrated_dispersions: Dict[float, float]):
    """
    Create interpolation function from calibrated dispersions.

    Call this function to recreate the interpolator after loading calibrators.
    """

    def get_dispersion_for_confidence(confidence: float) -> float:
        """Get interpolated dispersion for any confidence level"""
        if confidence <= 0.68:
            return calibrated_dispersions[0.68]
        elif confidence >= 0.95:
            return calibrated_dispersions[0.95]
        elif confidence <= 0.80:
            # interpolate between 68% and 80%
            t = (confidence - 0.68) / (0.80 - 0.68)
            return (1 - t) * calibrated_dispersions[0.68] + t * calibrated_dispersions[
                0.80
            ]
        else:
            # interpolate between 80% and 95%
            t = (confidence - 0.80) / (0.95 - 0.80)
            return (1 - t) * calibrated_dispersions[0.80] + t * calibrated_dispersions[
                0.95
            ]

    return get_dispersion_for_confidence


def fit_outcome_specific_temperatures(
    predictions: Union[pd.DataFrame, np.ndarray],
    actuals: Union[pd.Series, np.ndarray],
    min_samples_per_outcome: int = 20,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Fit separate temperature parameters for each outcome (home/draw/away).

    This allows draws to be calibrated independently, which is useful when
    the base model systematically under/overestimates draw probabilities.
    """

    # ========================================================================
    # INPUT VALIDATION AND CONVERSION
    # ========================================================================

    # convert predictions to numpy array
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns {required_cols}")
        predictions = predictions[required_cols].values
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    if predictions.shape[1] != 3:
        raise ValueError(f"Predictions must have 3 columns, got {predictions.shape[1]}")

    # convert actuals to indices
    if isinstance(actuals, pd.Series):
        actuals = actuals.values
    elif not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # handle string vs integer outcomes
    if actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        try:
            actuals = np.array([outcome_map[a] for a in actuals])
        except KeyError as e:
            raise ValueError(f"Unknown outcome: {e}")
    else:
        actuals = actuals.astype(int)
        if not all(a in [0, 1, 2] for a in actuals):
            raise ValueError("Integer outcomes must be 0 (H), 1 (D), or 2 (A)")

    # check sample sizes per outcome
    outcome_counts = np.bincount(actuals, minlength=3)

    if verbose:
        print("\n" + "=" * 70)
        print("OUTCOME-SPECIFIC TEMPERATURE SCALING")
        print("=" * 70)
        print(f"\nCalibration set: {len(actuals)} matches")
        print("Outcome distribution:")
        print(
            f"  Home wins: {outcome_counts[0]} ({outcome_counts[0] / len(actuals):.1%})"
        )
        print(
            f"  Draws:     {outcome_counts[1]} ({outcome_counts[1] / len(actuals):.1%})"
        )
        print(
            f"  Away wins: {outcome_counts[2]} ({outcome_counts[2] / len(actuals):.1%})"
        )

    # warn if insufficient samples
    for i, name in enumerate(["home wins", "draws", "away wins"]):
        if outcome_counts[i] < min_samples_per_outcome:
            warnings.warn(
                f"Only {outcome_counts[i]} {name} in calibration set "
                f"(recommended: {min_samples_per_outcome}+). "
                f"Temperature estimate may be unreliable.",
                UserWarning,
            )

    # ========================================================================
    # OPTIMISATION
    # ========================================================================

    # clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    logits = np.log(predictions)

    def negative_log_likelihood(temps: np.ndarray) -> float:
        """NLL with outcome-specific temperatures"""
        T_home, T_draw, T_away = temps

        # apply different temperature to each outcome
        scaled_logits = logits.copy()
        scaled_logits[:, 0] /= T_home
        scaled_logits[:, 1] /= T_draw
        scaled_logits[:, 2] /= T_away

        # softmax with numerical stability
        scaled_logits_stable = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits_stable)
        calibrated_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # nll
        nll = -np.log(calibrated_probs[np.arange(len(actuals)), actuals] + 1e-10).sum()
        return nll

    # optimise all three temperatures
    initial = np.array([1.0, 1.0, 1.0])
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0)]

    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not result.success:
        warnings.warn(
            f"Temperature optimisation did not converge: {result.message}", UserWarning
        )

    T_home, T_draw, T_away = result.x

    # ========================================================================
    # VALIDATION METRICS
    # ========================================================================

    uncalibrated_nll = negative_log_likelihood(np.array([1.0, 1.0, 1.0]))
    calibrated_nll = result.fun
    nll_improvement = uncalibrated_nll - calibrated_nll

    # calculate calibrated predictions for validation
    scaled_logits = logits.copy()
    scaled_logits[:, 0] /= T_home
    scaled_logits[:, 1] /= T_draw
    scaled_logits[:, 2] /= T_away

    scaled_logits_stable = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits_stable)
    calibrated_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # outcome-specific calibration metrics
    outcome_metrics = {}
    for i, name in enumerate(["home", "draw", "away"]):
        mask = actuals == i
        if mask.sum() > 0:
            # average predicted probability for this outcome
            avg_pred_uncal = predictions[mask, i].mean()
            avg_pred_cal = calibrated_probs[mask, i].mean()
            # actual rate (should be 1.0 for the correct outcome)
            actual_rate = 1.0

            outcome_metrics[name] = {
                "uncalibrated_prob": avg_pred_uncal,
                "calibrated_prob": avg_pred_cal,
                "actual_rate": actual_rate,
                "calibration_error_before": abs(avg_pred_uncal - actual_rate),
                "calibration_error_after": abs(avg_pred_cal - actual_rate),
            }

    # ========================================================================
    # DIAGNOSTIC OUTPUT
    # ========================================================================

    if verbose:
        print("\nOptimisation:")
        print(f"  Uncalibrated NLL: {uncalibrated_nll:.2f}")
        print(f"  Calibrated NLL:   {calibrated_nll:.2f}")
        print(f"  Improvement:      {nll_improvement:.2f}")

        if nll_improvement < 1.0:
            print("\n  Warning: Small improvement (<1.0)")
            print("  Base model may already be well-calibrated")
            print("  Or: calibration set may be too small/unrepresentative")

        print("\nOptimal temperatures:")

        # home wins
        print(f"  T_home = {T_home:.3f}", end="")
        if T_home > 1.5:
            print("  Model OVERCONFIDENT on home wins (will reduce home probs)")
        elif T_home < 0.7:
            print("  Model UNDERCONFIDENT on home wins (will increase home probs)")
        else:
            print("  Home calibration reasonable")

        # draws
        print(f"  T_draw = {T_draw:.3f}", end="")
        if T_draw > 1.5:
            print("  Model OVERCONFIDENT on draws (will reduce draw probs)")
        elif T_draw < 0.7:
            print("  Model UNDERCONFIDENT on draws (will BOOST draw probs!)")
        else:
            print("  Draw calibration reasonable")

        # away wins
        print(f"  T_away = {T_away:.3f}", end="")
        if T_away > 1.5:
            print("  Model OVERCONFIDENT on away wins (will reduce away probs)")
        elif T_away < 0.7:
            print("  Model UNDERCONFIDENT on away wins (will increase away probs)")
        else:
            print("  Away calibration reasonable")

        # per-outcome calibration metrics
        print("\n" + "-" * 70)
        print("Per-outcome calibration (for actual outcomes):")
        print(
            f"{'Outcome':<12} {'Before':<10} {'After':<10} {'Target':<10} {'Error Before':<12} {'Error After':<12}"
        )
        print("-" * 70)

        for name in ["home", "draw", "away"]:
            if name in outcome_metrics:
                m = outcome_metrics[name]
                print(
                    f"{name.capitalize():<12} "
                    f"{m['uncalibrated_prob']:>9.1%} "
                    f"{m['calibrated_prob']:>9.1%} "
                    f"{m['actual_rate']:>9.1%} "
                    f"{m['calibration_error_before']:>11.1%} "
                    f"{m['calibration_error_after']:>11.1%}"
                )

    # ========================================================================
    # RETURN PACKAGE
    # ========================================================================

    return {
        "T_home": T_home,
        "T_draw": T_draw,
        "T_away": T_away,
        "method": "outcome_specific",
        "nll_uncalibrated": uncalibrated_nll,
        "nll_calibrated": calibrated_nll,
        "nll_improvement": nll_improvement,
        "outcome_metrics": outcome_metrics,
        "n_samples": len(actuals),
        "outcome_counts": outcome_counts.tolist(),
        "optimisation_success": result.success,
    }


def apply_outcome_specific_scaling(
    predictions: Union[pd.DataFrame, np.ndarray],
    temperatures: Dict[str, float],
) -> np.ndarray:
    """Apply outcome-specific temperature scaling to predictions"""
    # convert to numpy array
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns {required_cols}")
        predictions = predictions[required_cols].values
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # clip and convert to logits
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    logits = np.log(predictions)

    # apply outcome-specific temperatures
    scaled_logits = logits.copy()
    scaled_logits[:, 0] /= temperatures["T_home"]
    scaled_logits[:, 1] /= temperatures["T_draw"]
    scaled_logits[:, 2] /= temperatures["T_away"]

    # softmax with numerical stability
    scaled_logits_stable = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits_stable)
    calibrated_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return calibrated_probs


def validate_calibration_on_holdout(
    temperatures: Dict[str, float],
    holdout_predictions: np.ndarray,
    holdout_actuals: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """Validate calibration on a holdout set"""
    # apply calibration
    calibrated_preds = apply_outcome_specific_scaling(holdout_predictions, temperatures)

    # convert actuals to indices if needed
    if holdout_actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        holdout_actuals = np.array([outcome_map[a] for a in holdout_actuals])

    # calculate metrics
    from ..evaluation.metrics import (
        calculate_brier_score,
        calculate_rps,
        calculate_log_loss,
    )

    # uncalibrated
    brier_uncal = calculate_brier_score(holdout_predictions, holdout_actuals)
    rps_uncal = calculate_rps(holdout_predictions, holdout_actuals)
    log_loss_uncal = calculate_log_loss(holdout_predictions, holdout_actuals)

    # calibrated
    brier_cal = calculate_brier_score(calibrated_preds, holdout_actuals)
    rps_cal = calculate_rps(calibrated_preds, holdout_actuals)
    log_loss_cal = calculate_log_loss(calibrated_preds, holdout_actuals)

    # draw-specific metrics
    draw_mask = holdout_actuals == 1
    if draw_mask.sum() > 0:
        avg_draw_prob_uncal = holdout_predictions[draw_mask, 1].mean()
        avg_draw_prob_cal = calibrated_preds[draw_mask, 1].mean()
        draw_actual_rate = draw_mask.mean()
    else:
        avg_draw_prob_uncal = avg_draw_prob_cal = draw_actual_rate = np.nan

    # accuracy
    pred_outcomes_uncal = np.argmax(holdout_predictions, axis=1)
    pred_outcomes_cal = np.argmax(calibrated_preds, axis=1)
    acc_uncal = (pred_outcomes_uncal == holdout_actuals).mean()
    acc_cal = (pred_outcomes_cal == holdout_actuals).mean()

    # draw prediction accuracy
    if draw_mask.sum() > 0:
        draw_acc_uncal = (pred_outcomes_uncal[draw_mask] == 1).mean()
        draw_acc_cal = (pred_outcomes_cal[draw_mask] == 1).mean()
    else:
        draw_acc_uncal = draw_acc_cal = np.nan

    metrics = {
        "brier_uncalibrated": brier_uncal,
        "brier_calibrated": brier_cal,
        "brier_improvement": brier_uncal - brier_cal,
        "rps_uncalibrated": rps_uncal,
        "rps_calibrated": rps_cal,
        "rps_improvement": rps_uncal - rps_cal,
        "log_loss_uncalibrated": log_loss_uncal,
        "log_loss_calibrated": log_loss_cal,
        "log_loss_improvement": log_loss_uncal - log_loss_cal,
        "accuracy_uncalibrated": acc_uncal,
        "accuracy_calibrated": acc_cal,
        "draw_accuracy_uncalibrated": draw_acc_uncal,
        "draw_accuracy_calibrated": draw_acc_cal,
        "draw_prob_uncalibrated": avg_draw_prob_uncal,
        "draw_prob_calibrated": avg_draw_prob_cal,
        "draw_actual_rate": draw_actual_rate,
        "n_samples": len(holdout_actuals),
        "n_draws": draw_mask.sum(),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("HOLDOUT VALIDATION")
        print("=" * 70)
        print(
            f"Holdout set: {len(holdout_actuals)} matches, {draw_mask.sum()} draws ({draw_mask.mean():.1%})"
        )

        print("\nOverall metrics:")
        print(f"{'Metric':<20} {'Uncalibrated':<15} {'Calibrated':<15} {'Change':<15}")
        print("-" * 70)
        print(
            f"{'Brier Score':<20} {brier_uncal:>14.4f} {brier_cal:>14.4f} {brier_uncal - brier_cal:>+14.4f}"
        )
        print(
            f"{'RPS':<20} {rps_uncal:>14.4f} {rps_cal:>14.4f} {rps_uncal - rps_cal:>+14.4f}"
        )
        print(
            f"{'Log Loss':<20} {log_loss_uncal:>14.4f} {log_loss_cal:>14.4f} {log_loss_uncal - log_loss_cal:>+14.4f}"
        )
        print(
            f"{'Accuracy':<20} {acc_uncal:>14.1%} {acc_cal:>14.1%} {(acc_cal - acc_uncal) * 100:>+13.1f}pp"
        )

        print("\nDraw-specific metrics:")
        print(
            f"{'Draw accuracy':<20} {draw_acc_uncal:>14.1%} {draw_acc_cal:>14.1%} {(draw_acc_cal - draw_acc_uncal) * 100:>+13.1f}pp"
        )
        print(
            f"{'Avg draw prob':<20} {avg_draw_prob_uncal:>14.1%} {avg_draw_prob_cal:>14.1%} {(avg_draw_prob_cal - avg_draw_prob_uncal) * 100:>+13.1f}pp"
        )
        print(f"{'Actual draw rate':<20} {draw_actual_rate:>14.1%}")

        # validation checks
        print("\n" + "-" * 70)
        if metrics["rps_improvement"] > 0.001:
            print(" GOOD: RPS improved on holdout set")
            print("  Calibration generalises well")
        elif metrics["rps_improvement"] > -0.001:
            print(" NEUTRAL: RPS unchanged on holdout set")
            print("  Calibration neither helps nor hurts")
        else:
            print(" WARNING: RPS degraded on holdout set")
            print("  Calibration may be overfitting to calibration set")
            print("  Consider using more calibration data or simpler calibration")

        if draw_mask.sum() > 0:
            draw_improvement = (draw_acc_cal - draw_acc_uncal) * 100
            if draw_improvement > 10:
                print(
                    f"\n EXCELLENT: Draw accuracy improved by {draw_improvement:.0f}pp"
                )
            elif draw_improvement > 0:
                print(f"\n GOOD: Draw accuracy improved by {draw_improvement:.0f}pp")
            else:
                print("\n  Draw accuracy unchanged or degraded")

    return metrics


def apply_calibration(
    predictions: Union[pd.DataFrame, np.ndarray],
    calibrators: Dict[str, Any],
) -> np.ndarray:
    """Apply calibration - handles both standard and outcome-specific"""
    if "temperature" not in calibrators:
        # no calibration available, return unchanged
        if isinstance(predictions, pd.DataFrame):
            return predictions[["home_win", "draw", "away_win"]].values
        return np.array(predictions)

    temperature = calibrators["temperature"]
    calibration_method = calibrators.get("calibration_method", "temperature_scaling")

    # detect calibration type
    if calibration_method == "outcome_specific" or isinstance(temperature, dict):
        # outcome-specific temperature scaling
        return apply_outcome_specific_scaling(predictions, temperature)
    else:
        # standard temperature scaling
        return apply_temperature_scaling(predictions, temperature)
