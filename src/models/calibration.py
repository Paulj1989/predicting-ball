# src/models/calibration.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from typing import Dict, Callable, Tuple, Any


def fit_temperature_scaler(
    predictions: np.ndarray, actuals: np.ndarray, verbose: bool = True
) -> float:
    """
    Fit temperature scaling parameter via NLL minimisation.

    Temperature scaling adjusts prediction confidence without changing
    the relative ordering of outcomes.
    """
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
        # NLL = -Σ log(p_i[y_i]) where y_i is the true class
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
            print("     → Model is overconfident (T > 1.5)")
        elif optimal_temperature < 0.8:
            print("     → Model is underconfident (T < 0.8)")
        else:
            print("     → Model confidence is reasonable")

    return optimal_temperature


def apply_temperature_scaling(
    predictions: np.ndarray, temperature: float
) -> np.ndarray:
    """
    Apply temperature scaling to predictions.

    Transforms predicted probabilities by:
    1. Converting to logits
    2. Scaling by temperature
    3. Applying softmax to renormalise
    """
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
                print(f"\n✓ Converged to dispersion={best_dispersion:.3f}")
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
            print(f"\n⚠ Did not fully converge after {max_iterations} iterations")
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
        """Get interpolated dispersion for any confidence level."""
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
