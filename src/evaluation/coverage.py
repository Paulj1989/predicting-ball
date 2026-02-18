# src/evaluation/coverage.py

from typing import Any

import numpy as np
import pandas as pd

from ..simulation.sampling import sample_goals_calibrated


def run_coverage_test(
    bootstrap_params: list[dict[str, Any]],
    test_data: pd.DataFrame,
    confidence: float = 0.80,
    n_samples: int = 5000,
    verbose: bool = True,
) -> float:
    """
    Test empirical coverage of prediction intervals.

    Generates prediction intervals from bootstrap samples and checks
    whether actual outcomes fall within the intervals at the specified
    confidence level.
    """
    # import here to avoid circular dependency
    from ..models.poisson import calculate_lambdas

    if verbose:
        print(f"\nTesting {confidence:.0%} prediction intervals on {len(test_data)} matches")

    n_matches = len(test_data)
    covered_both = 0

    for i in range(n_matches):
        match = test_data.iloc[i]
        actual_h = int(match["home_goals"])
        actual_a = int(match["away_goals"])

        # sample lambdas from bootstrap distribution
        lambda_h_samples = []
        lambda_a_samples = []

        # draw samples from bootstrap parameters
        sample_indices = np.random.choice(
            len(bootstrap_params),
            size=min(n_samples, len(bootstrap_params)),
            replace=True,
        )

        for idx in sample_indices:
            params = bootstrap_params[idx]
            lambda_h, lambda_a = calculate_lambdas(test_data.iloc[[i]], params)
            lambda_h_samples.append(lambda_h[0])
            lambda_a_samples.append(lambda_a[0])

        lambda_h_samples = np.array(lambda_h_samples)
        lambda_a_samples = np.array(lambda_a_samples)

        # sample goals from each lambda with dispersion
        home_goals_samples = []
        away_goals_samples = []

        for lh, la in zip(lambda_h_samples, lambda_a_samples, strict=False):
            hg = sample_goals_calibrated(lh, size=1)
            ag = sample_goals_calibrated(la, size=1)
            home_goals_samples.append(hg)
            away_goals_samples.append(ag)

        home_goals_samples = np.array(home_goals_samples)
        away_goals_samples = np.array(away_goals_samples)

        # calculate prediction intervals
        alpha = (1 - confidence) / 2
        h_lower, h_upper = np.percentile(home_goals_samples, [alpha * 100, (1 - alpha) * 100])
        a_lower, a_upper = np.percentile(away_goals_samples, [alpha * 100, (1 - alpha) * 100])

        # check coverage
        if (h_lower <= actual_h <= h_upper) and (a_lower <= actual_a <= a_upper):
            covered_both += 1

    empirical_coverage = covered_both / n_matches

    if verbose:
        print(f"Coverage: {empirical_coverage:.1%} (target: {confidence:.1%})")

        deviation = abs(empirical_coverage - confidence)
        if deviation < 0.05:
            print("✓ Well calibrated")
        elif deviation < 0.10:
            print("⚠ Moderately calibrated")
        else:
            print("✗ Poorly calibrated")

    return empirical_coverage


def test_base_poisson_coverage(
    params: dict[str, Any],
    test_data: pd.DataFrame,
    confidence: float = 0.80,
    verbose: bool = True,
) -> float:
    """
    Test coverage using base Poisson distribution (no bootstrap).

    This tests whether the model's point estimates produce well-calibrated
    prediction intervals without accounting for parameter uncertainty.
    """
    # import here to avoid circular dependency
    from scipy.stats import poisson

    from ..models.poisson import calculate_lambdas

    if verbose:
        print(f"\nTesting base Poisson {confidence:.0%} intervals (no bootstrap)")

    # calculate lambdas for all matches
    lambda_h, lambda_a = calculate_lambdas(test_data, params)

    actual_h = test_data["home_goals"].astype(int).values
    actual_a = test_data["away_goals"].astype(int).values

    covered = 0
    alpha = (1 - confidence) / 2

    for i in range(len(test_data)):
        # poisson quantiles
        h_lower = poisson.ppf(alpha, lambda_h[i])
        h_upper = poisson.ppf(1 - alpha, lambda_h[i])
        a_lower = poisson.ppf(alpha, lambda_a[i])
        a_upper = poisson.ppf(1 - alpha, lambda_a[i])

        if (h_lower <= actual_h[i] <= h_upper) and (a_lower <= actual_a[i] <= a_upper):
            covered += 1

    coverage = covered / len(test_data)

    if verbose:
        print(f"Base Poisson coverage: {coverage:.1%} (target: {confidence:.1%})")

        if coverage < confidence - 0.05:
            print("✗ Under-covering - intervals too narrow")
        elif coverage > confidence + 0.05:
            print("✗ Over-covering - intervals too wide")
        else:
            print("✓ Reasonably calibrated")

    return coverage


def diagnose_bootstrap_lambda_distribution(
    bootstrap_params: list[dict[str, Any]],
    match_data: pd.DataFrame,
    n_samples: int = 1000,
) -> dict[str, float]:
    """
    Diagnose bootstrap lambda distributions for a specific match.

    Useful for understanding whether bootstrap is appropriately
    quantifying parameter uncertainty.
    """
    # import here to avoid circular dependency
    from ..models.poisson import calculate_lambdas

    # get base model prediction
    base_params = bootstrap_params[0]
    lambda_h_base, lambda_a_base = calculate_lambdas(match_data, base_params)
    lambda_h_base = lambda_h_base[0]
    lambda_a_base = lambda_a_base[0]

    # sample from bootstrap distribution
    lambda_h_samples = []
    lambda_a_samples = []

    sample_indices = np.random.choice(
        len(bootstrap_params), size=min(n_samples, len(bootstrap_params)), replace=True
    )

    for idx in sample_indices:
        params = bootstrap_params[idx]
        lh, la = calculate_lambdas(match_data, params)
        lambda_h_samples.append(lh[0])
        lambda_a_samples.append(la[0])

    lambda_h_samples = np.array(lambda_h_samples)
    lambda_a_samples = np.array(lambda_a_samples)

    # calculate statistics
    diagnostics = {
        "lambda_h_mean": lambda_h_samples.mean(),
        "lambda_h_std": lambda_h_samples.std(),
        "lambda_h_base": lambda_h_base,
        "lambda_a_mean": lambda_a_samples.mean(),
        "lambda_a_std": lambda_a_samples.std(),
        "lambda_a_base": lambda_a_base,
        # variance compared to poisson
        "variance_ratio_h": lambda_h_samples.var() / lambda_h_base,
        "variance_ratio_a": lambda_a_samples.var() / lambda_a_base,
    }

    return diagnostics


def coverage_by_confidence_level(
    bootstrap_params: list[dict[str, Any]],
    test_data: pd.DataFrame,
    confidence_levels: list[float] | None = None,
    verbose: bool = True,
) -> dict[float, float]:
    """Test coverage at multiple confidence levels"""
    if confidence_levels is None:
        confidence_levels = [0.68, 0.8, 0.95]
    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("COVERAGE AT MULTIPLE CONFIDENCE LEVELS")
        print("=" * 60)

    for conf in confidence_levels:
        coverage = run_coverage_test(
            bootstrap_params, test_data, confidence=conf, verbose=False
        )
        results[conf] = coverage

        if verbose:
            deviation = abs(coverage - conf)
            status = "✓" if deviation < 0.05 else "⚠" if deviation < 0.10 else "✗"
            print(
                f"{status} {conf:.0%} intervals: {coverage:.1%} coverage (target: {conf:.0%})"
            )

    return results
