# src/simulation/sampling.py

import numpy as np
from typing import Union, Tuple
from scipy.stats import poisson


def sample_goals_calibrated(
    lambda_val: Union[float, np.ndarray], dispersion_factor: float, size: int = 1
) -> Union[int, np.ndarray]:
    """
    Sample goals accounting for overdispersion.

    Uses Poisson distribution when dispersion ≈ 1, and negative binomial
    when dispersion > 1.1 to account for extra variance in actual goals
    compared to the model's fitted lambdas.

    The hybrid architecture:
    - Model fitted on weighted performance (npxG/npG) for stable parameters
    - Sampling uses dispersion factor to match actual goal variance
    """
    # ensure lambda_val is array for consistent handling
    lambda_val = np.atleast_1d(lambda_val)
    is_scalar = lambda_val.shape == (1,)

    if dispersion_factor <= 1.1:
        # use poisson distribution
        result = np.random.poisson(lambda_val, size=(size, len(lambda_val)))
    else:
        # use negative binomial for overdispersion
        # negative binomial parameterisation: NB(r, p)
        # mean = r(1-p)/p, variance = r(1-p)/p^2
        # for overdispersion: var = dispersion * mean

        # calculate parameters
        r = lambda_val / (dispersion_factor - 1.0)
        r = np.maximum(r, 0.1)  # avoid numerical issues
        p = r / (r + lambda_val)

        result = np.random.negative_binomial(r, p, size=(size, len(lambda_val)))

    # return appropriate shape
    if size == 1:
        result = result.squeeze(0)  # remove size dimension
        if is_scalar:
            return result.item()  # return scalar if input was scalar
        return result
    else:
        if is_scalar:
            return result.squeeze(-1)  # remove lambda dimension if scalar
        return result


def sample_match_outcome(
    lambda_home: float,
    lambda_away: float,
    dispersion_factor: float = 1.0,
    max_goals: int = 10,
) -> Tuple[int, int]:
    """Sample a single match outcome"""
    home_goals = sample_goals_calibrated(lambda_home, dispersion_factor, size=1)
    away_goals = sample_goals_calibrated(lambda_away, dispersion_factor, size=1)

    # cap at max_goals (safety for extreme cases)
    home_goals = min(home_goals, max_goals)
    away_goals = min(away_goals, max_goals)

    return int(home_goals), int(away_goals)


def calculate_outcome_probabilities(
    lambda_home: float,
    lambda_away: float,
    dispersion_factor: float = 1.0,
    max_goals: int = 8,
    use_poisson: bool = True,
) -> Tuple[float, float, float]:
    """
    Calculate match outcome probabilities (home/draw/away).

    Can use either Poisson (fast, analytical) or negative binomial
    (more accurate for high dispersion, but slower).
    """
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    if use_poisson or dispersion_factor <= 1.1:
        # use Poisson (faster)
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

                if h > a:
                    home_win_prob += p
                elif h == a:
                    draw_prob += p
                else:
                    away_win_prob += p
    else:
        # use negative binomial (more accurate but slower)
        from scipy.stats import nbinom

        # calculate NB parameters
        r_h = lambda_home / (dispersion_factor - 1.0)
        r_h = max(r_h, 0.1)
        p_h = r_h / (r_h + lambda_home)

        r_a = lambda_away / (dispersion_factor - 1.0)
        r_a = max(r_a, 0.1)
        p_a = r_a / (r_a + lambda_away)

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = nbinom.pmf(h, r_h, p_h) * nbinom.pmf(a, r_a, p_a)

                if h > a:
                    home_win_prob += p
                elif h == a:
                    draw_prob += p
                else:
                    away_win_prob += p

    # normalise (should be close to 1.0 already)
    total = home_win_prob + draw_prob + away_win_prob

    return (home_win_prob / total, draw_prob / total, away_win_prob / total)


def test_sampling_distribution(
    lambda_val: float,
    dispersion_factor: float,
    n_samples: int = 10000,
    verbose: bool = True,
) -> dict:
    """
    Test sampling distribution properties.

    Useful for validating that the sampling correctly implements the
    desired overdispersion.
    """
    samples = sample_goals_calibrated(lambda_val, dispersion_factor, size=n_samples)

    empirical_mean = samples.mean()
    empirical_var = samples.var()

    # theoretical values
    theoretical_mean = lambda_val
    theoretical_var = lambda_val * dispersion_factor

    var_ratio = empirical_var / theoretical_var

    results = {
        "empirical_mean": empirical_mean,
        "empirical_var": empirical_var,
        "theoretical_mean": theoretical_mean,
        "theoretical_var": theoretical_var,
        "variance_ratio": var_ratio,
    }

    if verbose:
        print(
            f"\nTesting sampling with λ={lambda_val:.2f}, dispersion={dispersion_factor:.2f}"
        )
        print(
            f"Theoretical: mean={theoretical_mean:.2f}, variance={theoretical_var:.2f}"
        )
        print(f"Empirical:   mean={empirical_mean:.2f}, variance={empirical_var:.2f}")

        if abs(var_ratio - 1.0) < 0.1:
            print(f"✓ Variance ratio: {var_ratio:.2f} (target: 1.00)")
        else:
            print(f"✗ Variance ratio: {var_ratio:.2f} (target: 1.00)")

    return results
