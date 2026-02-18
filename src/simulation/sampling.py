# src/simulation/sampling.py


import numpy as np
from scipy.stats import poisson

from src.models.dixon_coles import tau_dixon_coles


def sample_goals_calibrated(lambda_val: float | np.ndarray, size: int = 1) -> int | np.ndarray:
    """Sample goals from a Poisson distribution with the given rate(s)"""
    lambda_val = np.atleast_1d(lambda_val)
    is_scalar = lambda_val.shape == (1,)

    result = np.random.poisson(lambda_val, size=(size, len(lambda_val)))

    if size == 1:
        result = result.squeeze(0)
        if is_scalar:
            return result.item()
        return result
    else:
        if is_scalar:
            return result.squeeze(-1)
        return result


def sample_match_outcome(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> tuple[int, int]:
    """Sample a single match outcome"""
    home_goals = sample_goals_calibrated(lambda_home, size=1)
    away_goals = sample_goals_calibrated(lambda_away, size=1)

    home_goals = min(home_goals, max_goals)
    away_goals = min(away_goals, max_goals)

    return int(home_goals), int(away_goals)


def sample_scoreline_dixon_coles(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.13,
    max_goals: int = 8,
) -> tuple[int, int]:
    """Sample a scoreline from the full Dixon-Coles joint PMF"""
    n = max_goals + 1

    pmf_h = np.array([poisson.pmf(g, lambda_home) for g in range(n)])
    pmf_a = np.array([poisson.pmf(g, lambda_away) for g in range(n)])

    # build joint probability grid with dixon-coles tau correction
    grid = np.outer(pmf_h, pmf_a)
    for h in range(min(2, n)):
        for a in range(min(2, n)):
            grid[h, a] *= tau_dixon_coles(h, a, lambda_home, lambda_away, rho)

    # flatten, normalise, and sample
    flat = grid.ravel()
    flat = np.maximum(flat, 0.0)
    flat /= flat.sum()

    idx = np.random.choice(len(flat), p=flat)
    home_goals, away_goals = divmod(idx, n)

    return int(home_goals), int(away_goals)
