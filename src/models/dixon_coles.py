# src/models/dixon_coles.py


import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


def tau_dixon_coles(
    home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float
) -> float:
    """Dixon-Coles tau correction function for low-scoring outcomes"""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


def calculate_match_probabilities_dixon_coles(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.13,
    max_goals: int = 8,
) -> tuple[float, float, float, dict[tuple[int, int], float]]:
    """Calculate match outcome probabilities with Dixon-Coles correction"""
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    score_probs = {}

    # calculate probabilities for all score combinations
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            # base poisson probability
            p_base = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

            # apply dixon-coles correction
            tau = tau_dixon_coles(h, a, lambda_home, lambda_away, rho)
            p = p_base * tau

            score_probs[(h, a)] = p

            # accumulate outcome probabilities
            if h > a:
                home_win_prob += p
            elif h == a:
                draw_prob += p
            else:
                away_win_prob += p

    # normalise probabilities to ensure they sum to 1.0
    total = home_win_prob + draw_prob + away_win_prob

    return (
        home_win_prob / total,
        draw_prob / total,
        away_win_prob / total,
        score_probs,
    )


def calculate_match_probabilities_dixon_coles_batch(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float = -0.13,
    max_goals: int = 8,
) -> np.ndarray:
    """Calculate outcome probabilities for arrays of matches.

    Vectorised version of calculate_match_probabilities_dixon_coles.
    Returns shape (n_matches, 3) with columns [home_win, draw, away_win].
    """
    goals = np.arange(max_goals + 1)

    # P(h goals) and P(a goals) for each match: shape (n, max_goals+1)
    P_h = poisson.pmf(goals[None, :], lambda_home[:, None])
    P_a = poisson.pmf(goals[None, :], lambda_away[:, None])

    # joint probability: shape (n, max_goals+1, max_goals+1)
    P_joint = P_h[:, :, None] * P_a[:, None, :]

    # tau correction (only 4 low-scoring outcomes are non-trivial)
    tau = np.ones_like(P_joint)
    tau[:, 0, 0] = 1 - lambda_home * lambda_away * rho
    tau[:, 0, 1] = 1 + lambda_home * rho
    tau[:, 1, 0] = 1 + lambda_away * rho
    tau[:, 1, 1] = 1 - rho

    P_corrected = P_joint * tau
    P_corrected /= P_corrected.sum(axis=(1, 2), keepdims=True)

    # score grid for outcome masks
    h_grid, a_grid = np.meshgrid(goals, goals, indexing="ij")

    return np.column_stack(
        [
            P_corrected[:, h_grid > a_grid].sum(axis=1),  # home win
            P_corrected[:, h_grid == a_grid].sum(axis=1),  # draw
            P_corrected[:, h_grid < a_grid].sum(axis=1),  # away win
        ]
    )


def fit_rho_parameter(
    df: pd.DataFrame,
    params: dict,
    initial_rho: float = -0.13,
    verbose: bool = True,
) -> float:
    """Optimise the rho correlation parameter using maximum likelihood"""
    from .poisson import calculate_lambdas

    if verbose:
        print("\n" + "=" * 60)
        print("FITTING DIXON-COLES RHO PARAMETER")
        print("=" * 60)

    # calculate lambdas for all matches
    lambda_home, lambda_away = calculate_lambdas(df, params)

    # get actual outcomes
    home_goals = df["home_goals"].fillna(0).astype(int).values
    away_goals = df["away_goals"].fillna(0).astype(int).values

    def negative_log_likelihood(rho: float) -> float:
        """Calculate negative log-likelihood for given rho"""
        log_lik = 0.0

        for i in range(len(df)):
            h, a = home_goals[i], away_goals[i]
            lam_h, lam_a = lambda_home[i], lambda_away[i]

            # base poisson probability
            p_base = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)

            # apply tau correction
            tau = tau_dixon_coles(h, a, lam_h, lam_a, rho)

            # total probability
            p = p_base * tau

            # add to log-likelihood (with safety for log(0))
            log_lik += np.log(max(p, 1e-10))

        return -log_lik

    # optimise rho (search range: -0.4 to 0.2)
    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(-0.4, 0.2),
        method="bounded",
    )

    optimal_rho = result.x

    if verbose:
        baseline_nll = negative_log_likelihood(-0.13)
        optimal_nll = result.fun

        print(f"  Initial rho: {initial_rho:.3f}")
        print(f"  Optimal rho: {optimal_rho:.3f}")
        print(f"  Baseline NLL (rho=-0.13): {baseline_nll:.2f}")
        print(f"  Optimal NLL: {optimal_nll:.2f}")
        print(f"  Improvement: {baseline_nll - optimal_nll:.2f}")

    return optimal_rho
