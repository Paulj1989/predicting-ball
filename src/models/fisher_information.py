# src/models/fisher_information.py
"""Fisher information matrix for MLE standard errors.

Computes analytical parameter uncertainty from the Hessian of the
Poisson log-likelihood, replacing the parametric bootstrap for
season simulation draws.
"""

from typing import Any

import numpy as np
import pandas as pd

from .poisson import calculate_lambdas


def compute_fisher_information(
    params: dict[str, Any],
    df: pd.DataFrame,
    hyperparams: dict[str, float],
) -> np.ndarray:
    """Compute the observed Fisher information matrix for the Poisson model.

    Parameters are ordered:
    [attack_0, ..., attack_n, defense_0, ..., defense_n, home_adv]
    """
    all_teams = params["teams"]
    nt = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n_params = 2 * nt + 1

    # fitted expected goals for every training match
    lambda_h, lambda_a = calculate_lambdas(df, params)

    # time decay weights (same formula used during model fitting)
    decay = hyperparams.get("time_decay", 0.001)
    days_ago = (df["date"].max() - df["date"]).dt.days.values
    weights = np.exp(-decay * days_ago)

    F = np.zeros((n_params, n_params))

    home_idx = df["home_team"].map(team_to_idx).values
    away_idx = df["away_team"].map(team_to_idx).values

    # accumulate fisher info from each match
    for m in range(len(df)):
        hi, ai = int(home_idx[m]), int(away_idx[m])
        w = weights[m]
        mu_h, mu_a = lambda_h[m], lambda_a[m]

        # home goals: att[home], def[away], home_adv
        h_idx = [hi, nt + ai, 2 * nt]
        for i in h_idx:
            for j in h_idx:
                F[i, j] += w * mu_h

        # away goals: att[away], def[home]
        a_idx = [ai, nt + hi]
        for i in a_idx:
            for j in a_idx:
                F[i, j] += w * mu_a

    # regularisation contribution (second derivative of quadratic prior)
    prior_decay_rate = hyperparams.get("prior_decay_rate", 10.0)
    base_prior = hyperparams.get("lambda_reg", 0.3)

    # pre-compute match counts per team
    home_counts = df["home_team"].value_counts()
    away_counts = df["away_team"].value_counts()

    for i, team in enumerate(all_teams):
        n_matches = home_counts.get(team, 0) + away_counts.get(team, 0)
        pw = base_prior / (1 + n_matches / prior_decay_rate)
        F[i, i] += 2 * pw
        F[nt + i, nt + i] += 2 * pw

    return F


def invert_fisher_with_constraints(
    F: np.ndarray,
    n_teams: int,
) -> np.ndarray:
    """Invert the Fisher matrix handling the identifiability constraint.

    Attack and defense ratings each sum to zero, making the Fisher matrix
    rank-deficient along two directions. We project these out before
    inversion to get meaningful standard errors.
    """
    n_params = F.shape[0]

    # null-space directions: uniform shift of all attacks / all defenses
    v_att = np.zeros(n_params)
    v_att[:n_teams] = 1.0 / np.sqrt(n_teams)

    v_def = np.zeros(n_params)
    v_def[n_teams : 2 * n_teams] = 1.0 / np.sqrt(n_teams)

    # large penalty along null-space directions
    constraint_weight = 1000.0
    F_constrained = (
        F
        + constraint_weight * np.outer(v_att, v_att)
        + constraint_weight * np.outer(v_def, v_def)
    )

    ridge = 1e-8 * np.eye(n_params)
    cov = np.linalg.inv(F_constrained + ridge)

    return cov


def build_state_vector(params: dict[str, Any]) -> np.ndarray:
    """Build the state vector from model params in canonical ordering.

    Order: [attack_0, ..., attack_n, defense_0, ..., defense_n, home_adv]
    Teams are ordered by params["teams"].
    """
    teams = params["teams"]
    return np.concatenate(
        [
            [params["attack"][t] for t in teams],
            [params["defense"][t] for t in teams],
            [params["home_adv"]],
        ]
    )


def draw_mle_samples(
    state_mean: np.ndarray,
    cov: np.ndarray,
    n_draws: int,
    seed: int | None = None,
) -> np.ndarray:
    """Draw parameter samples from the MLE posterior (multivariate normal).

    Returns an array of shape (n_draws, n_params). Ensures the covariance
    is positive definite before drawing.
    """
    if seed is not None:
        np.random.seed(seed)

    # nudge if not positive definite (numerical artifact)
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov = cov + (-eigvals.min() + 1e-8) * np.eye(len(cov))

    return np.random.multivariate_normal(state_mean, cov, size=n_draws)
