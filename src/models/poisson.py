# src/models/poisson.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, Optional, Tuple, Any


# ============================================================================
# TWO-STAGE FITTING FUNCTIONS
# ============================================================================


def fit_baseline_strengths(
    df_train: pd.DataFrame,
    hyperparams: Dict[str, float],
    promoted_priors: Optional[Dict[str, Dict[str, float]]] = None,
    home_adv_prior: Optional[float] = None,
    home_adv_std: Optional[float] = None,
    n_random_starts: int = 3,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Stage 1: Fit baseline team strengths (attack/defense) + home advantage"""

    from ..models.dixon_coles import tau_dixon_coles

    # ========================================================================
    # SETUP
    # ========================================================================

    all_teams = sorted(pd.unique(df_train[["home_team", "away_team"]].values.ravel()))

    if promoted_priors:
        for team in promoted_priors.keys():
            if team not in all_teams:
                all_teams.append(team)
        all_teams = sorted(all_teams)

    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    home_idx = df_train["home_team"].map(team_to_idx).values
    away_idx = df_train["away_team"].map(team_to_idx).values

    # fit on weighted goals
    home_g_weighted = np.round(df_train["home_goals_weighted"]).astype(int).values
    away_g_weighted = np.round(df_train["away_goals_weighted"]).astype(int).values

    # ========================================================================
    # PRIORS
    # ========================================================================

    attack_priors = np.zeros(n_teams)
    defense_priors = np.zeros(n_teams)

    matches_played = np.zeros(n_teams)
    for i, team in enumerate(all_teams):
        team_matches = df_train[
            (df_train["home_team"] == team) | (df_train["away_team"] == team)
        ]
        matches_played[i] = len(team_matches)

    prior_decay_rate = hyperparams.get("prior_decay_rate", 10.0)
    base_prior_weight = hyperparams.get("lambda_reg", 0.3)
    prior_weights = base_prior_weight / (1 + matches_played / prior_decay_rate)

    if promoted_priors:
        for team, priors in promoted_priors.items():
            # skip metadata keys (not actual teams)
            if team.startswith("_"):
                continue

            if team in team_to_idx:
                idx = team_to_idx[team]
                attack_priors[idx] = priors["attack_prior"]
                defense_priors[idx] = priors["defense_prior"]

            if matches_played[idx] == 0:
                prior_weights[idx] = base_prior_weight * 1000
            elif matches_played[idx] < 5:
                prior_weights[idx] = base_prior_weight * 10

    # home advantage prior
    if home_adv_prior is not None and home_adv_std is not None:
        lambda_home_adv = 1.0 / (home_adv_std**2)
        use_home_prior = True
    else:
        home_adv_prior = 0.30
        lambda_home_adv = 0.0
        use_home_prior = False

    # ========================================================================
    # TIME WEIGHTING
    # ========================================================================

    lambda_decay = hyperparams.get("time_decay", 0.001)
    time_weights = np.exp(
        -lambda_decay * (df_train["date"].max() - df_train["date"]).dt.days.values
    )
    time_weights = np.maximum(time_weights, 0.1)

    # ========================================================================
    # DIXON-COLES RHO
    # ========================================================================

    rho = hyperparams.get("rho", -0.13)

    # ========================================================================
    # OBJECTIVE FUNCTION (WITH DIXON-COLES)
    # ========================================================================

    def neg_loglik(x: np.ndarray) -> float:
        """Negative log-likelihood with Dixon-Coles correction"""
        attack = x[:n_teams]
        defense = x[n_teams : 2 * n_teams]
        home_adv = x[2 * n_teams]

        # identifiability constraint
        attack = attack - attack.mean()
        defense = defense - defense.mean()

        # calculate lambdas (baseline only: team + home advantage)
        home_strength = home_adv + attack[home_idx] + defense[away_idx]
        away_strength = attack[away_idx] + defense[home_idx]

        mu_h = np.clip(np.exp(home_strength), 0.1, 8.0)
        mu_a = np.clip(np.exp(away_strength), 0.1, 8.0)

        # vectorised tau calculation
        tau_vec = np.array(
            [
                tau_dixon_coles(
                    home_g_weighted[i], away_g_weighted[i], mu_h[i], mu_a[i], rho
                )
                for i in range(len(home_g_weighted))
            ]
        )

        # base poisson log-likelihoods
        ll_h = poisson.logpmf(home_g_weighted, mu_h)
        ll_a = poisson.logpmf(away_g_weighted, mu_a)

        # apply dixon-coles correction: log(p_base * tau) = log(p_base) + log(tau)
        ll_match = ll_h + ll_a + np.log(np.maximum(tau_vec, 1e-10))
        ll_total = np.sum(time_weights * ll_match)

        # regularisation: priors on attack/defense
        prior_penalty = np.sum(
            prior_weights
            * ((attack - attack_priors) ** 2 + (defense - defense_priors) ** 2)
        )

        # home advantage prior
        if use_home_prior:
            home_adv_penalty = lambda_home_adv * (home_adv - home_adv_prior) ** 2
        else:
            home_adv_penalty = 0.0

        return -(ll_total - prior_penalty - home_adv_penalty)

    # ========================================================================
    # OPTIMISATION
    # ========================================================================

    bounds = []
    for _ in range(n_teams):
        bounds.append((-3.0, 3.0))  # attack
    for _ in range(n_teams):
        bounds.append((-3.0, 3.0))  # defense
    bounds.append((0.05, 0.5))  # home_adv

    best_result = None
    best_nll = float("inf")

    for restart in range(n_random_starts):
        if restart == 0:
            # start from zero (with promoted priors if available)
            x0 = np.zeros(2 * n_teams + 1)
            if promoted_priors:
                for team, priors in promoted_priors.items():
                    # skip metadata keys
                    if team.startswith("_"):
                        continue

                    if team in team_to_idx:
                        idx = team_to_idx[team]
                        x0[idx] = priors["attack_prior"]
                        x0[n_teams + idx] = priors["defense_prior"]
            x0[-1] = home_adv_prior if home_adv_prior else 0.30
        else:
            # random starts
            x0 = np.random.randn(2 * n_teams + 1) * 0.3
            x0[-1] = home_adv_prior if home_adv_prior else 0.30

        result = minimize(
            neg_loglik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result

    if best_result is None or not best_result.success:
        if verbose:
            print("  ✗ Optimisation failed")
        return None

    # extract parameters
    attack = best_result.x[:n_teams]
    defense = best_result.x[n_teams : 2 * n_teams]
    home_adv = best_result.x[2 * n_teams]

    # apply identifiability constraint
    attack = attack - attack.mean()
    defense = defense - defense.mean()

    # ========================================================================
    # RETURN PARAMETERS
    # ========================================================================

    params = {
        "attack": {team: float(attack[i]) for i, team in enumerate(all_teams)},
        "defense": {team: float(defense[i]) for i, team in enumerate(all_teams)},
        "home_adv": float(home_adv),
        "rho": rho,
        "teams": all_teams,
        "success": True,
        "nll": float(best_nll),
    }

    if verbose:
        print("\n  ✓ Baseline strengths fitted (with Dixon-Coles)")
        print(f"    Home advantage: {home_adv:.3f}")
        print(f"    Rho: {rho:.3f}")
        print(f"    NLL: {best_nll:.2f}")

    return params


def fit_feature_coefficients(
    df_train: pd.DataFrame,
    baseline_params: Dict[str, Any],
    hyperparams: Dict[str, float],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Stage 2: Fit coefficients for match-specific features (odds, form)"""

    from .ratings import add_interpretable_ratings_to_params

    # ========================================================================
    # SETUP
    # ========================================================================

    all_teams = baseline_params["teams"]
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    home_idx = df_train["home_team"].map(team_to_idx).values
    away_idx = df_train["away_team"].map(team_to_idx).values

    home_g_weighted = np.round(df_train["home_goals_weighted"]).astype(int).values
    away_g_weighted = np.round(df_train["away_goals_weighted"]).astype(int).values

    # extract baseline parameters
    attack = np.array([baseline_params["attack"][t] for t in all_teams])
    defense = np.array([baseline_params["defense"][t] for t in all_teams])
    home_adv_baseline = baseline_params["home_adv"]

    # ========================================================================
    # FEATURES
    # ========================================================================

    home_log_odds_ratio = (
        df_train["home_log_odds_ratio"].fillna(0).values
        if "home_log_odds_ratio" in df_train
        else np.zeros(len(df_train))
    )

    home_form_w5 = (
        df_train["home_npxgd_w5"].fillna(0).values
        if "home_npxgd_w5" in df_train
        else np.zeros(len(df_train))
    )
    away_form_w5 = (
        df_train["away_npxgd_w5"].fillna(0).values
        if "away_npxgd_w5" in df_train
        else np.zeros(len(df_train))
    )

    use_form = "home_npxgd_w5" in df_train.columns

    # ========================================================================
    # TIME WEIGHTING
    # ========================================================================

    lambda_decay = hyperparams.get("time_decay", 0.001)
    time_weights = np.exp(
        -lambda_decay * (df_train["date"].max() - df_train["date"]).dt.days.values
    )
    time_weights = np.maximum(time_weights, 0.1)
    combined_weights = time_weights

    # ========================================================================
    # OBJECTIVE: FEATURE COEFFICIENTS ONLY
    # ========================================================================

    def neg_loglik_features(x: np.ndarray) -> float:
        """Fit feature coefficients with fixed team strengths"""
        beta_odds = x[0]
        beta_form = x[1] if use_form else 0.0

        # calculate lambdas (baseline + features)
        home_strength = (
            home_adv_baseline
            + attack[home_idx]
            + defense[away_idx]
            + beta_odds * home_log_odds_ratio
            + beta_form * home_form_w5
        )
        away_strength = (
            attack[away_idx]
            + defense[home_idx]
            - beta_odds * home_log_odds_ratio
            + beta_form * away_form_w5
        )

        mu_h = np.clip(np.exp(home_strength), 0.1, 8.0)
        mu_a = np.clip(np.exp(away_strength), 0.1, 8.0)

        # poisson log-likelihood
        ll_h = poisson.logpmf(home_g_weighted, mu_h)
        ll_a = poisson.logpmf(away_g_weighted, mu_a)
        ll_total = np.sum(combined_weights * (ll_h + ll_a))

        # light regularisation on features
        reg_features = 0.01 * (beta_odds**2 + beta_form**2)

        return -ll_total + reg_features

    # ========================================================================
    # OPTIMISATION
    # ========================================================================

    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: FITTING FEATURE COEFFICIENTS")
        print("=" * 60)
        print("  (Team strengths fixed from stage 1)")

    bounds = [
        (0.0, 1.5),  # beta_odds
        (-0.5, 0.5),  # beta_form
    ]

    x0 = np.array([0.5, 0.1])

    result = minimize(
        neg_loglik_features,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    beta_odds = result.x[0]
    beta_form = result.x[1] if use_form else 0.0

    # ========================================================================
    # COMBINE RESULTS
    # ========================================================================

    full_params = baseline_params.copy()
    full_params["beta_odds"] = beta_odds
    full_params["beta_form"] = beta_form
    full_params["log_likelihood_features"] = -result.fun

    # add calibration parameters
    home_g_actual = df_train["home_goals"].fillna(0).astype(int).values
    away_g_actual = df_train["away_goals"].fillna(0).astype(int).values

    # recalculate lambdas with features
    home_strength = (
        home_adv_baseline
        + attack[home_idx]
        + defense[away_idx]
        + beta_odds * home_log_odds_ratio
        + beta_form * home_form_w5
    )
    away_strength = (
        attack[away_idx]
        + defense[home_idx]
        - beta_odds * home_log_odds_ratio
        + beta_form * away_form_w5
    )

    lambda_h_fitted = np.exp(home_strength)
    lambda_a_fitted = np.exp(away_strength)

    # calculate dispersion
    residuals_h = home_g_actual - lambda_h_fitted
    residuals_a = away_g_actual - lambda_a_fitted

    var_h = np.var(residuals_h)
    var_a = np.var(residuals_a)
    mean_h = np.mean(home_g_actual)
    mean_a = np.mean(away_g_actual)

    dispersion_h = var_h / mean_h if mean_h > 0 else 1.0
    dispersion_a = var_a / mean_a if mean_a > 0 else 1.0
    dispersion_factor = (dispersion_h + dispersion_a) / 2

    full_params["dispersion_factor"] = dispersion_factor
    full_params["var_ratio_h"] = var_h / max(
        np.var(home_g_actual - lambda_h_fitted), 0.1
    )
    full_params["var_ratio_a"] = var_a / max(
        np.var(away_g_actual - lambda_a_fitted), 0.1
    )

    if verbose:
        print("\n  ✓ Feature coefficients fitted")
        print(f"    Beta (odds): {beta_odds:.3f}")
        print(f"    Beta (form): {beta_form:.3f}")
        print(f"    Dispersion factor: {dispersion_factor:.3f}")

    full_params = add_interpretable_ratings_to_params(full_params)

    return full_params


def fit_poisson_model_two_stage(
    df_train: pd.DataFrame,
    hyperparams: Dict[str, float],
    promoted_priors: Optional[Dict[str, Dict[str, float]]] = None,
    home_adv_prior: Optional[float] = None,
    home_adv_std: Optional[float] = None,
    n_random_starts: int = 3,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Two-stage Poisson model fitting.

    Stage 1: Baseline team strengths (team identity + home advantage only).
    Stage 2: Feature coefficients (odds, form) using fixed baseline strengths.
    """
    # stage 1: baseline
    baseline_params = fit_baseline_strengths(
        df_train=df_train,
        hyperparams=hyperparams,
        promoted_priors=promoted_priors,
        home_adv_prior=home_adv_prior,
        home_adv_std=home_adv_std,
        n_random_starts=n_random_starts,
        verbose=verbose,
    )

    if baseline_params is None:
        return None

    # stage 2: features
    full_params = fit_feature_coefficients(
        df_train=df_train,
        baseline_params=baseline_params,
        hyperparams=hyperparams,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("TWO-STAGE MODEL COMPLETE")
        print("=" * 60)
        print("✓ Team strengths reflect baseline ability (no feature contamination)")
        print("✓ Feature coefficients act as match-specific adjustments")
        print("\nFinal parameters:")
        print(f"  Home advantage: {full_params['home_adv']:.3f}")
        print(f"  Odds weight: {full_params['beta_odds']:.3f}")
        print(f"  Form weight: {full_params['beta_form']:.3f}")
        print(f"  Rho: {full_params['rho']:.3f}")
        print(f"  Dispersion factor: {full_params['dispersion_factor']:.3f}")

    return full_params


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def calculate_lambdas_single(
    home_team: str,
    away_team: str,
    params: Dict[str, Any],
    home_log_odds_ratio: float = 0.0,
    home_form_w5: float = 0.0,
    away_form_w5: float = 0.0,
) -> Tuple[float, float]:
    """Calculate expected goals for a single match"""
    att_h = params.get("attack", {}).get(home_team, 0.0)
    def_h = params.get("defense", {}).get(home_team, 0.0)
    att_a = params.get("attack", {}).get(away_team, 0.0)
    def_a = params.get("defense", {}).get(away_team, 0.0)

    home_adv = params.get("home_adv", 0.0)
    beta_odds = params.get("beta_odds", 0.0)
    beta_form = params.get("beta_form", 0.0)

    # calculate strengths
    home_strength = (
        att_h
        + def_a
        + home_adv
        + beta_odds * home_log_odds_ratio
        + beta_form * home_form_w5
    )
    away_strength = (
        att_a + def_h - beta_odds * home_log_odds_ratio + beta_form * away_form_w5
    )

    # convert to lambdas
    lambda_home = np.clip(np.exp(home_strength), 0.1, 8.0)
    lambda_away = np.clip(np.exp(away_strength), 0.1, 8.0)

    return lambda_home, lambda_away


def calculate_lambdas(
    df: pd.DataFrame, params: Dict[str, Any], fill_missing_with_mean: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate expected goals (lambdas) for matches using fitted parameters"""
    # get all teams (from data and parameters)
    df_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    param_teams = set(params.get("teams", []))
    all_teams = sorted(df_teams | param_teams)

    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    # extract parameter arrays
    attack_vals = [params.get("attack", {}).get(t, np.nan) for t in all_teams]
    defense_vals = [params.get("defense", {}).get(t, np.nan) for t in all_teams]

    attack_arr = np.array(attack_vals, dtype=float)
    defense_arr = np.array(defense_vals, dtype=float)

    # handle missing teams
    if fill_missing_with_mean:
        if np.isnan(attack_arr).any():
            mean_att = np.nanmean(attack_arr)
            attack_arr = np.where(np.isnan(attack_arr), mean_att, attack_arr)
        if np.isnan(defense_arr).any():
            mean_def = np.nanmean(defense_arr)
            defense_arr = np.where(np.isnan(defense_arr), mean_def, defense_arr)
    else:
        attack_arr = np.nan_to_num(attack_arr, nan=0.0)
        defense_arr = np.nan_to_num(defense_arr, nan=0.0)

    # map teams to indices
    home_idx = np.array([team_to_idx[t] for t in df["home_team"]], dtype=np.int64)
    away_idx = np.array([team_to_idx[t] for t in df["away_team"]], dtype=np.int64)

    # extract features
    home_log_odds_ratio = (
        df["home_log_odds_ratio"].fillna(0).values
        if "home_log_odds_ratio" in df
        else np.zeros(len(df))
    )

    # form features
    home_form_w5 = (
        df["home_npxgd_w5"].fillna(0).values
        if "home_npxgd_w5" in df
        else np.zeros(len(df))
    )
    away_form_w5 = (
        df["away_npxgd_w5"].fillna(0).values
        if "away_npxgd_w5" in df
        else np.zeros(len(df))
    )

    # calculate strengths
    home_strength = (
        attack_arr[home_idx]
        + defense_arr[away_idx]
        + params.get("home_adv", 0.0)
        + params.get("beta_odds", 0.0) * home_log_odds_ratio
        + params.get("beta_form", 0.0) * home_form_w5
    )
    away_strength = (
        attack_arr[away_idx]
        + defense_arr[home_idx]
        - params.get("beta_odds", 0.0) * home_log_odds_ratio
        + params.get("beta_form", 0.0) * away_form_w5
    )

    # convert to lambdas (expected goals)
    lambda_home = np.clip(np.exp(home_strength), 0.1, 8.0)
    lambda_away = np.clip(np.exp(away_strength), 0.1, 8.0)

    return lambda_home, lambda_away
