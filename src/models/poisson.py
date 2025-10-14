# src/models/poisson.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, Optional, Tuple, Any


def fit_poisson_model(
    df_train: pd.DataFrame,
    hyperparams: Dict[str, float],
    promoted_priors: Optional[Dict[str, Dict[str, float]]] = None,
    home_adv_prior: Optional[float] = None,
    home_adv_std: Optional[float] = None,
    n_random_starts: int = 3,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fit hybrid Poisson model on training data.

    The model fits on weighted performance metrics (npxG/npG composite) which
    provides stable signals for team strength estimation, then calibrates for
    actual discrete goals to ensure proper uncertainty quantification.
    """
    # ========================================================================
    # SETUP: Teams and Indices
    # ========================================================================
    all_teams = sorted(pd.unique(df_train[["home_team", "away_team"]].values.ravel()))

    # add promoted teams that might not appear in training data
    if promoted_priors:
        for team in promoted_priors.keys():
            if team not in all_teams:
                all_teams.append(team)
        all_teams = sorted(all_teams)

    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    # map teams to indices
    home_idx = df_train["home_team"].map(team_to_idx).values
    away_idx = df_train["away_team"].map(team_to_idx).values

    # fit on weighted performance (stable signal for parameters)
    home_g_weighted = np.round(df_train["home_goals_weighted"]).astype(int).values
    away_g_weighted = np.round(df_train["away_goals_weighted"]).astype(int).values

    # keep actual goals for calibration
    home_g_actual = df_train["home_goals"].fillna(0).astype(int).values
    away_g_actual = df_train["away_goals"].fillna(0).astype(int).values

    # ========================================================================
    # FEATURES
    # ========================================================================
    home_log_odds = (
        df_train["home_log_odds"].fillna(0).values
        if "home_log_odds" in df_train
        else np.zeros(len(df_train))
    )

    home_pens_att = df_train["home_pens_att"].fillna(0).values
    away_pens_att = df_train["away_pens_att"].fillna(0).values

    # ========================================================================
    # PRIORS: Team Attack/Defense
    # ========================================================================
    attack_priors = np.zeros(n_teams)
    defense_priors = np.zeros(n_teams)

    # calculate matches played per team for prior decay
    matches_played = np.zeros(n_teams)
    for i, team in enumerate(all_teams):
        team_matches = df_train[
            (df_train["home_team"] == team) | (df_train["away_team"] == team)
        ]
        matches_played[i] = len(team_matches)

    # decaying prior weights: base_weight / (1 + matches_played / decay_rate)
    prior_decay_rate = hyperparams.get("prior_decay_rate", 10.0)
    base_prior_weight = hyperparams.get("lambda_reg", 0.3)
    prior_weights = base_prior_weight / (1 + matches_played / prior_decay_rate)

    # set priors for promoted teams
    if promoted_priors:
        for team, priors in promoted_priors.items():
            if team in team_to_idx:
                idx = team_to_idx[team]
                attack_priors[idx] = priors["attack_prior"]
                defense_priors[idx] = priors["defense_prior"]

                # promoted teams with no/little data get very strong priors
                if matches_played[idx] == 0:
                    prior_weights[idx] = base_prior_weight * 1000
                    if verbose:
                        print(f"  {team}: No training data, using strong prior")
                elif matches_played[idx] < 5:
                    prior_weights[idx] = base_prior_weight * 10
                    if verbose:
                        print(
                            f"  {team}: Only {int(matches_played[idx])} matches, using strong prior"
                        )

    # ========================================================================
    # PRIOR: Home Advantage
    # ========================================================================
    if home_adv_prior is not None and home_adv_std is not None:
        # weight based on precision (inverse variance)
        lambda_home_adv = 1.0 / (home_adv_std**2)
        use_home_prior = True
        if verbose:
            print(
                f"  Using home advantage prior: {home_adv_prior:.3f} ± {home_adv_std:.3f}"
            )
            print(f"  Prior weight (precision): {lambda_home_adv:.2f}")
    else:
        home_adv_prior = 0.30  # default fallback
        lambda_home_adv = 0.0  # no regularisation
        use_home_prior = False
        if verbose:
            print(
                f"  Using default home advantage prior: {home_adv_prior:.3f} (no regularisation)"
            )

    # ========================================================================
    # PARAMETER BOUNDS
    # ========================================================================
    bounds = []
    for _ in range(n_teams):
        bounds.append((-3.0, 3.0))  # attack
    for _ in range(n_teams):
        bounds.append((-3.0, 3.0))  # defense
    bounds.extend(
        [
            (0.05, 0.5),  # home_adv
            (0.0, 1.5),  # beta_odds
            (-1.0, 1.0),  # beta_penalty
        ]
    )

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
    # OBJECTIVE FUNCTION
    # ========================================================================
    def neg_loglik(x: np.ndarray) -> float:
        """
        Negative log-likelihood with regularisation.
        Fits on weighted performance, includes home advantage prior.
        """
        attack = x[:n_teams]
        defense = x[n_teams : 2 * n_teams]
        home_adv = x[2 * n_teams]
        beta_odds = x[2 * n_teams + 1]
        beta_penalty = x[2 * n_teams + 2]

        # identifiability constraint
        attack = attack - attack.mean()
        defense = defense - defense.mean()

        # calculate lambdas
        home_strength = (
            home_adv
            + attack[home_idx]
            + defense[away_idx]
            + beta_odds * home_log_odds
            + beta_penalty * home_pens_att
        )
        away_strength = (
            attack[away_idx]
            + defense[home_idx]
            - beta_odds * home_log_odds
            + beta_penalty * away_pens_att
        )

        mu_h = np.clip(np.exp(home_strength), 0.2, 10.0)
        mu_a = np.clip(np.exp(away_strength), 0.2, 10.0)

        # log-likelihood (fit on weighted performance)
        ll_h = poisson.logpmf(home_g_weighted, mu_h)
        ll_a = poisson.logpmf(away_g_weighted, mu_a)
        ll_total = np.sum(combined_weights * (ll_h + ll_a))

        # regularisation: team parameters
        reg = np.sum(prior_weights * (attack - attack_priors) ** 2) + np.sum(
            prior_weights * (defense - defense_priors) ** 2
        )

        # regularisation: home advantage prior
        if use_home_prior:
            reg += lambda_home_adv * (home_adv - home_adv_prior) ** 2

        return -ll_total + reg

    # ========================================================================
    # OPTIMISATION WITH MULTIPLE RANDOM STARTS
    # ========================================================================
    best_result = None
    best_ll = -np.inf

    for start in range(n_random_starts):
        np.random.seed(42 + start if start > 0 else 42)

        # initialise around priors
        attack_init = np.random.normal(attack_priors, 0.2, n_teams)
        defense_init = np.random.normal(defense_priors, 0.2, n_teams)
        attack_init -= attack_init.mean()
        defense_init -= defense_init.mean()

        # initialise home advantage near prior
        if use_home_prior:
            home_adv_init = np.random.normal(home_adv_prior, home_adv_std)
            home_adv_init = np.clip(home_adv_init, 0.05, 0.5)
        else:
            home_adv_init = 0.30

        x0 = np.concatenate(
            [
                attack_init,
                defense_init,
                [home_adv_init, 0.75, 0.2],  # home_adv, beta_odds, beta_penalty
            ]
        )

        res = minimize(
            neg_loglik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800, "ftol": 1e-6},
        )

        if res.success and -res.fun > best_ll:
            best_result = res
            best_ll = -res.fun
            if verbose:
                print(
                    f"  Random start {start + 1}/{n_random_starts}: LL = {best_ll:.2f}"
                )

    if best_result is None:
        if verbose:
            print("  ✗ Optimisation failed for all random starts")
        return None

    # ========================================================================
    # EXTRACT PARAMETERS
    # ========================================================================
    attack = best_result.x[:n_teams]
    defense = best_result.x[n_teams : 2 * n_teams]
    attack = attack - attack.mean()
    defense = defense - defense.mean()

    # ========================================================================
    # CALIBRATION: Estimate Variance Inflation
    # ========================================================================
    if verbose:
        print("\n  Analysing variance for calibration...")

    # get predicted lambdas based on fitted model
    home_strength_fitted = (
        best_result.x[2 * n_teams]  # home_adv
        + attack[home_idx]
        + defense[away_idx]
        + best_result.x[2 * n_teams + 1] * home_log_odds  # beta_odds
        + best_result.x[2 * n_teams + 2] * home_pens_att  # beta_penalty
    )
    away_strength_fitted = (
        attack[away_idx]
        + defense[home_idx]
        - best_result.x[2 * n_teams + 1] * home_log_odds
        + best_result.x[2 * n_teams + 2] * away_pens_att
    )

    lambda_h_fitted = np.exp(home_strength_fitted)
    lambda_a_fitted = np.exp(away_strength_fitted)

    # calculate residuals for both weighted and actual
    residuals_weighted_h = home_g_weighted - lambda_h_fitted
    residuals_weighted_a = away_g_weighted - lambda_a_fitted
    residuals_actual_h = home_g_actual - lambda_h_fitted
    residuals_actual_a = away_g_actual - lambda_a_fitted

    # estimate variance ratio
    var_weighted_h = np.var(residuals_weighted_h)
    var_weighted_a = np.var(residuals_weighted_a)
    var_actual_h = np.var(residuals_actual_h)
    var_actual_a = np.var(residuals_actual_a)

    # overdispersion parameter for negative binomial
    mean_h = np.mean(home_g_actual)
    mean_a = np.mean(away_g_actual)
    dispersion_h = var_actual_h / mean_h if mean_h > 0 else 1.0
    dispersion_a = var_actual_a / mean_a if mean_a > 0 else 1.0

    initial_dispersion = (dispersion_h + dispersion_a) / 2

    if verbose:
        print(
            f"    Weighted performance - Home var: {var_weighted_h:.3f}, Away var: {var_weighted_a:.3f}"
        )
        print(
            f"    Actual goals - Home var: {var_actual_h:.3f}, Away var: {var_actual_a:.3f}"
        )
        print(
            f"    Variance ratio (actual/weighted): Home {var_actual_h / max(var_weighted_h, 0.1):.3f}, Away {var_actual_a / max(var_weighted_a, 0.1):.3f}"
        )
        print(f"    Initial dispersion estimate: {initial_dispersion:.3f}")

    # ========================================================================
    # BUILD RESULT DICTIONARY
    # ========================================================================
    result_dict = {
        "teams": all_teams,
        "attack": dict(zip(all_teams, attack)),
        "defense": dict(zip(all_teams, defense)),
        "defense_scaled": dict(zip(all_teams, -defense)),
        "overall": dict(zip(all_teams, 0.5 * attack - 0.5 * defense)),
        "home_adv": best_result.x[2 * n_teams],
        "beta_odds": best_result.x[2 * n_teams + 1],
        "beta_penalty": best_result.x[2 * n_teams + 2],
        "success": best_result.success,
        "log_likelihood": -best_result.fun,
        # calibration parameters
        "dispersion_factor": initial_dispersion,
        "var_ratio_h": var_actual_h / max(var_weighted_h, 0.1),
        "var_ratio_a": var_actual_a / max(var_weighted_a, 0.1),
    }

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    if verbose:
        print(f"\n{'=' * 60}")
        print("MODEL FITTED SUCCESSFULLY")
        print(f"{'=' * 60}")
        print(f"Parameters:")
        print(f"  Home advantage: {result_dict['home_adv']:.3f}")
        if use_home_prior:
            deviation = result_dict["home_adv"] - home_adv_prior
            print(f"    Prior: {home_adv_prior:.3f}, Deviation: {deviation:+.3f}")
        print(f"  Odds weight: {result_dict['beta_odds']:.3f}")
        print(f"  Penalty weight: {result_dict['beta_penalty']:.3f}")
        print(f"Calibration:")
        print(f"  Dispersion factor: {result_dict['dispersion_factor']:.3f}")
        print(f"  Log-likelihood: {result_dict['log_likelihood']:.2f}")

    return result_dict


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
    home_log_odds = (
        df["home_log_odds"].fillna(0).values
        if "home_log_odds" in df
        else np.zeros(len(df))
    )

    home_pens_att = (
        df["home_pens_att"].fillna(0).values
        if "home_pens_att" in df
        else np.zeros(len(df))
    )
    away_pens_att = (
        df["away_pens_att"].fillna(0).values
        if "away_pens_att" in df
        else np.zeros(len(df))
    )

    # calculate strengths
    home_strength = (
        attack_arr[home_idx]
        + defense_arr[away_idx]
        + params.get("home_adv", 0.0)
        + params.get("beta_odds", 0.0) * home_log_odds
        + params.get("beta_penalty", 0.0) * home_pens_att
    )
    away_strength = (
        attack_arr[away_idx]
        + defense_arr[home_idx]
        - params.get("beta_odds", 0.0) * home_log_odds
        + params.get("beta_penalty", 0.0) * away_pens_att
    )

    # convert to lambdas (expected goals)
    lambda_home = np.clip(np.exp(home_strength), 0.2, 10.0)
    lambda_away = np.clip(np.exp(away_strength), 0.2, 10.0)

    return lambda_home, lambda_away
