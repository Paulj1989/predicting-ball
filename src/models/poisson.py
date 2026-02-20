# src/models/poisson.py

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# ============================================================================
# HELPERS
# ============================================================================


def _compute_weighted_goals(
    df: pd.DataFrame, xg_weight: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute blended weighted goals from raw columns when available, else pre-computed column"""
    if all(c in df.columns for c in ("home_npxg", "away_npxg", "home_npg", "away_npg")):
        home_g = (
            xg_weight * df["home_npxg"].fillna(0).values
            + (1 - xg_weight) * df["home_npg"].fillna(0).values
        )
        away_g = (
            xg_weight * df["away_npxg"].fillna(0).values
            + (1 - xg_weight) * df["away_npg"].fillna(0).values
        )
    else:
        home_g = df["home_goals_weighted"].fillna(0).values
        away_g = df["away_goals_weighted"].fillna(0).values
    return home_g, away_g


# ============================================================================
# TWO-STAGE FITTING FUNCTIONS
# ============================================================================


def fit_baseline_strengths(
    df_train: pd.DataFrame,
    hyperparams: dict[str, float],
    promoted_priors: dict[str, dict[str, float]] | None = None,
    home_adv_prior: float | None = None,
    home_adv_std: float | None = None,
    n_random_starts: int = 3,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Stage 1: Fit baseline team strengths (attack/defense) + home advantage"""

    # ========================================================================
    # SETUP
    # ========================================================================

    all_teams = sorted(pd.unique(df_train[["home_team", "away_team"]].values.ravel()))

    if promoted_priors:
        for team in promoted_priors:
            if team not in all_teams:
                all_teams.append(team)
        all_teams = sorted(all_teams)

    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    home_idx = df_train["home_team"].map(team_to_idx).values
    away_idx = df_train["away_team"].map(team_to_idx).values

    xg_weight = hyperparams.get("xg_weight", 0.7)
    home_g_weighted, away_g_weighted = _compute_weighted_goals(df_train, xg_weight)

    # actual goals for tau correction (dixon-coles designed for discrete goal counts)
    home_g_actual = df_train["home_goals"].fillna(0).astype(int).values
    away_g_actual = df_train["away_goals"].fillna(0).astype(int).values

    # pre-compute outcome masks for vectorised tau (static across optimizer calls)
    dc_mask_00 = (home_g_actual == 0) & (away_g_actual == 0)
    dc_mask_01 = (home_g_actual == 0) & (away_g_actual == 1)
    dc_mask_10 = (home_g_actual == 1) & (away_g_actual == 0)
    dc_mask_11 = (home_g_actual == 1) & (away_g_actual == 1)

    # ========================================================================
    # PRIORS
    # ========================================================================

    attack_priors = np.zeros(n_teams)
    defense_priors = np.zeros(n_teams)

    current_season_year = df_train["season_end_year"].max()

    current_season_matches = np.zeros(n_teams)
    total_matches_played = np.zeros(n_teams)
    for i, team in enumerate(all_teams):
        team_mask = (df_train["home_team"] == team) | (df_train["away_team"] == team)
        total_matches_played[i] = team_mask.sum()
        current_season_matches[i] = (
            df_train.loc[team_mask, "season_end_year"] == current_season_year
        ).sum()

    prior_decay_rate = hyperparams.get("prior_decay_rate", 10.0)
    base_prior_weight = hyperparams.get("lambda_reg", 0.3)
    # prior erodes with current-season matches so it starts strong at matchweek 1
    # and fades naturally over the season — total_matches_played is used only to
    # identify teams with no bundesliga history (truly new/promoted sides)
    prior_weights = base_prior_weight / (1 + current_season_matches / prior_decay_rate)

    if promoted_priors:
        for team, priors in promoted_priors.items():
            # skip metadata keys (not actual teams)
            if team.startswith("_"):
                continue

            if team in team_to_idx:
                idx = team_to_idx[team]
                attack_priors[idx] = priors["attack_prior"]
                defense_priors[idx] = priors["defense_prior"]

            if total_matches_played[idx] == 0:
                prior_weights[idx] = base_prior_weight * 1000
            elif total_matches_played[idx] < 5:
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

    # ========================================================================
    # OBJECTIVE FUNCTION (WITH DIXON-COLES)
    # ========================================================================

    def neg_loglik(x: np.ndarray) -> float:
        """Negative log-likelihood with Dixon-Coles correction"""
        attack = x[:n_teams]
        defense = x[n_teams : 2 * n_teams]
        home_adv = x[2 * n_teams]
        rho = x[2 * n_teams + 1]

        # identifiability constraint
        attack = attack - attack.mean()
        defense = defense - defense.mean()

        # calculate lambdas (baseline only: team + home advantage)
        home_strength = home_adv + attack[home_idx] + defense[away_idx]
        away_strength = attack[away_idx] + defense[home_idx]

        mu_h = np.clip(np.exp(home_strength), 0.1, 8.0)
        mu_a = np.clip(np.exp(away_strength), 0.1, 8.0)

        # vectorised tau correction on actual goals — weighted goals distort the
        # discrete outcome distribution and bias rho if used here
        tau_vec = np.ones(len(home_g_actual))
        tau_vec[dc_mask_00] = 1 - mu_h[dc_mask_00] * mu_a[dc_mask_00] * rho
        tau_vec[dc_mask_01] = 1 + mu_h[dc_mask_01] * rho
        tau_vec[dc_mask_10] = 1 + mu_a[dc_mask_10] * rho
        tau_vec[dc_mask_11] = 1 - rho

        # continuous poisson deviance: valid for non-integer k (weighted goals)
        ll_h = (
            home_g_weighted * np.log(np.maximum(mu_h, 1e-10))
            - mu_h
            - gammaln(home_g_weighted + 1)
        )
        ll_a = (
            away_g_weighted * np.log(np.maximum(mu_a, 1e-10))
            - mu_a
            - gammaln(away_g_weighted + 1)
        )

        # apply dixon-coles correction: log(p_base * tau) = log(p_base) + log(tau)
        ll_match = ll_h + ll_a + np.log(np.maximum(tau_vec, 1e-10))
        ll_total = np.sum(time_weights * ll_match)

        # regularisation: priors on attack/defense
        prior_penalty = np.sum(
            prior_weights * ((attack - attack_priors) ** 2 + (defense - defense_priors) ** 2)
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
    bounds.append((-0.4, 0.0))  # rho (negative: draws more common than poisson predicts)

    best_result = None
    best_nll = float("inf")

    for restart in range(n_random_starts):
        if restart == 0:
            # start from zero (with promoted priors if available)
            x0 = np.zeros(2 * n_teams + 2)
            if promoted_priors:
                for team, priors in promoted_priors.items():
                    # skip metadata keys
                    if team.startswith("_"):
                        continue

                    if team in team_to_idx:
                        idx = team_to_idx[team]
                        x0[idx] = priors["attack_prior"]
                        x0[n_teams + idx] = priors["defense_prior"]
            x0[-2] = home_adv_prior if home_adv_prior else 0.30
            x0[-1] = -0.13  # start from literature value
        else:
            # random starts for team ratings; keep home_adv and rho initialised sensibly
            x0 = np.random.randn(2 * n_teams + 2) * 0.3
            x0[-2] = home_adv_prior if home_adv_prior else 0.30
            x0[-1] = -0.13

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
    rho = best_result.x[2 * n_teams + 1]

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
        "rho": float(rho),
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
    baseline_params: dict[str, Any],
    hyperparams: dict[str, float],
    blend_holdout_df: pd.DataFrame | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Stage 2: Fit form residual coefficient and odds blend weight.

    When blend_holdout_df is provided, the odds blend weight is fitted on that
    external data rather than df_train, preventing in-sample optimism in CV.
    """

    from .dixon_coles import calculate_match_probabilities_dixon_coles_batch
    from .ratings import add_interpretable_ratings_to_params

    # ========================================================================
    # SETUP
    # ========================================================================

    all_teams = baseline_params["teams"]
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    home_idx = df_train["home_team"].map(team_to_idx).values
    away_idx = df_train["away_team"].map(team_to_idx).values

    xg_weight = hyperparams.get("xg_weight", 0.7)
    home_g_weighted, away_g_weighted = _compute_weighted_goals(df_train, xg_weight)

    # actual goals for tau correction (consistent with stage 1)
    home_g_actual = df_train["home_goals"].fillna(0).astype(int).values
    away_g_actual = df_train["away_goals"].fillna(0).astype(int).values

    # pre-compute outcome masks for vectorised tau (static across optimizer calls)
    dc_mask_00 = (home_g_actual == 0) & (away_g_actual == 0)
    dc_mask_01 = (home_g_actual == 0) & (away_g_actual == 1)
    dc_mask_10 = (home_g_actual == 1) & (away_g_actual == 0)
    dc_mask_11 = (home_g_actual == 1) & (away_g_actual == 1)

    # extract baseline parameters
    attack = np.array([baseline_params["attack"][t] for t in all_teams])
    defense = np.array([baseline_params["defense"][t] for t in all_teams])
    home_adv_baseline = baseline_params["home_adv"]
    rho = baseline_params.get("rho", -0.13)

    # ========================================================================
    # FEATURES
    # ========================================================================

    home_npxgd_w5 = (
        df_train["home_npxgd_w5"].fillna(0).values
        if "home_npxgd_w5" in df_train
        else np.zeros(len(df_train))
    )
    away_npxgd_w5 = (
        df_train["away_npxgd_w5"].fillna(0).values
        if "away_npxgd_w5" in df_train
        else np.zeros(len(df_train))
    )

    # ========================================================================
    # BASELINE LAMBDAS AND RESIDUALS
    # ========================================================================

    baseline_home = home_adv_baseline + attack[home_idx] + defense[away_idx]
    baseline_away = attack[away_idx] + defense[home_idx]

    baseline_lambda_h = np.exp(np.clip(baseline_home, -3, 3))
    baseline_lambda_a = np.exp(np.clip(baseline_away, -3, 3))

    expected_npxgd_home = baseline_lambda_h - baseline_lambda_a
    expected_npxgd_away = baseline_lambda_a - baseline_lambda_h

    home_npxgd_residual = home_npxgd_w5 - expected_npxgd_home
    away_npxgd_residual = away_npxgd_w5 - expected_npxgd_away

    # ========================================================================
    # TIME WEIGHTING
    # ========================================================================

    lambda_decay = hyperparams.get("time_decay", 0.001)
    time_weights = np.exp(
        -lambda_decay * (df_train["date"].max() - df_train["date"]).dt.days.values
    )

    combined_weights = time_weights

    # ========================================================================
    # OBJECTIVE: FORM RESIDUAL COEFFICIENT
    # ========================================================================

    def neg_loglik_features(x: np.ndarray) -> float:
        """Fit form residual coefficient with fixed team strengths"""
        beta_form = x[0]

        # final strength with form residual adjustment
        home_strength = baseline_home + beta_form * home_npxgd_residual
        away_strength = baseline_away + beta_form * away_npxgd_residual

        mu_h = np.clip(np.exp(home_strength), 0.1, 8.0)
        mu_a = np.clip(np.exp(away_strength), 0.1, 8.0)

        # continuous poisson deviance: valid for non-integer k (weighted goals)
        ll_h = (
            home_g_weighted * np.log(np.maximum(mu_h, 1e-10))
            - mu_h
            - gammaln(home_g_weighted + 1)
        )
        ll_a = (
            away_g_weighted * np.log(np.maximum(mu_a, 1e-10))
            - mu_a
            - gammaln(away_g_weighted + 1)
        )

        # vectorised tau correction on actual goals, consistent with stage 1
        tau_vec = np.ones(len(home_g_actual))
        tau_vec[dc_mask_00] = 1 - mu_h[dc_mask_00] * mu_a[dc_mask_00] * rho
        tau_vec[dc_mask_01] = 1 + mu_h[dc_mask_01] * rho
        tau_vec[dc_mask_10] = 1 + mu_a[dc_mask_10] * rho
        tau_vec[dc_mask_11] = 1 - rho
        ll_total = np.sum(
            combined_weights * (ll_h + ll_a + np.log(np.maximum(tau_vec, 1e-10)))
        )

        # light regularisation
        reg_features = 0.01 * beta_form**2

        return -ll_total + reg_features

    # ========================================================================
    # OPTIMISATION: FORM COEFFICIENT
    # ========================================================================

    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: FITTING FEATURE COEFFICIENTS")
        print("=" * 60)
        print("  (Team strengths fixed from stage 1)")

    bounds = [(-0.5, 0.5)]  # beta_form
    x0 = np.array([0.1])

    result = minimize(
        neg_loglik_features,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    beta_form = result.x[0]

    # ========================================================================
    # ODDS BLEND WEIGHT
    # ========================================================================

    # use holdout data for blend fitting when provided to avoid in-sample optimism
    blend_data = blend_holdout_df if blend_holdout_df is not None else df_train

    # compute lambdas for blend_data using fitted team ratings + beta_form
    # identify unseen teams and substitute mean ratings rather than index-0 team
    known_mask_h = np.array([t in team_to_idx for t in blend_data["home_team"]])
    known_mask_a = np.array([t in team_to_idx for t in blend_data["away_team"]])
    blend_home_idx = np.array([team_to_idx.get(t, 0) for t in blend_data["home_team"]])
    blend_away_idx = np.array([team_to_idx.get(t, 0) for t in blend_data["away_team"]])

    blend_base_home = home_adv_baseline + attack[blend_home_idx] + defense[blend_away_idx]
    blend_base_away = attack[blend_away_idx] + defense[blend_home_idx]

    # substitute mean attack/defense for rows where either team is unseen
    mean_att = attack.mean()
    mean_def = defense.mean()
    blend_base_home = np.where(
        known_mask_h & known_mask_a,
        blend_base_home,
        home_adv_baseline + mean_att + mean_def,
    )
    blend_base_away = np.where(
        known_mask_h & known_mask_a,
        blend_base_away,
        mean_att + mean_def,
    )

    blend_bl_h = np.exp(np.clip(blend_base_home, -3, 3))
    blend_bl_a = np.exp(np.clip(blend_base_away, -3, 3))

    blend_npxgd_home = (
        blend_data["home_npxgd_w5"].fillna(0).values
        if "home_npxgd_w5" in blend_data.columns
        else np.zeros(len(blend_data))
    )
    blend_npxgd_away = (
        blend_data["away_npxgd_w5"].fillna(0).values
        if "away_npxgd_w5" in blend_data.columns
        else np.zeros(len(blend_data))
    )

    blend_exp_npxgd_home = blend_bl_h - blend_bl_a
    blend_exp_npxgd_away = blend_bl_a - blend_bl_h

    blend_resid_h = blend_npxgd_home - blend_exp_npxgd_home
    blend_resid_a = blend_npxgd_away - blend_exp_npxgd_away

    blend_lambda_h = np.clip(np.exp(blend_base_home + beta_form * blend_resid_h), 0.1, 8.0)
    blend_lambda_a = np.clip(np.exp(blend_base_away + beta_form * blend_resid_a), 0.1, 8.0)

    # compute model probabilities from blend data lambdas (vectorised)
    n_blend = len(blend_data)
    model_probs = calculate_match_probabilities_dixon_coles_batch(
        blend_lambda_h, blend_lambda_a, rho=rho
    )

    # extract odds-implied probabilities from blend data
    odds_home = (
        blend_data["odds_home_prob"].values
        if "odds_home_prob" in blend_data.columns
        else np.full(n_blend, np.nan)
    )
    odds_draw = (
        blend_data["odds_draw_prob"].values
        if "odds_draw_prob" in blend_data.columns
        else np.full(n_blend, np.nan)
    )
    odds_away = (
        blend_data["odds_away_prob"].values
        if "odds_away_prob" in blend_data.columns
        else np.full(n_blend, np.nan)
    )
    odds_probs = np.column_stack([odds_home, odds_draw, odds_away])

    # mask for matches with valid odds
    has_odds = ~np.isnan(odds_probs).any(axis=1)

    # time weights for blend data
    lambda_decay = hyperparams.get("time_decay", 0.001)
    blend_time_weights = np.exp(
        -lambda_decay * (blend_data["date"].max() - blend_data["date"]).dt.days.values
    )

    if has_odds.sum() > 10:
        # actual outcomes for RPS calculation
        results = blend_data["result"].values
        actual_matrix = np.zeros((n_blend, 3))
        for i, r in enumerate(results):
            if r == "H":
                actual_matrix[i] = [1, 0, 0]
            elif r == "D":
                actual_matrix[i] = [0, 1, 0]
            elif r == "A":
                actual_matrix[i] = [0, 0, 1]

        odds_weights = blend_time_weights[has_odds]
        model_p = model_probs[has_odds]
        odds_p = odds_probs[has_odds]
        actual_m = actual_matrix[has_odds]

        from scipy.optimize import minimize_scalar

        def blend_rps(w):
            """Time-weighted mean RPS for blended probabilities"""
            blended = w * model_p + (1 - w) * odds_p
            # rps for 3-outcome ordered categories
            cum_pred = np.cumsum(blended, axis=1)
            cum_actual = np.cumsum(actual_m, axis=1)
            rps_per_match = np.sum((cum_pred - cum_actual) ** 2, axis=1) / 2
            return np.average(rps_per_match, weights=odds_weights)

        blend_result = minimize_scalar(blend_rps, bounds=(0.0, 1.0), method="bounded")
        odds_blend_weight = float(blend_result.x)
    else:
        # not enough odds data, use model only
        odds_blend_weight = 1.0

    # ========================================================================
    # COMBINE RESULTS
    # ========================================================================

    full_params = baseline_params.copy()
    full_params["beta_form"] = beta_form
    full_params["odds_blend_weight"] = odds_blend_weight
    full_params["log_likelihood_features"] = -result.fun

    # diagnostic: pearson dispersion statistic (var(residuals) / mean(goals))
    # for true poisson, this should be close to 1.0; materially above 1.2
    # may indicate the model is missing structure worth investigating
    home_g_actual = df_train["home_goals"].fillna(0).astype(int).values
    away_g_actual = df_train["away_goals"].fillna(0).astype(int).values

    # recompute train-set lambdas for dispersion diagnostic (always on df_train)
    home_strength_final = baseline_home + beta_form * home_npxgd_residual
    away_strength_final = baseline_away + beta_form * away_npxgd_residual
    lambda_h_fitted = np.clip(np.exp(home_strength_final), 0.1, 8.0)
    lambda_a_fitted = np.clip(np.exp(away_strength_final), 0.1, 8.0)

    residuals_h = home_g_actual - lambda_h_fitted
    residuals_a = away_g_actual - lambda_a_fitted

    mean_h = np.mean(home_g_actual)
    mean_a = np.mean(away_g_actual)

    dispersion_h = np.var(residuals_h) / mean_h if mean_h > 0 else 1.0
    dispersion_a = np.var(residuals_a) / mean_a if mean_a > 0 else 1.0
    dispersion_factor = (dispersion_h + dispersion_a) / 2

    full_params["dispersion_factor"] = dispersion_factor

    if verbose:
        print("\n  ✓ Feature coefficients fitted")
        print(f"    Beta (form): {beta_form:.3f}")
        print(f"    Odds blend weight: {odds_blend_weight:.3f}")
        disp_flag = " ⚠ (> 1.2 — consider investigating)" if dispersion_factor > 1.2 else ""
        print(f"    Dispersion (diagnostic): {dispersion_factor:.3f}{disp_flag}")

    full_params = add_interpretable_ratings_to_params(full_params)

    return full_params


def fit_poisson_model_two_stage(
    df_train: pd.DataFrame,
    hyperparams: dict[str, float],
    promoted_priors: dict[str, dict[str, float]] | None = None,
    home_adv_prior: float | None = None,
    home_adv_std: float | None = None,
    n_random_starts: int = 3,
    blend_holdout_df: pd.DataFrame | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """
    Two-stage Poisson model fitting.

    Stage 1: Baseline team strengths (team identity + home advantage only).
    Stage 2: Form residual coefficient + odds blend weight using fixed baseline strengths.

    When blend_holdout_df is provided it is passed through to fit_feature_coefficients
    to fit the odds blend weight on external data, preventing in-sample optimism in CV.
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
        blend_holdout_df=blend_holdout_df,
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
        print(f"  Form weight: {full_params['beta_form']:.3f}")
        print(f"  Odds blend: {full_params['odds_blend_weight']:.3f}")
        print(f"  Rho: {full_params['rho']:.3f}")
        disp = full_params["dispersion_factor"]
        disp_flag = " ⚠" if disp > 1.2 else ""
        print(f"  Dispersion (diagnostic): {disp:.3f}{disp_flag}")

    return full_params


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def calculate_lambdas_single(
    home_team: str,
    away_team: str,
    params: dict[str, Any],
    home_npxgd_w5: float = 0.0,
    away_npxgd_w5: float = 0.0,
) -> tuple[float, float]:
    """Calculate expected goals for a single match"""
    att_h = params.get("attack", {}).get(home_team, 0.0)
    def_h = params.get("defense", {}).get(home_team, 0.0)
    att_a = params.get("attack", {}).get(away_team, 0.0)
    def_a = params.get("defense", {}).get(away_team, 0.0)

    home_adv = params.get("home_adv", 0.0)
    beta_form = params.get("beta_form", 0.0)

    # baseline strength (no features)
    baseline_home = att_h + def_a + home_adv
    baseline_away = att_a + def_h

    # expected npxgd from baseline lambdas
    baseline_lambda_h = np.exp(np.clip(baseline_home, -3, 3))
    baseline_lambda_a = np.exp(np.clip(baseline_away, -3, 3))
    expected_npxgd_home = baseline_lambda_h - baseline_lambda_a
    expected_npxgd_away = baseline_lambda_a - baseline_lambda_h

    # residual: actual form minus expected
    home_residual = home_npxgd_w5 - expected_npxgd_home
    away_residual = away_npxgd_w5 - expected_npxgd_away

    # final strength with form residual adjustment
    home_strength = baseline_home + beta_form * home_residual
    away_strength = baseline_away + beta_form * away_residual

    lambda_home = np.clip(np.exp(home_strength), 0.1, 8.0)
    lambda_away = np.clip(np.exp(away_strength), 0.1, 8.0)

    return lambda_home, lambda_away


def calculate_lambdas(
    df: pd.DataFrame, params: dict[str, Any], fill_missing_with_mean: bool = False
) -> tuple[np.ndarray, np.ndarray]:
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

    # form features
    home_npxgd_w5 = (
        df["home_npxgd_w5"].fillna(0).values if "home_npxgd_w5" in df else np.zeros(len(df))
    )
    away_npxgd_w5 = (
        df["away_npxgd_w5"].fillna(0).values if "away_npxgd_w5" in df else np.zeros(len(df))
    )

    home_adv = params.get("home_adv", 0.0)
    beta_form = params.get("beta_form", 0.0)

    # baseline strength (no features)
    baseline_home = attack_arr[home_idx] + defense_arr[away_idx] + home_adv
    baseline_away = attack_arr[away_idx] + defense_arr[home_idx]

    # expected npxgd from baseline lambdas
    baseline_lambda_h = np.exp(np.clip(baseline_home, -3, 3))
    baseline_lambda_a = np.exp(np.clip(baseline_away, -3, 3))
    expected_npxgd_home = baseline_lambda_h - baseline_lambda_a
    expected_npxgd_away = baseline_lambda_a - baseline_lambda_h

    # residual: actual form minus expected
    home_residual = home_npxgd_w5 - expected_npxgd_home
    away_residual = away_npxgd_w5 - expected_npxgd_away

    # final strength with form residual adjustment
    home_strength = baseline_home + beta_form * home_residual
    away_strength = baseline_away + beta_form * away_residual

    # convert to lambdas (expected goals)
    lambda_home = np.clip(np.exp(home_strength), 0.1, 8.0)
    lambda_away = np.clip(np.exp(away_strength), 0.1, 8.0)

    return lambda_home, lambda_away
