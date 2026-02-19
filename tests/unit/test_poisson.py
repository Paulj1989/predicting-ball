# tests/unit/test_poisson.py
"""Tests for the Poisson model module."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.poisson import (
    calculate_lambdas,
    calculate_lambdas_single,
    fit_baseline_strengths,
    fit_poisson_model_two_stage,
)


class TestCalculateLambdasSingle:
    """Tests for single-match lambda calculation."""

    def test_returns_tuple_of_two_floats(self, sample_model_params):
        """Should return (lambda_home, lambda_away) tuple."""
        lh, la = calculate_lambdas_single("Bayern", "Dortmund", sample_model_params)
        assert isinstance(lh, float)
        assert isinstance(la, float)

    def test_lambdas_positive(self, sample_model_params):
        """Lambdas should always be positive."""
        lh, la = calculate_lambdas_single("Bayern", "Dortmund", sample_model_params)
        assert lh > 0
        assert la > 0

    def test_lambdas_clipped(self, sample_model_params):
        """Lambdas should be clipped to [0.1, 8.0]."""
        lh, la = calculate_lambdas_single("Bayern", "Dortmund", sample_model_params)
        assert 0.1 <= lh <= 8.0
        assert 0.1 <= la <= 8.0

    def test_home_advantage_increases_home_lambda(self, sample_model_params):
        """Home advantage should boost the home team's expected goals."""
        # with home_adv
        lh_with, _ = calculate_lambdas_single("Bayern", "Dortmund", sample_model_params)

        # without home_adv
        params_no_ha = sample_model_params.copy()
        params_no_ha["home_adv"] = 0.0
        lh_without, _ = calculate_lambdas_single("Bayern", "Dortmund", params_no_ha)

        assert lh_with > lh_without

    def test_form_residual_affects_lambda(self, sample_model_params):
        """Non-zero npxGD residual should affect lambdas when beta_form > 0."""
        # with zero form (baseline only)
        lh_base, la_base = calculate_lambdas_single("Bayern", "Dortmund", sample_model_params)
        # with strong positive home form (exceeding expected)
        lh_form, la_form = calculate_lambdas_single(
            "Bayern",
            "Dortmund",
            sample_model_params,
            home_npxgd_w5=3.0,
            away_npxgd_w5=-1.0,
        )
        # form residual should shift lambdas
        assert lh_form != lh_base or la_form != la_base

    def test_unknown_team_uses_zero(self, sample_model_params):
        """Unknown teams should get 0 attack/defense."""
        lh, la = calculate_lambdas_single("Unknown FC", "Bayern", sample_model_params)
        assert 0.1 <= lh <= 8.0
        assert 0.1 <= la <= 8.0

    @given(
        st.floats(min_value=-2.0, max_value=2.0),
        st.floats(min_value=-2.0, max_value=2.0),
    )
    @settings(max_examples=20)
    def test_lambdas_always_in_range(self, form_h, form_a):
        """Lambdas should always be in [0.1, 8.0] regardless of features."""
        params = {
            "attack": {"A": 0.3, "B": -0.3},
            "defense": {"A": -0.2, "B": 0.2},
            "home_adv": 0.25,
            "beta_form": 0.1,
        }
        lh, la = calculate_lambdas_single(
            "A",
            "B",
            params,
            home_npxgd_w5=form_h,
            away_npxgd_w5=form_a,
        )
        assert 0.1 <= lh <= 8.0
        assert 0.1 <= la <= 8.0


class TestCalculateLambdas:
    """Tests for vectorised lambda calculation."""

    def test_returns_arrays(self, sample_model_params):
        """Should return two numpy arrays."""
        df = pd.DataFrame(
            {
                "home_team": ["Bayern", "Dortmund"],
                "away_team": ["Dortmund", "Bayern"],
            }
        )
        lh, la = calculate_lambdas(df, sample_model_params)
        assert isinstance(lh, np.ndarray)
        assert isinstance(la, np.ndarray)
        assert len(lh) == 2
        assert len(la) == 2

    def test_consistent_with_single(self, sample_model_params):
        """Vectorised results should match single-match calculation."""
        df = pd.DataFrame(
            {
                "home_team": ["Bayern"],
                "away_team": ["Dortmund"],
            }
        )
        lh_vec, la_vec = calculate_lambdas(df, sample_model_params)
        lh_single, la_single = calculate_lambdas_single(
            "Bayern", "Dortmund", sample_model_params
        )
        assert np.isclose(lh_vec[0], lh_single, atol=1e-6)
        assert np.isclose(la_vec[0], la_single, atol=1e-6)

    def test_handles_missing_features(self, sample_model_params):
        """Should work when feature columns are missing."""
        df = pd.DataFrame(
            {
                "home_team": ["Bayern"],
                "away_team": ["Dortmund"],
            }
        )
        lh, _la = calculate_lambdas(df, sample_model_params)
        assert len(lh) == 1
        assert all(0.1 <= v <= 8.0 for v in lh)

    def test_fill_missing_with_mean(self, sample_model_params):
        """fill_missing_with_mean should substitute for unknown teams."""
        df = pd.DataFrame(
            {
                "home_team": ["Unknown FC"],
                "away_team": ["Bayern"],
            }
        )
        lh, _la = calculate_lambdas(df, sample_model_params, fill_missing_with_mean=True)
        assert len(lh) == 1
        assert 0.1 <= lh[0] <= 8.0


class TestFitBaselineStrengths:
    """Tests for baseline model fitting."""

    def test_returns_params_dict(self, sample_training_data):
        """Should return a dict with expected keys."""
        from src.models.hyperparameters import get_default_hyperparameters

        hyperparams = get_default_hyperparameters()
        result = fit_baseline_strengths(
            sample_training_data, hyperparams, n_random_starts=1, verbose=False
        )
        assert result is not None
        assert "attack" in result
        assert "defense" in result
        assert "home_adv" in result
        assert "teams" in result
        assert result["success"] is True

    def test_attack_defense_centered(self, sample_training_data):
        """Attack and defense params should be mean-centered."""
        from src.models.hyperparameters import get_default_hyperparameters

        hyperparams = get_default_hyperparameters()
        result = fit_baseline_strengths(
            sample_training_data, hyperparams, n_random_starts=1, verbose=False
        )
        assert result is not None
        attack_mean = np.mean(list(result["attack"].values()))
        defense_mean = np.mean(list(result["defense"].values()))
        assert abs(attack_mean) < 0.1
        assert abs(defense_mean) < 0.1

    def test_xg_weight_affects_nll(self, sample_training_data):
        """Different xg_weight hyperparameter should produce different NLL (when raw cols present)."""
        from src.models.hyperparameters import get_default_hyperparameters

        hyperparams_07 = {**get_default_hyperparameters(), "xg_weight": 0.7}
        hyperparams_10 = {**get_default_hyperparameters(), "xg_weight": 1.0}

        result_07 = fit_baseline_strengths(
            sample_training_data, hyperparams_07, n_random_starts=1
        )
        result_10 = fit_baseline_strengths(
            sample_training_data, hyperparams_10, n_random_starts=1
        )

        assert result_07 is not None and result_10 is not None
        # different xg weights should yield different NLLs
        assert result_07["nll"] != result_10["nll"]

    def test_continuous_goals_produce_different_nll(self, sample_training_data):
        """Fractional weighted goals should produce a different NLL than rounded."""
        from src.models.hyperparameters import get_default_hyperparameters

        # drop raw xg columns to force the pre-computed fallback path
        raw_cols = ["home_npxg", "away_npxg", "home_npg", "away_npg"]

        # inject fractional weighted goals (e.g. 0.35 and 1.47)
        data_frac = sample_training_data.drop(columns=raw_cols, errors="ignore").copy()
        data_frac["home_goals_weighted"] = data_frac["home_goals_weighted"] + 0.3
        data_frac["away_goals_weighted"] = data_frac["away_goals_weighted"] + 0.3

        data_int = sample_training_data.drop(columns=raw_cols, errors="ignore").copy()
        data_int["home_goals_weighted"] = data_int["home_goals_weighted"].round()
        data_int["away_goals_weighted"] = data_int["away_goals_weighted"].round()

        hyperparams = get_default_hyperparameters()
        result_frac = fit_baseline_strengths(data_frac, hyperparams, n_random_starts=1)
        result_int = fit_baseline_strengths(data_int, hyperparams, n_random_starts=1)

        assert result_frac is not None and result_int is not None
        # continuous and rounded data should produce different NLLs
        assert result_frac["nll"] != result_int["nll"]


class TestFitPoissonModelTwoStage:
    """Tests for the full two-stage fitting pipeline."""

    def test_returns_full_params(self, sample_training_data):
        """Should return params with both stages completed."""
        from src.models.hyperparameters import get_default_hyperparameters

        hyperparams = get_default_hyperparameters()
        result = fit_poisson_model_two_stage(
            sample_training_data, hyperparams, n_random_starts=1, verbose=False
        )
        assert result is not None
        assert "attack" in result
        assert "defense" in result
        assert "home_adv" in result
        assert "odds_blend_weight" in result
        assert "beta_form" in result
        assert "dispersion_factor" in result
        assert "attack_rating" in result

    def test_odds_blend_weight_in_range(self, sample_training_data):
        """Odds blend weight should be between 0 and 1."""
        from src.models.hyperparameters import get_default_hyperparameters

        hyperparams = get_default_hyperparameters()
        result = fit_poisson_model_two_stage(
            sample_training_data, hyperparams, n_random_starts=1, verbose=False
        )
        assert result is not None
        assert 0 <= result["odds_blend_weight"] <= 1

    def test_fractional_weighted_goals_accepted(self, sample_training_data):
        """Fractional weighted goals should not raise and should produce valid params."""
        from src.models.hyperparameters import get_default_hyperparameters

        data = sample_training_data.copy()
        # explicit non-integer weighted goals
        data["home_goals_weighted"] = [1.47, 0.35, 2.13, 0.89] * (len(data) // 4) + [0.5] * (
            len(data) % 4
        )
        data["away_goals_weighted"] = [0.72, 1.61, 0.44, 1.88] * (len(data) // 4) + [1.1] * (
            len(data) % 4
        )

        hyperparams = get_default_hyperparameters()
        result = fit_poisson_model_two_stage(data, hyperparams, n_random_starts=1)
        assert result is not None
        assert result["success"] is True

    def test_blend_holdout_df_is_used(self, sample_training_data):
        """odds blend weight should differ when blend_holdout_df restricts the fitting data."""
        from src.models.hyperparameters import get_default_hyperparameters

        hyperparams = get_default_hyperparameters()
        n = len(sample_training_data)

        result_no_holdout = fit_poisson_model_two_stage(
            sample_training_data,
            hyperparams,
            n_random_starts=1,
        )
        result_with_holdout = fit_poisson_model_two_stage(
            sample_training_data,
            hyperparams,
            n_random_starts=1,
            blend_holdout_df=sample_training_data.iloc[: n // 2].copy(),
        )
        assert result_no_holdout is not None
        assert result_with_holdout is not None
        # fitting on a different data subset should produce a different blend weight
        assert (
            result_no_holdout["odds_blend_weight"] != result_with_holdout["odds_blend_weight"]
        )
