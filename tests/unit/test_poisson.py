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

    def test_odds_feature_affects_lambda(self, sample_model_params):
        """Non-zero log odds ratio should affect lambdas."""
        lh_no_odds, la_no_odds = calculate_lambdas_single(
            "Bayern", "Dortmund", sample_model_params, home_log_odds_ratio=0.0
        )
        lh_with_odds, la_with_odds = calculate_lambdas_single(
            "Bayern", "Dortmund", sample_model_params, home_log_odds_ratio=1.0
        )
        # positive odds ratio should boost home, reduce away
        assert lh_with_odds > lh_no_odds
        assert la_with_odds < la_no_odds

    def test_unknown_team_uses_zero(self, sample_model_params):
        """Unknown teams should get 0 attack/defense."""
        lh, la = calculate_lambdas_single("Unknown FC", "Bayern", sample_model_params)
        assert 0.1 <= lh <= 8.0
        assert 0.1 <= la <= 8.0

    @given(
        st.floats(min_value=-2.0, max_value=2.0),
        st.floats(min_value=-2.0, max_value=2.0),
        st.floats(min_value=-2.0, max_value=2.0),
    )
    @settings(max_examples=20)
    def test_lambdas_always_in_range(self, odds, form_h, form_a):
        """Lambdas should always be in [0.1, 8.0] regardless of features."""
        params = {
            "attack": {"A": 0.3, "B": -0.3},
            "defense": {"A": -0.2, "B": 0.2},
            "home_adv": 0.25,
            "beta_odds": 0.5,
            "beta_form": 0.1,
        }
        lh, la = calculate_lambdas_single(
            "A",
            "B",
            params,
            home_log_odds_ratio=odds,
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
        assert "beta_odds" in result
        assert "beta_form" in result
        assert "dispersion_factor" in result
        assert "attack_rating" in result

    def test_beta_odds_non_negative(self, sample_training_data):
        """Beta odds should be non-negative (bounded at 0)."""
        from src.models.hyperparameters import get_default_hyperparameters

        hyperparams = get_default_hyperparameters()
        result = fit_poisson_model_two_stage(
            sample_training_data, hyperparams, n_random_starts=1, verbose=False
        )
        assert result is not None
        assert result["beta_odds"] >= 0
