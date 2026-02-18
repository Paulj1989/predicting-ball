# tests/unit/test_fisher_information.py
"""Tests for Fisher information matrix computation."""

import numpy as np
import pandas as pd
import pytest

from src.models.fisher_information import (
    build_state_vector,
    compute_fisher_information,
    draw_mle_samples,
    invert_fisher_with_constraints,
)


@pytest.fixture
def simple_params():
    """Minimal model params for 3 teams"""
    return {
        "teams": ["A", "B", "C"],
        "attack": {"A": 0.3, "B": 0.0, "C": -0.3},
        "defense": {"A": -0.2, "B": 0.1, "C": 0.1},
        "home_adv": 0.25,
        "odds_blend_weight": 1.0,
        "beta_form": 0.0,
        "rho": -0.13,
    }


@pytest.fixture
def simple_hyperparams():
    """Standard hyperparameters for tests"""
    return {"time_decay": 0.005, "lambda_reg": 0.5, "prior_decay_rate": 15.0}


@pytest.fixture
def simple_train_data():
    """Minimal training data for 3 teams"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"] * 2),
            "home_team": ["A", "B", "C", "B", "A", "C"],
            "away_team": ["B", "C", "A", "A", "C", "B"],
            "home_goals": [2, 1, 0, 1, 3, 1],
            "away_goals": [1, 1, 2, 0, 0, 2],
            "home_goals_weighted": [2.0, 1.0, 0.0, 1.0, 3.0, 1.0],
            "away_goals_weighted": [1.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            "home_npxgd_w5": [0.0] * 6,
            "away_npxgd_w5": [0.0] * 6,
        }
    )


class TestComputeFisherInformation:
    """Tests for Fisher information matrix computation."""

    def test_returns_square_matrix(self, simple_params, simple_train_data, simple_hyperparams):
        """Matrix should be (2*n_teams+1) x (2*n_teams+1)"""
        F = compute_fisher_information(simple_params, simple_train_data, simple_hyperparams)
        n = 2 * 3 + 1  # 2*n_teams + 1 (home_adv)
        assert F.shape == (n, n)

    def test_matrix_is_symmetric(self, simple_params, simple_train_data, simple_hyperparams):
        """Fisher information should be symmetric"""
        F = compute_fisher_information(simple_params, simple_train_data, simple_hyperparams)
        np.testing.assert_allclose(F, F.T, atol=1e-10)

    def test_matrix_is_positive_semidefinite(
        self, simple_params, simple_train_data, simple_hyperparams
    ):
        """Fisher information should be positive semi-definite"""
        F = compute_fisher_information(simple_params, simple_train_data, simple_hyperparams)
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals >= -1e-10)

    def test_diagonal_is_positive(self, simple_params, simple_train_data, simple_hyperparams):
        """Diagonal entries (variances) should be positive"""
        F = compute_fisher_information(simple_params, simple_train_data, simple_hyperparams)
        assert np.all(np.diag(F) > 0)


class TestInvertFisherWithConstraints:
    """Tests for constrained Fisher matrix inversion."""

    def test_returns_correct_shape(self):
        """Covariance should match input dimensions"""
        F = np.eye(7) * 10.0  # 3 teams: 2*3+1
        cov = invert_fisher_with_constraints(F, n_teams=3)
        assert cov.shape == (7, 7)

    def test_covariance_is_symmetric(self):
        """Covariance matrix should be symmetric"""
        F = np.eye(7) * 10.0 + np.random.RandomState(42).randn(7, 7) * 0.1
        F = (F + F.T) / 2  # make symmetric
        F += np.eye(7) * 5  # ensure positive definite
        cov = invert_fisher_with_constraints(F, n_teams=3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_diagonal_is_positive(self):
        """Variance estimates should be positive"""
        F = np.eye(7) * 10.0
        cov = invert_fisher_with_constraints(F, n_teams=3)
        assert np.all(np.diag(cov) > 0)

    def test_constraint_reduces_attack_shift_variance(self):
        """Uniform attack shift direction should have near-zero variance"""
        F = np.eye(7) * 10.0
        cov = invert_fisher_with_constraints(F, n_teams=3)
        # uniform attack shift: [1,1,1, 0,0,0, 0] / sqrt(3)
        v_att = np.zeros(7)
        v_att[:3] = 1.0 / np.sqrt(3)
        # variance along this direction should be very small
        shift_var = v_att @ cov @ v_att
        assert shift_var < 0.01


class TestBuildStateVector:
    """Tests for state vector construction."""

    def test_correct_length(self, simple_params):
        """Should be 2*n_teams + 1"""
        state = build_state_vector(simple_params)
        assert len(state) == 2 * 3 + 1

    def test_correct_ordering(self, simple_params):
        """Should be [attacks, defenses, home_adv]"""
        state = build_state_vector(simple_params)
        assert state[0] == 0.3  # attack A
        assert state[1] == 0.0  # attack B
        assert state[2] == -0.3  # attack C
        assert state[3] == -0.2  # defense A
        assert state[4] == 0.1  # defense B
        assert state[5] == 0.1  # defense C
        assert state[6] == 0.25  # home_adv


class TestDrawMLESamples:
    """Tests for MLE posterior sampling."""

    def test_output_shape(self):
        """Should return (n_draws, n_params)"""
        mean = np.zeros(7)
        cov = np.eye(7) * 0.01
        samples = draw_mle_samples(mean, cov, n_draws=50, seed=42)
        assert samples.shape == (50, 7)

    def test_reproducible_with_seed(self):
        """Same seed should produce same samples"""
        mean = np.zeros(7)
        cov = np.eye(7) * 0.01
        s1 = draw_mle_samples(mean, cov, n_draws=10, seed=42)
        s2 = draw_mle_samples(mean, cov, n_draws=10, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_handles_near_singular_covariance(self):
        """Should handle covariance with small negative eigenvalues"""
        mean = np.zeros(7)
        cov = np.eye(7) * 0.01
        # introduce small negative eigenvalue
        cov[0, 0] = -1e-10
        samples = draw_mle_samples(mean, cov, n_draws=10, seed=42)
        assert samples.shape == (10, 7)
