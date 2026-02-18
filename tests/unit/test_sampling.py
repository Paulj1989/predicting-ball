# tests/unit/test_sampling.py
"""Tests for simulation sampling module."""

import numpy as np

from src.simulation.sampling import (
    sample_goals_calibrated,
    sample_match_outcome,
    sample_scoreline_dixon_coles,
)


class TestSampleGoalsCalibrated:
    """Tests for Poisson goal sampling."""

    def test_scalar_input_returns_scalar(self):
        """Scalar lambda with size=1 should return a Python int."""
        np.random.seed(42)
        result = sample_goals_calibrated(1.5, size=1)
        assert isinstance(result, (int, np.integer))

    def test_array_input_returns_array(self):
        """Array lambda should return array."""
        np.random.seed(42)
        result = sample_goals_calibrated(np.array([1.5, 1.2]), size=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_size_parameter(self):
        """size > 1 should return multiple samples."""
        np.random.seed(42)
        result = sample_goals_calibrated(1.5, size=100)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)

    def test_always_non_negative(self):
        """Goals should never be negative."""
        np.random.seed(42)
        result = sample_goals_calibrated(1.5, size=1000)
        assert np.all(result >= 0)

    def test_mean_close_to_lambda(self):
        """Mean of samples should be close to lambda."""
        np.random.seed(42)
        lam = 2.0
        samples = sample_goals_calibrated(lam, size=10000)
        assert isinstance(samples, np.ndarray)
        # poisson: mean ≈ variance ≈ lambda
        assert abs(samples.mean() - lam) < 0.15
        assert abs(samples.var() - lam) < 0.3

    def test_zero_lambda_gives_zero_goals(self):
        """Lambda of 0 should produce all zeros (or very close)."""
        np.random.seed(42)
        result = sample_goals_calibrated(0.1, size=1000)
        assert isinstance(result, np.ndarray)
        # with lambda=0.1, most should be 0
        assert (result == 0).sum() > 500


class TestSampleMatchOutcome:
    """Tests for single match outcome sampling."""

    def test_returns_tuple_of_two_ints(self):
        """Should return a tuple of two integers."""
        np.random.seed(42)
        home, away = sample_match_outcome(1.5, 1.2)
        assert isinstance(home, int)
        assert isinstance(away, int)

    def test_goals_non_negative(self):
        """Both goal counts should be non-negative."""
        np.random.seed(42)
        for _ in range(100):
            home, away = sample_match_outcome(1.5, 1.2)
            assert home >= 0
            assert away >= 0

    def test_goals_capped_at_max(self):
        """Goals should not exceed max_goals."""
        np.random.seed(42)
        for _ in range(100):
            home, away = sample_match_outcome(5.0, 5.0, max_goals=6)
            assert home <= 6
            assert away <= 6


class TestSampleScorelineDixonColes:
    """Tests for Dixon-Coles joint PMF scoreline sampling."""

    def test_returns_valid_tuple(self):
        """Should return a tuple of two non-negative integers."""
        np.random.seed(42)
        home, away = sample_scoreline_dixon_coles(1.5, 1.2)
        assert isinstance(home, int)
        assert isinstance(away, int)
        assert home >= 0
        assert away >= 0

    def test_goals_capped_at_max(self):
        """Neither goal count should exceed max_goals."""
        np.random.seed(42)
        for _ in range(200):
            home, away = sample_scoreline_dixon_coles(3.0, 3.0, max_goals=6)
            assert home <= 6
            assert away <= 6

    def test_rho_affects_draw_rate(self):
        """Negative rho should produce more draws than rho=0 for typical lambdas."""
        n = 10000

        np.random.seed(42)
        draws_with_rho = 0
        for _ in range(n):
            h, a = sample_scoreline_dixon_coles(1.3, 1.3, rho=-0.13)
            if h == a:
                draws_with_rho += 1

        np.random.seed(42)
        draws_no_rho = 0
        for _ in range(n):
            h, a = sample_scoreline_dixon_coles(1.3, 1.3, rho=0.0)
            if h == a:
                draws_no_rho += 1

        # rho=-0.13 boosts 0-0 and 1-1, so should produce more draws
        assert draws_with_rho > draws_no_rho

    def test_deterministic_with_seed(self):
        """Same seed should produce same result."""
        np.random.seed(123)
        result1 = sample_scoreline_dixon_coles(1.5, 1.2)
        np.random.seed(123)
        result2 = sample_scoreline_dixon_coles(1.5, 1.2)
        assert result1 == result2
