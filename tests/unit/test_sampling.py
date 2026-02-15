# tests/unit/test_sampling.py
"""Tests for simulation sampling module."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.simulation.sampling import (
    calculate_outcome_probabilities,
    sample_goals_calibrated,
    sample_match_outcome,
)


class TestSampleGoalsCalibrated:
    """Tests for calibrated goal sampling."""

    def test_scalar_input_returns_scalar(self):
        """Scalar lambda with size=1 should return a Python int."""
        np.random.seed(42)
        result = sample_goals_calibrated(1.5, dispersion_factor=1.0, size=1)
        assert isinstance(result, (int, np.integer))

    def test_array_input_returns_array(self):
        """Array lambda should return array."""
        np.random.seed(42)
        result = sample_goals_calibrated(np.array([1.5, 1.2]), dispersion_factor=1.0, size=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_size_parameter(self):
        """size > 1 should return multiple samples."""
        np.random.seed(42)
        result = sample_goals_calibrated(1.5, dispersion_factor=1.0, size=100)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)

    def test_always_non_negative(self):
        """Goals should never be negative."""
        np.random.seed(42)
        for disp in [1.0, 1.5, 2.0]:
            result = sample_goals_calibrated(1.5, dispersion_factor=disp, size=1000)
            assert np.all(result >= 0)

    def test_poisson_path_low_dispersion(self):
        """Dispersion <= 1.1 should use Poisson (variance ~ mean)."""
        np.random.seed(42)
        lam = 2.0
        samples = sample_goals_calibrated(lam, dispersion_factor=1.0, size=10000)
        assert isinstance(samples, np.ndarray)
        # poisson: mean ≈ variance ≈ lambda
        assert abs(samples.mean() - lam) < 0.15
        assert abs(samples.var() - lam) < 0.3

    def test_negative_binomial_path_high_dispersion(self):
        """Dispersion > 1.1 should use negative binomial (overdispersed)."""
        np.random.seed(42)
        lam = 2.0
        disp = 2.0
        samples = sample_goals_calibrated(lam, dispersion_factor=disp, size=10000)
        assert isinstance(samples, np.ndarray)
        # mean should still be close to lambda
        assert abs(samples.mean() - lam) < 0.2
        # variance should be greater than lambda (overdispersed)
        assert samples.var() > lam

    def test_zero_lambda_gives_zero_goals(self):
        """Lambda of 0 should produce all zeros (or very close)."""
        np.random.seed(42)
        result = sample_goals_calibrated(0.1, dispersion_factor=1.0, size=1000)
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

    def test_dispersion_factor_passed_through(self):
        """Should work with non-default dispersion factor."""
        np.random.seed(42)
        home, away = sample_match_outcome(1.5, 1.2, dispersion_factor=1.5)
        assert isinstance(home, int)
        assert isinstance(away, int)


class TestCalculateOutcomeProbabilities:
    """Tests for match outcome probability calculation."""

    def test_probabilities_sum_to_one(self):
        """Home/draw/away probabilities should sum to 1."""
        h, d, a = calculate_outcome_probabilities(1.5, 1.2)
        assert np.isclose(h + d + a, 1.0, atol=1e-6)

    def test_probabilities_in_valid_range(self):
        """All probabilities should be in [0, 1]."""
        h, d, a = calculate_outcome_probabilities(1.5, 1.2)
        assert 0 <= h <= 1
        assert 0 <= d <= 1
        assert 0 <= a <= 1

    def test_stronger_home_team_favoured(self):
        """Higher home lambda should give higher home win probability."""
        h1, _, _ = calculate_outcome_probabilities(1.0, 1.0)
        h2, _, _ = calculate_outcome_probabilities(2.5, 1.0)
        assert h2 > h1

    def test_equal_lambdas_symmetric(self):
        """Equal lambdas should give equal home/away probs."""
        h, _, a = calculate_outcome_probabilities(1.5, 1.5)
        assert np.isclose(h, a, atol=1e-6)

    def test_negative_binomial_path(self):
        """Should work with NB distribution when use_poisson=False and high dispersion."""
        h, d, a = calculate_outcome_probabilities(
            1.5, 1.2, dispersion_factor=2.0, use_poisson=False
        )
        assert np.isclose(h + d + a, 1.0, atol=1e-4)
        assert 0 <= h <= 1
        assert 0 <= d <= 1
        assert 0 <= a <= 1

    @given(
        st.floats(min_value=0.3, max_value=4.0),
        st.floats(min_value=0.3, max_value=4.0),
    )
    @settings(max_examples=20)
    def test_probabilities_always_sum_to_one(self, lambda_h, lambda_a):
        """Probabilities should always sum to 1 for any valid lambdas."""
        h, d, a = calculate_outcome_probabilities(lambda_h, lambda_a)
        assert np.isclose(h + d + a, 1.0, atol=1e-4)
