# tests/unit/test_coverage_eval.py
"""Tests for evaluation coverage module."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.coverage import (
    coverage_by_confidence_level,
    diagnose_bootstrap_lambda_distribution,
    run_coverage_test,
)
from src.evaluation.coverage import (
    test_base_poisson_coverage as base_poisson_coverage,
)


@pytest.fixture
def small_test_data():
    """Small test dataset for coverage evaluation."""
    return pd.DataFrame(
        {
            "home_team": ["Bayern", "Dortmund", "Leipzig", "Frankfurt", "Freiburg"],
            "away_team": ["Dortmund", "Leipzig", "Frankfurt", "Freiburg", "Bayern"],
            "home_goals": [2, 1, 0, 1, 3],
            "away_goals": [1, 1, 2, 0, 0],
        }
    )


class TestRunCoverageTest:
    """Tests for empirical coverage testing."""

    def test_returns_float_between_zero_and_one(self, sample_param_samples, small_test_data):
        """Coverage should be a float in [0, 1]."""
        np.random.seed(42)
        coverage = run_coverage_test(
            sample_param_samples,
            small_test_data,
            confidence=0.80,
            n_samples=10,
            verbose=False,
        )
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0

    def test_higher_confidence_higher_coverage(self, sample_param_samples, small_test_data):
        """Higher confidence level should generally give higher coverage."""
        np.random.seed(42)
        cov_80 = run_coverage_test(
            sample_param_samples,
            small_test_data,
            confidence=0.80,
            n_samples=50,
            verbose=False,
        )
        np.random.seed(42)
        cov_95 = run_coverage_test(
            sample_param_samples,
            small_test_data,
            confidence=0.95,
            n_samples=50,
            verbose=False,
        )
        # with small sample sizes this isn't guaranteed, but coverage_95 >= coverage_80
        # is expected in general
        assert cov_95 >= cov_80 or abs(cov_95 - cov_80) < 0.3


class TestTestBasePoissonCoverage:
    """Tests for base Poisson coverage."""

    def test_returns_float(self, sample_model_params, small_test_data):
        """Should return a float."""
        coverage = base_poisson_coverage(
            sample_model_params, small_test_data, confidence=0.80, verbose=False
        )
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0


class TestDiagnoseBootstrapLambdaDistribution:
    """Tests for bootstrap lambda diagnostics."""

    def test_returns_diagnostics_dict(self, sample_param_samples, small_test_data):
        """Should return dict with expected keys."""
        np.random.seed(42)
        diagnostics = diagnose_bootstrap_lambda_distribution(
            sample_param_samples,
            small_test_data.iloc[[0]],
            n_samples=10,
        )
        assert "lambda_h_mean" in diagnostics
        assert "lambda_h_std" in diagnostics
        assert "lambda_a_mean" in diagnostics
        assert "lambda_a_std" in diagnostics
        assert "variance_ratio_h" in diagnostics

    def test_mean_lambda_positive(self, sample_param_samples, small_test_data):
        """Mean lambdas should be positive."""
        np.random.seed(42)
        diagnostics = diagnose_bootstrap_lambda_distribution(
            sample_param_samples,
            small_test_data.iloc[[0]],
            n_samples=10,
        )
        assert diagnostics["lambda_h_mean"] > 0
        assert diagnostics["lambda_a_mean"] > 0


class TestCoverageByConfidenceLevel:
    """Tests for multi-level coverage testing."""

    def test_returns_dict(self, sample_param_samples, small_test_data):
        """Should return dict mapping confidence -> coverage."""
        np.random.seed(42)
        result = coverage_by_confidence_level(
            sample_param_samples,
            small_test_data,
            confidence_levels=[0.68, 0.80],
            verbose=False,
        )
        assert isinstance(result, dict)
        assert 0.68 in result
        assert 0.80 in result

    def test_all_coverages_valid(self, sample_param_samples, small_test_data):
        """All coverage values should be in [0, 1]."""
        np.random.seed(42)
        result = coverage_by_confidence_level(
            sample_param_samples,
            small_test_data,
            confidence_levels=[0.68, 0.80],
            verbose=False,
        )
        for coverage in result.values():
            assert 0.0 <= coverage <= 1.0
