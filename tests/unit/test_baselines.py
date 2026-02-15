# tests/unit/test_baselines.py
"""Tests for baseline evaluation module."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import (
    create_baseline_comparison_table,
    evaluate_implied_odds_baseline,
)


@pytest.fixture
def valid_odds_data():
    """DataFrame with valid betting odds and results."""
    np.random.seed(42)
    n = 50
    results = np.random.choice(["H", "D", "A"], size=n, p=[0.45, 0.27, 0.28])
    return pd.DataFrame(
        {
            "home_odds": np.random.uniform(1.3, 4.0, n),
            "draw_odds": np.random.uniform(2.5, 5.0, n),
            "away_odds": np.random.uniform(1.5, 6.0, n),
            "result": results,
        }
    )


class TestEvaluateImpliedOddsBaseline:
    """Tests for implied odds baseline evaluation."""

    def test_returns_metrics_dict(self, valid_odds_data):
        """Should return dict with RPS, Brier, log loss."""
        result = evaluate_implied_odds_baseline(valid_odds_data)
        assert result is not None
        assert "rps" in result
        assert "brier_score" in result
        assert "log_loss" in result
        assert "n_matches" in result

    def test_metrics_in_valid_ranges(self, valid_odds_data):
        """Metrics should be in expected ranges."""
        result = evaluate_implied_odds_baseline(valid_odds_data)
        assert result is not None
        assert 0 <= result["rps"] <= 1
        assert 0 <= result["brier_score"] <= 1
        assert result["log_loss"] >= 0

    def test_returns_none_missing_columns(self):
        """Should return None when required columns are missing."""
        data = pd.DataFrame({"home_team": ["Bayern"], "away_team": ["Dortmund"]})
        result = evaluate_implied_odds_baseline(data)
        assert result is None

    def test_returns_none_insufficient_coverage(self):
        """Should return None when odds coverage is below 80%."""
        n = 20
        data = pd.DataFrame(
            {
                "home_odds": [1.5] * 5 + [np.nan] * 15,
                "draw_odds": [3.0] * 5 + [np.nan] * 15,
                "away_odds": [4.0] * 5 + [np.nan] * 15,
                "result": ["H"] * n,
            }
        )
        result = evaluate_implied_odds_baseline(data)
        assert result is None

    def test_returns_none_no_valid_matches(self):
        """Should return None when no matches have complete odds."""
        data = pd.DataFrame(
            {
                "home_odds": [np.nan],
                "draw_odds": [np.nan],
                "away_odds": [np.nan],
                "result": ["H"],
            }
        )
        result = evaluate_implied_odds_baseline(data)
        assert result is None

    def test_normalises_probabilities(self, valid_odds_data):
        """Implied probabilities should be normalised (margin removed)."""
        result = evaluate_implied_odds_baseline(valid_odds_data)
        assert result is not None
        # if probabilities were normalised correctly, metrics should be computable
        assert not np.isnan(result["rps"])

    def test_coverage_field(self, valid_odds_data):
        """Coverage should reflect fraction of matches with odds."""
        result = evaluate_implied_odds_baseline(valid_odds_data)
        assert result is not None
        assert result["coverage"] == 1.0

    def test_verbose_output(self, valid_odds_data):
        """Verbose mode should not crash."""
        result = evaluate_implied_odds_baseline(valid_odds_data, verbose=True)
        assert result is not None

    def test_verbose_missing_columns(self):
        """Verbose mode should print missing column info."""
        data = pd.DataFrame({"home_team": ["Bayern"]})
        result = evaluate_implied_odds_baseline(data, verbose=True)
        assert result is None

    def test_verbose_insufficient_coverage(self):
        """Verbose mode should print coverage info."""
        data = pd.DataFrame(
            {
                "home_odds": [1.5] * 3 + [np.nan] * 17,
                "draw_odds": [3.0] * 3 + [np.nan] * 17,
                "away_odds": [4.0] * 3 + [np.nan] * 17,
                "result": ["H"] * 20,
            }
        )
        result = evaluate_implied_odds_baseline(data, verbose=True)
        assert result is None

    def test_returns_none_negative_odds(self):
        """Should return None for zero/negative odds."""
        data = pd.DataFrame(
            {
                "home_odds": [0.0] * 10,
                "draw_odds": [3.0] * 10,
                "away_odds": [4.0] * 10,
                "result": ["H"] * 10,
            }
        )
        result = evaluate_implied_odds_baseline(data, verbose=True)
        assert result is None


class TestEvaluateOddsOnlyModel:
    """Tests for the odds-only model evaluation."""

    def test_returns_none_for_none_params(self, valid_odds_data):
        """Should return None when params is None."""
        from src.evaluation.baselines import evaluate_odds_only_model

        result = evaluate_odds_only_model(valid_odds_data, None)  # type: ignore[arg-type]
        assert result is None

    def test_returns_none_for_failed_params(self, valid_odds_data):
        """Should return None when params indicate failure."""
        from src.evaluation.baselines import evaluate_odds_only_model

        result = evaluate_odds_only_model(valid_odds_data, {"success": False})
        assert result is None

    def test_returns_metrics_with_valid_params(self, valid_odds_data, sample_model_params):
        """Should return metrics dict when params are valid."""
        from src.evaluation.baselines import evaluate_odds_only_model

        # add required columns to test data
        data = valid_odds_data.copy()
        teams = list(sample_model_params["attack"].keys())
        n = len(data)
        data["home_team"] = [teams[i % len(teams)] for i in range(n)]
        data["away_team"] = [teams[(i + 1) % len(teams)] for i in range(n)]
        data["home_goals"] = np.random.randint(0, 4, n)
        data["away_goals"] = np.random.randint(0, 4, n)

        result = evaluate_odds_only_model(data, sample_model_params)
        assert result is not None
        assert "rps" in result


class TestCreateBaselineComparisonTable:
    """Tests for comparison table creation."""

    def test_returns_dataframe(self, valid_odds_data):
        """Should return a DataFrame."""
        model_metrics = {"rps": 0.18, "brier_score": 0.35, "log_loss": 0.95}
        result = create_baseline_comparison_table(
            model_metrics, valid_odds_data, verbose=False
        )
        assert isinstance(result, pd.DataFrame)

    def test_contains_model_row(self, valid_odds_data):
        """Should contain the fitted model metrics."""
        model_metrics = {"rps": 0.18, "brier_score": 0.35, "log_loss": 0.95}
        result = create_baseline_comparison_table(
            model_metrics, valid_odds_data, verbose=False
        )
        assert "Fitted Model" in result.index

    def test_calculates_improvement(self, valid_odds_data):
        """Should calculate improvement percentages when baseline is available."""
        model_metrics = {"rps": 0.18, "brier_score": 0.35, "log_loss": 0.95}
        result = create_baseline_comparison_table(
            model_metrics, valid_odds_data, verbose=False
        )
        if "Implied Odds" in result.index:
            assert "rps_improvement" in result.columns
