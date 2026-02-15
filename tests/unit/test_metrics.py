# tests/unit/test_metrics.py
"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_brier_score,
    calculate_log_loss,
    calculate_rps,
)


class TestRPS:
    """Tests for Ranked Probability Score."""

    def test_perfect_prediction(self):
        """Perfect predictions should give RPS of 0."""
        # predict exactly what happened
        predictions = np.array([[1.0, 0.0, 0.0]])  # home win
        actuals = np.array([0])  # home win (index 0)

        rps = calculate_rps(predictions, actuals)
        assert rps == 0.0

    def test_worst_prediction(self):
        """Completely wrong prediction should give high RPS."""
        # predict away win with certainty, actual is home win
        predictions = np.array([[0.0, 0.0, 1.0]])
        actuals = np.array([0])

        rps = calculate_rps(predictions, actuals)
        assert rps == 1.0  # maximum RPS for 3 outcomes

    def test_uniform_prediction(self):
        """Uniform prediction should give intermediate RPS."""
        predictions = np.array([[1 / 3, 1 / 3, 1 / 3]])
        actuals = np.array([0])

        rps = calculate_rps(predictions, actuals)
        # should be between 0 and 1
        assert 0 < rps < 1

    def test_dataframe_input(self, sample_predictions, sample_actuals):
        """Should work with DataFrame input."""
        rps = calculate_rps(sample_predictions, sample_actuals)
        assert isinstance(rps, float)
        assert 0 <= rps <= 1

    def test_string_actuals(self):
        """Should handle string outcome labels."""
        predictions = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
        actuals = np.array(["H", "D"])

        rps = calculate_rps(predictions, actuals)
        assert isinstance(rps, float)

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0.01, max_value=0.98),
                st.floats(min_value=0.01, max_value=0.98),
            ),
            min_size=1,
            max_size=10,
        )
    )
    def test_rps_bounded(self, prob_pairs):
        """RPS should always be between 0 and 1."""
        # construct valid probabilities
        predictions = []
        for p1, p2 in prob_pairs:
            # ensure they can sum to valid probabilities
            if p1 + p2 > 0.99:
                p1, p2 = p1 / 2, p2 / 2
            p3 = 1 - p1 - p2
            if p3 < 0:
                continue
            predictions.append([p1, p2, p3])

        if not predictions:
            return

        predictions = np.array(predictions)
        actuals = np.random.randint(0, 3, size=len(predictions))

        rps = calculate_rps(predictions, actuals)
        assert 0 <= rps <= 1


class TestBrierScore:
    """Tests for Brier Score."""

    def test_perfect_prediction(self):
        """Perfect prediction should give Brier score of 0."""
        predictions = np.array([[1.0, 0.0, 0.0]])
        actuals = np.array([0])

        brier = calculate_brier_score(predictions, actuals)
        assert brier == 0.0

    def test_worst_prediction(self):
        """Completely wrong prediction should give maximum Brier score."""
        predictions = np.array([[0.0, 0.0, 1.0]])
        actuals = np.array([0])

        brier = calculate_brier_score(predictions, actuals)
        assert brier == 1.0  # (0-1)^2 + (0-0)^2 + (1-0)^2 = 2, divided by 2 = 1

    def test_dataframe_input(self, sample_predictions, sample_actuals):
        """Should work with DataFrame input."""
        brier = calculate_brier_score(sample_predictions, sample_actuals)
        assert isinstance(brier, float)
        assert 0 <= brier <= 1


class TestLogLoss:
    """Tests for Log Loss."""

    def test_confident_correct_prediction(self):
        """High confidence correct prediction should have low log loss."""
        predictions = np.array([[0.99, 0.005, 0.005]])
        actuals = np.array([0])

        ll = calculate_log_loss(predictions, actuals)
        assert ll < 0.1  # very low

    def test_confident_wrong_prediction(self):
        """High confidence wrong prediction should have high log loss."""
        predictions = np.array([[0.01, 0.01, 0.98]])
        actuals = np.array([0])

        ll = calculate_log_loss(predictions, actuals)
        assert ll > 2  # very high

    def test_dataframe_input(self, sample_predictions, sample_actuals):
        """Should work with DataFrame input."""
        ll = calculate_log_loss(sample_predictions, sample_actuals)
        assert isinstance(ll, float)
        assert ll >= 0


class TestAccuracy:
    """Tests for classification accuracy."""

    def test_all_correct(self):
        """All correct predictions should give accuracy of 1."""
        predictions = np.array(
            [
                [0.8, 0.1, 0.1],  # predicts home
                [0.1, 0.8, 0.1],  # predicts draw
                [0.1, 0.1, 0.8],  # predicts away
            ]
        )
        actuals = np.array([0, 1, 2])  # H, D, A

        acc = calculate_accuracy(predictions, actuals)
        assert acc == 1.0

    def test_all_wrong(self):
        """All wrong predictions should give accuracy of 0."""
        predictions = np.array(
            [
                [0.1, 0.1, 0.8],  # predicts away
                [0.8, 0.1, 0.1],  # predicts home
                [0.1, 0.8, 0.1],  # predicts draw
            ]
        )
        actuals = np.array([0, 1, 2])  # H, D, A

        acc = calculate_accuracy(predictions, actuals)
        assert acc == 0.0

    def test_dataframe_input(self, sample_predictions, sample_actuals):
        """Should work with DataFrame input."""
        acc = calculate_accuracy(sample_predictions, sample_actuals)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1


class TestEvaluateModelComprehensive:
    """Tests for comprehensive model evaluation."""

    def test_returns_metrics_and_arrays(self, sample_model_params, sample_training_data):
        """Should return (metrics_dict, predictions_array, actuals_array)."""
        from src.evaluation.metrics import evaluate_model_comprehensive

        metrics, preds, actuals = evaluate_model_comprehensive(
            sample_model_params, sample_training_data
        )
        assert isinstance(metrics, dict)
        assert isinstance(preds, np.ndarray)
        assert isinstance(actuals, np.ndarray)

    def test_metrics_has_expected_keys(self, sample_model_params, sample_training_data):
        """Should contain standard metric keys."""
        from src.evaluation.metrics import evaluate_model_comprehensive

        metrics, _, _ = evaluate_model_comprehensive(sample_model_params, sample_training_data)
        assert "rps" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "accuracy" in metrics

    def test_metrics_in_valid_range(self, sample_model_params, sample_training_data):
        """Metrics should be in expected ranges."""
        from src.evaluation.metrics import evaluate_model_comprehensive

        metrics, _, _ = evaluate_model_comprehensive(sample_model_params, sample_training_data)
        assert 0 <= metrics["rps"] <= 1
        assert 0 <= metrics["brier_score"] <= 1
        assert metrics["log_loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 1

    def test_infers_result_from_goals(self, sample_model_params):
        """Should infer result from goals when result column is missing."""
        from src.evaluation.metrics import evaluate_model_comprehensive

        data = pd.DataFrame(
            {
                "home_team": ["Bayern", "Dortmund"],
                "away_team": ["Dortmund", "Bayern"],
                "home_goals": [2, 1],
                "away_goals": [1, 1],
            }
        )
        _metrics, _preds, actuals = evaluate_model_comprehensive(sample_model_params, data)
        assert len(actuals) == 2
        # first match: 2-1 = H (index 0), second match: 1-1 = D (index 1)
        assert set(actuals) == {0, 1}


class TestCompareMetrics:
    """Tests for model comparison."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        from src.evaluation.metrics import compare_metrics

        metrics_dict = {
            "model_a": {"rps": 0.20, "brier_score": 0.35, "log_loss": 0.95},
            "model_b": {"rps": 0.18, "brier_score": 0.33, "log_loss": 0.90},
        }
        result = compare_metrics(metrics_dict)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_calculates_improvement_vs_reference(self):
        """Should calculate improvement % when reference model exists."""
        from src.evaluation.metrics import compare_metrics

        metrics_dict = {
            "implied_odds": {"rps": 0.20, "brier_score": 0.35, "log_loss": 0.95},
            "my_model": {"rps": 0.18, "brier_score": 0.33, "log_loss": 0.90},
        }
        result = compare_metrics(metrics_dict, reference_model="implied_odds")
        assert "rps_vs_implied_odds" in result.columns


class TestCalculateMetricConfidenceInterval:
    """Tests for bootstrap confidence intervals."""

    def test_returns_three_floats(self):
        """Should return (point_estimate, lower, upper)."""
        from src.evaluation.metrics import calculate_metric_confidence_interval

        np.random.seed(42)
        preds = np.array([[0.5, 0.3, 0.2]] * 50)
        actuals = np.array([0] * 25 + [1] * 15 + [2] * 10)

        point, lower, upper = calculate_metric_confidence_interval(
            preds, actuals, calculate_rps, n_bootstrap=100
        )
        assert isinstance(point, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_lower_less_than_upper(self):
        """Lower bound should be less than upper bound."""
        from src.evaluation.metrics import calculate_metric_confidence_interval

        np.random.seed(42)
        preds = np.random.dirichlet([3, 2, 2], size=100)
        actuals = np.random.randint(0, 3, size=100)

        _point, lower, upper = calculate_metric_confidence_interval(
            preds, actuals, calculate_rps, n_bootstrap=200
        )
        assert lower <= upper

    def test_point_estimate_within_interval(self):
        """Point estimate should be within or near the CI."""
        from src.evaluation.metrics import calculate_metric_confidence_interval

        np.random.seed(42)
        preds = np.random.dirichlet([3, 2, 2], size=100)
        actuals = np.random.randint(0, 3, size=100)

        point, lower, upper = calculate_metric_confidence_interval(
            preds, actuals, calculate_rps, n_bootstrap=200
        )
        # point estimate should generally be within the CI
        assert lower - 0.05 <= point <= upper + 0.05
