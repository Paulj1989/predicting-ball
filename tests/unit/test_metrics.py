# tests/unit/test_metrics.py
"""Tests for evaluation metrics."""

import numpy as np
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
