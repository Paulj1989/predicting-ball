"""Tests for Diebold-Mariano significance testing."""

import numpy as np
import pytest

from src.evaluation.significance import diebold_mariano_test


class TestDieboldMarianoTest:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        model = rng.uniform(0, 0.3, 50)
        baseline = rng.uniform(0.05, 0.35, 50)
        result = diebold_mariano_test(model, baseline)
        assert set(result.keys()) == {
            "dm_statistic",
            "p_value",
            "mean_loss_difference",
            "se",
            "n",
            "max_lag",
            "significant",
        }

    def test_clearly_better_model_is_significant(self):
        # model always loses 0.1, baseline always loses 0.2
        n = 100
        model = np.full(n, 0.1)
        baseline = np.full(n, 0.2)
        result = diebold_mariano_test(model, baseline, alternative="less")
        assert result["dm_statistic"] < 0
        assert result["mean_loss_difference"] < 0
        assert result["significant"] is True

    def test_equal_models_not_significant(self):
        rng = np.random.default_rng(0)
        losses = rng.uniform(0, 0.3, 200)
        result = diebold_mariano_test(losses, losses, alternative="less")
        assert result["dm_statistic"] == pytest.approx(0.0, abs=1e-10)
        assert result["significant"] is False

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            diebold_mariano_test(np.array([0.1, 0.2]), np.array([0.1]))

    def test_raises_on_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 10"):
            diebold_mariano_test(np.array([0.1] * 5), np.array([0.2] * 5))

    def test_n_matches_actual_input(self):
        rng = np.random.default_rng(7)
        n = 80
        model = rng.uniform(0, 0.25, n)
        baseline = rng.uniform(0.05, 0.30, n)
        result = diebold_mariano_test(model, baseline)
        assert result["n"] == n

    def test_max_lag_defaults_to_sqrt_n(self):
        n = 100
        model = np.random.default_rng(1).uniform(0, 0.3, n)
        baseline = np.random.default_rng(2).uniform(0, 0.3, n)
        result = diebold_mariano_test(model, baseline)
        assert result["max_lag"] == int(np.floor(np.sqrt(n)))

    def test_custom_max_lag(self):
        model = np.random.default_rng(3).uniform(0, 0.3, 50)
        baseline = np.random.default_rng(4).uniform(0, 0.3, 50)
        result = diebold_mariano_test(model, baseline, max_lag=3)
        assert result["max_lag"] == 3

    def test_two_sided_alternative_equal_is_not_significant(self):
        rng = np.random.default_rng(0)
        losses = rng.uniform(0, 0.3, 200)
        result = diebold_mariano_test(losses, losses, alternative="two-sided")
        assert result["significant"] is False

    def test_greater_alternative_clearly_worse_model_is_significant(self):
        # model always loses more than baseline — should be significant under "greater"
        n = 100
        model = np.full(n, 0.2)
        baseline = np.full(n, 0.1)
        result = diebold_mariano_test(model, baseline, alternative="greater")
        assert result["dm_statistic"] > 0
        assert result["mean_loss_difference"] > 0
        assert result["significant"] is True

    def test_invalid_alternative_raises(self):
        model = np.ones(20) * 0.1
        baseline = np.ones(20) * 0.2
        with pytest.raises(ValueError, match="alternative must be"):
            diebold_mariano_test(model, baseline, alternative="wrong")

    def test_p_value_between_0_and_1(self):
        rng = np.random.default_rng(99)
        for _ in range(10):
            model = rng.uniform(0, 0.35, 60)
            baseline = rng.uniform(0, 0.35, 60)
            result = diebold_mariano_test(model, baseline)
            assert 0.0 <= result["p_value"] <= 1.0
