# tests/unit/test_calibration.py
"""Tests for model calibration module."""

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import (
    apply_calibration,
    apply_outcome_specific_scaling,
    apply_temperature_scaling,
    fit_outcome_specific_temperatures,
    fit_temperature_scaler,
    validate_calibration_on_holdout,
)


@pytest.fixture
def synthetic_predictions():
    """Synthetic predictions array (50 matches)."""
    np.random.seed(42)
    n = 50
    # generate somewhat realistic predictions
    raw = np.random.dirichlet([3, 2, 2], size=n)
    return raw


@pytest.fixture
def synthetic_actuals():
    """Synthetic actual outcomes (50 matches, mix of H/D/A)."""
    np.random.seed(42)
    return np.array(["H"] * 22 + ["D"] * 13 + ["A"] * 15)


class TestApplyTemperatureScaling:
    """Tests for temperature scaling application."""

    def test_temperature_one_preserves_predictions(self):
        """T=1 should return approximately the same predictions."""
        preds = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
        result = apply_temperature_scaling(preds, temperature=1.0)
        np.testing.assert_allclose(result, preds, atol=1e-6)

    def test_output_sums_to_one(self):
        """Each row should sum to 1 after scaling."""
        preds = np.array([[0.6, 0.25, 0.15], [0.3, 0.4, 0.3]])
        for T in [0.5, 1.0, 1.5, 2.0]:
            result = apply_temperature_scaling(preds, temperature=T)
            row_sums = result.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_high_temperature_flattens(self):
        """High T should move predictions towards uniform."""
        preds = np.array([[0.7, 0.2, 0.1]])
        flat = apply_temperature_scaling(preds, temperature=5.0)
        # should be closer to 1/3 each
        assert abs(flat[0, 0] - flat[0, 2]) < abs(preds[0, 0] - preds[0, 2])

    def test_low_temperature_sharpens(self):
        """Low T should make the dominant class more dominant."""
        preds = np.array([[0.5, 0.3, 0.2]])
        sharp = apply_temperature_scaling(preds, temperature=0.3)
        assert sharp[0, 0] > preds[0, 0]

    def test_dataframe_input(self):
        """Should accept DataFrame input."""
        df = pd.DataFrame({"home_win": [0.5, 0.6], "draw": [0.3, 0.2], "away_win": [0.2, 0.2]})
        result = apply_temperature_scaling(df, temperature=1.5)
        assert result.shape == (2, 3)

    def test_output_in_valid_range(self):
        """All probabilities should be in [0, 1]."""
        preds = np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
        for T in [0.2, 1.0, 3.0]:
            result = apply_temperature_scaling(preds, temperature=T)
            assert np.all(result >= 0)
            assert np.all(result <= 1)


class TestFitTemperatureScaler:
    """Tests for temperature fitting."""

    def test_returns_float(self, synthetic_predictions, synthetic_actuals):
        """Should return a float temperature."""
        T = fit_temperature_scaler(synthetic_predictions, synthetic_actuals, verbose=False)
        assert isinstance(T, float)
        assert 0.1 <= T <= 5.0

    def test_well_calibrated_gives_near_one(self):
        """Well-calibrated predictions should give T close to 1."""
        np.random.seed(42)
        n = 200
        # generate predictions that match actual frequencies
        preds = np.array([[0.45, 0.27, 0.28]] * n)
        actuals = np.array(["H"] * 90 + ["D"] * 54 + ["A"] * 56)
        T = fit_temperature_scaler(preds, actuals, verbose=False)
        assert 0.5 < T < 2.0

    def test_accepts_integer_actuals(self, synthetic_predictions):
        """Should work with integer outcome indices."""
        actuals = np.array([0] * 22 + [1] * 13 + [2] * 15)
        T = fit_temperature_scaler(synthetic_predictions, actuals, verbose=False)
        assert isinstance(T, float)

    def test_accepts_dataframe_input(self, synthetic_actuals):
        """Should work with DataFrame predictions."""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.dirichlet([3, 2, 2], size=50),
            columns=["home_win", "draw", "away_win"],  # type: ignore[arg-type]
        )
        T = fit_temperature_scaler(df, synthetic_actuals, verbose=False)
        assert isinstance(T, float)


class TestApplyOutcomeSpecificScaling:
    """Tests for outcome-specific temperature scaling."""

    def test_output_sums_to_one(self):
        """Each row should sum to 1."""
        preds = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
        temps = {"T_home": 1.2, "T_draw": 0.8, "T_away": 1.0}
        result = apply_outcome_specific_scaling(preds, temps)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_all_temperatures_one_preserves(self):
        """T=1 for all outcomes should preserve predictions."""
        preds = np.array([[0.5, 0.3, 0.2]])
        temps = {"T_home": 1.0, "T_draw": 1.0, "T_away": 1.0}
        result = apply_outcome_specific_scaling(preds, temps)
        np.testing.assert_allclose(result, preds, atol=1e-6)

    def test_different_temperatures_change_distribution(self):
        """Non-uniform temperatures should change the probability distribution."""
        preds = np.array([[0.5, 0.25, 0.25]])
        temps = {"T_home": 2.0, "T_draw": 0.5, "T_away": 1.0}
        result = apply_outcome_specific_scaling(preds, temps)
        # distribution should differ from input
        assert not np.allclose(result, preds, atol=0.01)


class TestFitOutcomeSpecificTemperatures:
    """Tests for fitting outcome-specific temperatures."""

    # small fixture has <20 draws and away wins — warning is expected, not the subject of these tests
    pytestmark = pytest.mark.filterwarnings("ignore:Only.*calibration set:UserWarning")

    def test_returns_expected_keys(self, synthetic_predictions, synthetic_actuals):
        """Should return dict with temperature keys."""
        result = fit_outcome_specific_temperatures(
            synthetic_predictions, synthetic_actuals, verbose=False
        )
        assert "T_home" in result
        assert "T_draw" in result
        assert "T_away" in result
        assert "method" in result
        assert result["method"] == "outcome_specific"

    def test_temperatures_in_valid_range(self, synthetic_predictions, synthetic_actuals):
        """Fitted temperatures should be in bounds."""
        result = fit_outcome_specific_temperatures(
            synthetic_predictions, synthetic_actuals, verbose=False
        )
        for key in ["T_home", "T_draw", "T_away"]:
            assert 0.1 <= result[key] <= 5.0

    def test_nll_improvement_non_negative(self, synthetic_predictions, synthetic_actuals):
        """Calibration should not increase NLL (or be very close)."""
        result = fit_outcome_specific_temperatures(
            synthetic_predictions, synthetic_actuals, verbose=False
        )
        # optimised nll should be <= uncalibrated nll
        # (outcome-specific temperatures still optimise on nll internally)
        assert result["nll_improvement"] >= -0.1


class TestFitTemperatureScalerEdgeCases:
    """Additional edge case tests for temperature scaling."""

    def test_raises_on_invalid_outcome(self):
        """Should raise ValueError for unknown outcome strings."""
        preds = np.array([[0.5, 0.3, 0.2]])
        actuals = np.array(["X"])
        with pytest.raises(ValueError, match="Unknown outcome"):
            fit_temperature_scaler(preds, actuals, verbose=False)

    def test_raises_on_invalid_integer_outcome(self):
        """Should raise ValueError for invalid integer outcomes."""
        preds = np.array([[0.5, 0.3, 0.2]])
        actuals = np.array([5])
        with pytest.raises(ValueError, match="Integer outcomes must be"):
            fit_temperature_scaler(preds, actuals, verbose=False)

    def test_raises_on_missing_columns(self):
        """Should raise ValueError for DataFrame with wrong columns."""
        df = pd.DataFrame({"a": [0.5], "b": [0.3], "c": [0.2]})
        with pytest.raises(ValueError, match="DataFrame must have columns"):
            apply_temperature_scaling(df, 1.0)


class TestFitOutcomeSpecificEdgeCases:
    """Additional tests for outcome-specific calibration."""

    # small fixture has <20 draws and away wins — warning is expected, not the subject of these tests
    pytestmark = pytest.mark.filterwarnings("ignore:Only.*calibration set:UserWarning")

    def test_raises_on_invalid_shape(self):
        """Should raise ValueError for wrong number of columns."""
        preds = np.array([[0.5, 0.5]])  # only 2 columns
        actuals = np.array([0])
        with pytest.raises(ValueError, match="Predictions must have 3 columns"):
            fit_outcome_specific_temperatures(preds, actuals, verbose=False)

    def test_accepts_series_actuals(self, synthetic_predictions):
        """Should accept pd.Series for actuals."""
        actuals = pd.Series(["H"] * 22 + ["D"] * 13 + ["A"] * 15)
        result = fit_outcome_specific_temperatures(
            synthetic_predictions, actuals, verbose=False
        )
        assert "T_home" in result

    def test_returns_outcome_metrics(self, synthetic_predictions, synthetic_actuals):
        """Should include per-outcome calibration metrics."""
        result = fit_outcome_specific_temperatures(
            synthetic_predictions, synthetic_actuals, verbose=False
        )
        assert "outcome_metrics" in result
        assert "n_samples" in result
        assert result["n_samples"] == 50

    def test_reports_sample_counts(self, synthetic_predictions, synthetic_actuals):
        """Should report outcome counts."""
        result = fit_outcome_specific_temperatures(
            synthetic_predictions, synthetic_actuals, verbose=False
        )
        assert "outcome_counts" in result
        assert sum(result["outcome_counts"]) == 50


class TestApplyOutcomeSpecificEdgeCases:
    """Additional tests for outcome-specific scaling application."""

    def test_raises_on_missing_columns(self):
        """Should raise ValueError for DataFrame with wrong columns."""
        df = pd.DataFrame({"a": [0.5], "b": [0.3], "c": [0.2]})
        temps = {"T_home": 1.0, "T_draw": 1.0, "T_away": 1.0}
        with pytest.raises(ValueError, match="DataFrame must have columns"):
            apply_outcome_specific_scaling(df, temps)

    def test_dataframe_input(self):
        """Should accept DataFrame input."""
        df = pd.DataFrame({"home_win": [0.5, 0.6], "draw": [0.3, 0.2], "away_win": [0.2, 0.2]})
        temps = {"T_home": 1.0, "T_draw": 1.0, "T_away": 1.0}
        result = apply_outcome_specific_scaling(df, temps)
        assert result.shape == (2, 3)


class TestValidateCalibrationOnHoldout:
    """Tests for holdout validation."""

    def test_returns_metrics_dict(self):
        """Should return dict with expected metric keys."""
        np.random.seed(42)
        preds = np.random.dirichlet([3, 2, 2], size=30)
        actuals = np.array(["H"] * 13 + ["D"] * 8 + ["A"] * 9)
        temps = {"T_home": 1.0, "T_draw": 0.9, "T_away": 1.1}

        metrics = validate_calibration_on_holdout(temps, preds, actuals, verbose=False)

        assert "brier_uncalibrated" in metrics
        assert "brier_calibrated" in metrics
        assert "rps_uncalibrated" in metrics
        assert "rps_calibrated" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == 30


class TestVerboseOutput:
    """Tests that exercise verbose paths for coverage."""

    def test_fit_temperature_verbose(self, synthetic_predictions, synthetic_actuals):
        """Verbose output should not crash."""
        T = fit_temperature_scaler(synthetic_predictions, synthetic_actuals, verbose=True)
        assert isinstance(T, float)

    @pytest.mark.filterwarnings("ignore:Only.*calibration set:UserWarning")
    def test_fit_outcome_specific_verbose(self, synthetic_predictions, synthetic_actuals):
        """Verbose output should not crash."""
        result = fit_outcome_specific_temperatures(
            synthetic_predictions, synthetic_actuals, verbose=True
        )
        assert "T_home" in result

    def test_validate_holdout_verbose(self):
        """Verbose validation output should not crash."""
        np.random.seed(42)
        preds = np.random.dirichlet([3, 2, 2], size=30)
        actuals = np.array(["H"] * 13 + ["D"] * 8 + ["A"] * 9)
        temps = {"T_home": 1.0, "T_draw": 0.9, "T_away": 1.1}
        metrics = validate_calibration_on_holdout(temps, preds, actuals, verbose=True)
        assert "n_samples" in metrics

    def test_fit_temperature_overconfident(self):
        """Test verbose path for overconfident model (T > 1.5)."""
        np.random.seed(42)
        n = 100
        # very overconfident predictions (high max prob but often wrong)
        preds = np.array([[0.9, 0.05, 0.05]] * n)
        actuals = np.array(["H"] * 40 + ["D"] * 30 + ["A"] * 30)
        T = fit_temperature_scaler(preds, actuals, verbose=True)
        assert T > 1.0  # should find model is overconfident

    def test_fit_temperature_underconfident(self):
        """Test verbose path for underconfident model (T < 0.8)."""
        np.random.seed(42)
        n = 100
        # nearly uniform predictions but results cluster
        preds = np.array([[0.34, 0.33, 0.33]] * n)
        actuals = np.array(["H"] * 80 + ["D"] * 10 + ["A"] * 10)
        T = fit_temperature_scaler(preds, actuals, verbose=True)
        assert isinstance(T, float)


class TestApplyCalibration:
    """Tests for calibration dispatch."""

    def test_standard_temperature_scaling(self):
        """Should apply standard scaling with float temperature."""
        preds = np.array([[0.5, 0.3, 0.2]])
        calibrators = {"temperature": 1.5, "calibration_method": "temperature_scaling"}
        result = apply_calibration(preds, calibrators)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_outcome_specific_scaling(self):
        """Should apply outcome-specific scaling with dict temperature."""
        preds = np.array([[0.5, 0.3, 0.2]])
        calibrators = {
            "temperature": {"T_home": 1.0, "T_draw": 0.8, "T_away": 1.2},
            "calibration_method": "outcome_specific",
        }
        result = apply_calibration(preds, calibrators)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_no_temperature_returns_unchanged(self):
        """Should return unchanged predictions when no temperature key."""
        preds = np.array([[0.5, 0.3, 0.2]])
        calibrators = {}
        result = apply_calibration(preds, calibrators)
        np.testing.assert_allclose(result, preds)

    def test_dataframe_input_no_calibration(self):
        """Should handle DataFrame input when no calibration applied."""
        df = pd.DataFrame({"home_win": [0.5], "draw": [0.3], "away_win": [0.2]})
        calibrators = {}
        result = apply_calibration(df, calibrators)
        assert result.shape == (1, 3)
