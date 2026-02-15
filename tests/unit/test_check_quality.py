# tests/unit/test_check_quality.py
"""Tests for quality gate checks."""

import pickle
import sys
from pathlib import Path

# add project root so scripts.modeling can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modeling.check_quality import (
    check_baseline_comparison,
    check_calibration_health,
    check_rps_threshold,
)


class TestCheckBaselineComparison:
    """Tests for the baseline comparison quality check."""

    def test_pass_when_model_beats_baseline_all_seasons(self):
        """Should pass when model RPS is lower than baseline in every season"""
        metrics = {
            "per_season": [
                {"season": 2024, "rps": 0.18, "baseline_rps": 0.22},
                {"season": 2025, "rps": 0.19, "baseline_rps": 0.21},
            ]
        }
        result = check_baseline_comparison(metrics)
        assert result["status"] == "pass"
        assert result["name"] == "baseline_comparison"

    def test_warn_when_model_worse_than_baseline(self):
        """Should warn when model RPS exceeds baseline in any season"""
        metrics = {
            "per_season": [
                {"season": 2024, "rps": 0.18, "baseline_rps": 0.22},
                {"season": 2025, "rps": 0.25, "baseline_rps": 0.21},
            ]
        }
        result = check_baseline_comparison(metrics)
        assert result["status"] == "warn"
        assert "1 season(s)" in result["details"]
        assert "2025" in result["details"]

    def test_skip_season_when_baseline_rps_is_none(self):
        """Should gracefully skip seasons where baseline_rps is None"""
        metrics = {
            "per_season": [
                {"season": 2024, "rps": 0.18, "baseline_rps": None},
                {"season": 2025, "rps": 0.19, "baseline_rps": 0.21},
            ]
        }
        result = check_baseline_comparison(metrics)
        assert result["status"] == "pass"

    def test_skip_season_when_model_rps_is_none(self):
        """Should gracefully skip seasons where model rps is None"""
        metrics = {
            "per_season": [
                {"season": 2024, "rps": None, "baseline_rps": 0.22},
            ]
        }
        result = check_baseline_comparison(metrics)
        assert result["status"] == "pass"

    def test_pass_with_empty_per_season_list(self):
        """Should pass when per_season list is empty"""
        metrics = {"per_season": []}
        result = check_baseline_comparison(metrics)
        assert result["status"] == "pass"

    def test_pass_with_missing_per_season_key(self):
        """Should pass when per_season key is absent"""
        result = check_baseline_comparison({})
        assert result["status"] == "pass"

    def test_warn_includes_all_failing_seasons(self):
        """Should report all seasons where the model is worse"""
        metrics = {
            "per_season": [
                {"season": 2024, "rps": 0.25, "baseline_rps": 0.22},
                {"season": 2025, "rps": 0.24, "baseline_rps": 0.21},
            ]
        }
        result = check_baseline_comparison(metrics)
        assert result["status"] == "warn"
        assert "2 season(s)" in result["details"]


class TestCheckRpsThreshold:
    """Tests for the RPS threshold quality check."""

    def test_pass_when_rps_below_threshold(self):
        """Should pass when average RPS is under the threshold"""
        metrics = {"average": {"rps": 0.19}}
        result = check_rps_threshold(metrics, threshold=0.22)
        assert result["status"] == "pass"
        assert result["name"] == "rps_threshold"
        assert "0.1900" in result["details"]

    def test_warn_when_rps_exceeds_threshold(self):
        """Should warn when average RPS is above the threshold"""
        metrics = {"average": {"rps": 0.25}}
        result = check_rps_threshold(metrics, threshold=0.22)
        assert result["status"] == "warn"
        assert "exceeds" in result["details"]

    def test_warn_when_average_rps_missing(self):
        """Should warn when average RPS is not present"""
        result = check_rps_threshold({}, threshold=0.22)
        assert result["status"] == "warn"
        assert "not found" in result["details"]

    def test_warn_when_rps_key_missing_from_average(self):
        """Should warn when rps key is missing inside average dict"""
        metrics = {"average": {"brier": 0.20}}
        result = check_rps_threshold(metrics, threshold=0.22)
        assert result["status"] == "warn"

    def test_pass_when_rps_exactly_at_threshold(self):
        """Should pass when average RPS equals the threshold exactly"""
        metrics = {"average": {"rps": 0.22}}
        result = check_rps_threshold(metrics, threshold=0.22)
        assert result["status"] == "pass"


class TestCheckCalibrationHealth:
    """Tests for the calibration health quality check."""

    def test_pass_when_no_calibrator_path(self):
        """Should pass when calibrator_path is None"""
        result = check_calibration_health(None)
        assert result["status"] == "pass"
        assert result["name"] == "calibration_health"
        assert "skipping" in result["details"].lower()

    def test_warn_when_file_does_not_exist(self, tmp_path):
        """Should warn when calibrator file is missing"""
        missing_path = str(tmp_path / "nonexistent.pkl")
        result = check_calibration_health(missing_path)
        assert result["status"] == "warn"
        assert "not found" in result["details"]

    def test_pass_when_calibration_improved_rps(self, tmp_path):
        """Should pass when rps_improvement_holdout is negative (calibration helped)"""
        cal_path = tmp_path / "calibrators.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump({"rps_improvement_holdout": -0.005}, f)

        result = check_calibration_health(str(cal_path))
        assert result["status"] == "pass"
        assert "improved" in result["details"].lower()

    def test_warn_when_calibration_degraded_rps(self, tmp_path):
        """Should warn when rps_improvement_holdout is positive (calibration hurt)"""
        cal_path = tmp_path / "calibrators.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump({"rps_improvement_holdout": 0.01}, f)

        result = check_calibration_health(str(cal_path))
        assert result["status"] == "warn"
        assert "degraded" in result["details"].lower()

    def test_pass_when_rps_improvement_missing(self, tmp_path):
        """Should pass when rps_improvement_holdout key is absent from calibrators"""
        cal_path = tmp_path / "calibrators.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump({"best_method": "isotonic"}, f)

        result = check_calibration_health(str(cal_path))
        assert result["status"] == "pass"
        assert "skipping" in result["details"].lower()

    def test_warn_when_pickle_is_corrupt(self, tmp_path):
        """Should warn when calibrator file cannot be unpickled"""
        cal_path = tmp_path / "calibrators.pkl"
        cal_path.write_bytes(b"not a valid pickle")

        result = check_calibration_health(str(cal_path))
        assert result["status"] == "warn"
        assert "failed to load" in result["details"].lower()

    def test_pass_when_improvement_is_zero(self, tmp_path):
        """Should pass when rps_improvement_holdout is exactly zero"""
        cal_path = tmp_path / "calibrators.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump({"rps_improvement_holdout": 0.0}, f)

        result = check_calibration_health(str(cal_path))
        assert result["status"] == "pass"
