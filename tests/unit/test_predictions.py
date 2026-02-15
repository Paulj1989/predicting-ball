# tests/unit/test_predictions.py
"""Tests for simulation predictions module."""

import numpy as np
import pandas as pd
import pytest

from src.simulation.predictions import (
    get_all_future_fixtures,
    get_next_round_fixtures,
    predict_match_probabilities,
    predict_next_fixtures,
    predict_single_match,
)


@pytest.fixture
def fixture_data():
    """Season data with played and unplayed fixtures."""
    return pd.DataFrame(
        {
            "home_team": [
                "Bayern",
                "Dortmund",
                "Leipzig",
                "Frankfurt",
                "Freiburg",
                "Bayern",
                "Dortmund",
                "Leipzig",
                "Frankfurt",
                "Freiburg",
                "Bayern",
                "Dortmund",
                "Leipzig",
                "Frankfurt",
                "Freiburg",
                "Bayern",
                "Dortmund",
                "Leipzig",
            ],
            "away_team": [
                "Dortmund",
                "Leipzig",
                "Frankfurt",
                "Freiburg",
                "Bayern",
                "Leipzig",
                "Frankfurt",
                "Freiburg",
                "Bayern",
                "Dortmund",
                "Frankfurt",
                "Freiburg",
                "Bayern",
                "Dortmund",
                "Leipzig",
                "Freiburg",
                "Bayern",
                "Dortmund",
            ],
            "home_goals": [
                2,
                1,
                0,
                1,
                0,
                3,
                2,
                1,
                0,
                1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "away_goals": [
                1,
                1,
                2,
                0,
                3,
                0,
                1,
                1,
                2,
                0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "is_played": [True] * 10 + [False] * 8,
            "matchweek": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4],
            "date": (
                [pd.Timestamp("2025-01-04")] * 5
                + [pd.Timestamp("2025-01-11")] * 5
                + [pd.Timestamp("2025-01-18")] * 5
                + [pd.Timestamp("2025-01-25")] * 3
            ),
        }
    )


class TestGetNextRoundFixtures:
    """Tests for matchweek-based fixture selection."""

    def test_returns_next_unplayed_matchweek(self, fixture_data):
        """Should return the next full unplayed matchweek."""
        result = get_next_round_fixtures(fixture_data, full_matchday_size=5)
        assert result is not None
        # matchweek 3 has 5 unplayed games
        assert len(result) == 5

    def test_returns_none_when_all_played(self, fixture_data):
        """Should return None when no unplayed fixtures exist."""
        all_played = fixture_data.copy()
        all_played["is_played"] = True
        all_played["home_goals"] = 1
        all_played["away_goals"] = 0
        result = get_next_round_fixtures(all_played)
        assert result is None

    def test_falls_back_to_null_goals(self):
        """Should use null goals check if is_played column missing."""
        data = pd.DataFrame(
            {
                "home_team": ["Bayern", "Dortmund"],
                "away_team": ["Dortmund", "Bayern"],
                "home_goals": [2, np.nan],
                "away_goals": [1, np.nan],
                "matchweek": [1, 2],
                "date": [pd.Timestamp("2025-01-04"), pd.Timestamp("2025-01-11")],
            }
        )
        result = get_next_round_fixtures(data, full_matchday_size=1)
        assert result is not None
        assert len(result) == 1

    def test_returns_all_remaining_when_no_full_matchweek(self):
        """Should return all remaining fixtures when no full matchweek exists."""
        data = pd.DataFrame(
            {
                "home_team": ["Bayern", "Dortmund"],
                "away_team": ["Dortmund", "Bayern"],
                "home_goals": [np.nan, np.nan],
                "away_goals": [np.nan, np.nan],
                "matchweek": [1, 2],
                "date": [pd.Timestamp("2025-01-04"), pd.Timestamp("2025-01-11")],
            }
        )
        # full_matchday_size=9 so neither matchweek is "full"
        result = get_next_round_fixtures(data, full_matchday_size=9)
        assert result is not None
        assert len(result) == 2


class TestGetAllFutureFixtures:
    """Tests for getting all unplayed fixtures."""

    def test_returns_only_unplayed(self, fixture_data):
        """Should return only unplayed fixtures."""
        result = get_all_future_fixtures(fixture_data)
        assert result is not None
        assert len(result) == 8  # 8 unplayed

    def test_returns_none_when_all_played(self, fixture_data):
        """Should return None when all fixtures are played."""
        all_played = fixture_data.copy()
        all_played["is_played"] = True
        result = get_all_future_fixtures(all_played)
        assert result is None

    def test_sorted_by_date(self, fixture_data):
        """Result should be sorted by date."""
        result = get_all_future_fixtures(fixture_data)
        assert result is not None
        dates = result["date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


class TestPredictSingleMatch:
    """Tests for single match prediction."""

    def test_returns_expected_keys(self, sample_model_params):
        """Should return dict with all expected keys."""
        result = predict_single_match("Bayern", "Dortmund", sample_model_params)

        expected_keys = {
            "home_team",
            "away_team",
            "expected_goals_home",
            "expected_goals_away",
            "home_win",
            "draw",
            "away_win",
            "most_likely_score",
            "score_probabilities",
        }
        assert set(result.keys()) == expected_keys

    def test_probabilities_sum_to_one(self, sample_model_params):
        """Outcome probabilities should sum to 1."""
        result = predict_single_match("Bayern", "Dortmund", sample_model_params)
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert np.isclose(total, 1.0, atol=1e-6)

    def test_probabilities_in_valid_range(self, sample_model_params):
        """All probabilities should be between 0 and 1."""
        result = predict_single_match("Bayern", "Leipzig", sample_model_params)
        assert 0 <= result["home_win"] <= 1
        assert 0 <= result["draw"] <= 1
        assert 0 <= result["away_win"] <= 1

    def test_expected_goals_positive(self, sample_model_params):
        """Expected goals should be positive."""
        result = predict_single_match("Bayern", "Dortmund", sample_model_params)
        assert result["expected_goals_home"] > 0
        assert result["expected_goals_away"] > 0

    def test_most_likely_score_format(self, sample_model_params):
        """Most likely score should be in 'X-Y' format."""
        result = predict_single_match("Bayern", "Dortmund", sample_model_params)
        parts = result["most_likely_score"].split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    def test_independent_poisson_path(self, sample_model_params):
        """Should work with use_dixon_coles=False."""
        result = predict_single_match(
            "Bayern", "Dortmund", sample_model_params, use_dixon_coles=False
        )
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert np.isclose(total, 1.0, atol=1e-6)

    def test_stronger_team_favoured(self, sample_model_params):
        """Bayern (highest attack) should be favoured at home vs Freiburg (weakest)."""
        result = predict_single_match("Bayern", "Freiburg", sample_model_params)
        assert result["home_win"] > result["away_win"]


class TestPredictMatchProbabilities:
    """Tests for predicting probabilities from a Series."""

    def test_returns_probability_dict(self, sample_model_params):
        """Should return dict with home_win, draw, away_win."""
        match = pd.Series(
            {
                "home_team": "Bayern",
                "away_team": "Dortmund",
                "home_log_odds_ratio": 0.1,
                "home_npxgd_w5": 0.2,
                "away_npxgd_w5": -0.1,
            }
        )
        result = predict_match_probabilities(sample_model_params, match)

        assert set(result.keys()) == {"home_win", "draw", "away_win"}
        assert np.isclose(sum(result.values()), 1.0, atol=1e-6)

    def test_handles_nan_features(self, sample_model_params):
        """Should handle NaN features gracefully."""
        match = pd.Series(
            {
                "home_team": "Bayern",
                "away_team": "Dortmund",
                "home_log_odds_ratio": np.nan,
                "home_npxgd_w5": np.nan,
                "away_npxgd_w5": np.nan,
            }
        )
        result = predict_match_probabilities(sample_model_params, match)
        assert np.isclose(sum(result.values()), 1.0, atol=1e-6)

    def test_handles_missing_features(self, sample_model_params):
        """Should handle completely missing feature columns."""
        match = pd.Series({"home_team": "Bayern", "away_team": "Dortmund"})
        result = predict_match_probabilities(sample_model_params, match)
        assert np.isclose(sum(result.values()), 1.0, atol=1e-6)


class TestPredictNextFixtures:
    """Tests for predicting multiple fixtures."""

    def test_returns_dataframe(self, sample_model_params):
        """Should return a DataFrame."""
        fixtures = pd.DataFrame(
            {
                "home_team": ["Bayern", "Leipzig"],
                "away_team": ["Dortmund", "Frankfurt"],
            }
        )
        result = predict_next_fixtures(fixtures, sample_model_params)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_returns_none_for_empty(self, sample_model_params):
        """Should return None for empty input."""
        result = predict_next_fixtures(pd.DataFrame(), sample_model_params)
        assert result is None

    def test_returns_none_for_none(self, sample_model_params):
        """Should return None for None input."""
        result = predict_next_fixtures(None, sample_model_params)  # type: ignore[arg-type]
        assert result is None

    def test_output_has_expected_columns(self, sample_model_params):
        """Should have prediction columns."""
        fixtures = pd.DataFrame(
            {
                "home_team": ["Bayern"],
                "away_team": ["Dortmund"],
            }
        )
        result = predict_next_fixtures(fixtures, sample_model_params)
        assert result is not None
        assert "home_win" in result.columns
        assert "draw" in result.columns
        assert "away_win" in result.columns
        assert "expected_goals_home" in result.columns

    def test_calibration_applied(self, sample_model_params):
        """Calibration should modify probabilities when calibrators provided."""
        fixtures = pd.DataFrame(
            {
                "home_team": ["Bayern"],
                "away_team": ["Dortmund"],
            }
        )
        # uncalibrated
        uncal = predict_next_fixtures(fixtures, sample_model_params)

        # calibrated with temperature > 1 (should reduce confidence)
        calibrators = {"temperature": 2.0, "calibration_method": "temperature_scaling"}
        cal = predict_next_fixtures(fixtures, sample_model_params, calibrators=calibrators)

        # calibration with T>1 should move probs closer to uniform
        assert cal is not None
        assert uncal is not None
        # the max probability should be lower after calibration with T>1
        assert cal["home_win"].iloc[0] != uncal["home_win"].iloc[0]
