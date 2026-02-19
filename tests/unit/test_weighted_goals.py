# tests/unit/test_weighted_goals.py
"""Tests for weighted goals calculation."""

import numpy as np
import pandas as pd
import pytest

from src.features.weighted_goals import calculate_weighted_goals


@pytest.fixture
def raw_goals_data():
    """DataFrame with raw npxG, npG, and penalty columns."""
    return pd.DataFrame(
        {
            "home_npxg": [1.5, 0.8, 2.1],
            "away_npxg": [0.9, 1.4, 0.6],
            "home_goals": [2, 0, 3],
            "away_goals": [1, 1, 0],
            "home_pens_made": [0, 0, 1],
            "away_pens_made": [1, 0, 0],
        }
    )


class TestCalculateWeightedGoals:
    def test_default_weight_is_point_seven(self, raw_goals_data):
        """Default xg_weight=0.7 should reproduce the original 70/30 blend."""
        result = calculate_weighted_goals(raw_goals_data)
        home_npg = raw_goals_data["home_goals"] - raw_goals_data["home_pens_made"]
        expected = 0.7 * raw_goals_data["home_npxg"] + 0.3 * home_npg
        assert np.allclose(result["home_goals_weighted"], expected)

    def test_different_weight_produces_different_result(self, raw_goals_data):
        """xg_weight=0.9 should differ from the default xg_weight=0.7."""
        result_07 = calculate_weighted_goals(raw_goals_data, xg_weight=0.7)
        result_09 = calculate_weighted_goals(raw_goals_data, xg_weight=0.9)
        assert not np.allclose(
            result_07["home_goals_weighted"], result_09["home_goals_weighted"]
        )

    def test_weight_one_gives_pure_xg(self, raw_goals_data):
        """xg_weight=1.0 should return pure npxG."""
        result = calculate_weighted_goals(raw_goals_data, xg_weight=1.0)
        assert np.allclose(result["home_goals_weighted"], raw_goals_data["home_npxg"])
        assert np.allclose(result["away_goals_weighted"], raw_goals_data["away_npxg"])

    def test_output_is_continuous(self, raw_goals_data):
        """Weighted goals should be continuous floats, not rounded integers."""
        result = calculate_weighted_goals(raw_goals_data)
        # at least one value should be non-integer
        vals = result["home_goals_weighted"].values
        assert any(v != round(v) for v in vals)
