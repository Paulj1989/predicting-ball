# tests/unit/test_ratings.py
"""Tests for interpretable ratings module."""

import numpy as np

from src.models.ratings import (
    add_interpretable_ratings_to_params,
    create_interpretable_ratings,
)


class TestCreateInterpretableRatings:
    """Tests for z-score based rating transformation."""

    def test_returns_expected_keys(self):
        """Should return attack, defense, and overall rating dicts."""
        params = {
            "teams": ["A", "B", "C"],
            "attack": {"A": 0.3, "B": 0.0, "C": -0.3},
            "defense": {"A": -0.2, "B": 0.0, "C": 0.2},
        }
        ratings = create_interpretable_ratings(params)

        assert set(ratings.keys()) == {"attack", "defense", "overall"}
        for key in ["attack", "defense", "overall"]:
            assert set(ratings[key].keys()) == {"A", "B", "C"}

    def test_ratings_clipped_to_minus_one_one(self):
        """All ratings should be in [-1, 1]."""
        params = {
            "teams": ["A", "B", "C", "D"],
            "attack": {"A": 2.0, "B": -2.0, "C": 0.5, "D": -0.5},
            "defense": {"A": -1.5, "B": 1.5, "C": 0.0, "D": 0.0},
        }
        ratings = create_interpretable_ratings(params)

        for category in ["attack", "defense", "overall"]:
            for val in ratings[category].values():
                assert -1.0 <= val <= 1.0

    def test_mean_approximately_zero(self):
        """Z-score normalisation should center ratings at 0."""
        params = {
            "teams": ["A", "B", "C", "D", "E"],
            "attack": {"A": 0.4, "B": 0.2, "C": 0.0, "D": -0.2, "E": -0.4},
            "defense": {"A": -0.3, "B": -0.1, "C": 0.0, "D": 0.1, "E": 0.3},
        }
        ratings = create_interpretable_ratings(params)

        attack_mean = np.mean(list(ratings["attack"].values()))
        defense_mean = np.mean(list(ratings["defense"].values()))
        assert abs(attack_mean) < 1e-10
        assert abs(defense_mean) < 1e-10

    def test_defense_sign_flipped(self):
        """Positive defense param (bad) should give negative rating (bad)."""
        params = {
            "teams": ["Good", "Bad"],
            "attack": {"Good": 0.0, "Bad": 0.0},
            # lower defense param = better, so sign flip means Good > Bad
            "defense": {"Good": -0.5, "Bad": 0.5},
        }
        ratings = create_interpretable_ratings(params)

        # after sign flip: Good's raw = 0.5 (good), Bad's raw = -0.5 (bad)
        assert ratings["defense"]["Good"] > ratings["defense"]["Bad"]

    def test_overall_is_mean_of_attack_and_defense(self):
        """Overall rating should be average of attack and defense."""
        params = {
            "teams": ["A", "B", "C"],
            "attack": {"A": 0.3, "B": 0.0, "C": -0.3},
            "defense": {"A": -0.2, "B": 0.0, "C": 0.2},
        }
        ratings = create_interpretable_ratings(params)

        for team in ["A", "B", "C"]:
            expected = (ratings["attack"][team] + ratings["defense"][team]) / 2
            assert np.isclose(ratings["overall"][team], expected)

    def test_equal_teams_get_identical_ratings(self):
        """All teams with identical params should get identical ratings."""
        params = {
            "teams": ["A", "B", "C"],
            "attack": {"A": 0.1, "B": 0.1, "C": 0.1},
            "defense": {"A": 0.1, "B": 0.1, "C": 0.1},
        }
        ratings = create_interpretable_ratings(params)

        # all teams should get the same rating (degenerate case)
        for cat in ["attack", "defense", "overall"]:
            values = list(ratings[cat].values())
            assert all(v == values[0] for v in values)


class TestAddInterpretableRatingsToParams:
    """Tests for adding ratings to params dict."""

    def test_adds_rating_keys(self):
        """Should add attack_rating, defense_rating, overall_rating to params."""
        params = {
            "teams": ["A", "B"],
            "attack": {"A": 0.3, "B": -0.3},
            "defense": {"A": -0.2, "B": 0.2},
        }
        result = add_interpretable_ratings_to_params(params)

        assert "attack_rating" in result
        assert "defense_rating" in result
        assert "overall_rating" in result

    def test_returns_same_dict(self):
        """Should modify and return the same dict."""
        params = {
            "teams": ["A", "B"],
            "attack": {"A": 0.3, "B": -0.3},
            "defense": {"A": -0.2, "B": 0.2},
        }
        result = add_interpretable_ratings_to_params(params)
        assert result is params

    def test_preserves_existing_keys(self):
        """Should not remove existing params."""
        params = {
            "teams": ["A", "B"],
            "attack": {"A": 0.3, "B": -0.3},
            "defense": {"A": -0.2, "B": 0.2},
            "home_adv": 0.25,
            "rho": -0.13,
        }
        result = add_interpretable_ratings_to_params(params)

        assert result["home_adv"] == 0.25
        assert result["rho"] == -0.13
