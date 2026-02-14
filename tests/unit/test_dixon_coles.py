# tests/unit/test_dixon_coles.py
"""Tests for Dixon-Coles model components."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from src.models.dixon_coles import (
    calculate_match_probabilities_dixon_coles,
    tau_dixon_coles,
)


class TestTauDixonColes:
    """Tests for the tau correction function."""

    def test_tau_0_0_negative_rho(self):
        """0-0 with negative rho should increase probability."""
        tau = tau_dixon_coles(0, 0, 1.5, 1.2, rho=-0.1)
        # tau = 1 - lambda_home * lambda_away * rho
        # tau = 1 - 1.5 * 1.2 * (-0.1) = 1 + 0.18 = 1.18
        assert tau > 1.0

    def test_tau_0_1_negative_rho(self):
        """0-1 with negative rho should decrease probability."""
        tau = tau_dixon_coles(0, 1, 1.5, 1.2, rho=-0.1)
        # tau = 1 + lambda_home * rho = 1 + 1.5 * (-0.1) = 0.85
        assert tau < 1.0

    def test_tau_1_0_negative_rho(self):
        """1-0 with negative rho should decrease probability."""
        tau = tau_dixon_coles(1, 0, 1.5, 1.2, rho=-0.1)
        # tau = 1 + lambda_away * rho = 1 + 1.2 * (-0.1) = 0.88
        assert tau < 1.0

    def test_tau_1_1_negative_rho(self):
        """1-1 with negative rho should increase probability."""
        tau = tau_dixon_coles(1, 1, 1.5, 1.2, rho=-0.1)
        # tau = 1 - rho = 1 - (-0.1) = 1.1
        assert tau > 1.0

    def test_tau_higher_scores_unchanged(self):
        """Scores > 1 should return tau = 1.0."""
        assert tau_dixon_coles(2, 0, 1.5, 1.2, -0.1) == 1.0
        assert tau_dixon_coles(0, 2, 1.5, 1.2, -0.1) == 1.0
        assert tau_dixon_coles(3, 2, 1.5, 1.2, -0.1) == 1.0
        assert tau_dixon_coles(2, 2, 1.5, 1.2, -0.1) == 1.0

    @given(
        st.integers(min_value=0, max_value=10),
        st.integers(min_value=0, max_value=10),
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=-0.3, max_value=0.1),
    )
    def test_tau_always_positive(self, home_goals, away_goals, lambda_home, lambda_away, rho):
        """Tau should always be positive for valid inputs."""
        tau = tau_dixon_coles(home_goals, away_goals, lambda_home, lambda_away, rho)
        assert tau > 0


class TestMatchProbabilities:
    """Tests for match probability calculations."""

    def test_probabilities_sum_to_one(self):
        """Outcome probabilities should sum to 1."""
        home, draw, away, _ = calculate_match_probabilities_dixon_coles(
            lambda_home=1.5, lambda_away=1.2, rho=-0.13
        )
        assert np.isclose(home + draw + away, 1.0, atol=1e-10)

    def test_stronger_home_team(self):
        """Higher home lambda should give higher home win probability."""
        home1, _, _, _ = calculate_match_probabilities_dixon_coles(1.5, 1.0, -0.13)
        home2, _, _, _ = calculate_match_probabilities_dixon_coles(2.5, 1.0, -0.13)
        assert home2 > home1

    def test_stronger_away_team(self):
        """Higher away lambda should give higher away win probability."""
        _, _, away1, _ = calculate_match_probabilities_dixon_coles(1.5, 1.0, -0.13)
        _, _, away2, _ = calculate_match_probabilities_dixon_coles(1.5, 2.0, -0.13)
        assert away2 > away1

    def test_equal_teams_symmetric(self):
        """Equal lambdas should give roughly symmetric home/away probabilities."""
        home, _draw, away, _ = calculate_match_probabilities_dixon_coles(
            lambda_home=1.5, lambda_away=1.5, rho=-0.13
        )
        # home and away should be close (within rounding)
        assert abs(home - away) < 0.01

    def test_score_probabilities_sum_correctly(self):
        """Individual score probabilities should sum to outcome probabilities."""
        home, draw, away, scores = calculate_match_probabilities_dixon_coles(
            lambda_home=1.5, lambda_away=1.2, rho=-0.13
        )

        home_from_scores = sum(p for (h, a), p in scores.items() if h > a)
        draw_from_scores = sum(p for (h, a), p in scores.items() if h == a)
        away_from_scores = sum(p for (h, a), p in scores.items() if h < a)

        # after normalisation they should match
        total = home_from_scores + draw_from_scores + away_from_scores
        assert np.isclose(home_from_scores / total, home, atol=1e-10)
        assert np.isclose(draw_from_scores / total, draw, atol=1e-10)
        assert np.isclose(away_from_scores / total, away, atol=1e-10)

    @given(
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=-0.3, max_value=-0.05),
    )
    def test_probabilities_valid_range(self, lambda_home, lambda_away, rho):
        """All probabilities should be between 0 and 1."""
        home, draw, away, scores = calculate_match_probabilities_dixon_coles(
            lambda_home=lambda_home, lambda_away=lambda_away, rho=rho
        )

        assert 0 <= home <= 1
        assert 0 <= draw <= 1
        assert 0 <= away <= 1
        assert np.isclose(home + draw + away, 1.0, atol=1e-10)

        for prob in scores.values():
            assert prob >= 0
