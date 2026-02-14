# tests/property/test_probabilities.py
"""Property-based tests for probability calculations."""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.models.dixon_coles import calculate_match_probabilities_dixon_coles


class TestProbabilityInvariants:
    """Property-based tests for probability invariants."""

    @given(
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=-0.25, max_value=-0.05),
    )
    @settings(max_examples=200)
    def test_probabilities_sum_to_one(self, lambda_home, lambda_away, rho):
        """Outcome probabilities must always sum to 1."""
        home, draw, away, _ = calculate_match_probabilities_dixon_coles(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=rho,
        )

        total = home + draw + away
        assert np.isclose(total, 1.0, atol=1e-9), f"Sum was {total}, not 1.0"

    @given(
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=-0.25, max_value=-0.05),
    )
    @settings(max_examples=200)
    def test_probabilities_non_negative(self, lambda_home, lambda_away, rho):
        """All probabilities must be non-negative."""
        home, draw, away, scores = calculate_match_probabilities_dixon_coles(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=rho,
        )

        assert home >= 0, f"Home probability was negative: {home}"
        assert draw >= 0, f"Draw probability was negative: {draw}"
        assert away >= 0, f"Away probability was negative: {away}"

        for (h, a), prob in scores.items():
            assert prob >= 0, f"Score {h}-{a} probability was negative: {prob}"

    @given(
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=-0.25, max_value=-0.05),
    )
    @settings(max_examples=200)
    def test_probabilities_at_most_one(self, lambda_home, lambda_away, rho):
        """Individual probabilities must be at most 1."""
        home, draw, away, _ = calculate_match_probabilities_dixon_coles(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=rho,
        )

        assert home <= 1.0, f"Home probability exceeded 1: {home}"
        assert draw <= 1.0, f"Draw probability exceeded 1: {draw}"
        assert away <= 1.0, f"Away probability exceeded 1: {away}"

    @given(
        st.floats(min_value=0.5, max_value=3.0),
        st.floats(min_value=-0.25, max_value=-0.05),
    )
    @settings(max_examples=100)
    def test_equal_teams_near_symmetric(self, lambda_val, rho):
        """Equal team strengths should produce near-symmetric probabilities."""
        home, _draw, away, _ = calculate_match_probabilities_dixon_coles(
            lambda_home=lambda_val,
            lambda_away=lambda_val,
            rho=rho,
        )

        # home and away should be very close for equal teams
        assert (
            abs(home - away) < 0.02
        ), f"Asymmetric probabilities for equal teams: home={home}, away={away}"

    @given(
        st.floats(min_value=1.0, max_value=3.0),
        st.floats(min_value=0.3, max_value=0.9),
        st.floats(min_value=-0.25, max_value=-0.05),
    )
    @settings(max_examples=100)
    def test_stronger_home_team_favoured(self, lambda_home, lambda_away, rho):
        """Stronger home team should have higher win probability than away."""
        assume(lambda_home > lambda_away + 0.3)  # ensure meaningful difference

        home, _draw, away, _ = calculate_match_probabilities_dixon_coles(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=rho,
        )

        assert home > away, f"Stronger home team not favoured: home={home}, away={away}"

    @given(
        st.floats(min_value=0.3, max_value=0.9),
        st.floats(min_value=1.0, max_value=3.0),
        st.floats(min_value=-0.25, max_value=-0.05),
    )
    @settings(max_examples=100)
    def test_stronger_away_team_favoured(self, lambda_home, lambda_away, rho):
        """Stronger away team should have higher win probability than home."""
        assume(lambda_away > lambda_home + 0.3)  # ensure meaningful difference

        home, _draw, away, _ = calculate_match_probabilities_dixon_coles(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=rho,
        )

        assert away > home, f"Stronger away team not favoured: home={home}, away={away}"
