# tests/unit/test_hot_simulation.py
"""Tests for hot simulation with MLE posterior draws."""

import numpy as np
import pandas as pd
import pytest

from src.simulation.hot_simulation import simulate_season_hot


@pytest.fixture
def simple_fixtures():
    """Minimal season fixture list (3 teams, 6 matches)"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-08-01",
                    "2024-08-01",
                    "2024-08-01",
                    "2024-08-15",
                    "2024-08-15",
                    "2024-08-15",
                ]
            ),
            "matchweek": [1, 1, 1, 2, 2, 2],
            "home_team": ["A", "B", "C", "B", "A", "C"],
            "away_team": ["B", "C", "A", "A", "C", "B"],
        }
    )


@pytest.fixture
def state_mean():
    """State vector for 3 teams: [att_A, att_B, att_C, def_A, def_B, def_C, ha]"""
    return np.array([0.3, 0.0, -0.3, -0.2, 0.1, 0.1, 0.25])


@pytest.fixture
def state_cov():
    """Small covariance for 3-team state (7x7)"""
    return np.eye(7) * 0.01


@pytest.fixture
def state_teams():
    """Team ordering matching the state vector"""
    return ["A", "B", "C"]


@pytest.fixture
def current_standings():
    """Empty standings (simulating full season from scratch)"""
    return {
        "A": {
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "games_played": 0,
        },
        "B": {
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "games_played": 0,
        },
        "C": {
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "games_played": 0,
        },
    }


class TestSimulateSeasonHot:
    """Tests for hot season simulation."""

    def test_returns_correct_structure(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """Should return (results_dict, teams_list)"""
        results, teams = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=10,
            seed=42,
        )
        assert isinstance(results, dict)
        assert "points" in results
        assert "goals_for" in results
        assert "goals_against" in results
        assert "positions" in results
        assert isinstance(teams, list)

    def test_output_shapes(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """Arrays should be (n_simulations, n_teams)"""
        n_sims = 50
        results, teams = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=n_sims,
            seed=42,
        )
        n_teams = len(teams)
        assert results["points"].shape == (n_sims, n_teams)
        assert results["positions"].shape == (n_sims, n_teams)

    def test_points_are_non_negative(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """No team should have negative points"""
        results, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=100,
            seed=42,
        )
        assert np.all(results["points"] >= 0)

    def test_positions_are_valid(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """Positions should be 1..n_teams with no duplicates per sim"""
        results, teams = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=100,
            seed=42,
        )
        n_teams = len(teams)
        for s in range(100):
            positions = results["positions"][s]
            assert set(positions) == set(range(1, n_teams + 1))

    def test_total_points_bounded(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """Total points should be bounded: each match awards 2 (draw) or 3 points"""
        results, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=100,
            seed=42,
        )
        n_matches = len(simple_fixtures)
        # draws award 2 points (1+1), decisive results award 3 (3+0)
        min_total = n_matches * 2
        max_total = n_matches * 3
        for s in range(100):
            total = results["points"][s].sum()
            assert min_total <= total <= max_total

    def test_hot_produces_wider_spread_than_cold(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """Hot simulation should produce wider point distributions than cold"""
        results_cold, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=2000,
            K_att=0.0,
            K_def=0.0,
            seed=42,
        )
        results_hot, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=2000,
            K_att=0.05,
            K_def=0.025,
            seed=42,
        )
        cold_spread = results_cold["points"].std(axis=0).mean()
        hot_spread = results_hot["points"].std(axis=0).mean()
        assert hot_spread > cold_spread

    def test_k_zero_is_cold_simulation(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """K=0 should give same results as no hot updates"""
        results, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=100,
            K_att=0.0,
            K_def=0.0,
            seed=42,
        )
        # just check it runs and produces valid results
        assert results["points"].shape[0] == 100

    def test_accumulates_current_standings(
        self, simple_fixtures, state_mean, state_cov, state_teams
    ):
        """Should add simulated points on top of current standings"""
        standings_with_points = {
            "A": {
                "points": 10,
                "goals_for": 5,
                "goals_against": 2,
                "goal_diff": 3,
                "games_played": 4,
            },
            "B": {
                "points": 7,
                "goals_for": 3,
                "goals_against": 3,
                "goal_diff": 0,
                "games_played": 4,
            },
            "C": {
                "points": 3,
                "goals_for": 2,
                "goals_against": 5,
                "goal_diff": -3,
                "games_played": 4,
            },
        }
        results, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            standings_with_points,
            state_teams=state_teams,
            n_simulations=100,
            seed=42,
        )
        # all teams should have at least their starting points
        teams = ["A", "B", "C"]
        for i, team in enumerate(sorted(teams)):
            min_pts = standings_with_points[team]["points"]
            assert np.all(results["points"][:, i] >= min_pts)

    def test_runs_with_minimal_args(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """Should run without error using only required arguments"""
        results, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=10,
            seed=42,
        )
        assert results["points"].shape[0] == 10

    def test_rho_affects_draw_rate(
        self, simple_fixtures, state_mean, state_cov, current_standings, state_teams
    ):
        """Negative rho should produce higher draw rate than rho=0"""
        n_sims = 3000

        results_rho, _teams = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=n_sims,
            K_att=0.0,
            K_def=0.0,
            rho=-0.13,
            seed=42,
        )

        results_no_rho, _ = simulate_season_hot(
            simple_fixtures,
            state_mean,
            state_cov,
            current_standings,
            state_teams=state_teams,
            n_simulations=n_sims,
            K_att=0.0,
            K_def=0.0,
            rho=0.0,
            seed=42,
        )

        # count draws: matches where both teams got 1 point
        # total points per sim: each draw adds 2 pts, each decisive result adds 3
        n_matches = len(simple_fixtures)
        # max possible total = 3 * n_matches (all decisive)
        # each draw reduces total by 1 (2 pts instead of 3)
        total_rho = results_rho["points"].sum(axis=1)
        total_no_rho = results_no_rho["points"].sum(axis=1)

        # more draws means lower total points on average
        draws_rho = (n_matches * 3 - total_rho).mean()
        draws_no_rho = (n_matches * 3 - total_no_rho).mean()

        assert draws_rho > draws_no_rho
