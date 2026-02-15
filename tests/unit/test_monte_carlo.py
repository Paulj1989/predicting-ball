# tests/unit/test_monte_carlo.py
"""Tests for Monte Carlo simulation module."""

import numpy as np
import pandas as pd
import pytest

from src.simulation.monte_carlo import (
    create_final_summary,
    get_current_standings,
    simulate_remaining_season_calibrated,
)


@pytest.fixture
def played_matches():
    """Simple set of played matches."""
    return pd.DataFrame(
        {
            "home_team": ["Bayern", "Dortmund", "Leipzig"],
            "away_team": ["Dortmund", "Leipzig", "Bayern"],
            "home_goals": [2, 1, 0],
            "away_goals": [1, 1, 2],
        }
    )


@pytest.fixture
def future_fixtures():
    """Unplayed future fixtures."""
    return pd.DataFrame(
        {
            "home_team": ["Bayern", "Dortmund"],
            "away_team": ["Leipzig", "Bayern"],
        }
    )


class TestGetCurrentStandings:
    """Tests for current standings calculation."""

    def test_returns_dict_with_all_teams(self, played_matches):
        """Should return standings for every team in the data."""
        standings = get_current_standings(played_matches)
        assert set(standings.keys()) == {"Bayern", "Dortmund", "Leipzig"}

    def test_points_calculation(self, played_matches):
        """Points should follow 3-1-0 system."""
        standings = get_current_standings(played_matches)
        # Bayern: beat Dortmund 2-1 at home (3pts), beat Leipzig 0-2 away (3pts) = 6
        assert standings["Bayern"]["points"] == 6
        # Dortmund: lost to Bayern 1-2 away (0pts), drew with Leipzig 1-1 at home (1pt) = 1
        assert standings["Dortmund"]["points"] == 1
        # Leipzig: drew with Dortmund 1-1 away (1pt), lost to Bayern 0-2 at home (0pts) = 1
        assert standings["Leipzig"]["points"] == 1

    def test_goal_difference(self, played_matches):
        """Goal difference should be GF - GA."""
        standings = get_current_standings(played_matches)
        for team in standings:
            gf = standings[team]["goals_for"]
            ga = standings[team]["goals_against"]
            gd = standings[team]["goal_diff"]
            assert gd == gf - ga

    def test_games_played(self, played_matches):
        """Games played should count home and away appearances."""
        standings = get_current_standings(played_matches)
        for team in standings:
            assert standings[team]["games_played"] == 2

    def test_empty_dataframe(self):
        """Should return empty dict for no matches."""
        standings = get_current_standings(
            pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals"])  # type: ignore[arg-type]
        )
        assert standings == {}

    def test_single_match_home_win(self):
        """Single match: home win gives 3pts to home, 0 to away."""
        df = pd.DataFrame(
            {
                "home_team": ["A"],
                "away_team": ["B"],
                "home_goals": [3],
                "away_goals": [0],
            }
        )
        standings = get_current_standings(df)
        assert standings["A"]["points"] == 3
        assert standings["B"]["points"] == 0
        assert standings["A"]["goals_for"] == 3
        assert standings["B"]["goals_against"] == 3

    def test_single_match_draw(self):
        """Single match: draw gives 1pt each."""
        df = pd.DataFrame(
            {
                "home_team": ["A"],
                "away_team": ["B"],
                "home_goals": [1],
                "away_goals": [1],
            }
        )
        standings = get_current_standings(df)
        assert standings["A"]["points"] == 1
        assert standings["B"]["points"] == 1


class TestSimulateRemainingSeason:
    """Tests for season simulation."""

    def test_returns_results_and_teams(
        self, future_fixtures, sample_bootstrap_params, played_matches
    ):
        """Should return results dict and team list."""
        standings = get_current_standings(played_matches)
        results, teams = simulate_remaining_season_calibrated(
            future_fixtures,
            sample_bootstrap_params,
            standings,
            n_simulations=10,
            seed=42,
        )
        assert results is not None
        assert teams is not None
        assert "points" in results
        assert "positions" in results

    def test_correct_number_of_simulations(
        self, future_fixtures, sample_bootstrap_params, played_matches
    ):
        """Results should have n_simulations rows."""
        standings = get_current_standings(played_matches)
        results, _teams = simulate_remaining_season_calibrated(
            future_fixtures,
            sample_bootstrap_params,
            standings,
            n_simulations=20,
            seed=42,
        )
        assert results is not None
        assert results["points"].shape[0] == 20

    def test_returns_none_for_empty_bootstrap(self, future_fixtures, played_matches):
        """Should return None when bootstrap params are empty."""
        standings = get_current_standings(played_matches)
        results, teams = simulate_remaining_season_calibrated(
            future_fixtures, [], standings, n_simulations=10
        )
        assert results is None
        assert teams is None

    def test_points_non_decreasing_from_base(
        self, future_fixtures, sample_bootstrap_params, played_matches
    ):
        """Simulated points should be >= base standing points."""
        standings = get_current_standings(played_matches)
        results, teams = simulate_remaining_season_calibrated(
            future_fixtures,
            sample_bootstrap_params,
            standings,
            n_simulations=10,
            seed=42,
        )
        assert results is not None
        assert teams is not None
        for i, team in enumerate(teams):
            base_pts = standings.get(team, {}).get("points", 0)
            assert np.all(results["points"][:, i] >= base_pts)


class TestCreateFinalSummary:
    """Tests for summary table creation."""

    def test_returns_dataframe(self, sample_model_params):
        """Should return a DataFrame."""
        teams = ["Bayern", "Dortmund", "Leipzig"]
        n_sims = 100
        results = {
            "points": np.random.randint(20, 80, size=(n_sims, 3)).astype(float),
            "goals_for": np.random.randint(30, 80, size=(n_sims, 3)).astype(float),
            "goals_against": np.random.randint(20, 60, size=(n_sims, 3)).astype(float),
            "positions": np.zeros((n_sims, 3), dtype=int),
        }
        # set positions (1-indexed)
        for s in range(n_sims):
            order = np.argsort(-results["points"][s])
            for pos, idx in enumerate(order):
                results["positions"][s, idx] = pos + 1

        standings = {
            t: {"points": 0, "goals_for": 0, "goals_against": 0, "games_played": 0}
            for t in teams
        }

        # add rating keys to params
        params = sample_model_params.copy()
        for t in teams:
            if t not in params["attack_rating"]:
                params["attack_rating"][t] = 0.0
                params["defense_rating"][t] = 0.0
                params["overall_rating"][t] = 0.0

        df = create_final_summary(results, params, teams, standings)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_has_expected_columns(self, sample_model_params):
        """Should contain projection and probability columns."""
        teams = ["Bayern", "Dortmund", "Leipzig"]
        n_sims = 50
        results = {
            "points": np.random.randint(20, 80, size=(n_sims, 3)).astype(float),
            "goals_for": np.random.randint(30, 80, size=(n_sims, 3)).astype(float),
            "goals_against": np.random.randint(20, 60, size=(n_sims, 3)).astype(float),
            "positions": np.ones((n_sims, 3), dtype=int),
        }
        standings = {t: {"points": 0} for t in teams}

        params = sample_model_params.copy()
        for t in teams:
            if t not in params["attack_rating"]:
                params["attack_rating"][t] = 0.0
                params["defense_rating"][t] = 0.0
                params["overall_rating"][t] = 0.0

        df = create_final_summary(results, params, teams, standings)
        assert "team" in df.columns
        assert "projected_points" in df.columns
        assert "title_prob" in df.columns
        assert "ucl_prob" in df.columns
        assert "relegation_prob" in df.columns

    def test_probabilities_bounded(self, sample_model_params):
        """Probabilities should be in [0, 1]."""
        teams = ["Bayern", "Dortmund", "Leipzig"]
        n_sims = 50
        results = {
            "points": np.random.randint(20, 80, size=(n_sims, 3)).astype(float),
            "goals_for": np.random.randint(30, 80, size=(n_sims, 3)).astype(float),
            "goals_against": np.random.randint(20, 60, size=(n_sims, 3)).astype(float),
            "positions": np.zeros((n_sims, 3), dtype=int),
        }
        for s in range(n_sims):
            order = np.argsort(-results["points"][s])
            for pos, idx in enumerate(order):
                results["positions"][s, idx] = pos + 1

        standings = {t: {"points": 0} for t in teams}
        params = sample_model_params.copy()
        for t in teams:
            if t not in params["attack_rating"]:
                params["attack_rating"][t] = 0.0
                params["defense_rating"][t] = 0.0
                params["overall_rating"][t] = 0.0

        df = create_final_summary(results, params, teams, standings)
        for col in ["title_prob", "ucl_prob", "relegation_prob"]:
            assert df[col].between(0, 1).all()
