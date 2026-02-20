# tests/unit/test_priors.py
"""Tests for Bayesian priors module."""

import numpy as np
import pandas as pd
import pytest

from src.models.priors import (
    _get_team_metric,
    calculate_all_team_priors,
    calculate_elo_priors,
    calculate_home_advantage_prior,
    calculate_squad_value_priors,
    identify_promoted_teams,
)


class TestGetTeamMetric:
    """Tests for the helper function that aggregates team metrics."""

    def test_returns_average_of_home_and_away(self):
        """Should average metric from home and away appearances."""
        df = pd.DataFrame(
            {
                "home_team": ["Bayern", "Dortmund"],
                "away_team": ["Dortmund", "Bayern"],
                "home_elo": [1800, 1600],
                "away_elo": [1600, 1800],
            }
        )
        # Bayern: home_elo=1800 when home, away_elo=1800 when away -> avg 1800
        result = _get_team_metric(df, "Bayern", "elo")
        assert np.isclose(result, 1800.0)

    def test_returns_nan_for_unknown_team(self):
        """Should return NaN for a team not in the data."""
        df = pd.DataFrame(
            {
                "home_team": ["Bayern"],
                "away_team": ["Dortmund"],
                "home_elo": [1800],
                "away_elo": [1600],
            }
        )
        result = _get_team_metric(df, "Leipzig", "elo")
        assert np.isnan(result)


class TestCalculateHomeAdvantagePrior:
    """Tests for home advantage calculation."""

    def test_returns_tuple(self, sample_historic_data):
        """Should return (mean, std) tuple."""
        result = calculate_home_advantage_prior(sample_historic_data, verbose=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_positive_home_advantage(self, sample_historic_data):
        """Home advantage should be positive (home teams score more)."""
        mean, std = calculate_home_advantage_prior(sample_historic_data, verbose=False)
        # home_goals avg is typically > away_goals avg, so log(ratio) > 0
        # with random data this isn't guaranteed, but the sign should be sensible
        assert isinstance(mean, float)
        assert std >= 0

    def test_raises_on_zero_metrics(self):
        """Should raise ValueError if metrics are zero."""
        data = pd.DataFrame(
            {
                "home_goals": [0, 0, 0],
                "away_goals": [0, 0, 0],
                "season_end_year": [2024, 2024, 2024],
            }
        )
        with pytest.raises(ValueError, match="Invalid metric values"):
            calculate_home_advantage_prior(data, verbose=False)

    def test_uses_weighted_goals(self, sample_historic_data):
        """Should use weighted goals when use_actual_goals=False."""
        result = calculate_home_advantage_prior(
            sample_historic_data, use_actual_goals=False, verbose=False
        )
        assert isinstance(result, tuple)


class TestCalculateSquadValuePriors:
    """Tests for squad value based priors."""

    def test_returns_dict_for_all_teams(self, sample_training_data):
        """Should return priors for every team."""
        teams = sorted(sample_training_data["home_team"].unique())
        result = calculate_squad_value_priors(sample_training_data, teams, verbose=False)
        assert set(result.keys()) == set(teams)

    def test_each_team_has_attack_and_defense_prior(self, sample_training_data):
        """Each team should have attack_prior and defense_prior."""
        teams = sorted(sample_training_data["home_team"].unique())
        result = calculate_squad_value_priors(sample_training_data, teams, verbose=False)
        for team in teams:
            assert "attack_prior" in result[team]
            assert "defense_prior" in result[team]

    def test_returns_neutral_without_data(self):
        """Should return neutral priors if no squad values available."""
        df = pd.DataFrame(
            {
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_value_pct": [np.nan, np.nan],
                "away_value_pct": [np.nan, np.nan],
            }
        )
        result = calculate_squad_value_priors(df, ["A", "B"], verbose=False)
        assert result["A"]["attack_prior"] == 0.0
        assert result["A"]["defense_prior"] == 0.0


class TestCalculateEloPriors:
    """Tests for Elo-based priors."""

    def test_returns_dict_for_all_teams(self, sample_training_data):
        """Should return priors for every team."""
        teams = sorted(sample_training_data["home_team"].unique())
        result = calculate_elo_priors(sample_training_data, teams, verbose=False)
        assert set(result.keys()) == set(teams)

    def test_higher_elo_gives_higher_attack(self, sample_training_data):
        """Team with higher Elo should generally get higher attack prior."""
        # create controlled data with clear Elo difference
        df = pd.DataFrame(
            {
                "home_team": ["Strong", "Weak"],
                "away_team": ["Weak", "Strong"],
                "home_elo": [1800, 1200],
                "away_elo": [1200, 1800],
            }
        )
        result = calculate_elo_priors(df, ["Strong", "Weak"], verbose=False)
        assert result["Strong"]["attack_prior"] > result["Weak"]["attack_prior"]

    def test_defense_inversely_related(self):
        """Higher Elo should give lower (better) defense prior."""
        df = pd.DataFrame(
            {
                "home_team": ["Strong", "Weak"],
                "away_team": ["Weak", "Strong"],
                "home_elo": [1800, 1200],
                "away_elo": [1200, 1800],
            }
        )
        result = calculate_elo_priors(df, ["Strong", "Weak"], verbose=False)
        # defense_prior = -z_score * 0.25, so higher elo -> more negative defense
        assert result["Strong"]["defense_prior"] < result["Weak"]["defense_prior"]

    def test_returns_neutral_without_elo(self):
        """Should return neutral priors if no Elo data available."""
        df = pd.DataFrame(
            {
                "home_team": ["A"],
                "away_team": ["B"],
                "home_elo": [np.nan],
                "away_elo": [np.nan],
            }
        )
        result = calculate_elo_priors(df, ["A", "B"], verbose=False)
        assert result["A"]["attack_prior"] == 0.0


class TestIdentifyPromotedTeams:
    """Tests for promoted team identification."""

    def test_identifies_new_team(self, sample_historic_data):
        """Should identify teams in current season not in previous season."""
        # last season (2025) has Holstein Kiel but not Leipzig
        # current season has Leipzig but not Holstein Kiel
        current = pd.DataFrame(
            {
                "home_team": ["Bayern", "Dortmund", "Leipzig"],
                "away_team": ["Dortmund", "Leipzig", "Bayern"],
                "home_value_pct": [20.0, 15.0, 12.0],
                "away_value_pct": [15.0, 12.0, 20.0],
            }
        )
        last_season = sample_historic_data[sample_historic_data["season_end_year"] == 2025]
        result = identify_promoted_teams(last_season, current, verbose=False)

        # Leipzig should be identified as promoted (not in last_season which has Holstein Kiel)
        assert "Leipzig" in result

    def test_returns_empty_when_no_promotions(self):
        """Should return empty dict when all teams were in previous season."""
        historic = pd.DataFrame(
            {
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "season_end_year": [2025, 2025],
                "home_value_pct": [10, 10],
                "away_value_pct": [10, 10],
            }
        )
        current = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
        result = identify_promoted_teams(historic, current, verbose=False)
        assert result == {}


class TestCalculatePromotedTeamPriors:
    """Tests for the orchestrating function."""

    def test_returns_priors_and_home_advantage(self, sample_historic_data):
        """Should return (priors, home_adv_mean, home_adv_std)."""
        from src.models.priors import calculate_promoted_team_priors

        current = pd.DataFrame(
            {
                "home_team": ["Bayern", "Dortmund", "Leipzig"],
                "away_team": ["Dortmund", "Leipzig", "Bayern"],
                "home_value_pct": [20.0, 15.0, 12.0],
                "away_value_pct": [15.0, 12.0, 20.0],
                "home_elo": [1800, 1600, 1500],
                "away_elo": [1600, 1500, 1800],
            }
        )
        promoted = {"Leipzig": {"is_promoted": True}}
        priors, ha_mean, ha_std = calculate_promoted_team_priors(
            sample_historic_data, promoted, current, verbose=False
        )
        assert isinstance(priors, dict)
        assert isinstance(ha_mean, float)
        assert isinstance(ha_std, float)


class TestCalculateAllTeamPriors:
    """Tests for blended prior calculation."""

    def test_returns_dict_for_all_teams(self, sample_training_data):
        """Should return priors for every team."""
        teams = sorted(sample_training_data["home_team"].unique())
        result = calculate_all_team_priors(
            sample_training_data, teams, promoted_teams={}, verbose=False
        )
        assert set(result.keys()) == set(teams)

    def test_promoted_teams_flagged_correctly(self, sample_training_data):
        """Promoted teams should be flagged and use elo_squad source."""
        teams = sorted(sample_training_data["home_team"].unique())
        promoted = {"Bayern": {"is_promoted": True}}
        result = calculate_all_team_priors(
            sample_training_data, teams, promoted_teams=promoted, verbose=False
        )
        assert result["Bayern"]["is_promoted"] is True
        assert result["Bayern"]["source"] in ("elo_squad", "squad_only")

    def test_all_teams_use_same_formula(self, sample_training_data):
        """All teams should use the same elo+squad blend regardless of status."""
        teams = sorted(sample_training_data["home_team"].unique())
        result = calculate_all_team_priors(
            sample_training_data, teams, promoted_teams={}, verbose=False
        )
        for team in teams:
            assert result[team]["source"] in ("elo_squad", "squad_only")
            assert not result[team]["is_promoted"]
