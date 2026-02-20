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


class TestCreateInterpretableRatingsReferenceTeams:
    """Tests for reference_teams normalisation behaviour."""

    def test_all_params_teams_get_ratings(self):
        """All teams in params receive ratings even if not in reference_teams."""
        params = {
            "teams": ["Current1", "Current2", "Current3", "Historical"],
            "attack": {"Current1": 0.5, "Current2": 0.1, "Current3": -0.2, "Historical": -0.9},
            "defense": {"Current1": -0.3, "Current2": 0.0, "Current3": 0.2, "Historical": 0.1},
        }
        ratings = create_interpretable_ratings(
            params, reference_teams=["Current1", "Current2", "Current3"]
        )

        assert set(ratings["attack"].keys()) == {
            "Current1",
            "Current2",
            "Current3",
            "Historical",
        }

    def test_reference_teams_are_centred_at_zero(self):
        """Mean of ratings for the reference population should be exactly 0."""
        params = {
            "teams": ["Strong", "Mid", "Weak", "Relegated"],
            "attack": {"Strong": 0.5, "Mid": 0.0, "Weak": -0.3, "Relegated": -0.9},
            "defense": {"Strong": -0.3, "Mid": 0.0, "Weak": 0.2, "Relegated": 0.3},
        }
        reference = ["Strong", "Mid", "Weak"]

        ratings = create_interpretable_ratings(params, reference_teams=reference)

        ref_att_mean = np.mean([ratings["attack"][t] for t in reference])
        ref_def_mean = np.mean([ratings["defense"][t] for t in reference])
        assert abs(ref_att_mean) < 1e-10
        assert abs(ref_def_mean) < 1e-10

    def test_weak_historical_teams_inflate_current_ratings_without_reference(self):
        """Without reference_teams, weak historical teams pull the mean down and make
        all current teams appear artificially above average."""
        params = {
            "teams": ["Strong", "Mid", "Weak", "Relegated1", "Relegated2", "Relegated3"],
            "attack": {
                "Strong": 0.5,
                "Mid": 0.1,
                "Weak": -0.2,
                "Relegated1": -0.8,
                "Relegated2": -0.9,
                "Relegated3": -1.0,
            },
            "defense": {
                "Strong": -0.3,
                "Mid": 0.0,
                "Weak": 0.2,
                "Relegated1": 0.0,
                "Relegated2": 0.0,
                "Relegated3": 0.0,
            },
        }
        current = ["Strong", "Mid", "Weak"]

        without_ref = create_interpretable_ratings(params)
        with_ref = create_interpretable_ratings(params, reference_teams=current)

        # without reference_teams, weak historical clubs pull mean attack down
        # → all current team z-scores shift positive → mean > 0
        current_att_without = np.mean([without_ref["attack"][t] for t in current])
        assert current_att_without > 0

        # with reference_teams, current teams are centred at 0
        current_att_with = np.mean([with_ref["attack"][t] for t in current])
        assert abs(current_att_with) < 1e-10

    def test_reference_teams_changes_individual_ratings(self):
        """Same params but different reference population produces different ratings."""
        params = {
            "teams": ["A", "B", "C", "D"],
            "attack": {"A": 0.5, "B": 0.3, "C": 0.1, "D": -0.9},
            "defense": {"A": -0.3, "B": 0.0, "C": 0.2, "D": 0.0},
        }
        all_ref = create_interpretable_ratings(params)
        subset_ref = create_interpretable_ratings(params, reference_teams=["A", "B", "C"])

        # D's extreme weakness shifts the all-team normalisation; removing it changes ratings
        assert all_ref["attack"]["A"] != subset_ref["attack"]["A"]

    def test_non_reference_team_above_average_when_strong(self):
        """A historical team stronger than the reference average gets a positive rating."""
        params = {
            "teams": ["RefA", "RefB", "RefC", "Historical"],
            "attack": {"RefA": 0.2, "RefB": 0.0, "RefC": -0.2, "Historical": 1.0},
            "defense": {"RefA": -0.1, "RefB": 0.0, "RefC": 0.1, "Historical": 0.0},
        }
        ratings = create_interpretable_ratings(
            params, reference_teams=["RefA", "RefB", "RefC"]
        )

        # historical team's attack is above the reference mean → positive rating
        assert ratings["attack"]["Historical"] > 0

    def test_non_reference_team_below_average_when_weak(self):
        """A historical team weaker than the reference average gets a negative rating."""
        params = {
            "teams": ["RefA", "RefB", "RefC", "Historical"],
            "attack": {"RefA": 0.2, "RefB": 0.0, "RefC": -0.2, "Historical": -1.0},
            "defense": {"RefA": -0.1, "RefB": 0.0, "RefC": 0.1, "Historical": 0.0},
        }
        ratings = create_interpretable_ratings(
            params, reference_teams=["RefA", "RefB", "RefC"]
        )

        assert ratings["attack"]["Historical"] < 0

    def test_no_reference_teams_matches_all_teams_default(self):
        """Omitting reference_teams produces the same result as passing all teams."""
        params = {
            "teams": ["A", "B", "C"],
            "attack": {"A": 0.3, "B": 0.0, "C": -0.3},
            "defense": {"A": -0.2, "B": 0.0, "C": 0.2},
        }
        default = create_interpretable_ratings(params)
        explicit_all = create_interpretable_ratings(params, reference_teams=["A", "B", "C"])

        for cat in ["attack", "defense", "overall"]:
            for team in ["A", "B", "C"]:
                assert np.isclose(default[cat][team], explicit_all[cat][team])

    def test_reference_teams_ordering_preserved_within_population(self):
        """Relative ordering among current teams should be the same regardless of
        whether relegated teams are included in the reference population."""
        params = {
            "teams": ["Top", "Mid", "Bot", "Relegated"],
            "attack": {"Top": 0.6, "Mid": 0.2, "Bot": -0.2, "Relegated": -1.5},
            "defense": {"Top": -0.4, "Mid": 0.0, "Bot": 0.3, "Relegated": 0.2},
        }
        with_relegated = create_interpretable_ratings(params)
        current_only = create_interpretable_ratings(
            params, reference_teams=["Top", "Mid", "Bot"]
        )

        for ratings in [with_relegated, current_only]:
            assert (
                ratings["attack"]["Top"] > ratings["attack"]["Mid"] > ratings["attack"]["Bot"]
            )


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

    def test_reference_teams_passed_through_to_normalisation(self):
        """reference_teams kwarg should affect the ratings written into params."""
        params = {
            "teams": ["Current1", "Current2", "Historical"],
            "attack": {"Current1": 0.4, "Current2": -0.1, "Historical": -1.2},
            "defense": {"Current1": -0.2, "Current2": 0.1, "Historical": 0.2},
        }
        # two separate calls with different reference populations
        result_all = add_interpretable_ratings_to_params(
            {**params, "attack": dict(params["attack"]), "defense": dict(params["defense"])}
        )
        result_ref = add_interpretable_ratings_to_params(
            {**params, "attack": dict(params["attack"]), "defense": dict(params["defense"])},
            reference_teams=["Current1", "Current2"],
        )

        # weak historical team skews normalisation → ratings differ
        assert (
            result_all["attack_rating"]["Current1"] != result_ref["attack_rating"]["Current1"]
        )

    def test_reference_teams_centres_current_team_ratings_at_zero(self):
        """With reference_teams, the mean of those teams' ratings should be ~0."""
        params = {
            "teams": ["A", "B", "C", "D"],
            "attack": {"A": 0.5, "B": 0.2, "C": -0.1, "D": -1.5},
            "defense": {"A": -0.3, "B": 0.0, "C": 0.2, "D": 0.5},
        }
        current = ["A", "B", "C"]
        result = add_interpretable_ratings_to_params(params, reference_teams=current)

        mean_att = np.mean([result["attack_rating"][t] for t in current])
        mean_def = np.mean([result["defense_rating"][t] for t in current])
        assert abs(mean_att) < 1e-10
        assert abs(mean_def) < 1e-10
