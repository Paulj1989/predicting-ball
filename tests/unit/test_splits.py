"""Tests for walk-forward split generator."""

import itertools

import pandas as pd
import pytest

from src.validation.splits import matchweek_walkforward_splits


def make_data(
    n_seasons: int = 5,
    matchweeks_per_season: int = 10,
    matches_per_mw: int = 9,
) -> pd.DataFrame:
    """Create synthetic multi-season match data."""
    rows = []
    base_date = pd.Timestamp("2020-08-01")
    for season_offset in range(n_seasons):
        season = 2021 + season_offset
        for mw in range(1, matchweeks_per_season + 1):
            match_date = base_date + pd.Timedelta(
                weeks=(season_offset * matchweeks_per_season + mw - 1)
            )
            for i in range(matches_per_mw):
                rows.append(
                    {
                        "home_team": f"Team{i}",
                        "away_team": f"Team{i + 1}",
                        "home_goals": 1,
                        "away_goals": 0,
                        "date": match_date,
                        "season_end_year": season,
                        "matchweek": mw,
                    }
                )
    return pd.DataFrame(rows)


class TestMatchweekWalkforwardSplits:
    def test_basic_yields_train_predict_pairs(self):
        data = make_data(n_seasons=5, matchweeks_per_season=4, matches_per_mw=3)
        splits = list(matchweek_walkforward_splits(data, min_train_seasons=3))
        assert len(splits) > 0
        for train, predict in splits:
            assert isinstance(train, pd.DataFrame)
            assert isinstance(predict, pd.DataFrame)
            assert len(train) > 0
            assert len(predict) > 0

    def test_no_temporal_leakage(self):
        data = make_data(n_seasons=5, matchweeks_per_season=4, matches_per_mw=3)
        splits = list(matchweek_walkforward_splits(data, min_train_seasons=3))
        for train, predict in splits:
            assert train["date"].max() < predict["date"].min(), (
                "Training data extends into prediction period"
            )

    def test_predict_data_is_always_single_matchweek(self):
        data = make_data(n_seasons=5, matchweeks_per_season=4, matches_per_mw=3)
        splits = list(matchweek_walkforward_splits(data, min_train_seasons=3))
        for _train, predict in splits:
            assert predict["matchweek"].nunique() == 1
            assert predict["season_end_year"].nunique() == 1

    def test_train_set_grows_monotonically(self):
        data = make_data(n_seasons=5, matchweeks_per_season=4, matches_per_mw=3)
        splits = list(matchweek_walkforward_splits(data, min_train_seasons=3))
        train_sizes = [len(t) for t, _ in splits]
        assert all(a <= b for a, b in itertools.pairwise(train_sizes)), (
            "Training set should grow (or stay same) at each step"
        )

    def test_first_prediction_is_from_correct_season(self):
        data = make_data(n_seasons=5, matchweeks_per_season=4, matches_per_mw=3)
        splits = list(matchweek_walkforward_splits(data, min_train_seasons=3))
        first_predict = splits[0][1]
        # with 5 seasons (2021-2025) and min_train_seasons=3, first predict season = 2024
        assert first_predict["season_end_year"].iloc[0] == 2024

    def test_raises_if_not_enough_seasons(self):
        data = make_data(n_seasons=3, matchweeks_per_season=4, matches_per_mw=3)
        with pytest.raises(ValueError, match="more than 3 seasons"):
            list(matchweek_walkforward_splits(data, min_train_seasons=3))

    def test_total_predicted_matches_covers_remaining_seasons(self):
        n_seasons = 5
        mw_per_season = 4
        matches_per_mw = 3
        data = make_data(
            n_seasons=n_seasons,
            matchweeks_per_season=mw_per_season,
            matches_per_mw=matches_per_mw,
        )
        splits = list(matchweek_walkforward_splits(data, min_train_seasons=3))
        total_predicted = sum(len(p) for _, p in splits)
        # seasons 4 and 5 (indices 3, 4) = 2 seasons x 4 mw x 3 matches
        expected = 2 * mw_per_season * matches_per_mw
        assert total_predicted == expected

    def test_works_with_custom_column_names(self):
        data = make_data(n_seasons=5, matchweeks_per_season=3, matches_per_mw=2)
        data = data.rename(
            columns={
                "matchweek": "round",
                "season_end_year": "season",
                "date": "match_date",
            }
        )
        splits = list(
            matchweek_walkforward_splits(
                data,
                min_train_seasons=3,
                matchweek_column="round",
                season_column="season",
                date_column="match_date",
            )
        )
        assert len(splits) > 0

    def test_first_split_train_size_is_exactly_min_seasons(self):
        mw_per_season = 4
        matches_per_mw = 3
        data = make_data(
            n_seasons=5, matchweeks_per_season=mw_per_season, matches_per_mw=matches_per_mw
        )
        splits = list(matchweek_walkforward_splits(data, min_train_seasons=3))
        # first prediction is season 2024 mw1 — training data is seasons 2021-2023 only
        first_train = splits[0][0]
        expected_train_size = 3 * mw_per_season * matches_per_mw
        assert len(first_train) == expected_train_size
