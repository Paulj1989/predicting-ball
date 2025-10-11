# src/processing/model_preparation.py

import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
import logging
from typing import Tuple, List


def prepare_bundesliga_data(
    db_path: str = "data/club_football.duckdb",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load integrated data and engineer features for modeling.

    Creates a comprehensive feature set including:
    - Match outcome features (points, totals)
    - Squad value features (ratios, logs, percentiles)
    - Odds features (ratios, margins)
    - Rolling npxG statistics
    - Home vs away performance metrics
    - Red card indicators
    """
    logger = logging.getLogger(__name__)

    # load integrated data from database
    with duckdb.connect(db_path) as con:
        df = con.execute("""
            SELECT * FROM processed.integrated_data
            ORDER BY date
        """).df()

    if df.empty:
        raise ValueError("No data found in processed.integrated_data")

    # prepare base dataframe
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_played"] = df["home_goals"].notna()

    # add all feature sets
    logger.info("Engineering features")
    df = add_all_features(df)

    # verify feature creation
    _verify_features(df)

    # split into historic and current season
    current_season = df["season_end_year"].max()
    historic_df = df[
        (df["season_end_year"] < current_season) & (df["is_played"] == True)
    ].copy()
    current_df = df[df["season_end_year"] == current_season].copy()

    # remove matches without npxG from historic (required for modeling)
    historic_df = historic_df.dropna(subset=["home_npxg", "away_npxg"])

    # log summary
    logger.info(
        f"Historic: {len(historic_df)} matches from {sorted(historic_df['season_end_year'].unique())}"
    )
    logger.info(
        f"Current ({current_season}): {current_df['is_played'].sum()} played, {(~current_df['is_played']).sum()} future"
    )

    return historic_df, current_df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for modeling"""
    logger = logging.getLogger(__name__)

    logger.info("  Adding basic match features")
    df = add_basic_match_features(df)

    logger.info("  Adding squad value features")
    df = add_squad_value_features(df)

    logger.info("  Adding odds features")
    df = add_odds_features(df)

    logger.info("  Adding card features")
    df = add_card_features(df)

    logger.info("  Adding rolling npxG features")
    df = add_rolling_npxg_features(df, windows=[5, 10])

    logger.info("  Adding home vs away features")
    df = add_home_vs_away_features(df)

    return df


def _verify_features(df: pd.DataFrame):
    """Verify that key features were created successfully"""
    feature_checks = [
        ("home_points", "Points"),
        ("home_npxg_w10_mean", "npxG (10 games)"),
        ("home_npxg_per_game", "Home npxG"),
        ("odds_home_away_ratio", "Odds ratio"),
        ("value_ratio", "Value ratio"),
    ]

    print("\n  Feature verification:")
    for col_name, description in feature_checks:
        if col_name in df.columns:
            non_null = df[col_name].notna().sum()
            non_zero = (df[col_name] > 0).sum()
            mean_val = df[col_name].mean()
            print(
                f"    {description:20s}: {non_null:5d} non-null, {non_zero:5d} non-zero, mean={mean_val:.3f}"
            )
        else:
            print(f"    {description:20s}: MISSING")


def add_basic_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic match outcome features"""
    mask_played = df["is_played"]

    if mask_played.any():
        # points allocation
        df.loc[mask_played, "home_points"] = 0
        df.loc[mask_played, "away_points"] = 0
        df.loc[mask_played & (df["result"] == "H"), "home_points"] = 3
        df.loc[mask_played & (df["result"] == "A"), "away_points"] = 3
        df.loc[mask_played & (df["result"] == "D"), "home_points"] = 1
        df.loc[mask_played & (df["result"] == "D"), "away_points"] = 1

        # total goals
        df.loc[mask_played, "total_goals"] = (
            df.loc[mask_played, "home_goals"] + df.loc[mask_played, "away_goals"]
        )

    return df


def add_squad_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add squad value derived features.

    Creates:
    - Percentile features (relative to season average)
    - Log-transformed features
    - Ratio and difference features
    """
    # fill missing values
    for col in ["home_value", "away_value"]:
        if col in df.columns:
            # fill with season median first
            df[col] = df.groupby("season_end_year")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # then overall median
            df[col] = df[col].fillna(df[col].median())
            # fallback to 50M euros
            df[col] = df[col].fillna(50_000_000)

    # calculate season average for normalisation
    season_avg_value = (
        df.groupby("season_end_year")[["home_value", "away_value"]]
        .transform(lambda x: x.mean())
        .mean(axis=1)
    )

    # percentile features (relative to season average)
    df["home_value_pct"] = (
        (df["home_value"] - season_avg_value) / season_avg_value
    ) * 100
    df["away_value_pct"] = (
        (df["away_value"] - season_avg_value) / season_avg_value
    ) * 100
    df["value_pct_diff"] = df["home_value_pct"] - df["away_value_pct"]

    # log-transformed features
    df["home_value_log"] = np.log(df["home_value"].clip(lower=1))
    df["away_value_log"] = np.log(df["away_value"].clip(lower=1))
    df["value_diff_log"] = df["home_value_log"] - df["away_value_log"]

    # ratio and difference features
    df["value_ratio"] = df["home_value"] / df["away_value"].clip(lower=1)
    df["value_diff"] = df["home_value"] - df["away_value"]

    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add betting odds derived features.

    Creates:
    - Home/away odds ratio
    - Win probability margin (home - away)
    """
    if "home_odds" in df.columns and "away_odds" in df.columns:
        # odds ratio (how much more likely is home vs away)
        df["odds_home_away_ratio"] = df["home_odds"] / df["away_odds"]

        # win margin (difference in implied probabilities)
        if "odds_home_prob" in df.columns and "odds_away_prob" in df.columns:
            df["odds_home_win_margin"] = df["odds_home_prob"] - df["odds_away_prob"]

    return df


def add_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add red card indicator features"""
    mask_played = df["is_played"]

    if "home_cards_red" in df.columns:
        df.loc[mask_played, "home_had_red"] = (
            df.loc[mask_played, "home_cards_red"] > 0
        ).astype(int)
        df.loc[mask_played, "away_had_red"] = (
            df.loc[mask_played, "away_cards_red"] > 0
        ).astype(int)

    return df


def add_rolling_npxg_features(
    df: pd.DataFrame, windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """Add rolling npxG statistics"""
    df = df.sort_values("date").copy()

    # initialise columns
    for window in windows:
        for prefix in ["home", "away"]:
            df[f"{prefix}_npxg_w{window}_mean"] = 0.0
            df[f"{prefix}_npxg_w{window}_std"] = 0.0
            df[f"{prefix}_npxga_w{window}_mean"] = 0.0
            df[f"{prefix}_npxga_w{window}_std"] = 0.0

    # only use matches with npxg data
    played_df = df[df["is_played"] & df["home_npxg"].notna()].copy()
    if len(played_df) == 0:
        return df

    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())

    # calculate npxg stats for each team
    for team in teams:
        team_data = _calculate_team_npxg(played_df, team, windows)
        df = _merge_team_npxg(df, team_data, team, windows)

    return df


def _calculate_team_npxg(
    played_df: pd.DataFrame, team: str, windows: List[int]
) -> pd.DataFrame:
    """Calculate rolling npxG statistics for a single team"""
    # get all matches with npxg data
    team_home = played_df[played_df["home_team"] == team].copy()
    team_away = played_df[played_df["away_team"] == team].copy()

    team_home["is_home"] = True
    team_away["is_home"] = False

    team_matches = pd.concat([team_home, team_away]).sort_values("date")

    if len(team_matches) == 0:
        return pd.DataFrame()

    # extract npxg for/against
    team_matches["npxg"] = np.where(
        team_matches["is_home"], team_matches["home_npxg"], team_matches["away_npxg"]
    )
    team_matches["npxga"] = np.where(
        team_matches["is_home"], team_matches["away_npxg"], team_matches["home_npxg"]
    )

    # calculate rolling statistics
    for window in windows:
        team_matches[f"npxg_w{window}_mean"] = (
            team_matches["npxg"].rolling(window, min_periods=1).mean()
        )
        team_matches[f"npxg_w{window}_std"] = (
            team_matches["npxg"].rolling(window, min_periods=1).std()
        )
        team_matches[f"npxga_w{window}_mean"] = (
            team_matches["npxga"].rolling(window, min_periods=1).mean()
        )
        team_matches[f"npxga_w{window}_std"] = (
            team_matches["npxga"].rolling(window, min_periods=1).std()
        )

        # shift to exclude current match
        for col in [
            f"npxg_w{window}_mean",
            f"npxg_w{window}_std",
            f"npxga_w{window}_mean",
            f"npxga_w{window}_std",
        ]:
            team_matches[col] = team_matches[col].shift(1).fillna(0)

    return team_matches


def _merge_team_npxg(
    df: pd.DataFrame, team_data: pd.DataFrame, team: str, windows: List[int]
) -> pd.DataFrame:
    """Merge team npxG data back to main dataframe"""
    if team_data.empty:
        return df

    for is_home, prefix in [(True, "home"), (False, "away")]:
        mask = team_data["is_home"] == is_home
        if not mask.any():
            continue

        team_subset = team_data[mask]
        indices = df[
            (df[f"{prefix}_team"] == team) & df[f"{prefix}_team"].notna()
        ].index

        if len(team_subset) != len(indices):
            continue

        for window in windows:
            df.loc[indices, f"{prefix}_npxg_w{window}_mean"] = team_subset[
                f"npxg_w{window}_mean"
            ].values
            df.loc[indices, f"{prefix}_npxg_w{window}_std"] = team_subset[
                f"npxg_w{window}_std"
            ].values
            df.loc[indices, f"{prefix}_npxga_w{window}_mean"] = team_subset[
                f"npxga_w{window}_mean"
            ].values
            df.loc[indices, f"{prefix}_npxga_w{window}_std"] = team_subset[
                f"npxga_w{window}_std"
            ].values

    return df


def add_home_vs_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate home vs away performance"""
    df = df.sort_values("date").copy()

    # initialise columns
    for team_type in ["home", "away"]:
        df[f"{team_type}_npxg_per_game"] = 0.0
        df[f"{team_type}_npxga_per_game"] = 0.0

    played_df = df[df["is_played"]].copy()
    if len(played_df) == 0:
        return df

    # calculate by season and team
    for season in df["season_end_year"].unique():
        season_mask = (df["season_end_year"] == season) & df["is_played"]
        season_df = df[season_mask].copy()

        teams = pd.unique(season_df[["home_team", "away_team"]].values.ravel())

        for team in teams:
            df = _calculate_home_vs_away_stats(df, season_df, team)

    return df


def _calculate_home_vs_away_stats(
    df: pd.DataFrame, season_df: pd.DataFrame, team: str
) -> pd.DataFrame:
    """Calculate home vs away statistics for a single team"""
    # home performance
    team_home_matches = season_df[season_df["home_team"] == team].sort_values("date")

    for i, (idx, match) in enumerate(team_home_matches.iterrows()):
        if i > 0:  # need at least one previous match
            prev_matches = team_home_matches.iloc[:i]

            # calculate cumulative stats
            npxg_pg = prev_matches["home_npxg"].mean()
            npxga_pg = prev_matches["away_npxg"].mean()

            df.at[idx, "home_npxg_per_game"] = npxg_pg
            df.at[idx, "home_npxga_per_game"] = npxga_pg

    # away performance
    team_away_matches = season_df[season_df["away_team"] == team].sort_values("date")

    for i, (idx, match) in enumerate(team_away_matches.iterrows()):
        if i > 0:
            prev_matches = team_away_matches.iloc[:i]

            npxg_pg = prev_matches["away_npxg"].mean()
            npxga_pg = prev_matches["home_npxg"].mean()

            df.at[idx, "away_npxg_per_game"] = npxg_pg
            df.at[idx, "away_npxga_per_game"] = npxga_pg

    return df


def test_data_preparation():
    """Test the data preparation pipeline"""
    logging.basicConfig(level=logging.INFO)

    historic_df, current_df = prepare_bundesliga_data()

    print("\n" + "=" * 60)
    print("DATA PREPARATION TEST")
    print("=" * 60)

    print(f"\nHistoric data: {historic_df.shape}")
    print(f"Seasons: {sorted(historic_df['season_end_year'].unique())}")

    # check key features
    key_features = [
        "home_points",
        "home_npxg_w10_mean",
        "away_npxg_w10_mean",
        "home_npxg_per_game",
        "away_npxg_per_game",
        "value_ratio",
        "odds_home_away_ratio",
    ]

    print("\nFeature validation:")
    for feat in key_features:
        if feat in historic_df.columns:
            non_null = historic_df[feat].notna().sum()
            non_zero = (historic_df[feat] > 0).sum()
            mean_val = historic_df[feat].mean()
            print(
                f"  {feat}: {non_null} non-null, {non_zero} non-zero, mean={mean_val:.3f}"
            )
        else:
            print(f"  {feat}: MISSING")

    return historic_df, current_df


if __name__ == "__main__":
    historic_df, current_df = test_data_preparation()
