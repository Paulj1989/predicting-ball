# src/features/npxgd_features.py

import pandas as pd
import numpy as np
from typing import List


def add_rolling_npxgd(df: pd.DataFrame, windows: List[int] = [5, 10]) -> pd.DataFrame:
    """
    Add rolling npxGD (non-penalty xG difference) statistics.
    Calculates mean npxGD over specified windows for each team.

    Creates features:
    - home_npxgd_w5: home team's avg npxGD over last 5 matches
    - home_npxgd_w10: home team's avg npxGD over last 10 matches
    - away_npxgd_w5: away team's avg npxGD over last 5 matches
    - away_npxgd_w10: away team's avg npxGD over last 10 matches
    """
    df = df.sort_values("date").copy()

    # initialise columns
    for window in windows:
        for prefix in ["home", "away"]:
            df[f"{prefix}_npxgd_w{window}"] = np.nan

    # only use matches with npxg data
    played_df = df[
        df["is_played"] & df["home_npxg"].notna() & df["away_npxg"].notna()
    ].copy()
    if len(played_df) == 0:
        return df

    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())

    # calculate npxgd stats for each team
    for team in teams:
        df = _calculate_and_merge_team_npxgd(df, played_df, team, windows)

    return df


def _calculate_and_merge_team_npxgd(
    df: pd.DataFrame, played_df: pd.DataFrame, team: str, windows: List[int]
) -> pd.DataFrame:
    """Calculate and merge rolling npxGD statistics for a single team"""
    # get all matches with npxg data for this team
    team_home = played_df[played_df["home_team"] == team].copy()
    team_away = played_df[played_df["away_team"] == team].copy()

    if len(team_home) == 0 and len(team_away) == 0:
        return df

    # add venue indicator and keep original index
    team_home["is_home"] = True
    team_home["orig_idx"] = team_home.index
    team_away["is_home"] = False
    team_away["orig_idx"] = team_away.index

    # combine and sort by date
    team_matches = (
        pd.concat([team_home, team_away]).sort_values("date").reset_index(drop=True)
    )

    # calculate npxGD (npxG for - npxG against)
    team_matches["npxgd"] = np.where(
        team_matches["is_home"],
        team_matches["home_npxg"] - team_matches["away_npxg"],  # home
        team_matches["away_npxg"] - team_matches["home_npxg"],  # away
    )

    # calculate rolling mean for each window
    for window in windows:
        team_matches[f"npxgd_w{window}"] = (
            team_matches["npxgd"].rolling(window, min_periods=1).mean()
        )

        # shift to exclude current match (we want form going INTO the match)
        team_matches[f"npxgd_w{window}"] = team_matches[f"npxgd_w{window}"].shift(1)

    # merge back to main dataframe using original indices
    for _, row in team_matches.iterrows():
        idx = row["orig_idx"]
        is_home = row["is_home"]
        prefix = "home" if is_home else "away"

        for window in windows:
            df.at[idx, f"{prefix}_npxgd_w{window}"] = row[f"npxgd_w{window}"]

    return df


def add_venue_npxgd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate venue-specific npxGD performance.

    Creates:
    - home_venue_npxgd_per_game: home team's avg npxGD when playing at home
    - away_venue_npxgd_per_game: away team's avg npxGD when playing away
    """
    df = df.sort_values("date").copy()

    # initialise columns
    df["home_venue_npxgd_per_game"] = np.nan
    df["away_venue_npxgd_per_game"] = np.nan

    played_df = df[
        df["is_played"] & df["home_npxg"].notna() & df["away_npxg"].notna()
    ].copy()
    if len(played_df) == 0:
        return df

    # calculate by season and team
    for season in df["season_end_year"].unique():
        season_mask = (
            (df["season_end_year"] == season)
            & df["is_played"]
            & df["home_npxg"].notna()
            & df["away_npxg"].notna()
        )
        season_df = df[season_mask].copy()

        if len(season_df) == 0:
            continue

        teams = pd.unique(season_df[["home_team", "away_team"]].values.ravel())

        for team in teams:
            df = _calculate_venue_npxgd_stats(df, season_df, team)

    return df


def _calculate_venue_npxgd_stats(
    df: pd.DataFrame, season_df: pd.DataFrame, team: str
) -> pd.DataFrame:
    """Calculate venue-specific npxGD statistics for a single team"""
    # home venue performance (team playing at home)
    team_home_matches = season_df[season_df["home_team"] == team].sort_values("date")

    for i, (idx, match) in enumerate(team_home_matches.iterrows()):
        if i > 0:  # need at least one previous home match
            prev_matches = team_home_matches.iloc[:i]

            # calculate npxGD when playing at home: home_npxg - away_npxg
            npxgd_per_game = (
                prev_matches["home_npxg"] - prev_matches["away_npxg"]
            ).mean()

            df.at[idx, "home_venue_npxgd_per_game"] = npxgd_per_game

    # away venue performance (team playing away)
    team_away_matches = season_df[season_df["away_team"] == team].sort_values("date")

    for i, (idx, match) in enumerate(team_away_matches.iterrows()):
        if i > 0:  # need at least one previous away match
            prev_matches = team_away_matches.iloc[:i]

            # calculate npxGD when playing away: away_npxg - home_npxg
            npxgd_per_game = (
                prev_matches["away_npxg"] - prev_matches["home_npxg"]
            ).mean()

            df.at[idx, "away_venue_npxgd_per_game"] = npxgd_per_game

    return df
