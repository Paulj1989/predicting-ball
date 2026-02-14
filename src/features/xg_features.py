# src/features/xg_features.py


import numpy as np
import pandas as pd


def add_rolling_npxgd(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Add rolling npxGD (non-penalty xG difference) statistics.
    Calculates mean npxGD over specified windows for each team, respecting season boundaries.

    Features:
    - home_npxgd_w5: home team's avg npxGD over last 5 matches (within season)
    - home_npxgd_w10: home team's avg npxGD over last 10 matches (within season)
    - away_npxgd_w5: away team's avg npxGD over last 5 matches (within season)
    - away_npxgd_w10: away team's avg npxGD over last 10 matches (within season)

    Initialisation:
    - First game of season uses previous season's average npxGD (or 0.0 if unavailable)
    - Expanding window: games 1-N use 1-N games until reaching window size
    """
    if windows is None:
        windows = [5, 10]
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
    df: pd.DataFrame, played_df: pd.DataFrame, team: str, windows: list[int]
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

    # =========================================================================
    # SEASON-AWARE ROLLING CALCULATION
    # =========================================================================

    # initialise rolling columns
    for window in windows:
        team_matches[f"npxgd_w{window}"] = np.nan

    # calculate previous season averages for initialisation
    season_avg_npxgd = {}
    for season in team_matches["season_end_year"].unique():
        season_matches = team_matches[team_matches["season_end_year"] == season]
        if len(season_matches) > 0:
            season_avg_npxgd[season] = season_matches["npxgd"].mean()

    # process each season separately
    for season in sorted(team_matches["season_end_year"].unique()):
        season_mask = team_matches["season_end_year"] == season
        season_indices = team_matches[season_mask].index

        if len(season_indices) == 0:
            continue

        # get initialisation value (previous season's average or 0.0)
        prev_season = season - 1
        init_value = season_avg_npxgd.get(prev_season, 0.0)

        # calculate rolling stats within this season only
        for window in windows:
            # extract npxgd values for this season
            season_npxgd = team_matches.loc[season_indices, "npxgd"].values

            # calculate rolling mean with expanding window
            rolling_values = []
            for i in range(len(season_npxgd)):
                if i == 0:
                    # first game: use previous season average (or 0.0)
                    rolling_values.append(init_value)
                else:
                    # use previous games (expanding window up to size)
                    start_idx = max(0, i - window)
                    window_values = season_npxgd[start_idx:i]
                    rolling_values.append(window_values.mean())

            # assign back to dataframe
            team_matches.loc[season_indices, f"npxgd_w{window}"] = rolling_values

    # =========================================================================
    # MERGE BACK TO MAIN DATAFRAME
    # =========================================================================

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
            df = _calculate_venue_npxgd_stats(df, season_df, team, season)

    return df


def _calculate_venue_npxgd_stats(
    df: pd.DataFrame, season_df: pd.DataFrame, team: str, season: int
) -> pd.DataFrame:
    """Calculate venue-specific npxGD statistics for a single team"""

    # =========================================================================
    # CALCULATE PREVIOUS SEASON BASELINE
    # =========================================================================

    prev_season = season - 1
    prev_season_df = df[
        (df["season_end_year"] == prev_season)
        & df["is_played"]
        & df["home_npxg"].notna()
        & df["away_npxg"].notna()
    ]

    # previous season home venue average
    prev_home_matches = prev_season_df[prev_season_df["home_team"] == team]
    if len(prev_home_matches) > 0:
        prev_home_avg = (
            prev_home_matches["home_npxg"] - prev_home_matches["away_npxg"]
        ).mean()
    else:
        prev_home_avg = 0.0

    # previous season away venue average
    prev_away_matches = prev_season_df[prev_season_df["away_team"] == team]
    if len(prev_away_matches) > 0:
        prev_away_avg = (
            prev_away_matches["away_npxg"] - prev_away_matches["home_npxg"]
        ).mean()
    else:
        prev_away_avg = 0.0

    # =========================================================================
    # HOME VENUE PERFORMANCE
    # =========================================================================

    team_home_matches = season_df[season_df["home_team"] == team].sort_values("date")

    for i, (idx, _match) in enumerate(team_home_matches.iterrows()):
        if i == 0:
            # first home game: use previous season's home average
            df.at[idx, "home_venue_npxgd_per_game"] = prev_home_avg
        else:
            # expanding window: use all previous home games this season
            prev_matches = team_home_matches.iloc[:i]
            npxgd_per_game = (
                prev_matches["home_npxg"] - prev_matches["away_npxg"]
            ).mean()
            df.at[idx, "home_venue_npxgd_per_game"] = npxgd_per_game

    # =========================================================================
    # AWAY VENUE PERFORMANCE
    # =========================================================================

    team_away_matches = season_df[season_df["away_team"] == team].sort_values("date")

    for i, (idx, _match) in enumerate(team_away_matches.iterrows()):
        if i == 0:
            # first away game: use previous season's away average
            df.at[idx, "away_venue_npxgd_per_game"] = prev_away_avg
        else:
            # expanding window: use all previous away games this season
            prev_matches = team_away_matches.iloc[:i]
            npxgd_per_game = (
                prev_matches["away_npxg"] - prev_matches["home_npxg"]
            ).mean()
            df.at[idx, "away_venue_npxgd_per_game"] = npxgd_per_game

    return df
