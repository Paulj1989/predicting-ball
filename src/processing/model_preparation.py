# src/processing/model_preparation.py

import logging

import duckdb
import pandas as pd

from src.features.feature_builder import prepare_model_features


def prepare_bundesliga_data(
    db_path: str = "data/pb.duckdb",
    windows: list | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and engineer features for modeling.

    Pipeline:
    1. Load match features from database (models.match_features)
    2. Add basic derived fields (points, total_goals)
    3. Build all features via feature builder
    4. Split into historic vs current season
    5. Filter for modeling requirements

    Creates features:
    - Match outcome features (points, totals)
    - Squad value features (ratios, logs, percentiles)
    - Odds features (ratios, margins)
    - Rolling npxGD statistics (5 and 10 game windows)
    - Venue-specific npxGD (home at home, away away)
    - Red card indicators
    """
    if windows is None:
        windows = [5, 10]
    logger = logging.getLogger(__name__)

    # load match features from database (Bundesliga only)
    with duckdb.connect(db_path) as con:
        df = con.execute("""
            SELECT * FROM models.match_features
            WHERE competition = 'Bundesliga'
            ORDER BY date
        """).df()

    if df.empty:
        raise ValueError("No data found in models.match_features")

    # prepare base dataframe
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_played"] = df["home_goals"].notna()

    # rename time column to kickoff_time for consistency
    if "time" in df.columns:
        df = df.rename(columns={"time": "kickoff_time"})

    # add basic match features
    if verbose:
        logger.info("Adding basic match features")
    df = _add_basic_match_features(df)

    # build all features using feature builder
    if verbose:
        logger.info("Engineering features")
    df = prepare_model_features(df, windows=windows, verbose=verbose)

    # verify feature creation
    if verbose:
        _verify_features(df)

    # split into historic and current season
    current_season = df["season_end_year"].max()
    historic_df = df[(df["season_end_year"] < current_season) & (df["is_played"])].copy()
    current_df = df[df["season_end_year"] == current_season].copy()

    # remove matches without npxG from historic (required for modeling)
    historic_df = historic_df.dropna(subset=["home_npxg", "away_npxg"])

    # log summary
    logger.info(
        f"Historic: {len(historic_df)} matches from {sorted(historic_df['season_end_year'].unique())}"
    )
    logger.info(
        f"Current ({current_season}): {current_df['is_played'].sum()} played, "
        f"{(~current_df['is_played']).sum()} future"
    )

    return historic_df, current_df


def _add_basic_match_features(df: pd.DataFrame) -> pd.DataFrame:
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


def _verify_features(df: pd.DataFrame):
    """Verify that key features were created successfully"""
    feature_checks = [
        ("home_points", "Points"),
        ("home_npxgd_w5", "npxGD (5 games)"),
        ("home_npxgd_w10", "npxGD (10 games)"),
        ("home_venue_npxgd_per_game", "Home venue npxGD"),
        ("away_venue_npxgd_per_game", "Away venue npxGD"),
        ("odds_home_away_ratio", "Odds ratio"),
        ("value_ratio", "Value ratio"),
        ("home_elo", "Elo ratings"),
    ]

    print("\n  Feature verification:")
    for col_name, description in feature_checks:
        if col_name in df.columns:
            non_null = df[col_name].notna().sum()
            non_zero = (df[col_name] != 0).sum()
            mean_val = df[col_name].mean()
            print(
                f"    {description:20s}: {non_null:5d} non-null, {non_zero:5d} non-zero, "
                f"mean={mean_val:.3f}"
            )
        else:
            print(f"    {description:20s}: MISSING")


def test_data_preparation():
    """Test the data preparation pipeline"""
    logging.basicConfig(level=logging.INFO)

    historic_df, current_df = prepare_bundesliga_data(verbose=True)

    print("\n" + "=" * 60)
    print("DATA PREPARATION TEST")
    print("=" * 60)

    print(f"\nHistoric data: {historic_df.shape}")
    print(f"Seasons: {sorted(historic_df['season_end_year'].unique())}")

    # check key features
    key_features = [
        "home_points",
        "home_npxgd_w5",
        "home_npxgd_w10",
        "away_npxgd_w5",
        "away_npxgd_w10",
        "home_venue_npxgd_per_game",
        "away_venue_npxgd_per_game",
        "value_ratio",
        "odds_home_away_ratio",
        "home_elo",
        "away_elo",
    ]

    print("\nFeature validation:")
    for feat in key_features:
        if feat in historic_df.columns:
            non_null = historic_df[feat].notna().sum()
            non_zero = (historic_df[feat] != 0).sum()
            mean_val = historic_df[feat].mean()
            min_val = historic_df[feat].min()
            max_val = historic_df[feat].max()
            print(
                f"  {feat}: {non_null} non-null, {non_zero} non-zero, "
                f"mean={mean_val:.3f}, min={min_val:.3f}, max={max_val:.3f}"
            )
        else:
            print(f"  {feat}: MISSING")

    return historic_df, current_df


if __name__ == "__main__":
    historic_df, current_df = test_data_preparation()
