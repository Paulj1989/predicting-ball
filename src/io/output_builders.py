# src/io/output_builders.py

import json
from datetime import datetime

import pandas as pd


def create_matches_dataframe(
    all_predictions: pd.DataFrame | None,
    all_fixtures: pd.DataFrame | None,
    next_fixtures: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create the latest_buli_matches.parquet DataFrame by merging predictions with fixtures"""
    if all_predictions is None or len(all_predictions) == 0:
        return pd.DataFrame()

    df = all_predictions.copy()

    if all_fixtures is not None and len(all_fixtures) > 0:
        fixture_cols = ["home_team", "away_team"]
        if "date" in all_fixtures.columns:
            fixture_cols.append("date")
        if "match_id" in all_fixtures.columns:
            fixture_cols.append("match_id")
        if "matchweek" in all_fixtures.columns:
            fixture_cols.append("matchweek")
        if "kickoff_time" in all_fixtures.columns:
            fixture_cols.append("kickoff_time")

        fixtures_subset = all_fixtures[fixture_cols].copy()
        df = df.merge(fixtures_subset, on=["home_team", "away_team"], how="left")

    if next_fixtures is not None and len(next_fixtures) > 0:
        next_matches = set(
            zip(next_fixtures["home_team"], next_fixtures["away_team"], strict=False)
        )
        df["is_next_round"] = df.apply(
            lambda row: (row["home_team"], row["away_team"]) in next_matches,
            axis=1,
        )
    else:
        # mark the earliest date as next round when no explicit next_fixtures given
        if "date" in df.columns:
            min_date = df["date"].min()
            df["is_next_round"] = df["date"] == min_date
        else:
            df["is_next_round"] = False

    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = None
    if "matchweek" not in df.columns:
        df["matchweek"] = None

    output_cols = [
        "match_id",
        "date",
        "matchweek",
        "kickoff_time",
        "home_team",
        "away_team",
        "expected_goals_home",
        "expected_goals_away",
        "home_win",
        "draw",
        "away_win",
        "most_likely_score",
        "is_next_round",
    ]
    available_cols = [col for col in output_cols if col in df.columns]
    return df[available_cols]


def create_projections_dataframe(
    summary: pd.DataFrame,
    model_params: dict,
    current_standings: dict,
) -> pd.DataFrame:
    """Create the latest_buli_projections.parquet DataFrame with team ratings and standings"""
    df = summary.copy()

    attack_ratings = model_params.get("attack_rating", {})
    defense_ratings = model_params.get("defense_rating", {})
    overall_ratings = model_params.get("overall_rating", {})

    df["attack_rating"] = df["team"].map(lambda t: attack_ratings.get(t, 0.0))
    df["defense_rating"] = df["team"].map(lambda t: defense_ratings.get(t, 0.0))
    df["overall_rating"] = df["team"].map(lambda t: overall_ratings.get(t, 0.0))

    if current_standings is not None and len(current_standings) > 0:
        df["current_points"] = df["team"].map(
            lambda t: current_standings.get(t, {}).get("points", 0)
        )
        df["current_gd"] = df["team"].map(
            lambda t: current_standings.get(t, {}).get("goal_diff", 0)
        )
        df["matches_played"] = df["team"].map(
            lambda t: current_standings.get(t, {}).get("games_played", 0)
        )

    output_cols = [
        "team",
        "projected_points",
        "projected_gd",
        "attack_rating",
        "defense_rating",
        "overall_rating",
        "title_prob",
        "ucl_prob",
        "relegation_prob",
        "current_points",
        "current_gd",
        "matches_played",
    ]
    available_cols = [col for col in output_cols if col in df.columns]
    return df[available_cols]


def create_run_snapshot(
    all_predictions: pd.DataFrame | None,
    all_fixtures: pd.DataFrame | None,
    model_params: dict,
    model: dict,
    calibrators: dict | None,
    run_timestamp: datetime,
    validation_metrics: dict | None = None,
) -> pd.DataFrame:
    """
    Create the buli_run_{timestamp}.parquet snapshot for pbdb ingestion.

    Denormalized structure with one row per match prediction, plus metadata.
    """
    if all_predictions is None or len(all_predictions) == 0:
        return pd.DataFrame()

    df = all_predictions.copy()

    if all_fixtures is not None and len(all_fixtures) > 0:
        fixture_cols = ["home_team", "away_team"]
        if "date" in all_fixtures.columns:
            fixture_cols.append("date")
        if "match_id" in all_fixtures.columns:
            fixture_cols.append("match_id")
        if "matchweek" in all_fixtures.columns:
            fixture_cols.append("matchweek")
        if "kickoff_time" in all_fixtures.columns:
            fixture_cols.append("kickoff_time")

        fixtures_subset = all_fixtures[fixture_cols].copy()
        df = df.merge(fixtures_subset, on=["home_team", "away_team"], how="left")

    attack_ratings = model_params.get("attack_rating", {})
    defense_ratings = model_params.get("defense_rating", {})
    overall_ratings = model_params.get("overall_rating", {})

    df["home_attack_rating"] = df["home_team"].map(lambda t: attack_ratings.get(t, 0.0))
    df["home_defense_rating"] = df["home_team"].map(lambda t: defense_ratings.get(t, 0.0))
    df["home_overall_rating"] = df["home_team"].map(lambda t: overall_ratings.get(t, 0.0))

    df["away_attack_rating"] = df["away_team"].map(lambda t: attack_ratings.get(t, 0.0))
    df["away_defense_rating"] = df["away_team"].map(lambda t: defense_ratings.get(t, 0.0))
    df["away_overall_rating"] = df["away_team"].map(lambda t: overall_ratings.get(t, 0.0))

    run_id = run_timestamp.strftime("%Y%m%d_%H%M%S")
    df["run_id"] = run_id
    df["run_timestamp"] = run_timestamp
    df["model_version"] = model.get("trained_at", run_timestamp).strftime("%Y-%m-%d %H:%M:%S")

    hyperparams = model.get("hyperparams", {})
    df["hyperparameters_json"] = json.dumps(hyperparams)

    if validation_metrics:
        df["validation_metrics_json"] = json.dumps(validation_metrics)
    else:
        df["validation_metrics_json"] = json.dumps({"note": "No validation metrics provided"})

    calibration_metrics = {}
    if calibrators:
        calibration_metrics = {
            "method": calibrators.get("calibration_method", "unknown"),
            "rps_improvement": calibrators.get("rps_improvement_holdout"),
            "brier_improvement": calibrators.get("brier_improvement_holdout"),
        }
    df["calibration_metrics_json"] = json.dumps(calibration_metrics)

    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = None

    return df
