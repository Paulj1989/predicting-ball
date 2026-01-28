#!/usr/bin/env python3
"""
Generate Predictions
====================

Generate season projections and match predictions, then upload to DO Spaces.

Usage:
    python scripts/modeling/generate_predictions.py --model-path outputs/models/production_model.pkl
    python scripts/modeling/generate_predictions.py --model-path outputs/models/production_model.pkl --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.model_io import load_calibrators, load_model
from src.io.spaces import (
    INCOMING_PREFIX,
    SERVING_PREFIX,
    get_public_url,
    upload_dataframe_as_parquet,
    upload_pickle,
)
from src.processing.model_preparation import prepare_bundesliga_data
from src.simulation import (
    create_final_summary,
    get_current_standings,
    parametric_bootstrap_with_residuals,
    predict_next_fixtures,
    simulate_remaining_season_calibrated,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate predictions and season projections"
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model file"
    )

    parser.add_argument(
        "--calibrator-path",
        type=str,
        default=None,
        help="Optional path to calibrators file",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=250,
        help="Number of bootstrap iterations (default: 250)",
    )

    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10000,
        help="Number of season simulations (default: 10000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="Rolling window sizes for npxGD features (default: use model's windows)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate files locally without uploading to DO Spaces",
    )

    parser.add_argument(
        "--validation-metrics",
        type=str,
        default=None,
        help="Path to validation metrics JSON file (from validate_model.py)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Local output directory for dry-run mode (default: outputs/predictions)",
    )

    return parser.parse_args()


def create_matches_dataframe(
    all_predictions: pd.DataFrame,
    all_fixtures: pd.DataFrame,
    next_fixtures: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create the latest_buli_matches.parquet DataFrame by merging predictions with fixtures"""
    if all_predictions is None or len(all_predictions) == 0:
        return pd.DataFrame()

    # merge predictions with original fixtures to get date, match_id, etc.
    # predictions have: home_team, away_team, expected_goals_*, probabilities
    # fixtures have: date, match_id, home_team, away_team, etc.
    df = all_predictions.copy()

    if all_fixtures is not None and len(all_fixtures) > 0:
        # select fixture columns to merge
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

        # merge on home_team and away_team
        df = df.merge(
            fixtures_subset,
            on=["home_team", "away_team"],
            how="left",
        )

    # determine next round matches
    if next_fixtures is not None and len(next_fixtures) > 0:
        # create set of (home_team, away_team) tuples for next round
        next_matches = set(
            zip(next_fixtures["home_team"], next_fixtures["away_team"])
        )
        df["is_next_round"] = df.apply(
            lambda row: (row["home_team"], row["away_team"]) in next_matches,
            axis=1,
        )
    else:
        # if no next_fixtures, mark the earliest date as next round
        if "date" in df.columns:
            min_date = df["date"].min()
            df["is_next_round"] = df["date"] == min_date
        else:
            df["is_next_round"] = False

    # ensure optional columns exist (nullable)
    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = None
    if "matchweek" not in df.columns:
        df["matchweek"] = None

    # select and order columns
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

    # only include columns that exist
    available_cols = [col for col in output_cols if col in df.columns]
    return df[available_cols]


def create_projections_dataframe(
    summary: pd.DataFrame,
    model_params: dict,
    current_standings: dict,
) -> pd.DataFrame:
    """Create the latest_buli_projections.parquet DataFrame with team ratings and standings"""
    df = summary.copy()

    # add team ratings from model params
    # model params has interpretable ratings: attack_rating, defense_rating, overall_rating
    # these are z-scores scaled to approximately [-1, 1] with defense flipped (positive = good)
    attack_ratings = model_params.get("attack_rating", {})
    defense_ratings = model_params.get("defense_rating", {})
    overall_ratings = model_params.get("overall_rating", {})

    df["attack_rating"] = df["team"].map(lambda t: attack_ratings.get(t, 0.0))
    df["defense_rating"] = df["team"].map(lambda t: defense_ratings.get(t, 0.0))
    df["overall_rating"] = df["team"].map(lambda t: overall_ratings.get(t, 0.0))

    # add current standings info (current_standings is a dict, not DataFrame)
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

    # select and order columns
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

    # only include columns that exist
    available_cols = [col for col in output_cols if col in df.columns]
    return df[available_cols]


def create_run_snapshot(
    all_predictions: pd.DataFrame,
    all_fixtures: pd.DataFrame,
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

    # merge with fixtures to get date, match_id, etc.
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

    # add team ratings for home and away teams
    # model params has interpretable ratings: attack_rating, defense_rating, overall_rating
    # these are z-scores scaled to approximately [-1, 1] with defense flipped (positive = good)
    attack_ratings = model_params.get("attack_rating", {})
    defense_ratings = model_params.get("defense_rating", {})
    overall_ratings = model_params.get("overall_rating", {})

    df["home_attack_rating"] = df["home_team"].map(lambda t: attack_ratings.get(t, 0.0))
    df["home_defense_rating"] = df["home_team"].map(lambda t: defense_ratings.get(t, 0.0))
    df["home_overall_rating"] = df["home_team"].map(lambda t: overall_ratings.get(t, 0.0))

    df["away_attack_rating"] = df["away_team"].map(lambda t: attack_ratings.get(t, 0.0))
    df["away_defense_rating"] = df["away_team"].map(lambda t: defense_ratings.get(t, 0.0))
    df["away_overall_rating"] = df["away_team"].map(lambda t: overall_ratings.get(t, 0.0))

    # add run metadata (same for all rows)
    run_id = run_timestamp.strftime("%Y%m%d_%H%M%S")
    df["run_id"] = run_id
    df["run_timestamp"] = run_timestamp
    df["model_version"] = model.get("trained_at", run_timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # add hyperparameters as JSON string
    hyperparams = model.get("hyperparams", {})
    df["hyperparameters_json"] = json.dumps(hyperparams)

    # add validation metrics (already native python types from validate_model.py)
    if validation_metrics:
        df["validation_metrics_json"] = json.dumps(validation_metrics)
    else:
        df["validation_metrics_json"] = json.dumps(
            {"note": "No validation metrics provided"}
        )

    # add calibration metrics if available
    calibration_metrics = {}
    if calibrators:
        calibration_metrics = {
            "method": calibrators.get("calibration_method", "unknown"),
            "rps_improvement": calibrators.get("rps_improvement_holdout"),
            "brier_improvement": calibrators.get("brier_improvement_holdout"),
        }
    df["calibration_metrics_json"] = json.dumps(calibration_metrics)

    # ensure kickoff_time column exists
    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = None

    return df


def upload_to_spaces(
    matches_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    model: dict,
    calibrators: dict | None,
    model_path: str,
    calibrator_path: str | None,
    run_timestamp: datetime,
) -> dict:
    """
    Upload all artifacts to DO Spaces.

    Returns dict with URLs for uploaded files.
    """
    urls = {}

    # upload matches parquet (public)
    if len(matches_df) > 0:
        key = f"{SERVING_PREFIX}latest_buli_matches.parquet"
        upload_dataframe_as_parquet(matches_df, key, public=True)
        urls["matches"] = get_public_url(key)
        print(f"   Uploaded: {key}")

    # upload projections parquet (public)
    if len(projections_df) > 0:
        key = f"{SERVING_PREFIX}latest_buli_projections.parquet"
        upload_dataframe_as_parquet(projections_df, key, public=True)
        urls["projections"] = get_public_url(key)
        print(f"   Uploaded: {key}")

    # upload model pkl (private)
    key = f"{SERVING_PREFIX}buli_model.pkl"
    upload_pickle(model, key, public=False)
    urls["model"] = key
    print(f"   Uploaded: {key}")

    # upload calibrators pkl (private) if available
    if calibrators:
        key = f"{SERVING_PREFIX}buli_calibrators.pkl"
        upload_pickle(calibrators, key, public=False)
        urls["calibrators"] = key
        print(f"   Uploaded: {key}")

    # upload run snapshot (private) for pbdb ingestion
    if len(snapshot_df) > 0:
        timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
        key = f"{INCOMING_PREFIX}buli_run_{timestamp_str}.parquet"
        upload_dataframe_as_parquet(snapshot_df, key, public=False)
        urls["snapshot"] = key
        print(f"   Uploaded: {key}")

    # upload projections snapshot (private) for pbdb ingestion
    if len(projections_df) > 0:
        timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
        key = f"{INCOMING_PREFIX}buli_projections_{timestamp_str}.parquet"
        upload_dataframe_as_parquet(projections_df, key, public=False)
        urls["projections_snapshot"] = key
        print(f"   Uploaded: {key}")

    return urls


def save_local_outputs(
    matches_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    model: dict,
    calibrators: dict | None,
    output_dir: Path,
    run_timestamp: datetime,
):
    """Save outputs locally for dry-run mode."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(matches_df) > 0:
        matches_df.to_parquet(output_dir / "latest_buli_matches.parquet", index=False)
        print(f"   Saved: {output_dir / 'latest_buli_matches.parquet'}")

    if len(projections_df) > 0:
        projections_df.to_parquet(
            output_dir / "latest_buli_projections.parquet", index=False
        )
        print(f"   Saved: {output_dir / 'latest_buli_projections.parquet'}")

    if len(snapshot_df) > 0:
        timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
        snapshot_df.to_parquet(
            output_dir / f"buli_run_{timestamp_str}.parquet", index=False
        )
        print(f"   Saved: {output_dir / f'buli_run_{timestamp_str}.parquet'}")


def main():
    """Main prediction pipeline"""
    args = parse_args()

    print("=" * 70)
    print("GENERATING PREDICTIONS")
    if args.dry_run:
        print("(DRY RUN - no upload to DO Spaces)")
    print("=" * 70)

    run_timestamp = datetime.now()

    # set random seed for reproducibility
    np.random.seed(args.seed)

    # load validation metrics if provided
    validation_metrics = None
    if args.validation_metrics:
        try:
            with open(args.validation_metrics, "r") as f:
                validation_metrics = json.load(f)
            print(f"   Loaded validation metrics from: {args.validation_metrics}")
        except Exception as e:
            print(f"   Warning: Could not load validation metrics: {e}")

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n1. Loading model...")
    model = load_model(args.model_path)

    print(f"   Model loaded: {args.model_path}")

    # get model details safely
    n_teams = len(model.get("teams", model.get("params", {}).get("teams", [])))
    print(f"   Teams: {n_teams}")

    # get windows from model if not provided
    windows = args.windows if args.windows else model.get("windows", [5, 10])
    print(f"   Using rolling windows: {windows}")

    # load calibrators if provided
    calibrators = None
    if args.calibrator_path:
        print(f"   Loading calibrators: {args.calibrator_path}")
        calibrators = load_calibrators(args.calibrator_path)
        print("   Calibrators loaded")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n2. Loading data...")

    historic_data, current_season = prepare_bundesliga_data(
        windows=windows, verbose=False
    )

    # verify weighted goals exists
    if "home_goals_weighted" not in historic_data.columns:
        print("\n   Error: home_goals_weighted not found in data")
        print(
            "   Make sure prepare_bundesliga_data includes weighted goals calculation"
        )
        sys.exit(1)

    current_played = current_season[current_season["is_played"] == True].copy()
    current_future = current_season[current_season["is_played"] == False].copy()

    current_season_year = current_season["season_end_year"].iloc[0]

    print(f"   Current season: {current_season_year - 1}/{current_season_year}")
    print(f"   Played matches: {len(current_played)}")
    print(f"   Remaining matches: {len(current_future)}")

    if len(current_future) == 0:
        print("\n   WARNING: No remaining matches found")
        print("   Season may be complete or no future fixtures available")

    # ========================================================================
    # BOOTSTRAP FOR UNCERTAINTY
    # ========================================================================
    print(f"\n3. Running bootstrap ({args.n_bootstrap} iterations)...")

    # use recent data for bootstrap
    all_train = pd.concat([historic_data, current_played], ignore_index=True)

    print(f"   Training data: {len(all_train)} matches")

    bootstrap_params = parametric_bootstrap_with_residuals(
        all_train,
        model["params"],
        model["hyperparams"],
        promoted_priors=model.get("promoted_priors"),
        n_bootstrap=args.n_bootstrap,
        verbose=True,
    )

    print(f"   Bootstrap complete: {len(bootstrap_params)} samples")

    # ========================================================================
    # SIMULATE REMAINING SEASON
    # ========================================================================
    print(f"\n4. Simulating season ({args.n_simulations:,} iterations)...")

    current_standings = get_current_standings(current_played)

    print(f"   Current standings calculated for {len(current_standings)} teams")

    if len(current_future) > 0:
        results, teams = simulate_remaining_season_calibrated(
            current_future,
            bootstrap_params,
            current_standings,
            n_simulations=args.n_simulations,
            seed=args.seed,
        )

        print("   Simulation complete")
    else:
        print("   Skipping simulation (no remaining fixtures)")
        # create empty results structure
        teams = current_standings.index.tolist()
        results = {
            team: {"points": [], "goal_diff": [], "position": []} for team in teams
        }

    # ========================================================================
    # CREATE PROJECTIONS
    # ========================================================================
    print("\n5. Creating projections...")

    summary = create_final_summary(results, model["params"], teams, current_standings)

    print(f"   Projections created for {len(summary)} teams")

    # ========================================================================
    # PREDICT FIXTURES
    # ========================================================================
    print("\n6. Predicting fixtures...")

    from src.simulation.predictions import (
        get_all_future_fixtures,
        get_next_round_fixtures,
    )

    next_fixtures = get_next_round_fixtures(current_season)
    all_fixtures = get_all_future_fixtures(current_season)

    next_predictions = None
    all_predictions = None

    if next_fixtures is not None and len(next_fixtures) > 0:
        next_predictions = predict_next_fixtures(
            next_fixtures, model["params"], calibrators=calibrators
        )
        print(f"   Next matchday: {len(next_predictions)} fixtures")

    if all_fixtures is not None and len(all_fixtures) > 0:
        all_predictions = predict_next_fixtures(
            all_fixtures, model["params"], calibrators=calibrators
        )
        print(f"   All future: {len(all_predictions)} fixtures")

    # ========================================================================
    # CREATE OUTPUT DATAFRAMES
    # ========================================================================
    print("\n7. Creating output DataFrames...")

    matches_df = create_matches_dataframe(all_predictions, all_fixtures, next_fixtures)
    print(f"   Matches DataFrame: {len(matches_df)} rows")

    projections_df = create_projections_dataframe(
        summary, model["params"], current_standings
    )
    print(f"   Projections DataFrame: {len(projections_df)} rows")

    snapshot_df = create_run_snapshot(
        all_predictions,
        all_fixtures,
        model["params"],
        model,
        calibrators,
        run_timestamp,
        validation_metrics,
    )
    print(f"   Snapshot DataFrame: {len(snapshot_df)} rows")

    # ========================================================================
    # UPLOAD OR SAVE
    # ========================================================================
    if args.dry_run:
        print("\n8. Saving outputs locally (dry-run)...")
        output_dir = Path(args.output_dir)
        save_local_outputs(
            matches_df,
            projections_df,
            snapshot_df,
            model,
            calibrators,
            output_dir,
            run_timestamp,
        )
    else:
        print("\n8. Uploading to DO Spaces...")
        try:
            urls = upload_to_spaces(
                matches_df,
                projections_df,
                snapshot_df,
                model,
                calibrators,
                args.model_path,
                args.calibrator_path,
                run_timestamp,
            )
            print("\n   Public URLs:")
            if "matches" in urls:
                print(f"   - Matches: {urls['matches']}")
            if "projections" in urls:
                print(f"   - Projections: {urls['projections']}")
            if "model" in urls:
                print(f"   - Model: {urls['model']}")
            if "calibrators" in urls:
                print(f"   - Calibrators: {urls['calibrators']}")
        except Exception as e:
            print(f"\n   ERROR uploading to DO Spaces: {e}")
            print("   Falling back to local save...")
            output_dir = Path(args.output_dir)
            save_local_outputs(
                matches_df,
                projections_df,
                snapshot_df,
                model,
                calibrators,
                output_dir,
                run_timestamp,
            )

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("TOP 5 PROJECTED STANDINGS")
    print("=" * 70)

    display_cols = [
        "team",
        "projected_points",
        "projected_gd",
        "title_prob",
        "ucl_prob",
    ]

    # only display columns that exist
    available_cols = [col for col in display_cols if col in summary.columns]
    print(summary[available_cols].head().to_string(index=False))

    if next_predictions is not None:
        print("\n" + "=" * 70)
        print("NEXT MATCHDAY PREDICTIONS")
        print("=" * 70)

        display_cols = [
            "home_team",
            "away_team",
            "expected_goals_home",
            "expected_goals_away",
            "home_win",
            "draw",
            "away_win",
            "most_likely_score",
        ]

        # only display columns that exist
        available_cols = [
            col for col in display_cols if col in next_predictions.columns
        ]

        # format probabilities as percentages for display
        display_df = next_predictions[available_cols].copy()
        if "home_win" in display_df.columns:
            display_df["home_win"] = display_df["home_win"].map(lambda x: f"{x:.1%}")
        if "draw" in display_df.columns:
            display_df["draw"] = display_df["draw"].map(lambda x: f"{x:.1%}")
        if "away_win" in display_df.columns:
            display_df["away_win"] = display_df["away_win"].map(lambda x: f"{x:.1%}")
        if "expected_goals_home" in display_df.columns:
            display_df["expected_goals_home"] = display_df["expected_goals_home"].map(
                lambda x: f"{x:.2f}"
            )
        if "expected_goals_away" in display_df.columns:
            display_df["expected_goals_away"] = display_df["expected_goals_away"].map(
                lambda x: f"{x:.2f}"
            )

        print(display_df.to_string(index=False))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("PREDICTIONS COMPLETE")
    print("=" * 70)

    if args.dry_run:
        print(f"\nDry-run outputs saved to: {args.output_dir}")
    else:
        print("\nOutputs uploaded to DO Spaces:")
        print("  - serving/latest_buli_matches.parquet (public)")
        print("  - serving/latest_buli_projections.parquet (public)")
        print("  - serving/buli_model.pkl (private)")
        if calibrators:
            print("  - serving/buli_calibrators.pkl (private)")
        timestamp_str = run_timestamp.strftime('%Y%m%d_%H%M%S')
        print(f"  - incoming/buli_run_{timestamp_str}.parquet (for pbdb)")
        print(f"  - incoming/buli_projections_{timestamp_str}.parquet (for pbdb)")

    if calibrators:
        print("\nPredictions were calibrated")
    else:
        print("\nNo calibrators applied (use --calibrator-path to enable)")


if __name__ == "__main__":
    main()
