#!/usr/bin/env python3
"""
Generate Predictions
====================

Generate season projections and match predictions, then upload to DO Spaces.

Usage:
    python scripts/modeling/generate_predictions.py --model-path outputs/models/buli_model.pkl
    python scripts/modeling/generate_predictions.py --model-path outputs/models/buli_model.pkl --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.model_io import load_calibrators, load_model
from src.io.output_builders import (
    create_matches_dataframe,
    create_projections_dataframe,
    create_run_snapshot,
)
from src.io.spaces import (
    INCOMING_PREFIX,
    SERVING_PREFIX,
    get_public_url,
    upload_dataframe_as_parquet,
    upload_pickle,
)
from src.models.fisher_information import (
    build_state_vector,
    compute_fisher_information,
    invert_fisher_with_constraints,
)
from src.processing.model_preparation import prepare_bundesliga_data
from src.simulation import (
    create_final_summary,
    get_current_standings,
    predict_next_fixtures,
)
from src.simulation.hot_simulation import simulate_season_hot


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate predictions and season projections")

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
        "--hot-k-att",
        type=float,
        default=0.05,
        help="Attack learning rate for hot simulation (0 = cold simulation, default: 0.05)",
    )

    parser.add_argument(
        "--hot-k-def",
        type=float,
        default=0.025,
        help="Defence learning rate for hot simulation (default: 0.025)",
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
        projections_df.to_parquet(output_dir / "latest_buli_projections.parquet", index=False)
        print(f"   Saved: {output_dir / 'latest_buli_projections.parquet'}")

    if len(snapshot_df) > 0:
        timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
        snapshot_df.to_parquet(output_dir / f"buli_run_{timestamp_str}.parquet", index=False)
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
            with open(args.validation_metrics) as f:
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

    historic_data, current_season = prepare_bundesliga_data(windows=windows, verbose=False)

    # verify weighted goals exists
    if "home_goals_weighted" not in historic_data.columns:
        print("\n   Error: home_goals_weighted not found in data")
        print("   Make sure prepare_bundesliga_data includes weighted goals calculation")
        sys.exit(1)

    current_played = current_season[current_season["is_played"]].copy()
    current_future = current_season[~current_season["is_played"]].copy()

    current_season_year = current_season["season_end_year"].iloc[0]

    print(f"   Current season: {current_season_year - 1}/{current_season_year}")
    print(f"   Played matches: {len(current_played)}")
    print(f"   Remaining matches: {len(current_future)}")

    if len(current_future) == 0:
        print("\n   WARNING: No remaining matches found")
        print("   Season may be complete or no future fixtures available")

    # ========================================================================
    # MLE STANDARD ERRORS FOR UNCERTAINTY
    # ========================================================================
    print("\n3. Computing MLE standard errors...")

    all_train = pd.concat([historic_data, current_played], ignore_index=True)
    print(f"   Training data: {len(all_train)} matches")

    fisher = compute_fisher_information(model["params"], all_train, model["hyperparams"])

    n_model_teams = len(model["params"]["teams"])
    mle_cov = invert_fisher_with_constraints(fisher, n_model_teams)
    state_mean = build_state_vector(model["params"])

    print(f"   Fisher information computed ({fisher.shape[0]}x{fisher.shape[0]} matrix)")
    print("   Covariance matrix inverted successfully")

    # ========================================================================
    # SIMULATE REMAINING SEASON
    # ========================================================================
    sim_mode = "hot" if (args.hot_k_att > 0 or args.hot_k_def > 0) else "cold"
    print(
        f"\n4. Simulating season ({args.n_simulations:,} iterations, {sim_mode},"
        f" K_att={args.hot_k_att}, K_def={args.hot_k_def})..."
    )

    current_standings = get_current_standings(current_played)
    print(f"   Current standings calculated for {len(current_standings)} teams")

    dispersion = model["params"].get("dispersion_factor", 1.0)
    rho = model["params"].get("rho", -0.13)
    disp_flag = " ⚠ (> 1.2)" if dispersion > 1.2 else ""
    print(f"   Dispersion (diagnostic): {dispersion:.3f}{disp_flag}")
    print(f"   Rho: {rho:.3f}")

    if len(current_future) > 0:
        results, teams = simulate_season_hot(
            current_future,
            state_mean,
            mle_cov,
            current_standings,
            state_teams=model["params"]["teams"],
            n_simulations=args.n_simulations,
            K_att=args.hot_k_att,
            K_def=args.hot_k_def,
            rho=rho,
            seed=args.seed,
        )
        print("   Simulation complete")
    else:
        print("   Skipping simulation (no remaining fixtures)")
        teams = list(current_standings.keys())
        n_teams_sim = len(teams)
        results = {
            "points": np.empty((0, n_teams_sim)),
            "goals_for": np.empty((0, n_teams_sim)),
            "goals_against": np.empty((0, n_teams_sim)),
            "positions": np.empty((0, n_teams_sim)),
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
        if next_predictions is not None:
            print(f"   Next matchday: {len(next_predictions)} fixtures")

    if all_fixtures is not None and len(all_fixtures) > 0:
        all_predictions = predict_next_fixtures(
            all_fixtures, model["params"], calibrators=calibrators
        )
        if all_predictions is not None:
            print(f"   All future: {len(all_predictions)} fixtures")

    # ========================================================================
    # CREATE OUTPUT DATAFRAMES
    # ========================================================================
    print("\n7. Creating output DataFrames...")

    if all_predictions is None or all_fixtures is None:
        # no remaining fixtures means the season is complete
        print("   Season complete — no remaining fixtures to predict.")

    matches_df = create_matches_dataframe(all_predictions, all_fixtures, next_fixtures)
    print(f"   Matches DataFrame: {len(matches_df)} rows")

    projections_df = create_projections_dataframe(summary, model["params"], current_standings)
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
        available_cols = [col for col in display_cols if col in next_predictions.columns]

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
        timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
        print(f"  - incoming/buli_run_{timestamp_str}.parquet (for pbdb)")
        print(f"  - incoming/buli_projections_{timestamp_str}.parquet (for pbdb)")

    if calibrators:
        print("\nPredictions were calibrated")
    else:
        print("\nNo calibrators applied (use --calibrator-path to enable)")


if __name__ == "__main__":
    main()
