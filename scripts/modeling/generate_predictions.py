#!/usr/bin/env python3
"""
Generate Predictions
====================

Generate season projections and next match predictions.

Usage:
    python scripts/modeling/generate_predictions.py --model-path outputs/models/production_model.pkl
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from src.simulation import (
    parametric_bootstrap_with_residuals,
    simulate_remaining_season_calibrated,
    predict_next_fixtures,
    get_current_standings,
    create_final_summary,
)
from src.processing.model_preparation import prepare_bundesliga_data
from src.io.model_io import load_model, load_calibrators
from src.visualisation import (
    create_standings_table,
    create_next_fixtures_table
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
        default=500,
        help="Number of bootstrap iterations (default: 500)",
    )

    parser.add_argument(
        "--n-simulations",
        type=int,
        default=100000,
        help="Number of season simulations (default: 100000)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory for prediction outputs (default: outputs/predictions)",
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

    return parser.parse_args()


def main():
    """Main prediction pipeline"""
    args = parse_args()

    print("=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    # set random seed for reproducibility
    np.random.seed(args.seed)

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n1. Loading model...")
    model = load_model(args.model_path)

    print(f"   ✓ Model loaded: {args.model_path}")

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
        print("   ✓ Calibrators loaded")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n2. Loading data...")

    # data already includes all features (weighted performance, npxGD, etc.)
    historic_data, current_season = prepare_bundesliga_data(
        windows=windows, verbose=False
    )

    # verify weighted performance exists
    if "home_goals_weighted" not in historic_data.columns:
        print("\n   ✗ Error: home_goals_weighted not found in data")
        print(
            "   Make sure prepare_bundesliga_data includes weighted performance calculation"
        )
        sys.exit(1)

    current_played = current_season[current_season["is_played"] == True].copy()
    current_future = current_season[current_season["is_played"] == False].copy()

    current_season_year = current_season["season_end_year"].iloc[0]

    print(f"   Current season: {current_season_year - 1}/{current_season_year}")
    print(f"   Played matches: {len(current_played)}")
    print(f"   Remaining matches: {len(current_future)}")

    if len(current_future) == 0:
        print("\n   ⚠ Warning: No remaining matches found")
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

    print(f"   ✓ Bootstrap complete: {len(bootstrap_params)} samples")

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

        print("   ✓ Simulation complete")
    else:
        print("   ⚠ Skipping simulation (no remaining fixtures)")
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

    print(f"   ✓ Projections created for {len(summary)} teams")

    # ========================================================================
    # PREDICT NEXT FIXTURES
    # ========================================================================
    print("\n6. Predicting next fixtures...")

    from src.simulation.predictions import get_next_round_fixtures

    next_fixtures = get_next_round_fixtures(current_season)

    if next_fixtures is not None and len(next_fixtures) > 0:
        next_predictions = predict_next_fixtures(
            next_fixtures, model["params"], calibrators=calibrators
        )
        print(f"   ✓ {len(next_predictions)} fixtures predicted")
    else:
        next_predictions = None
        print("   No upcoming fixtures found")

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================
    print("\n7. Saving outputs...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # save projections csv
    summary.to_csv(output_dir / "season_projections.csv", index=False)
    print(f"   ✓ Saved: season_projections.csv")

    # save next fixtures csv
    if next_predictions is not None:
        next_predictions.to_csv(output_dir / "next_fixtures.csv", index=False)
        print(f"   ✓ Saved: next_fixtures.csv")

    # ========================================================================
    # CREATE VISUALISATIONS
    # ========================================================================
    print("\n8. Creating visualisations...")

    # standings table (great_tables)
    try:
        create_standings_table(summary, save_path=str(tables_dir / "standings_table"))
        print("   ✓ Created: standings table")
    except Exception as e:
        print(f"   ⚠ Could not create standings table: {e}")

    # next fixtures table (great_tables)
    if next_predictions is not None:
        try:
            create_next_fixtures_table(
                next_predictions, save_path=str(tables_dir / "next_fixtures_table")
            )
            print("   ✓ Created: next fixtures table")
        except Exception as e:
            print(f"   ⚠ Could not create next fixtures table: {e}")

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
    print("\nOutputs saved to:")
    print(f"  CSV files: {output_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables: {tables_dir}")

    if calibrators:
        print("\n✓ Predictions were calibrated")
    else:
        print("\n⚠ No calibrators applied (use --calibrator-path to enable)")

    print("\nGenerated visualisations:")
    print("  - Standings table")
    if next_predictions is not None:
        print("  - Next fixtures table")


if __name__ == "__main__":
    main()
