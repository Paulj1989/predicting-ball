#!/usr/bin/env python3
"""
Validate Model
==============

Run backtesting and validation on trained model.

Usage:
    python scripts/modeling/validate_model.py --model-path outputs/models/production_model.pkl
"""

import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.validation import (
    backtest_multiple_seasons,
    create_validation_report,
    analyse_prediction_errors,
    analyse_performance_by_team,
)
from src.processing.model_preparation import prepare_bundesliga_data
from src.io.model_io import load_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Validate trained model on historical data"
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model file"
    )

    parser.add_argument(
        "--test-seasons",
        type=int,
        nargs="+",
        default=None,
        help="Seasons to test (default: last 2 seasons)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation",
        help="Directory for validation outputs (default: outputs/validation)",
    )

    parser.add_argument(
        "--calibrator-path",
        type=str,
        default=None,
        help="Optional path to calibrators file",
    )

    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="Rolling window sizes for npxGD features (default: use model's windows)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed prediction quality analysis",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["rps", "log_loss", "brier"],
        default="rps",
        help="Primary metric to report and emphasise (default: rps)",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save validation metrics as JSON (for pipeline integration)",
    )

    return parser.parse_args()


def main():
    """Main validation pipeline"""
    args = parse_args()

    print("=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70)

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n1. Loading model...")
    model = load_model(args.model_path)

    print(f"   ✓ Model loaded: {args.model_path}")

    # get model details safely
    n_teams = len(model.get("teams", model.get("params", {}).get("teams", [])))
    train_shape = model.get("train_data_shape", (0, 0))

    print(f"   Teams: {n_teams}")
    print(f"   Trained on: {train_shape[0]} matches")

    # get windows from model if not provided
    windows = args.windows if args.windows else model.get("windows", [5, 10])
    print(f"   Using rolling windows: {windows}")

    # load calibrators if provided
    calibrators = None
    if args.calibrator_path:
        print(f"\n   Loading calibrators from: {args.calibrator_path}")
        from src.io.model_io import load_calibrators

        calibrators = load_calibrators(args.calibrator_path)
        print("   ✓ Calibrators loaded")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n2. Loading data...")

    historic_data, current_season = prepare_bundesliga_data(
        windows=windows, verbose=False
    )

    # verify weighted goals exists
    if "home_goals_weighted" not in historic_data.columns:
        print("\n   ✗ Error: home_goals_weighted not found in data")
        print(
            "   Make sure prepare_bundesliga_data includes weighted goals calculation"
        )
        sys.exit(1)

    # combine all data
    current_played = current_season[current_season["is_played"] == True].copy()
    all_data = pd.concat([historic_data, current_played], ignore_index=True)

    print(f"   Total matches available: {len(all_data)}")
    print(
        f"   Date range: {all_data['date'].min().date()} to {all_data['date'].max().date()}"
    )
    print(f"   Seasons: {sorted(all_data['season_end_year'].unique())}")

    # ========================================================================
    # DETERMINE TEST SEASONS
    # ========================================================================
    if args.test_seasons:
        test_seasons = args.test_seasons
    else:
        # use last 2 seasons by default
        all_seasons = sorted(all_data["season_end_year"].unique())
        test_seasons = all_seasons[-2:]

    print(f"\n3. Test seasons: {test_seasons}")

    # verify test seasons have data
    available_seasons = set(all_data["season_end_year"].unique())
    missing_seasons = set(test_seasons) - available_seasons

    if missing_seasons:
        print(f"\n   ✗ Error: Requested seasons not available: {missing_seasons}")
        print(f"   Available seasons: {sorted(available_seasons)}")
        sys.exit(1)

    # ========================================================================
    # RUN BACKTESTING
    # ========================================================================
    print("\n4. Running backtesting...")

    results = backtest_multiple_seasons(
        model, all_data, test_seasons, calibrators=calibrators, verbose=True
    )

    if args.debug:
        # check actual prediction distributions
        print("\n" + "=" * 60)
        print("PREDICTION QUALITY ANALYSIS")
        print("=" * 60)

        for result in results:
            season = result["season"]
            preds = np.array(result["predictions"])
            actuals = np.array(result["actuals"])

            print(f"\nSeason {season}:")
            print(f"  Predictions shape: {preds.shape}")
            print(f"  NaN count: {np.isnan(preds).sum()}")

            # check draw probabilities specifically
            draw_probs = preds[:, 1]
            print("\n  Draw probabilities:")
            print(f"    Min: {draw_probs.min():.4f}")
            print(f"    Max: {draw_probs.max():.4f}")
            print(f"    Mean: {draw_probs.mean():.4f}")
            print(f"    Median: {np.median(draw_probs):.4f}")

            # distribution of predictions
            print("\n  Prediction distribution:")
            predicted_outcomes = np.argmax(preds, axis=1)
            print(f"    Home wins: {(predicted_outcomes == 0).sum()}")
            print(f"    Draws: {(predicted_outcomes == 1).sum()}")
            print(f"    Away wins: {(predicted_outcomes == 2).sum()}")

            # sample predictions
            print("\n  Sample predictions (first 5):")
            for i in range(min(5, len(preds))):
                h, d, a = preds[i]
                actual = ["H", "D", "A"][actuals[i]]
                print(
                    f"    Match {i}: H={h:.3f} D={d:.3f} A={a:.3f} (actual: {actual})"
                )

            # check if probabilities sum to 1
            prob_sums = preds.sum(axis=1)
            print("\n  Probability sums (should be ~1.0):")
            print(f"    Min: {prob_sums.min():.4f}")
            print(f"    Max: {prob_sums.max():.4f}")
            print(f"    Mean: {prob_sums.mean():.4f}")

            # check for zero draws
            draw_preds = preds[:, 1]
            max_draw = draw_preds.max() if len(draw_preds) > 0 else 0
            print(
                f"     Draw predictions: min={draw_preds.min():.4f}, max={max_draw:.4f}, mean={draw_preds.mean():.4f}"
            )

            if max_draw < 0.15:
                print(
                    f"     WARNING: Maximum draw probability is very low ({max_draw:.3f})"
                )

    if len(results) == 0:
        print("\n✗ No validation results obtained")
        sys.exit(1)

    print(f"   ✓ Backtesting complete: {len(results)} seasons validated")

    # ========================================================================
    # CREATE OUTPUTS
    # ========================================================================
    print("\n5. Creating validation outputs...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # create validation report
    report = create_validation_report(
        results, save_path=output_dir / "validation_report.csv"
    )
    print("   ✓ Validation report saved")

    # analyse errors for most recent season
    most_recent_result = results[-1]

    print(f"\n   Analysing errors for season {most_recent_result['season']}...")

    season_data = all_data[all_data["season_end_year"] == most_recent_result["season"]]

    error_analysis = analyse_prediction_errors(
        most_recent_result["predictions"],
        most_recent_result["actuals"],
        season_data,
        verbose=True,
    )

    team_analysis = analyse_performance_by_team(
        most_recent_result["predictions"],
        most_recent_result["actuals"],
        season_data,
        verbose=True,
    )

    team_analysis.to_csv(
        output_dir / f"team_analysis_{most_recent_result['season']}.csv", index=False
    )
    print("   ✓ Team analysis saved")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {output_dir}")
    print("\nFiles created:")
    print("  - validation_report.csv")
    print(f"  - team_analysis_{most_recent_result['season']}.csv")

    # print summary stats
    print("\nValidation Metrics:")

    metric_key_map = {"rps": "rps", "log_loss": "log_loss", "brier": "brier_score"}
    metric_display_map = {"rps": "RPS", "log_loss": "Log Loss", "brier": "Brier"}

    for result in results:
        season = result["season"]

        # get metrics
        rps = result["metrics"]["rps"]
        brier = result["metrics"].get("brier_score", None)
        log_loss = result["metrics"].get("log_loss", None)

        # get primary metric
        primary_key = metric_key_map[args.metric]
        primary_value = result["metrics"].get(primary_key, "N/A")

        baseline_info = ""
        if result.get("baseline_metrics") and args.metric in [
            "rps",
            "log_loss",
            "brier",
        ]:
            baseline_key = (
                "rps" if args.metric == "rps" else metric_key_map[args.metric]
            )
            if baseline_key in result["baseline_metrics"]:
                baseline_value = result["baseline_metrics"][baseline_key]
                if isinstance(primary_value, (int, float)) and isinstance(
                    baseline_value, (int, float)
                ):
                    improvement = (
                        (baseline_value - primary_value) / baseline_value * 100
                    )
                    baseline_info = f" (vs baseline: {improvement:+.1f}%)"

        # format based on selected metric
        if args.metric == "rps":
            print(
                f"  Season {season}: RPS={rps:.4f} (PRIMARY), Brier={brier if brier is not None else 'N/A'}, Log Loss={log_loss if log_loss is not None else 'N/A'}{baseline_info}"
            )
        elif args.metric == "log_loss":
            print(
                f"  Season {season}: Log Loss={log_loss if log_loss is not None else 'N/A'} (PRIMARY), Brier={brier if brier is not None else 'N/A'}, RPS={rps:.4f}{baseline_info}"
            )
        else:  # brier
            print(
                f"  Season {season}: Brier={brier if brier is not None else 'N/A'} (PRIMARY), RPS={rps:.4f}, Log Loss={log_loss if log_loss is not None else 'N/A'}{baseline_info}"
            )

    # calculate average performance
    print("\nAverage Performance:")

    # calculate averages for all metrics
    avg_rps = np.mean([r["metrics"]["rps"] for r in results])
    avg_brier = np.mean(
        [
            r["metrics"].get("brier_score", 0)
            for r in results
            if r["metrics"].get("brier_score") is not None
        ]
    )
    avg_log_loss = np.mean(
        [
            r["metrics"].get("log_loss", 0)
            for r in results
            if r["metrics"].get("log_loss") is not None
        ]
    )

    # display based on selected metric
    if args.metric == "rps":
        print(f"  RPS: {avg_rps:.4f} (PRIMARY)")
        print(f"  Brier: {avg_brier:.4f}")
        print(f"  Log Loss: {avg_log_loss:.4f}")
    elif args.metric == "log_loss":
        print(f"  Log Loss: {avg_log_loss:.4f} (PRIMARY)")
        print(f"  Brier: {avg_brier:.4f}")
        print(f"  RPS: {avg_rps:.4f}")
    else:  # brier
        print(f"  Brier: {avg_brier:.4f} (PRIMARY)")
        print(f"  RPS: {avg_rps:.4f}")
        print(f"  Log Loss: {avg_log_loss:.4f}")

    # baseline comparison for primary metric
    if all(r.get("baseline_metrics") for r in results):
        primary_key = metric_key_map[args.metric]
        if all(primary_key in r.get("baseline_metrics", {}) for r in results):
            avg_baseline = np.mean(
                [r["baseline_metrics"][primary_key] for r in results]
            )

            if args.metric == "rps":
                avg_primary = avg_rps
            elif args.metric == "log_loss":
                avg_primary = avg_log_loss
            else:
                avg_primary = avg_brier

            improvement = (avg_baseline - avg_primary) / avg_baseline * 100
            print(f"\n  Baseline {metric_display_map[args.metric]}: {avg_baseline:.4f}")
            print(f"  Improvement: {improvement:+.1f}%")

    # print calibrator info if used
    if calibrators:
        print("\n Predictions were calibrated")
    else:
        print("\n No calibrators applied (use --calibrator-path to enable)")

    # save metrics as JSON if requested
    if args.output_json:
        metrics_output = {
            "test_seasons": test_seasons,
            "per_season": [
                {
                    "season": r["season"],
                    "n_matches": r["n_matches"],
                    "rps": r["metrics"]["rps"],
                    "brier_score": r["metrics"].get("brier_score"),
                    "log_loss": r["metrics"].get("log_loss"),
                    "accuracy": r["metrics"].get("accuracy"),
                    "baseline_rps": r.get("baseline_metrics", {}).get("rps"),
                }
                for r in results
            ],
            "average": {
                "rps": avg_rps,
                "brier_score": avg_brier,
                "log_loss": avg_log_loss,
            },
            "calibrated": calibrators is not None,
        }

        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(metrics_output, f, indent=2)
        print(f"\nValidation metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
