#!/usr/bin/env python3
"""
Run Calibration
===============

Fit post-hoc calibrators for improved probability estimates.

Usage:
    python scripts/run_calibration.py --model-path outputs/models/production_model.pkl
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.models.calibration import (
    fit_isotonic_calibrator,
    calibrate_dispersion_for_coverage,
    calibrate_model_comprehensively,
)
from src.evaluation import (
    evaluate_model_comprehensive,
    create_calibration_report,
)
from src.processing.model_preparation import prepare_bundesliga_data
from src.validation.splits import create_calibration_split
from src.io.model_io import load_model, save_calibrators


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fit calibrators for trained model")

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model file"
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.15,
        help="Fraction of data to use for calibration (default: 0.15)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Directory to save calibrators (default: outputs/models)",
    )

    parser.add_argument(
        "--calibrator-name",
        type=str,
        default="calibrators",
        help="Name for saved calibrator file (default: calibrators)",
    )

    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive calibration (multiple confidence levels)",
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
    """Main calibration pipeline"""
    args = parse_args()

    print("=" * 70)
    print("POST-HOC CALIBRATION")
    print("=" * 70)

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n1. Loading model...")
    model = load_model(args.model_path)

    print(f"   ✓ Model loaded: {args.model_path}")
    print(f"   Teams: {len(model.get('teams', []))}")

    # get windows from model if not provided
    windows = args.windows if args.windows else model.get("windows", [5, 10])
    print(f"   Using rolling windows: {windows}")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n2. Loading calibration data...")

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

    # use all available data
    current_played = current_season[current_season["is_played"] == True].copy()
    all_data = pd.concat([historic_data, current_played], ignore_index=True)

    print(f"   Total matches: {len(all_data)}")
    print(
        f"   Date range: {all_data['date'].min().date()} to {all_data['date'].max().date()}"
    )

    # split off calibration set
    fit_data, calibration_data = create_calibration_split(
        all_data, calibration_fraction=args.calibration_fraction
    )

    print(f"   Fitting set: {len(fit_data)} matches")
    print(f"   Calibration set: {len(calibration_data)} matches")

    # ========================================================================
    # FIT ISOTONIC CALIBRATORS
    # ========================================================================
    print("\n3. Fitting isotonic regression calibrators...")

    # get predictions on calibration set
    metrics_uncal, cal_predictions, cal_actuals = evaluate_model_comprehensive(
        model["params"], calibration_data
    )

    print(f"   Uncalibrated Brier: {metrics_uncal['brier_score']:.4f}")
    print(f"   Uncalibrated RPS:   {metrics_uncal.get('rps', 'N/A')}")

    # fit calibrators
    calibrators = fit_isotonic_calibrator(cal_predictions, cal_actuals)

    # test improvement
    from src.models.calibration import apply_calibration
    from src.evaluation.metrics import calculate_brier_score

    cal_preds_calibrated = apply_calibration(cal_predictions, calibrators)
    brier_cal = calculate_brier_score(cal_preds_calibrated, cal_actuals)

    print(f"   Calibrated Brier:   {brier_cal:.4f}")
    print(f"   Improvement:        {metrics_uncal['brier_score'] - brier_cal:.4f}")

    # ========================================================================
    # DISPERSION CALIBRATION (Optional)
    # ========================================================================
    dispersion_calibrated = None

    if args.comprehensive:
        print("\n4. Running comprehensive dispersion calibration...")

        dispersion_dict, _ = calibrate_model_comprehensively(
            model["params"], calibration_data, verbose=True
        )

        dispersion_calibrated = {
            "dispersion_dict": dispersion_dict,
            # don't save the function - it will be recreated when needed
        }

        print("   ✓ Dispersion calibration complete")
    else:
        print("\n4. Skipping comprehensive calibration (use --comprehensive to enable)")

    # ========================================================================
    # SAVE CALIBRATORS
    # ========================================================================
    print("\n5. Saving calibrators...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrator_package = {
        "isotonic_calibrators": calibrators,
        "dispersion_calibrated": dispersion_calibrated,
        "calibration_data_size": len(calibration_data),
        "calibration_improvement": metrics_uncal["brier_score"] - brier_cal,
        "windows": windows,
        "calibration_date": pd.Timestamp.now(),
    }

    calibrator_path = output_dir / f"{args.calibrator_name}.pkl"
    save_calibrators(calibrator_package, calibrator_path)

    print(f"   ✓ Calibrators saved: {calibrator_path}")

    # ========================================================================
    # CREATE CALIBRATION REPORT
    # ========================================================================
    print("\n6. Creating calibration report...")

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    report = create_calibration_report(
        cal_preds_calibrated,
        cal_actuals,
        model_name="Calibrated Model",
        save_path=figures_dir / "calibration_report.png",
    )

    print(f"   ✓ Report saved: {figures_dir / 'calibration_report.png'}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Calibrators saved to: {calibrator_path}")
    print(f"Calibration set size: {len(calibration_data)} matches")
    print(f"Brier improvement: {metrics_uncal['brier_score'] - brier_cal:.4f}")

    if dispersion_calibrated:
        print("Dispersion calibration: ✓ Enabled")
    else:
        print("Dispersion calibration: Not applied (use --comprehensive)")

    print("\nNext steps:")
    print(f"  - Validate with calibration:")
    print(f"    python scripts/validate_model.py --model-path {args.model_path} \\")
    print(f"           --calibrator-path {calibrator_path}")
    print(f"  - Generate predictions with calibration:")
    print(
        f"    python scripts/generate_predictions.py --model-path {args.model_path} \\"
    )
    print(f"           --calibrator-path {calibrator_path}")


if __name__ == "__main__":
    main()
