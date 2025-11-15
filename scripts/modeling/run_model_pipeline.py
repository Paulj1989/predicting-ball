#!/usr/bin/env python3
"""
Run Model Pipeline
=====================

Execute the full modelling pipeline:
1. Train model
2. Run calibration
3. Validate
4. Generate predictions

Usage:
    python scripts/modeling/run_model_pipeline.py [--tune]
"""

import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run complete modelling pipeline")

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Optimise hyperparameters (takes longer)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of hyperparameter optimisation trials (default: 30)",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=250,
        help="Number of bootstrap samples (default: 250)",
    )

    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation step"
    )

    return parser.parse_args()


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n{description} failed")
        sys.exit(1)

    print(f"\n{description} completed successfully")


def main():
    """Run complete pipeline"""
    args = parse_args()

    # get the modeling directory
    modeling_dir = Path(__file__).parent

    print("=" * 70)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 70)

    # ========================================================================
    # STEP 1 - TRAIN MODEL
    # ========================================================================
    train_cmd = [
        "python",
        str(modeling_dir / "train_model.py"),
    ]

    if args.tune:
        train_cmd.extend(["--tune", "--n-trials", str(args.n_trials)])

    run_command(train_cmd, "STEP 1: TRAINING MODEL")

    # ========================================================================
    # STEP 2 - CALIBRATE MODEL
    # ========================================================================
    calibrate_cmd = [
        "python",
        str(modeling_dir / "run_calibration.py"),
        "--model-path",
        "outputs/models/production_model.pkl",
        "--comprehensive",
        "--outcome-specific"
    ]

    run_command(calibrate_cmd, "STEP 2: CALIBRATING MODEL")

    # ========================================================================
    # STEP 3 - VALIDATE MODEL (OPTIONAL)
    # ========================================================================
    if not args.skip_validation:
        validate_cmd = [
            "python",
            str(modeling_dir / "validate_model.py"),
            "--model-path",
            "outputs/models/production_model.pkl",
            "--calibrator-path",
            "outputs/models/calibrators.pkl",
        ]

        run_command(validate_cmd, "STEP 3: VALIDATING MODEL")

    # ========================================================================
    # STEP 4 - GENERATE PREDICTIONS
    # ========================================================================
    predict_cmd = [
        "python",
        str(modeling_dir / "generate_predictions.py"),
        "--model-path",
        "outputs/models/production_model.pkl",
        "--calibrator-path",
        "outputs/models/calibrators.pkl",
        "--n-bootstrap",
        str(args.n_bootstrap),
    ]

    run_command(predict_cmd, "STEP 4: GENERATING PREDICTIONS")

    # summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print("  - Model: outputs/models/production_model.pkl")
    print("  - Calibrators: outputs/models/calibrators.pkl")
    if not args.skip_validation:
        print("  - Validation: outputs/validation/")
    print("  - Predictions: outputs/predictions/")
    print("  - Figures: outputs/figures/")


if __name__ == "__main__":
    main()
