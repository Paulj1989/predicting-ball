#!/usr/bin/env python3
"""
Calibration Check
=================

Quick pipeline check that calibrators are still effective on recent matches.
Exits non-zero if calibration has degraded.

Usage:
    python scripts/evaluation/check_calibration.py
    python scripts/evaluation/check_calibration.py --n-recent 30
"""

import argparse
import sys

from src.evaluation.metrics import calculate_rps, evaluate_model_comprehensive
from src.io.model_io import load_calibrators, load_model
from src.models.calibration import apply_calibration
from src.processing.model_preparation import prepare_bundesliga_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check calibration on recent matches")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/production_model.pkl",
        help="Path to trained model (default: outputs/models/production_model.pkl)",
    )
    parser.add_argument(
        "--calibrator-path",
        type=str,
        default="outputs/models/calibrators.pkl",
        help="Path to calibrators (default: outputs/models/calibrators.pkl)",
    )
    parser.add_argument(
        "--n-recent",
        type=int,
        default=50,
        help="Number of recent played matches to check (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(args.model_path)
    calibrators = load_calibrators(args.calibrator_path)

    _, current_season = prepare_bundesliga_data(verbose=False)
    recent = current_season[current_season["is_played"]].tail(args.n_recent)

    if len(recent) == 0:
        print("Calibration check: no recent played matches — skipping")
        sys.exit(0)

    _, predictions, actuals = evaluate_model_comprehensive(
        model["params"], recent, use_dixon_coles=True
    )

    calibrated = apply_calibration(predictions, calibrators)

    n = len(actuals)
    rps_before = calculate_rps(predictions, actuals)
    rps_after = calculate_rps(calibrated, actuals)
    improvement = rps_before - rps_after
    ok = improvement >= -0.01

    print(
        f"Calibration check ({n} matches): "
        f"RPS {rps_before:.4f} → {rps_after:.4f} ({improvement:+.4f}) "
        f"[{'OK' if ok else 'FAIL'}]"
    )

    if not ok:
        print("  Calibration has degraded — consider recalibrating with recent data")
        sys.exit(1)


if __name__ == "__main__":
    main()
