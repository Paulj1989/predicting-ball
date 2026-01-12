#!/usr/bin/env python3
"""
Run Calibration
===============

Fit post-hoc calibrators for improved probability estimates.

Now includes outcome-specific temperature scaling for better draw calibration.

Usage:
    python scripts/modeling/run_calibration.py --model-path outputs/models/production_model.pkl
    python scripts/modeling/run_calibration.py --model-path outputs/models/production_model.pkl --outcome-specific
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.models.calibration import (
    fit_temperature_scaler,
    apply_temperature_scaling,
    fit_outcome_specific_temperatures,
    apply_outcome_specific_scaling,
    validate_calibration_on_holdout,
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
        default=0.3,
        help="Fraction of data to use for calibration (default: 0.3)",
    )

    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.15,
        help="Fraction of data to use for holdout validation (default: 0.15)",
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
        "--outcome-specific",
        action="store_true",
        help="Use outcome-specific temperature scaling (recommended for draw issues)",
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

    parser.add_argument(
        "--metric",
        type=str,
        choices=["rps", "log_loss", "brier"],
        default="rps",
        help="Primary metric to report and emphasise (default: rps)",
    )

    return parser.parse_args()


def main():
    """Main calibration pipeline"""
    args = parse_args()

    print("=" * 70)
    if args.outcome_specific:
        print("POST-HOC CALIBRATION (OUTCOME-SPECIFIC TEMPERATURE SCALING)")
    else:
        print("POST-HOC CALIBRATION (STANDARD TEMPERATURE SCALING)")
    print("=" * 70)

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n1. Loading model...")
    model = load_model(args.model_path)

    print(f"   ✓ Model loaded: {args.model_path}")
    print(f"   Teams: {len(model.get('teams', []))}")

    # check dixon-coles
    rho = model["params"].get("rho", None)
    if rho is not None:
        print(f"   Dixon-Coles rho: {rho:.4f}")
    else:
        print("   WARNING: Dixon-Coles not found in model")

    # get windows from model if not provided
    windows = args.windows if args.windows else model.get("windows", [5, 10])
    print(f"   Using rolling windows: {windows}")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n2. Loading calibration data...")

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

    # use all available data
    current_played = current_season[current_season["is_played"] == True].copy()
    all_data = pd.concat([historic_data, current_played], ignore_index=True)

    print(f"   Total matches: {len(all_data)}")
    print(
        f"   Date range: {all_data['date'].min().date()} to {all_data['date'].max().date()}"
    )

    # three-way split: fit, calibration, holdout
    total_test_fraction = args.calibration_fraction + args.holdout_fraction
    fit_data, test_data = create_calibration_split(
        all_data, calibration_fraction=total_test_fraction
    )

    # split test_data into calibration and holdout
    n_test = len(test_data)
    n_cal = int(n_test * (args.calibration_fraction / total_test_fraction))

    calibration_data = test_data.iloc[:n_cal]
    holdout_data = test_data.iloc[n_cal:]

    print(f"   Fitting set: {len(fit_data)} matches")
    print(f"   Calibration set: {len(calibration_data)} matches")
    print(f"   Holdout validation: {len(holdout_data)} matches")

    # check draw rate
    cal_draw_rate = (
        calibration_data["home_goals"] == calibration_data["away_goals"]
    ).mean()
    holdout_draw_rate = (
        holdout_data["home_goals"] == holdout_data["away_goals"]
    ).mean()
    print(f"   Draw rate (cal): {cal_draw_rate:.1%}")
    print(f"   Draw rate (holdout): {holdout_draw_rate:.1%}")

    # ========================================================================
    # GET PREDICTIONS
    # ========================================================================
    print("\n3. Generating predictions...")

    # calibration set predictions
    metrics_uncal_cal, cal_predictions, cal_actuals = evaluate_model_comprehensive(
        model["params"], calibration_data, use_dixon_coles=True
    )

    # holdout set predictions
    metrics_uncal_holdout, holdout_predictions, holdout_actuals = (
        evaluate_model_comprehensive(
            model["params"], holdout_data, use_dixon_coles=True
        )
    )

    print(f"\n   Uncalibrated metrics (calibration set):")

    # emphasise selected metric
    metric_map = {"rps": "RPS", "log_loss": "Log Loss", "brier": "Brier"}
    metric_display = metric_map[args.metric]

    if args.metric == "rps":
        print(f"     {metric_display}: {metrics_uncal_cal.get('rps', 'N/A')} (PRIMARY)")
        print(f"     Brier: {metrics_uncal_cal['brier_score']:.4f}")
        print(f"     Log Loss: {metrics_uncal_cal.get('log_loss', 'N/A')}")
    elif args.metric == "log_loss":
        print(
            f"     {metric_display}: {metrics_uncal_cal.get('log_loss', 'N/A')} (PRIMARY)"
        )
        print(f"     Brier: {metrics_uncal_cal['brier_score']:.4f}")
        print(f"     RPS: {metrics_uncal_cal.get('rps', 'N/A')}")
    else:  # brier
        print(
            f"     {metric_display}: {metrics_uncal_cal['brier_score']:.4f} (PRIMARY)"
        )
        print(f"     RPS: {metrics_uncal_cal.get('rps', 'N/A')}")
        print(f"     Log Loss: {metrics_uncal_cal.get('log_loss', 'N/A')}")

    # check draw prediction rate
    pred_draws_cal = (np.argmax(cal_predictions, axis=1) == 1).sum()
    actual_draws_cal = (cal_actuals == 1).sum()
    print(
        f"     Predicted draws: {pred_draws_cal}/{len(cal_actuals)} ({pred_draws_cal / len(cal_actuals):.1%})"
    )
    print(
        f"     Actual draws: {actual_draws_cal}/{len(cal_actuals)} ({actual_draws_cal / len(cal_actuals):.1%})"
    )

    # ========================================================================
    # FIT CALIBRATION
    # ========================================================================
    if args.outcome_specific:
        print("\n4. Fitting outcome-specific temperature scaling...")

        temperatures = fit_outcome_specific_temperatures(
            cal_predictions, cal_actuals, verbose=True
        )

        # apply to calibration set
        cal_preds_calibrated = apply_outcome_specific_scaling(
            cal_predictions, temperatures
        )

        # store in calibrator package
        calibration_method = "outcome_specific"
        temperature_value = temperatures

    else:
        print("\n4. Fitting standard temperature scaling...")

        temperature = fit_temperature_scaler(cal_predictions, cal_actuals, verbose=True)

        # apply to calibration set
        cal_preds_calibrated = apply_temperature_scaling(cal_predictions, temperature)

        # store in calibrator package
        calibration_method = "temperature_scaling"
        temperature_value = temperature

        # convert to dict format for consistency
        temperatures = {
            "T_home": temperature,
            "T_draw": temperature,
            "T_away": temperature,
            "method": "standard",
        }

    # calculate calibrated metrics on calibration set
    from src.evaluation.metrics import (
        calculate_brier_score,
        calculate_rps,
        calculate_log_loss,
    )

    brier_cal = calculate_brier_score(cal_preds_calibrated, cal_actuals)
    rps_cal = calculate_rps(cal_preds_calibrated, cal_actuals)
    log_loss_cal = calculate_log_loss(cal_preds_calibrated, cal_actuals)

    # draw metrics on calibration set
    pred_draws_cal_calibrated = (np.argmax(cal_preds_calibrated, axis=1) == 1).sum()
    draw_acc_cal_before = (
        (np.argmax(cal_predictions, axis=1) == 1) & (cal_actuals == 1)
    ).sum() / max((cal_actuals == 1).sum(), 1)
    draw_acc_cal_after = (
        (np.argmax(cal_preds_calibrated, axis=1) == 1) & (cal_actuals == 1)
    ).sum() / max((cal_actuals == 1).sum(), 1)

    print("\n   Calibrated metrics (calibration set):")

    # emphasise selected metric with improvement
    if args.metric == "rps":
        rps_uncal = metrics_uncal_cal.get("rps", rps_cal)
        print(f"     RPS: {rps_cal:.4f} (Δ = {rps_cal - rps_uncal:+.4f}) (PRIMARY)")
        print(
            f"     Brier: {brier_cal:.4f} (Δ = {brier_cal - metrics_uncal_cal['brier_score']:+.4f})"
        )
        print(f"     Log Loss: {log_loss_cal:.4f}")
    elif args.metric == "log_loss":
        log_loss_uncal = metrics_uncal_cal.get("log_loss", log_loss_cal)
        print(
            f"     Log Loss: {log_loss_cal:.4f} (Δ = {log_loss_cal - log_loss_uncal:+.4f}) (PRIMARY)"
        )
        print(
            f"     Brier: {brier_cal:.4f} (Δ = {brier_cal - metrics_uncal_cal['brier_score']:+.4f})"
        )
        print(f"     RPS: {rps_cal:.4f}")
    else:  # brier
        print(
            f"     Brier: {brier_cal:.4f} (Δ = {brier_cal - metrics_uncal_cal['brier_score']:+.4f}) (PRIMARY)"
        )
        rps_uncal = metrics_uncal_cal.get("rps", rps_cal)
        print(f"     RPS: {rps_cal:.4f} (Δ = {rps_cal - rps_uncal:+.4f})")
        print(f"     Log Loss: {log_loss_cal:.4f}")
    print(
        f"     Predicted draws: {pred_draws_cal_calibrated}/{len(cal_actuals)} ({pred_draws_cal_calibrated / len(cal_actuals):.1%})"
    )
    print(
        f"     Draw accuracy: {draw_acc_cal_before:.1%} → {draw_acc_cal_after:.1%} ({(draw_acc_cal_after - draw_acc_cal_before) * 100:+.0f}pp)"
    )

    # ========================================================================
    # VALIDATE ON HOLDOUT
    # ========================================================================
    print("\n5. Validating on holdout set...")

    holdout_metrics = validate_calibration_on_holdout(
        temperatures,
        holdout_predictions,
        holdout_actuals,
        verbose=True,
    )

    # ========================================================================
    # DISPERSION CALIBRATION (Optional)
    # ========================================================================
    dispersion_calibrated = None

    if args.comprehensive:
        print("\n6. Running comprehensive dispersion calibration...")

        dispersion_dict, _ = calibrate_model_comprehensively(
            model["params"], calibration_data, verbose=True
        )

        dispersion_calibrated = {
            "dispersion_dict": dispersion_dict,
        }

        print("   ✓ Dispersion calibration complete")
    else:
        print("\n6. Skipping comprehensive calibration (use --comprehensive to enable)")

    # ========================================================================
    # SAVE CALIBRATORS
    # ========================================================================
    print("\n7. Saving calibrators...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrator_package = {
        "temperature": temperature_value,
        "calibration_method": calibration_method,
        "dispersion_calibrated": dispersion_calibrated,
        "calibration_data_size": len(calibration_data),
        "holdout_data_size": len(holdout_data),
        "brier_uncalibrated_cal": metrics_uncal_cal["brier_score"],
        "brier_calibrated_cal": brier_cal,
        "brier_uncalibrated_holdout": holdout_metrics["brier_uncalibrated"],
        "brier_calibrated_holdout": holdout_metrics["brier_calibrated"],
        "rps_improvement_holdout": holdout_metrics["rps_improvement"],
        "draw_accuracy_improvement_holdout": holdout_metrics["draw_accuracy_calibrated"]
        - holdout_metrics["draw_accuracy_uncalibrated"],
        "windows": windows,
        "calibration_date": pd.Timestamp.now(),
    }

    calibrator_path = output_dir / f"{args.calibrator_name}.pkl"
    save_calibrators(calibrator_package, calibrator_path)

    print(f"   ✓ Calibrators saved: {calibrator_path}")

    # ========================================================================
    # CREATE CALIBRATION REPORT
    # ========================================================================
    print("\n8. Creating calibration report...")

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # use holdout set for report (more honest assessment)
    holdout_preds_calibrated = apply_outcome_specific_scaling(
        holdout_predictions, temperatures
    )

    try:
        report = create_calibration_report(
            holdout_preds_calibrated,
            holdout_actuals,
            model_name="Calibrated Model (Holdout Set)",
            save_path=figures_dir / "calibration_report.png",
        )
        print(f"   ✓ Report saved: {figures_dir / 'calibration_report.png'}")
    except Exception as e:
        print(f"   Could not create calibration report: {e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)

    if args.outcome_specific:
        print("Method: Outcome-Specific Temperature Scaling")
        print("Temperatures:")
        print(f"  T_home = {temperatures['T_home']:.3f}")
        print(f"  T_draw = {temperatures['T_draw']:.3f}")
        print(f"  T_away = {temperatures['T_away']:.3f}")
    else:
        print("Method: Standard Temperature Scaling")
        print(f"Optimal temperature: {temperature_value:.3f}")

    print(f"\nCalibration set: {len(calibration_data)} matches")
    print(f"Holdout set: {len(holdout_data)} matches")

    print("\nHoldout validation (most honest assessment):")

    # emphasise selected metric first
    if args.metric == "rps":
        print(f"  RPS improvement: {holdout_metrics['rps_improvement']:+.4f} (PRIMARY)")
        print(f"  Brier improvement: {holdout_metrics['brier_improvement']:+.4f}")
        print(
            f"  Log Loss improvement: {holdout_metrics.get('log_loss_improvement', 'N/A')}"
        )
    elif args.metric == "log_loss":
        print(
            f"  Log Loss improvement: {holdout_metrics.get('log_loss_improvement', 'N/A')} (PRIMARY)"
        )
        print(f"  Brier improvement: {holdout_metrics['brier_improvement']:+.4f}")
        print(f"  RPS improvement: {holdout_metrics['rps_improvement']:+.4f}")
    else:  # brier
        print(
            f"  Brier improvement: {holdout_metrics['brier_improvement']:+.4f} (PRIMARY)"
        )
        print(f"  RPS improvement: {holdout_metrics['rps_improvement']:+.4f}")
        print(
            f"  Log Loss improvement: {holdout_metrics.get('log_loss_improvement', 'N/A')}"
        )
    print(
        f"  Draw accuracy: {holdout_metrics['draw_accuracy_uncalibrated']:.1%} → {holdout_metrics['draw_accuracy_calibrated']:.1%}"
    )
    print(
        f"    Improvement: {(holdout_metrics['draw_accuracy_calibrated'] - holdout_metrics['draw_accuracy_uncalibrated']) * 100:+.0f} percentage points"
    )

    if dispersion_calibrated:
        print("\nDispersion calibration: ✓ Enabled")
    else:
        print("\nDispersion calibration: Not applied (use --comprehensive)")

    # draw-specific assessment
    draw_improvement = (
        holdout_metrics["draw_accuracy_calibrated"]
        - holdout_metrics["draw_accuracy_uncalibrated"]
    ) * 100

    print(f"\n Draw accuracy improved by {draw_improvement:.0f}pp on holdout")


if __name__ == "__main__":
    main()
