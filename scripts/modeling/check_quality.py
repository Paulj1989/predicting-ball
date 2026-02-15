#!/usr/bin/env python3
"""
Check Quality
=============

Soft quality gate that validates model metrics before prediction upload.
Always allows the pipeline to continue but exits with code 1 if any checks
produce warnings.

Usage:
    python scripts/modeling/check_quality.py --validation-metrics outputs/validation/metrics.json
    python scripts/modeling/check_quality.py --validation-metrics outputs/validation/metrics.json \
        --calibrator-path outputs/models/calibrators.pkl --rps-threshold 0.20
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run quality gate checks on model validation metrics"
    )

    parser.add_argument(
        "--validation-metrics",
        type=str,
        required=True,
        help="Path to validation metrics JSON file from validate_model.py",
    )

    parser.add_argument(
        "--calibrator-path",
        type=str,
        default=None,
        help="Optional path to calibrators pickle file",
    )

    parser.add_argument(
        "--rps-threshold",
        type=float,
        default=0.22,
        help="Maximum acceptable average RPS (default: 0.22)",
    )

    return parser.parse_args()


def check_baseline_comparison(validation_metrics):
    """Check whether the model beats the baseline RPS for each season"""
    seasons_worse = []

    for season_data in validation_metrics.get("per_season", []):
        season = season_data.get("season")
        model_rps = season_data.get("rps")
        baseline_rps = season_data.get("baseline_rps")

        if model_rps is None or baseline_rps is None:
            continue

        if model_rps > baseline_rps:
            seasons_worse.append(
                {
                    "season": season,
                    "model_rps": model_rps,
                    "baseline_rps": baseline_rps,
                }
            )

    if seasons_worse:
        details = "; ".join(
            f"season {s['season']}: model={s['model_rps']:.4f} vs baseline={s['baseline_rps']:.4f}"
            for s in seasons_worse
        )
        return {
            "name": "baseline_comparison",
            "status": "warn",
            "details": f"Model worse than baseline in {len(seasons_worse)} season(s): {details}",
        }

    return {
        "name": "baseline_comparison",
        "status": "pass",
        "details": "Model beats baseline in all seasons",
    }


def check_rps_threshold(validation_metrics, threshold):
    """Check whether average RPS exceeds the configured ceiling"""
    avg_rps = validation_metrics.get("average", {}).get("rps")

    if avg_rps is None:
        return {
            "name": "rps_threshold",
            "status": "warn",
            "details": "Average RPS not found in validation metrics",
        }

    if avg_rps > threshold:
        return {
            "name": "rps_threshold",
            "status": "warn",
            "details": f"Average RPS {avg_rps:.4f} exceeds threshold {threshold:.4f}",
        }

    return {
        "name": "rps_threshold",
        "status": "pass",
        "details": f"Average RPS {avg_rps:.4f} is within threshold {threshold:.4f}",
    }


def check_calibration_health(calibrator_path):
    """Check whether calibration is improving or degrading RPS"""
    if calibrator_path is None:
        return {
            "name": "calibration_health",
            "status": "pass",
            "details": "No calibrator path provided, skipping check",
        }

    path = Path(calibrator_path)
    if not path.exists():
        return {
            "name": "calibration_health",
            "status": "warn",
            "details": f"Calibrator file not found: {calibrator_path}",
        }

    try:
        with open(path, "rb") as f:
            calibrators = pickle.load(f)
    except Exception as e:
        return {
            "name": "calibration_health",
            "status": "warn",
            "details": f"Failed to load calibrators: {e}",
        }

    rps_improvement = calibrators.get("rps_improvement_holdout")

    if rps_improvement is None:
        return {
            "name": "calibration_health",
            "status": "pass",
            "details": "No rps_improvement_holdout found in calibrators, skipping check",
        }

    # positive means calibration made RPS worse (lower RPS is better)
    if rps_improvement > 0:
        return {
            "name": "calibration_health",
            "status": "warn",
            "details": f"Calibration degraded RPS by {rps_improvement:.4f} on holdout set",
        }

    return {
        "name": "calibration_health",
        "status": "pass",
        "details": f"Calibration improved RPS by {abs(rps_improvement):.4f} on holdout set",
    }


def format_report(results):
    """Format check results as a readable report"""
    lines = []
    lines.append("=" * 60)
    lines.append("QUALITY GATE REPORT")
    lines.append("=" * 60)

    for result in results:
        icon = "PASS" if result["status"] == "pass" else "WARN"
        lines.append(f"\n  [{icon}] {result['name']}")
        lines.append(f"         {result['details']}")

    lines.append("\n" + "-" * 60)

    warnings = [r for r in results if r["status"] == "warn"]
    if warnings:
        lines.append(f"  {len(warnings)} warning(s) detected")
    else:
        lines.append("  All checks passed")

    lines.append("=" * 60)

    return "\n".join(lines)


def format_markdown(results):
    """Format check results as markdown for GitHub step summary"""
    lines = []
    lines.append("## Quality Gate Report")
    lines.append("")
    lines.append("| Check | Status | Details |")
    lines.append("|-------|--------|---------|")

    for result in results:
        icon = "✅" if result["status"] == "pass" else "⚠️"
        lines.append(f"| {result['name']} | {icon} {result['status']} | {result['details']} |")

    warnings = [r for r in results if r["status"] == "warn"]
    lines.append("")
    if warnings:
        lines.append(f"**{len(warnings)} warning(s) detected**")
    else:
        lines.append("**All checks passed**")

    return "\n".join(lines)


def main():
    """Run quality gate checks"""
    args = parse_args()

    # load validation metrics
    metrics_path = Path(args.validation_metrics)
    if not metrics_path.exists():
        print(f"Error: Validation metrics file not found: {args.validation_metrics}")
        sys.exit(1)

    with open(metrics_path) as f:
        validation_metrics = json.load(f)

    # run checks
    results = [
        check_baseline_comparison(validation_metrics),
        check_rps_threshold(validation_metrics, args.rps_threshold),
        check_calibration_health(args.calibrator_path),
    ]

    # print report
    report = format_report(results)
    print(report)

    # save report as json
    output_dir = Path("outputs/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    # write to github step summary if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        markdown = format_markdown(results)
        with open(summary_path, "a") as f:
            f.write(markdown + "\n")
        print("Markdown summary written to GITHUB_STEP_SUMMARY")

    # exit with 1 if any warnings
    has_warnings = any(r["status"] == "warn" for r in results)
    sys.exit(1 if has_warnings else 0)


if __name__ == "__main__":
    main()
