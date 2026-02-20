#!/usr/bin/env python3
"""
Weekly Live Monitoring
======================

Scores recent production predictions from pb.duckdb against actual results
and reports rolling performance trends. Run weekly (Monday 06:00 UTC).

Usage:
    python scripts/evaluation/run_monitoring.py
    python scripts/evaluation/run_monitoring.py --db-path data/pb.duckdb
    python scripts/evaluation/run_monitoring.py --lookback-days 60 --drift-threshold 0.225
    python scripts/evaluation/run_monitoring.py --competition "Bundesliga"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.evaluation.significance import diebold_mariano_test

# -------------------------------------------------------------------
# SQL: create the last-pre-match-prediction-per-match view
# -------------------------------------------------------------------
CREATE_LAST_PREDICTION_VIEW = """
CREATE OR REPLACE VIEW models.last_prediction_per_match AS
WITH ranked AS (
    SELECT
        ps.run_id,
        ps.match_id,
        ps.home_prob,
        ps.draw_prob,
        ps.away_prob,
        ps.actual_outcome,
        ps.result_rps,
        ps.result_brier,
        ps.result_log_loss,
        ps.result_correct,
        ph.predicted_at,
        mf.date        AS match_date,
        mf.matchweek,
        mf.season_end_year,
        mf.competition,
        mf.market_avg_home_odds,
        mf.market_avg_draw_odds,
        mf.market_avg_away_odds,
        ROW_NUMBER() OVER (
            PARTITION BY ps.match_id
            ORDER BY ph.predicted_at DESC
        ) AS rn
    FROM models.prediction_scores ps
    JOIN models.prediction_history ph
        ON ps.run_id = ph.run_id AND ps.match_id = ph.match_id
    JOIN models.match_features mf
        ON ps.match_id = mf.match_id
    -- only predictions made strictly before the match
    WHERE CAST(ph.predicted_at AS DATE) < mf.date::DATE
)
SELECT * EXCLUDE (rn) FROM ranked WHERE rn = 1
"""

# -------------------------------------------------------------------
# SQL: fetch scored predictions within the lookback window
# -------------------------------------------------------------------
SCORED_QUERY = """
SELECT
    lp.match_id,
    lp.match_date,
    lp.season_end_year,
    lp.matchweek,
    lp.predicted_at,
    lp.home_prob,
    lp.draw_prob,
    lp.away_prob,
    lp.actual_outcome,
    lp.result_log_loss AS model_log_loss,
    lp.market_avg_home_odds,
    lp.market_avg_draw_odds,
    lp.market_avg_away_odds
FROM models.last_prediction_per_match lp
WHERE lp.match_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
  AND lp.actual_outcome IS NOT NULL
  AND lp.competition = '{competition}'
ORDER BY lp.match_date
"""

# -------------------------------------------------------------------
# SQL: recent per-match data for rolling RPS trend (last ~20 matchweeks)
# -------------------------------------------------------------------
ROLLING_QUERY = """
SELECT
    season_end_year AS season,
    matchweek,
    home_prob,
    draw_prob,
    away_prob,
    actual_outcome,
    match_date
FROM models.last_prediction_per_match
WHERE actual_outcome IS NOT NULL
  AND competition = '{competition}'
  AND match_date >= CURRENT_DATE - INTERVAL '200 days'
ORDER BY match_date DESC
"""


def _compute_rps(probs: np.ndarray, actual_outcome: str) -> float | None:
    """Compute RPS for a single match given probability array and actual outcome."""
    idx = {"H": 0, "D": 1, "A": 2}.get(actual_outcome)
    if idx is None:
        return None
    one_hot = np.zeros(3)
    one_hot[idx] = 1.0
    return float(np.sum((np.cumsum(probs) - np.cumsum(one_hot)) ** 2) / 2)


def _compute_implied_rps(row: pd.Series) -> float | None:
    """Compute RPS from market average decimal odds for one match."""
    h = row["market_avg_home_odds"]
    d = row["market_avg_draw_odds"]
    a = row["market_avg_away_odds"]
    if pd.isna(h) or pd.isna(d) or pd.isna(a):
        return None
    # decimal odds must be > 1 to be valid
    if h <= 1 or d <= 1 or a <= 1:
        return None
    # convert to implied probs and normalise to strip overround
    raw = np.array([1.0 / h, 1.0 / d, 1.0 / a])
    probs = raw / raw.sum()
    return _compute_rps(probs, row["actual_outcome"])


def _compute_model_rps(row: pd.Series) -> float | None:
    """Compute RPS from stored model probabilities for one match."""
    probs = np.array([row["home_prob"], row["draw_prob"], row["away_prob"]])
    return _compute_rps(probs, row["actual_outcome"])


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Weekly live monitoring from pb.duckdb")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/pb.duckdb",
        help="Path to pb.duckdb (default: data/pb.duckdb)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/monitoring",
        help="Directory for monitoring outputs (default: outputs/evaluation/monitoring)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=60,
        help="Days of scored predictions to analyse (default: 60)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.225,
        help="RPS threshold that triggers a drift alert (default: 0.225)",
    )
    parser.add_argument(
        "--competition",
        type=str,
        default="Bundesliga",
        help="Competition name to filter predictions (default: Bundesliga)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"LIVE MONITORING REPORT — {args.competition.upper()}")
    print("=" * 70)

    conn = duckdb.connect(str(db_path), read_only=False)

    # create the last-prediction view (idempotent — OR REPLACE ensures fresh data)
    conn.execute(CREATE_LAST_PREDICTION_VIEW)

    # -------------------------------------------------------------------------
    # fetch recent scored predictions
    # -------------------------------------------------------------------------
    query = SCORED_QUERY.format(lookback_days=args.lookback_days, competition=args.competition)
    df = conn.execute(query).df()

    if len(df) == 0:
        print(f"No scored predictions in last {args.lookback_days} days.")
        conn.close()
        sys.exit(0)

    # compute model rps from raw stored probabilities — same formula as baseline
    df["model_rps"] = df.apply(_compute_model_rps, axis=1)

    print(f"\nScored predictions (last {args.lookback_days} days): {len(df)}")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")

    # -------------------------------------------------------------------------
    # compute baseline RPS inline from odds columns
    # -------------------------------------------------------------------------
    df["baseline_rps"] = df.apply(_compute_implied_rps, axis=1)
    baseline_available = df["baseline_rps"].notna().sum()
    print(
        f"Odds coverage: {baseline_available}/{len(df)} matches ({100 * baseline_available / len(df):.0f}%)"
    )

    # -------------------------------------------------------------------------
    # headline metrics
    # -------------------------------------------------------------------------
    mean_model_rps = float(df["model_rps"].dropna().mean())
    mean_model_ll = float(df["model_log_loss"].mean())

    print("\n=== Headline metrics ===")
    print(
        f"  Mean RPS:       {mean_model_rps:.4f}  (alert threshold: {args.drift_threshold:.4f})"
    )
    print(f"  Mean Log Loss:  {mean_model_ll:.4f}")

    mean_baseline_rps = None
    improvement_pct = None
    dm = None

    if baseline_available >= 20:
        mean_baseline_rps = float(df.loc[df["baseline_rps"].notna(), "baseline_rps"].mean())
        improvement_pct = (mean_baseline_rps - mean_model_rps) / mean_baseline_rps * 100
        print(f"  Baseline RPS:   {mean_baseline_rps:.4f}")
        print(f"  vs. Baseline:   {improvement_pct:+.1f}%")

        mask = df["baseline_rps"].notna()
        dm = diebold_mariano_test(
            df.loc[mask, "model_rps"].values,
            df.loc[mask, "baseline_rps"].values,
            alternative="less",
        )
        print(
            f"\n  DM test: statistic={dm['dm_statistic']:.3f}, p={dm['p_value']:.3f} "
            f"({'significant' if dm['significant'] else 'not significant'})"
        )
    else:
        print("  Baseline: insufficient coverage")

    # -------------------------------------------------------------------------
    # per-outcome calibration check
    # -------------------------------------------------------------------------
    print("\n=== Per-outcome frequency ===")
    outcome_col = {"H": "home_prob", "D": "draw_prob", "A": "away_prob"}
    outcome_label = {"H": "Home win", "D": "Draw", "A": "Away win"}
    calibration_ok = True
    for code, label in outcome_label.items():
        freq = (df["actual_outcome"] == code).mean()
        mean_pred = df[outcome_col[code]].mean()
        diff = abs(mean_pred - freq)
        # flag if mean predicted probability deviates from observed frequency by > 5pp
        flag = "WARNING" if diff > 0.05 else "ok"
        if diff > 0.05:
            calibration_ok = False
        print(
            f"  [{flag}] {label}: predicted {mean_pred:.3f}, actual {freq:.3f} (diff {diff:+.3f})"
        )

    # -------------------------------------------------------------------------
    # recent rolling matchweek trend
    # -------------------------------------------------------------------------
    rolling_raw = conn.execute(ROLLING_QUERY.format(competition=args.competition)).df()
    conn.close()

    rolling_raw["rps"] = rolling_raw.apply(_compute_model_rps, axis=1)
    rolling_df = (
        rolling_raw.groupby(["season", "matchweek"])
        .agg(
            n_matches=("rps", "count"),
            mean_model_rps=("rps", "mean"),
            matchweek_date=("match_date", "min"),
        )
        .reset_index()
        .sort_values("matchweek_date", ascending=False)
    )

    print("\n=== Rolling matchweek RPS (most recent 10) ===")
    print(f"  {'Season':>6}  {'MW':>4}  {'N':>5}  {'Mean RPS':>10}")
    for _, row in rolling_df.head(10).iterrows():
        print(
            f"  {int(row['season']):>6}  {int(row['matchweek']):>4}  {int(row['n_matches']):>5}  {row['mean_model_rps']:>10.4f}"
        )

    # -------------------------------------------------------------------------
    # drift detection
    # -------------------------------------------------------------------------
    drift_detected = mean_model_rps > args.drift_threshold

    print("\n=== Drift check ===")
    if drift_detected:
        print(
            f"  DRIFT ALERT: mean RPS {mean_model_rps:.4f} exceeds threshold {args.drift_threshold:.4f}"
        )
    else:
        print(
            f"  No drift: mean RPS {mean_model_rps:.4f} is within threshold {args.drift_threshold:.4f}"
        )

    if not calibration_ok:
        print("  CALIBRATION WARNING: predicted vs actual outcome frequencies diverge > 5pp")

    # -------------------------------------------------------------------------
    # save report
    # -------------------------------------------------------------------------
    report = {
        "n_scored_matches": len(df),
        "lookback_days": args.lookback_days,
        "date_range": {
            "start": str(df["match_date"].min()),
            "end": str(df["match_date"].max()),
        },
        "mean_model_rps": mean_model_rps,
        "mean_model_log_loss": mean_model_ll,
        "mean_baseline_rps": mean_baseline_rps,
        "rps_improvement_pct": improvement_pct,
        "dm_test": dm,
        "drift_detected": drift_detected,
        "drift_threshold": args.drift_threshold,
        "calibration_ok": calibration_ok,
        "per_outcome": {
            code: {
                "predicted": float(df[col].mean()),
                "actual": float((df["actual_outcome"] == code).mean()),
            }
            for code, col in outcome_col.items()
        },
    }

    report_path = output_dir / "monitoring_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # write to GitHub step summary when running in CI
    if "GITHUB_STEP_SUMMARY" in os.environ:
        summary_path = os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary_path, "a") as f:
            f.write("\n## Monitoring Summary\n\n")
            f.write("| Metric | Value |\n|:---|:---|\n")
            f.write(f"| Scored predictions (last {args.lookback_days}d) | {len(df)} |\n")
            f.write(f"| Mean RPS | {mean_model_rps:.4f} |\n")
            if mean_baseline_rps is not None:
                f.write(f"| Baseline RPS | {mean_baseline_rps:.4f} |\n")
                f.write(f"| vs. Baseline | {improvement_pct:+.1f}% |\n")
            if dm is not None:
                sig = "Yes" if dm["significant"] else "No"
                f.write(f"| DM test significant | {sig} |\n")
            f.write(f"| Drift detected | {drift_detected} |\n")

    # non-zero exit lets CI create a GitHub issue for drift/calibration problems
    if drift_detected or not calibration_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
