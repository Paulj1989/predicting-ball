#!/usr/bin/env python3
"""
Walk-Forward Backtest
=====================

Re-fit the model at each matchweek boundary and evaluate predictions on the
immediately following matchweek — the only genuinely out-of-sample setting.

Run this once to validate the model methodology on historical data, or
whenever a fundamental model change is made.

Usage:
    python scripts/evaluation/run_backtest.py
    python scripts/evaluation/run_backtest.py --n-seasons 4 --output-dir outputs/evaluation/backtest
    python scripts/evaluation/run_backtest.py --hyperparams-from outputs/models/buli_model.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.calibration_plots import save_reliability_diagrams
from src.evaluation.metrics import (
    calculate_metric_confidence_interval,
    calculate_rps,
    evaluate_model_comprehensive,
)
from src.evaluation.significance import diebold_mariano_test
from src.models import (
    calculate_promoted_team_priors,
    fit_poisson_model_two_stage,
    get_default_hyperparameters,
    identify_promoted_teams,
)
from src.models.calibration import (
    apply_calibration,
    fit_outcome_specific_temperatures,
)
from src.processing.model_preparation import prepare_bundesliga_data
from src.simulation.predictions import predict_match_probabilities
from src.validation.splits import matchweek_walkforward_splits


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest: re-fit model at each matchweek, evaluate next matchweek"
    )
    parser.add_argument(
        "--n-seasons",
        type=int,
        default=3,
        help="Number of seasons to use as backtest window (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/backtest",
        help="Directory for outputs (default: outputs/evaluation/backtest)",
    )
    parser.add_argument(
        "--hyperparams-from",
        type=str,
        default=None,
        help="Path to saved model .pkl — hyperparameters taken from this model",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["rps", "log_loss", "brier"],
        default="rps",
        help="Primary metric to report (default: rps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the first 5 matchweek steps (smoke test)",
    )
    return parser.parse_args()


def _infer_outcome(row: pd.Series) -> str | None:
    """Infer H/D/A outcome from goals columns."""
    hg, ag = row.get("home_goals"), row.get("away_goals")
    if pd.isna(hg) or pd.isna(ag):
        return None
    if hg > ag:
        return "H"
    elif ag > hg:
        return "A"
    return "D"


def _compute_per_match_rps(probs: list[float], outcome: str) -> float:
    """Compute RPS for a single match."""
    outcome_map = {"H": 0, "D": 1, "A": 2}
    idx = outcome_map[outcome]
    pred = np.array(probs, dtype=float)
    cum_pred = np.cumsum(pred)
    one_hot = np.zeros(3)
    one_hot[idx] = 1.0
    cum_actual = np.cumsum(one_hot)
    return float(np.sum((cum_pred - cum_actual) ** 2) / 2)


def _compute_implied_probs(row: pd.Series) -> list[float] | None:
    """Extract odds-implied probabilities from pre-computed odds_*_prob columns."""
    h = row.get("odds_home_prob")
    d = row.get("odds_draw_prob")
    a = row.get("odds_away_prob")
    if pd.isna(h) or pd.isna(d) or pd.isna(a):
        return None
    total = h + d + a
    if total <= 0:
        return None
    return [h / total, d / total, a / total]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WALK-FORWARD BACKTEST")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # load data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    all_data, _ = prepare_bundesliga_data(verbose=False)

    # restrict to played matches only (need actual results)
    played = all_data[all_data["home_goals"].notna() & all_data["away_goals"].notna()].copy()

    seasons = sorted(played["season_end_year"].unique())
    print(f"Available seasons: {seasons}")
    print(f"Total played matches: {len(played)}")

    # -------------------------------------------------------------------------
    # load hyperparameters
    # -------------------------------------------------------------------------
    if args.hyperparams_from:
        model_path = Path(args.hyperparams_from)
    else:
        # auto-detect from outputs/models/
        model_path = Path("outputs/models/buli_model.pkl")

    if model_path.exists():
        with open(model_path, "rb") as f:
            saved_model = pickle.load(f)
        hyperparams = saved_model.get("hyperparams", get_default_hyperparameters())
        print(f"Hyperparameters loaded from {model_path}")
    else:
        hyperparams = get_default_hyperparameters()
        print("No saved model found — using default hyperparameters")

    for key, val in hyperparams.items():
        print(f"  {key}={round(val, 4)}")

    # -------------------------------------------------------------------------
    # walk-forward loop
    # -------------------------------------------------------------------------
    n_backtest_seasons = args.n_seasons
    # need at least 3 seasons of training data before the first prediction
    max_backtest = len(seasons) - 3
    if max_backtest < 1:
        print(f"Error: need at least 4 seasons for walk-forward, got {len(seasons)}")
        sys.exit(1)
    if n_backtest_seasons > max_backtest:
        print(
            f"Note: capping --n-seasons from {n_backtest_seasons} to {max_backtest} "
            f"({len(seasons)} seasons available, need at least 3 for training)"
        )
        n_backtest_seasons = max_backtest
    min_train = len(seasons) - n_backtest_seasons

    print(f"\nRunning walk-forward with {n_backtest_seasons}-season backtest window...")
    records = []
    step = 0

    for train_data, predict_data in matchweek_walkforward_splits(
        played,
        min_train_seasons=min_train,
    ):
        season = predict_data["season_end_year"].iloc[0]
        matchweek = predict_data["matchweek"].iloc[0]
        train_cutoff = train_data["date"].max()

        print(
            f"  Step {step + 1}: season={season} mw={matchweek} (train n={len(train_data)})",
            end=" ",
            flush=True,
        )

        # fit priors — identify_promoted_teams wants only the last season before current
        current_season_data = played[played["season_end_year"] == season]
        last_historic_season = train_data[
            train_data["season_end_year"] == train_data["season_end_year"].max()
        ]
        promoted = identify_promoted_teams(
            last_historic_season, current_season_data, verbose=False
        )
        # calculate_promoted_team_priors returns (all_priors, home_adv_prior, home_adv_std)
        all_priors, home_adv_prior, home_adv_std = calculate_promoted_team_priors(
            train_data, promoted, current_season_data, verbose=False
        )

        # fit model (no tuning — hyperparams fixed)
        model_result = fit_poisson_model_two_stage(
            train_data,
            hyperparams,
            promoted_priors=all_priors,
            home_adv_prior=home_adv_prior,
            home_adv_std=home_adv_std,
            verbose=False,
        )

        # fit_poisson_model_two_stage returns the params dict directly, or None on failure
        if model_result is None:
            print("SKIPPED (model fit failed)")
            step += 1
            if args.dry_run and step >= 5:
                break
            continue

        params = model_result

        # calibrate on the tail of training data; the model was fit on all of
        # train_data, so cal_data is genuinely out-of-sample from the model's perspective
        n_cal = max(50, int(len(train_data) * 0.15))
        cal_data = train_data.sort_values("date").iloc[-n_cal:]
        fit_data_size = len(train_data) - n_cal
        calibrators = None
        if n_cal >= 30 and fit_data_size >= 100:
            _, cal_preds, cal_actuals = evaluate_model_comprehensive(
                params, cal_data, use_dixon_coles=True
            )
            calibrators = fit_outcome_specific_temperatures(
                cal_preds, cal_actuals, verbose=False
            )

        # predict each match in predict_data
        n_predicted = 0
        n_failed = 0
        for _, match in predict_data.iterrows():
            outcome = match.get("result") or _infer_outcome(match)
            if outcome not in ("H", "D", "A"):
                n_failed += 1
                continue

            pred = predict_match_probabilities(params, match, use_dixon_coles=True)
            model_probs = [pred["home_win"], pred["draw"], pred["away_win"]]

            if calibrators:
                cal_array = apply_calibration(np.array([model_probs]), calibrators)
                model_probs = cal_array[0].tolist()

            implied = _compute_implied_probs(match)
            model_rps = _compute_per_match_rps(model_probs, outcome)
            baseline_rps = _compute_per_match_rps(implied, outcome) if implied else None
            outcome_idx = {"H": 0, "D": 1, "A": 2}[outcome]
            model_ll = float(-np.log(np.clip(model_probs[outcome_idx], 1e-15, 1.0)))

            records.append(
                {
                    "step": step,
                    "season": int(season),
                    "matchweek": int(matchweek),
                    "train_cutoff_date": str(train_cutoff.date()),
                    "n_train_matches": len(train_data),
                    "match_id": match.get("match_id", ""),
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "date": str(match["date"])[:10],
                    "pred_home_win": model_probs[0],
                    "pred_draw": model_probs[1],
                    "pred_away_win": model_probs[2],
                    "implied_home_prob": implied[0] if implied else None,
                    "implied_draw_prob": implied[1] if implied else None,
                    "implied_away_prob": implied[2] if implied else None,
                    "actual_outcome": outcome,
                    "model_rps": model_rps,
                    "baseline_rps": baseline_rps,
                    "model_log_loss": model_ll,
                    "calibrated": calibrators is not None,
                }
            )
            n_predicted += 1

        print(f"→ {n_predicted} predicted", end="")
        if n_failed:
            print(f" ({n_failed} skipped)", end="")
        print()

        step += 1
        if args.dry_run and step >= 5:
            print("\n[dry-run] Stopping after 5 steps")
            break

    # -------------------------------------------------------------------------
    # aggregate results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not records:
        print("No predictions recorded — check input data.")
        sys.exit(1)

    results_df = pd.DataFrame(records)
    n_total = len(results_df)
    model_ll_arr = results_df["model_log_loss"].values

    # bootstrap CIs on RPS
    outcome_int = np.array([{"H": 0, "D": 1, "A": 2}[o] for o in results_df["actual_outcome"]])
    preds_arr = results_df[["pred_home_win", "pred_draw", "pred_away_win"]].values
    rps_est, rps_lo, rps_hi = calculate_metric_confidence_interval(
        preds_arr,
        outcome_int,
        calculate_rps,
        n_bootstrap=1000,
    )

    print(f"\nTotal predictions:  {n_total}")
    print(f"Matchweek steps:    {step}")
    print(f"Mean RPS:           {rps_est:.4f} [95% CI: {rps_lo:.4f}, {rps_hi:.4f}]")

    # DM test vs baseline
    baseline_mask = results_df["baseline_rps"].notna()
    dm = None
    if baseline_mask.sum() >= 20:
        dm = diebold_mariano_test(
            results_df.loc[baseline_mask, "model_rps"].values,
            results_df.loc[baseline_mask, "baseline_rps"].values,
            alternative="less",
        )
        baseline_rps_mean = results_df.loc[baseline_mask, "baseline_rps"].mean()
        improvement_pct = (baseline_rps_mean - rps_est) / baseline_rps_mean * 100
        print(f"Baseline RPS:       {baseline_rps_mean:.4f}")
        print(f"Improvement:        {improvement_pct:+.1f}%")
        print(f"DM statistic:       {dm['dm_statistic']:.3f}")
        print(f"DM p-value:         {dm['p_value']:.3f}")
        print(f"Significant:        {'YES' if dm['significant'] else 'NO'}")
    else:
        print("Baseline: insufficient odds coverage for DM test")

    # per-season breakdown
    print("\nPer-season breakdown:")
    print(
        f"  {'Season':>8}  {'Matchweeks':>10}  {'Matches':>8}  {'Mean RPS':>10}  {'Baseline':>10}"
    )
    for season, grp in results_df.groupby("season"):
        bl = (
            grp["baseline_rps"].mean()
            if grp["baseline_rps"].notna().sum() > 5
            else float("nan")
        )
        print(
            f"  {season:>8}  {grp['matchweek'].nunique():>10}  "
            f"{len(grp):>8}  {grp['model_rps'].mean():>10.4f}  {bl:>10.4f}"
        )

    # -------------------------------------------------------------------------
    # save outputs
    # -------------------------------------------------------------------------
    results_path = output_dir / "backtest_results.parquet"
    results_df.to_parquet(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    summary = {
        "n_predictions": n_total,
        "n_steps": step,
        "n_seasons_backtest": args.n_seasons,
        "mean_rps": float(rps_est),
        "rps_ci_lower": float(rps_lo),
        "rps_ci_upper": float(rps_hi),
        "mean_log_loss": float(model_ll_arr.mean()),
        "dm_test": dm,
        "per_season": [
            {
                "season": int(str(s)),
                "n_matches": len(g),
                "mean_rps": float(g["model_rps"].mean()),
                "mean_baseline_rps": (
                    float(g["baseline_rps"].mean())
                    if g["baseline_rps"].notna().sum() > 0
                    else None
                ),
            }
            for s, g in results_df.groupby("season")
        ],
    }
    summary_path = output_dir / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    # reliability diagrams
    diagram_path = output_dir / "reliability_diagrams.png"
    save_reliability_diagrams(
        preds_arr,
        outcome_int,
        diagram_path,
        title="Backtest Reliability Diagrams",
    )
    print(f"Reliability diagrams saved to {diagram_path}")

    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
