#!/usr/bin/env python3
"""
Inspect Model
=============

In-sample diagnostics for a trained model. Examines the odds blend weight
contribution and per-team performance on the current season.

Note: all analysis is in-sample — the model was trained on this data.
For genuine out-of-sample evaluation use scripts/evaluation/run_backtest.py.

Usage:
    python scripts/evaluation/inspect_model.py --model-path outputs/models/production_model.pkl
    python scripts/evaluation/inspect_model.py --model-path outputs/models/production_model.pkl --output-dir outputs/evaluation/inspection
"""

import argparse
import copy
from pathlib import Path

from src.evaluation.metrics import evaluate_model_comprehensive
from src.io.model_io import load_model
from src.processing.model_preparation import prepare_bundesliga_data
from src.validation import analyse_performance_by_team


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="In-sample model diagnostics")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/inspection",
        help="Directory for outputs (default: outputs/evaluation/inspection)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INSPECT MODEL — IN-SAMPLE DIAGNOSTICS")
    print("=" * 70)
    print("Note: all results below are in-sample (model trained on this data).")
    print("For out-of-sample evaluation use scripts/evaluation/run_backtest.py.")

    model = load_model(args.model_path)
    params = model["params"]

    _, current_season = prepare_bundesliga_data(verbose=False)
    played = current_season[current_season["is_played"]].copy()

    if len(played) == 0:
        print("\nNo played matches in current season — nothing to inspect.")
        return

    season = int(played["season_end_year"].iloc[0])
    print(f"\nSeason: {season}  |  Played matches: {len(played)}")

    # -------------------------------------------------------------------------
    # odds blend ablation — compare pure model, production blend, pure odds
    # -------------------------------------------------------------------------
    print("\n=== Odds blend ablation ===")
    production_w = params.get("odds_blend_weight", 1.0)
    ablation_weights = [
        ("Pure model  (w=1.00)", 1.0),
        (f"Production  (w={production_w:.2f})", production_w),
        ("Pure odds   (w=0.00)", 0.0),
    ]

    print(f"  {'Strategy':<35}  {'Mean RPS':>10}")
    print(f"  {'-' * 35}  {'-' * 10}")
    for label, w in ablation_weights:
        ablation_params = copy.deepcopy(params)
        ablation_params["odds_blend_weight"] = w
        metrics, _, _ = evaluate_model_comprehensive(
            ablation_params, played, use_dixon_coles=True
        )
        print(f"  {label:<35}  {metrics['rps']:>10.4f}")

    # -------------------------------------------------------------------------
    # team performance
    # -------------------------------------------------------------------------
    print("\n=== Team performance ===")
    _, predictions, actuals = evaluate_model_comprehensive(
        params, played, use_dixon_coles=True
    )
    team_df = analyse_performance_by_team(predictions, actuals, played, verbose=True)

    csv_path = output_dir / f"team_analysis_{season}.csv"
    team_df.to_csv(csv_path, index=False)
    print(f"\nFull team analysis saved to {csv_path}")


if __name__ == "__main__":
    main()
