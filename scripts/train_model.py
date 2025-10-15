#!/usr/bin/env python3
"""
Train Final Model
=================

Fit the production model on all available data.

Usage:
    python scripts/train_model.py [--optimise-hyperparams] [--n-trials 30]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    fit_poisson_model,
    calculate_home_advantage_prior,
    calculate_promoted_team_priors,
    identify_promoted_teams,
    optimise_hyperparameters,
    get_default_hyperparameters,
)
from src.features import prepare_model_features
from src.processing.model_preparation import prepare_bundesliga_data
from src.io.model_io import save_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train final production model on all available data"
    )

    parser.add_argument(
        "--optimise-hyperparams",
        action="store_true",
        help="Optimise hyperparameters via cross-validation",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for hyperparameter optimisation (default: 30)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Directory to save trained model (default: outputs/models)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="production_model",
        help="Name for saved model file (default: production_model)",
    )

    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Rolling window sizes for npxGD features (default: 5 10)",
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 70)
    print("TRAINING FINAL PRODUCTION MODEL")
    print("=" * 70)

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n1. Loading data...")
    historic_data, current_season = prepare_bundesliga_data(
        windows=args.windows, verbose=True
    )

    print(f"   Historic data: {len(historic_data)} matches")
    print(f"   Current season: {len(current_season)} matches")

    # verify that weighted performance was calculated
    if "home_goals_weighted" not in historic_data.columns:
        print("\n   ✗ Error: home_goals_weighted not found in data")
        print(
            "   Make sure prepare_bundesliga_data includes weighted performance calculation"
        )
        sys.exit(1)

    print("   ✓ Weighted performance calculated")

    # ========================================================================
    # PREPARE TRAINING DATA
    # ========================================================================
    print("\n2. Preparing training data...")

    # combine historic and played current season matches
    current_played = current_season[current_season["is_played"] == True].copy()
    all_train_data = pd.concat([historic_data, current_played], ignore_index=True)

    print(f"   Total training data: {len(all_train_data)} matches")
    print(
        f"   Date range: {all_train_data['date'].min().date()} to {all_train_data['date'].max().date()}"
    )

    # ========================================================================
    # HYPERPARAMETER OPTIMISATION (Optional)
    # ========================================================================
    if args.optimise_hyperparams:
        print("\n3. Optimising hyperparameters...")
        hyperparams = optimise_hyperparameters(
            all_train_data, n_trials=args.n_trials, metric="rps", verbose=True
        )
    else:
        print("\n3. Using default hyperparameters...")
        hyperparams = get_default_hyperparameters()
        for key, val in hyperparams.items():
            print(f"   {key}: {val}")

    # ========================================================================
    # CALCULATE PRIORS
    # ========================================================================
    print("\n4. Calculating priors...")

    # home advantage prior
    home_adv_prior, home_adv_std = calculate_home_advantage_prior(
        all_train_data, use_actual_goals=True, verbose=True
    )

    # promoted teams
    last_historic_season = historic_data[
        historic_data["season_end_year"] == historic_data["season_end_year"].max()
    ]

    promoted_teams = identify_promoted_teams(last_historic_season, current_season)

    if promoted_teams:
        print(f"   Promoted teams detected: {', '.join(promoted_teams)}")
        promoted_priors, _, _ = calculate_promoted_team_priors(
            all_train_data, promoted_teams, current_season
        )
    else:
        promoted_priors = None
        print("   No promoted teams detected")

    # ========================================================================
    # FIT MODEL
    # ========================================================================
    print("\n5. Fitting model on all training data...")

    fitted_params = fit_poisson_model(
        all_train_data,
        hyperparams,
        promoted_priors=promoted_priors,
        home_adv_prior=home_adv_prior,
        home_adv_std=home_adv_std,
        n_random_starts=5,
        verbose=True,
    )

    if not fitted_params or not fitted_params.get("success", False):
        print("\n✗ Model fitting failed")
        sys.exit(1)

    print("   ✓ Model fitted successfully")

    # ========================================================================
    # PACKAGE MODEL
    # ========================================================================
    print("\n6. Packaging model...")

    model_package = {
        "params": fitted_params,
        "hyperparams": hyperparams,
        "promoted_priors": promoted_priors,
        "home_adv_prior": home_adv_prior,
        "home_adv_std": home_adv_std,
        "windows": args.windows,
        "train_data_shape": all_train_data.shape,
        "train_date_range": (
            all_train_data["date"].min(),
            all_train_data["date"].max(),
        ),
        "n_teams": len(fitted_params.get("teams", [])),
        "teams": fitted_params.get("teams", []),
    }

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n7. Saving model...")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / f"{args.model_name}.pkl"
    save_model(model_package, model_path)

    print(f"   ✓ Model saved: {model_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {model_path}")
    print(f"Teams: {len(fitted_params.get('teams', []))}")
    print(f"Training matches: {len(all_train_data)}")

    # display log-likelihood if available
    if "log_likelihood" in fitted_params:
        print(f"Final log-likelihood: {fitted_params['log_likelihood']:.2f}")

    print(f"Rolling windows used: {args.windows}")

    # display team strengths if available
    if "team_params" in fitted_params:
        print("\nTop 5 teams by attack strength:")
        attack_strengths = fitted_params["team_params"]
        attack_sorted = sorted(
            attack_strengths.items(), key=lambda x: x[1]["attack"], reverse=True
        )
        for i, (team, params) in enumerate(attack_sorted[:5], 1):
            print(f"  {i}. {team}: {params['attack']:.3f}")

    # display fitted parameters structure (debugging)
    print("\nFitted parameters structure:")
    for key in fitted_params.keys():
        value_type = type(fitted_params[key]).__name__
        if isinstance(fitted_params[key], (list, dict)):
            length = len(fitted_params[key])
            print(f"  {key}: {value_type} (length: {length})")
        else:
            print(f"  {key}: {value_type}")

    print("\nNext steps:")
    print(f"  - Validate: python scripts/validate_model.py --model-path {model_path}")
    print(f"  - Calibrate: python scripts/run_calibration.py --model-path {model_path}")
    print(
        f"  - Predict: python scripts/generate_predictions.py --model-path {model_path}"
    )


if __name__ == "__main__":
    main()
