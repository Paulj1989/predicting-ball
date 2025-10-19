#!/usr/bin/env python3
"""
Train Final Model
=================

Fit the production model on all available data.

Usage:
    python scripts/modeling/train_model.py [--tune] [--dry-run]
    python scripts/modeling/train_model.py --tune --n-trials 50
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import (
    fit_poisson_model,
    calculate_home_advantage_prior,
    calculate_promoted_team_priors,
    identify_promoted_teams,
    optimise_hyperparameters,
    get_default_hyperparameters,
)
from src.processing.model_preparation import prepare_bundesliga_data
from src.io.model_io import save_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train final production model on all available data"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run full hyperparameter optimisation via cross-validation",
    )

    parser.add_argument(
        "--optimise-hyperparams",
        action="store_true",
        help="Alias for --tune (for backwards compatibility)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for hyperparameter optimisation (default: 30)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save to outputs/test/ instead of outputs/models/",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save trained model (overrides dry-run)",
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


def load_previous_hyperparameters(output_dir: Path) -> dict:
    """Load previously optimised hyperparameters"""
    params_path = output_dir / "best_hyperparams.pkl"

    if params_path.exists():
        with open(params_path, "rb") as f:
            saved_params = pickle.load(f)
        return saved_params

    return None


def save_hyperparameters(hyperparams: dict, output_dir: Path, dry_run: bool = False):
    """Save optimised hyperparameters for future use"""
    if not dry_run:
        params_path = output_dir / "best_hyperparams.pkl"

        save_data = {
            "hyperparams": hyperparams,
            "optimised_at": datetime.now(),
            "note": "Use these for weekly training without --tune flag",
        }

        with open(params_path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"   Hyperparameters saved: {params_path}")


def main():
    """Main training pipeline"""
    args = parse_args()

    # handle alias
    if args.optimise_hyperparams:
        args.tune = True

    # determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.dry_run:
        output_dir = Path("outputs/test")
        print("DRY RUN MODE - outputs will be saved to outputs/test/")
    else:
        output_dir = Path("outputs/models")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING FINAL PRODUCTION MODEL")
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN MODE - production not affected")
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

    if "home_goals_weighted" not in historic_data.columns:
        print("\n   Error: home_goals_weighted not found in data")
        print(
            "   Make sure prepare_bundesliga_data includes weighted performance calculation"
        )
        sys.exit(1)

    print("   Weighted performance calculated")

    # ========================================================================
    # PREPARE TRAINING DATA
    # ========================================================================
    print("\n2. Preparing training data...")

    current_played = current_season[current_season["is_played"] == True].copy()
    all_train_data = pd.concat([historic_data, current_played], ignore_index=True)

    print(f"   Total training data: {len(all_train_data)} matches")
    print(
        f"   Date range: {all_train_data['date'].min().date()} to {all_train_data['date'].max().date()}"
    )

    # ========================================================================
    # HYPERPARAMETER OPTIMISATION OR LOADING
    # ========================================================================
    if args.tune:
        print("\n3. Running full hyperparameter optimisation...")

        hyperparams = optimise_hyperparameters(
            all_train_data, n_trials=args.n_trials, metric="rps", verbose=True
        )

        save_hyperparameters(hyperparams, output_dir, dry_run=args.dry_run)

    else:
        print("\n3. Loading hyperparameters...")

        # always check production directory for best params
        prod_dir = Path("outputs/models")
        previous_params = load_previous_hyperparameters(prod_dir)

        if previous_params:
            hyperparams = previous_params["hyperparams"]
            optimised_date = previous_params.get("optimised_at", "unknown")
            print(f"   Loaded previously optimised hyperparameters")
            print(f"   Optimised at: {optimised_date}")
            print("\n   Using hyperparameters:")
            for key, val in hyperparams.items():
                print(f"      {key}: {val}")
        else:
            print("   No previously optimised hyperparameters found")
            print("   Using default hyperparameters (consider running with --tune)")
            hyperparams = get_default_hyperparameters()
            print("\n   Default hyperparameters:")
            for key, val in hyperparams.items():
                print(f"      {key}: {val}")

    # ========================================================================
    # CALCULATE PRIORS
    # ========================================================================
    print("\n4. Calculating priors...")

    home_adv_prior, home_adv_std = calculate_home_advantage_prior(
        all_train_data, use_actual_goals=True, verbose=True
    )

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
        print("\nModel fitting failed")
        sys.exit(1)

    print("   Model fitted successfully")

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
        "trained_at": datetime.now(),
        "hyperparams_optimised": args.tune,
    }

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n7. Saving model...")

    model_path = output_dir / f"{args.model_name}.pkl"
    save_model(model_package, model_path)

    print(f"   Model saved: {model_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    if args.dry_run:
        print("DRY RUN - Files saved to outputs/test/")
        print("   Production model not affected")
        print("=" * 70)

    print(f"Model saved to: {model_path}")
    print(f"Teams: {len(fitted_params.get('teams', []))}")
    print(f"Training matches: {len(all_train_data)}")
    print(
        f"Hyperparameters {'optimised' if args.tune else 'loaded from previous optimisation'}"
    )

    if "log_likelihood" in fitted_params:
        print(f"Final log-likelihood: {fitted_params['log_likelihood']:.2f}")

    print(f"Rolling windows used: {args.windows}")

    if "team_params" in fitted_params:
        print("\nTop 5 teams by attack strength:")
        attack_strengths = fitted_params["team_params"]
        attack_sorted = sorted(
            attack_strengths.items(), key=lambda x: x[1]["attack"], reverse=True
        )
        for i, (team, params) in enumerate(attack_sorted[:5], 1):
            print(f"  {i}. {team}: {params['attack']:.3f}")

    if not args.dry_run:
        print("\nNext steps:")
        print(
            f"  - Calibrate: python scripts/run_calibration.py --model-path {model_path}"
        )
        print(
            f"  - Validate: python scripts/validate_model.py --model-path {model_path}"
        )
        print(
            f"  - Predict: python scripts/generate_predictions.py --model-path {model_path}"
        )


if __name__ == "__main__":
    main()
