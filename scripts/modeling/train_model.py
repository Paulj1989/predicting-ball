#!/usr/bin/env python3
"""
Train Final Model
=================

Fit production model

Usage:
    python scripts/modeling/train_model.py [--tune] [--dry-run]
    python scripts/modeling/train_model.py --tune --n-trials 50
    python scripts/modeling/train_model.py --prev-model outputs/models/production_model.pkl
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import (
    fit_poisson_model_two_stage,
    calculate_home_advantage_prior,
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

    parser.add_argument(
        "--prev-model",
        type=str,
        default=None,
        help="Path to previous season's model for seasonal prior blending",
    )

    parser.add_argument(
        "--skip-prev-season",
        action="store_true",
        help="Skip extracting previous season ratings (use squad values only)",
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
            "   Make sure prepare_bundesliga_data includes weighted goals calculation"
        )
        sys.exit(1)

    print("   Weighted goals calculated")

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
            print("   Loaded previously optimised hyperparameters")
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
    # EXTRACT PREVIOUS SEASON RATINGS
    # ========================================================================
    print("\n4. Extracting previous season ratings...")

    previous_season_params = None

    if not args.skip_prev_season and len(historic_data) > 0:
        last_complete_season = historic_data["season_end_year"].max()
        last_season_data = historic_data[
            historic_data["season_end_year"] == last_complete_season
        ]

        print(f"   Season: {last_complete_season}")
        print(f"   Matches: {len(last_season_data)}")

        # calculate home advantage from all historic data
        home_adv_prior_temp, home_adv_std_temp = calculate_home_advantage_prior(
            historic_data, use_actual_goals=True, verbose=False
        )

        # fit lightweight model on last season only
        print(f"   Fitting model on season {last_complete_season}...")
        prev_params = fit_poisson_model_two_stage(
            last_season_data,
            hyperparams,
            promoted_priors=None,
            home_adv_prior=home_adv_prior_temp,
            home_adv_std=home_adv_std_temp,
            n_random_starts=3,
            verbose=True,
        )

        if prev_params and prev_params.get("success"):
            previous_season_params = prev_params
            print(f"   ✓ Extracted ratings for {len(prev_params['teams'])} teams")
        else:
            print("   ✗ Fit failed, will use squad values only")
    else:
        if args.skip_prev_season:
            print("   Skipped (--skip-prev-season)")
        else:
            print("   No historic data available")

    # ========================================================================
    # IDENTIFY PROMOTED TEAMS
    # ========================================================================
    print("\n5. Identifying promoted teams...")

    last_historic_season = historic_data[
        historic_data["season_end_year"] == historic_data["season_end_year"].max()
    ]

    promoted_teams_info = identify_promoted_teams(last_historic_season, current_season)

    # build promoted_priors dict with is_promoted flag
    from src.models.priors import calculate_squad_value_priors

    all_teams = sorted(
        pd.unique(
            pd.concat([all_train_data, current_season])[
                ["home_team", "away_team"]
            ].values.ravel()
        )
    )

    squad_priors = calculate_squad_value_priors(
        all_train_data, all_teams, verbose=False
    )

    # create promoted_priors dict
    promoted_priors = {}
    for team in promoted_teams_info.keys():
        if team in squad_priors:
            promoted_priors[team] = {
                "attack_prior": squad_priors[team]["attack_prior"],
                "defense_prior": squad_priors[team]["defense_prior"],
                "is_promoted": True,
            }

    # add previous season params to dict
    if previous_season_params:
        promoted_priors["_previous_season_params"] = previous_season_params
        print("   Previous season params will be used for 70/30 blending")

    if promoted_priors:
        print(
            f"   Promoted teams: {len([k for k in promoted_priors if k != '_previous_season_params'])}"
        )

    # ========================================================================
    # CALCULATE HOME ADVANTAGE PRIOR
    # ========================================================================
    home_adv_prior, home_adv_std = calculate_home_advantage_prior(
        all_train_data, use_actual_goals=False, verbose=True
    )

    # ========================================================================
    # FIT MODEL
    # ========================================================================
    print("\n6. Fitting model..")

    fitted_params = fit_poisson_model_two_stage(
        all_train_data,
        hyperparams,
        promoted_priors=promoted_priors,  # contains promoted teams + prev params
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
    print("\n7. Packaging model...")

    model_package = {
        "params": fitted_params,
        "hyperparams": hyperparams,
        "promoted_teams": promoted_teams_info,
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
        "uses_seasonal_priors": previous_season_params is not None,
        "blend_weight_prev": 0.7 if previous_season_params else 0.0,
    }

    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n8. Saving model...")

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

    print(f"Model: {model_path}")
    print(f"Teams: {len(fitted_params.get('teams', []))}")
    print(f"Matches: {len(all_train_data)}")
    print(f"Hyperparams: {'Optimised' if args.tune else 'Loaded from Previous Tuning'}")
    print(f"Priors: {'70/30 Blend' if previous_season_params else 'Squad Values Only'}")

    if "log_likelihood" in fitted_params:
        print(f"Log-likelihood: {fitted_params['log_likelihood']:.2f}")

    if "team_params" in fitted_params:
        print("\nTop 5 attack:")
        attack_strengths = fitted_params["team_params"]
        attack_sorted = sorted(
            attack_strengths.items(), key=lambda x: x[1]["attack"], reverse=True
        )
        for i, (team, params) in enumerate(attack_sorted[:5], 1):
            print(f"  {i}. {team}: {params['attack']:.3f}")

    if not args.dry_run:
        print("\nNext training (with this as prior):")
        print(f"  --prev-model {model_path}")


if __name__ == "__main__":
    main()
