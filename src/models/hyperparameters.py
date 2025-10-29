# src/models/hyperparameters.py

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict


def get_default_hyperparameters() -> Dict[str, float]:
    """Get default hyperparameters for the model"""
    return {
        "time_decay": 0.001,
        "lambda_reg": 0.5,
        "prior_decay_rate": 15.0,
        "rho": -0.13,
    }


def optimise_hyperparameters(
    train_val_data: pd.DataFrame,
    n_trials: int = 30,
    n_jobs: int = -1,
    metric: str = "rps",
    use_two_stage: bool = True,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Optimise hyperparameters using time-series cross-validation.

    Uses Optuna with Tree-structured Parzen Estimator (TPE) sampling to
    efficiently search the hyperparameter space. Employs median pruning
    to stop unpromising trials early.

    Search space:
        - time_decay: [0.0005, 0.002] (log scale)
        - lambda_reg: [0.2, 0.8] (linear scale)
        - prior_decay_rate: [5.0, 17.0] (linear scale)
        - rho: [-0.3, -0.1] (linear scale) - Dixon-Coles correlation
    """
    # import here to avoid circular dependency
    from .poisson import fit_poisson_model_two_stage

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMISATION")
        print("=" * 60)
        print(f"Running {n_trials} trials with {n_jobs} parallel jobs")
        print(f"Optimising {metric.upper()}")
        print(f"Fitting method: {'two-stage' if use_two_stage else 'joint'}")
        print("Includes Dixon-Coles rho parameter\n")

    # suppress optuna logging if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna"""
        # suggest hyperparameters
        hyperparams = {
            "time_decay": trial.suggest_float("time_decay", 0.0005, 0.002, log=True),
            "lambda_reg": trial.suggest_float("lambda_reg", 0.2, 0.8),
            "prior_decay_rate": trial.suggest_float("prior_decay_rate", 5.0, 17.0),
            "rho": trial.suggest_float("rho", -0.3, -0.1),
        }

        # prepare data
        train_full = train_val_data.dropna(
            subset=["home_goals_weighted", "away_goals_weighted"]
        )
        train_full = train_full.sort_values("date").reset_index(drop=True)

        # time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=306)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(train_full)):
            train_fold = train_full.iloc[train_idx]
            val_fold = train_full.iloc[val_idx]

            params = fit_poisson_model_two_stage(train_fold, hyperparams, verbose=False)

            if params is None or not params["success"]:
                return float("inf")

            # evaluate on validation fold
            from ..evaluation.metrics import evaluate_model_comprehensive

            metrics_dict, _, _ = evaluate_model_comprehensive(
                params, val_fold, use_dixon_coles=True
            )
            cv_scores.append(metrics_dict[metric])

            # report intermediate value for pruning
            trial.report(metrics_dict[metric], fold_idx)

            # check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    # create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=15),
    )

    # optimise
    study.optimize(
        objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose
    )

    if verbose:
        print("\n" + "=" * 60)
        print("OPTIMISATION COMPLETE")
        print("=" * 60)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best CV {metric.upper()}: {study.best_value:.4f}")

        half_life_years = np.log(2) / study.best_params["time_decay"] / 365.25
        print("\nOptimal hyperparameters:")
        print(
            f"  time_decay: {study.best_params['time_decay']:.4f} ({half_life_years:.1f} year half-life)"
        )
        print(f"  lambda_reg: {study.best_params['lambda_reg']:.4f}")
        print(
            f"  prior_decay_rate: {study.best_params['prior_decay_rate']:.2f} matches"
        )
        print(f"  rho (Dixon-Coles): {study.best_params['rho']:.4f}")

        # interpret rho
        if study.best_params["rho"] < -0.18:
            print("    -> Strong draw correction (more draws predicted)")
        elif study.best_params["rho"] > -0.08:
            print("    -> Weak draw correction")
        else:
            print("    -> Typical Bundesliga draw correction")

    return study.best_params
