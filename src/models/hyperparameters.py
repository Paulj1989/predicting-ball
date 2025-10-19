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
        "lambda_reg": 0.3,
        "prior_decay_rate": 15.0,
    }


def optimise_hyperparameters(
    train_val_data: pd.DataFrame,
    n_trials: int = 30,
    n_jobs: int = -1,
    metric: str = "rps",
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Optimise hyperparameters using time-series cross-validation.

    Uses Optuna with Tree-structured Parzen Estimator (TPE) sampling to
    efficiently search the hyperparameter space. Employs median pruning
    to stop unpromising trials early.

    Search space:
        - time_decay: [0.0003, 0.001] (log scale)
        - lambda_reg: [0.05, 0.8] (linear scale)
        - prior_decay_rate: [5.0, 25.0] (linear scale)
    """
    # import here to avoid circular dependency
    from .poisson import fit_poisson_model
    # from ..evaluation.metrics import calculate_rps

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMISATION")
        print("=" * 60)
        print(f"Running {n_trials} trials with {n_jobs} parallel jobs")
        print(f"Optimising {metric.upper()}\n")

    # suppress optuna logging if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna"""
        # suggest hyperparameters
        hyperparams = {
            "time_decay": trial.suggest_float("time_decay", 0.0003, 0.001, log=True),
            "lambda_reg": trial.suggest_float("lambda_reg", 0.05, 0.8),
            "prior_decay_rate": trial.suggest_float("prior_decay_rate", 5.0, 25.0),
        }

        # prepare data
        train_full = train_val_data.dropna(
            subset=["home_goals_weighted", "away_goals_weighted"]
        )
        train_full = train_full.sort_values("date").reset_index(drop=True)

        # time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=10, test_size=108)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(train_full)):
            train_fold = train_full.iloc[train_idx]
            val_fold = train_full.iloc[val_idx]

            # fit model
            params = fit_poisson_model(train_fold, hyperparams, verbose=False)

            if params is None or not params["success"]:
                return float("inf")

            # evaluate on validation fold
            # import evaluation function
            from ..evaluation.metrics import evaluate_model_comprehensive

            metrics_dict, _, _ = evaluate_model_comprehensive(params, val_fold)
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

    return study.best_params
