# src/models/hyperparameters.py


import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit


def get_default_hyperparameters() -> dict[str, float]:
    """Get default hyperparameters for the model"""
    return {
        "time_decay": 0.005,
        "lambda_reg": 0.5,
        "prior_decay_rate": 17.0,
        "xg_weight": 0.7,
    }


def optimise_hyperparameters(
    train_val_data: pd.DataFrame,
    n_trials: int = 50,
    n_jobs: int = -1,
    metric: str = "rps",
    use_two_stage: bool = True,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Optimise hyperparameters using time-series cross-validation.

    Uses Optuna with Tree-structured Parzen Estimator (TPE) sampling to
    efficiently search the hyperparameter space. Employs median pruning
    to stop unpromising trials early.

    Search space:
        - time_decay: [0.001, 0.01] (log scale)
        - lambda_reg: [0.05, 1.0] (linear scale)
        - xg_weight: [0.5, 1.0] (linear scale)

    Fixed values (not tuned):
        - prior_decay_rate: 17 matches
        - rho: fitted jointly with team ratings in stage 1 MLE
    """
    # import here to avoid circular dependency
    from ..evaluation.metrics import evaluate_model_comprehensive
    from .poisson import fit_poisson_model_two_stage

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMISATION")
        print("=" * 60)
        print(f"Running {n_trials} trials with {n_jobs} parallel jobs")
        print(f"Optimising {metric.upper()}\n")

    # suppress optuna logging if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # precompute once — train_val_data is constant across all trials
    train_full = (
        train_val_data.dropna(subset=["home_goals_weighted", "away_goals_weighted"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # oldest 30% of data for odds blend weight — held out from cv loop to prevent
    # in-sample optimism in cv scores
    holdout_size = max(30, int(0.3 * len(train_full)))
    blend_holdout = train_full.iloc[:holdout_size].copy()

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna"""
        # suggest hyperparameters (rho fitted by MLE, prior_decay_rate fixed)
        hyperparams = {
            "time_decay": trial.suggest_float("time_decay", 0.001, 0.01, log=True),
            "lambda_reg": trial.suggest_float("lambda_reg", 0.05, 1.0),
            "prior_decay_rate": 17.0,
            "xg_weight": trial.suggest_float("xg_weight", 0.5, 0.8),
        }

        # time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=153)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(train_full)):
            train_fold = train_full.iloc[train_idx]
            val_fold = train_full.iloc[val_idx]

            # n_random_starts=1 is sufficient for cv evaluation
            params = fit_poisson_model_two_stage(
                train_fold,
                hyperparams,
                n_random_starts=1,
                blend_holdout_df=blend_holdout,
                verbose=False,
            )

            if params is None or not params["success"]:
                return float("inf")

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
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)

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
        print(f"  xg_weight: {study.best_params['xg_weight']:.4f}")
        print("  prior_decay_rate: 17 matches (fixed)")
        print("  rho: fitted jointly with team ratings (see model params)")

    return {
        **study.best_params,
        "prior_decay_rate": 17.0,
    }
