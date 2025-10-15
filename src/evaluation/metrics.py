# src/evaluation/metrics.py

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Dict, Tuple, Any


def calculate_rps(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Ranked Probability Score (RPS).

    RPS evaluates ordered outcome predictions (home/draw/away).
    It measures the mean squared error of cumulative probability distributions.
    """
    n_samples = len(actuals)
    n_outcomes = predictions.shape[1]

    rps_sum = 0.0

    for i in range(n_samples):
        # cumulative predicted probabilities
        cum_pred = np.cumsum(predictions[i])

        # cumulative actual probabilities (one-hot encoded)
        actual_one_hot = np.zeros(n_outcomes)
        actual_one_hot[actuals[i]] = 1
        cum_actual = np.cumsum(actual_one_hot)

        # squared differences of cumulative distributions
        rps_sum += np.sum((cum_pred - cum_actual) ** 2)

    return rps_sum / n_samples


def calculate_brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Brier Score.

    Brier score measures the mean squared error between predicted probabilities
    and actual outcomes.
    """
    n_samples = len(actuals)
    n_outcomes = predictions.shape[1]

    # one-hot encode actuals
    one_hot_actuals = np.zeros_like(predictions)
    for i, actual in enumerate(actuals):
        one_hot_actuals[i, actual] = 1

    # mean squared error
    brier = np.mean((predictions - one_hot_actuals) ** 2)

    return brier


def calculate_log_loss(
    predictions: np.ndarray, actuals: np.ndarray, epsilon: float = 1e-15
) -> float:
    """
    Calculate Log Loss (cross-entropy loss).

    Log loss heavily penalises confident wrong predictions.
    """
    # clip predictions to avoid log(0)
    predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)

    # extract probability of actual outcome
    log_loss = -np.mean(
        [np.log(predictions_clipped[i, actuals[i]]) for i in range(len(actuals))]
    )

    return log_loss


def calculate_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Predicts the most likely outcome and calculates % correct.
    """
    predicted_outcomes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_outcomes == actuals)
    return accuracy


def evaluate_model_comprehensive(
    params: Dict[str, Any], test_data: pd.DataFrame
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Comprehensive model evaluation.

    Calculates all standard metrics on a test set.
    """
    # import here to avoid circular dependency
    from ..models.poisson import calculate_lambdas

    home_teams = test_data["home_team"].values
    away_teams = test_data["away_team"].values
    home_goals = test_data["home_goals"].astype(int).values
    away_goals = test_data["away_goals"].astype(int).values

    # get features
    home_log_odds = (
        test_data["home_log_odds"].fillna(0).values
        if "home_log_odds" in test_data
        else np.zeros(len(test_data))
    )

    home_pens_att = (
        test_data["home_pens_att"].fillna(0).values
        if "home_pens_att" in test_data
        else np.zeros(len(test_data))
    )
    away_pens_att = (
        test_data["away_pens_att"].fillna(0).values
        if "away_pens_att" in test_data
        else np.zeros(len(test_data))
    )

    predictions = []
    actuals = []

    for i in range(len(test_data)):
        # get team parameters
        att_h = params["attack"].get(home_teams[i], 0)
        def_h = params["defense"].get(home_teams[i], 0)
        att_a = params["attack"].get(away_teams[i], 0)
        def_a = params["defense"].get(away_teams[i], 0)

        # calculate expected goals
        lambda_h = np.exp(
            att_h
            + def_a
            + params["home_adv"]
            + params["beta_odds"] * home_log_odds[i]
            + params.get("beta_penalty", 0.0) * home_pens_att[i]
        )
        lambda_a = np.exp(
            att_a
            + def_h
            - params["beta_odds"] * home_log_odds[i]
            + params.get("beta_penalty", 0.0) * away_pens_att[i]
        )

        lambda_h = np.clip(lambda_h, 0.1, 10.0)
        lambda_a = np.clip(lambda_a, 0.1, 10.0)

        # calculate outcome probabilities
        home_win_prob = draw_prob = away_win_prob = 0

        for h in range(8):
            for a in range(8):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                if h > a:
                    home_win_prob += p
                elif h == a:
                    draw_prob += p
                else:
                    away_win_prob += p

        total = home_win_prob + draw_prob + away_win_prob
        predictions.append(
            [home_win_prob / total, draw_prob / total, away_win_prob / total]
        )

        # actual outcome
        if home_goals[i] > away_goals[i]:
            actuals.append(0)
        elif home_goals[i] == away_goals[i]:
            actuals.append(1)
        else:
            actuals.append(2)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # calculate all metrics
    metrics = {
        "rps": calculate_rps(predictions, actuals),
        "brier_score": calculate_brier_score(predictions, actuals),
        "log_loss": calculate_log_loss(predictions, actuals),
        "accuracy": calculate_accuracy(predictions, actuals),
    }

    return metrics, predictions, actuals


def compare_metrics(
    metrics_dict: Dict[str, Dict[str, float]], reference_model: str = "implied_odds"
) -> pd.DataFrame:
    """Compare multiple models across metrics"""
    rows = []

    for model_name, metrics in metrics_dict.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # calculate improvements vs reference
    if reference_model in metrics_dict:
        ref_metrics = metrics_dict[reference_model]

        for metric in ["rps", "brier_score", "log_loss"]:
            if metric in df.columns:
                df[f"{metric}_vs_{reference_model}"] = (
                    (ref_metrics[metric] - df[metric]) / ref_metrics[metric] * 100
                )

    return df


def calculate_metric_confidence_interval(
    predictions: np.ndarray,
    actuals: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Calculate confidence interval for a metric via bootstrap"""
    point_estimate = metric_func(predictions, actuals)

    bootstrap_metrics = []
    n_samples = len(actuals)

    for _ in range(n_bootstrap):
        # resample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_metric = metric_func(predictions[idx], actuals[idx])
        bootstrap_metrics.append(boot_metric)

    # calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)

    return point_estimate, lower, upper
