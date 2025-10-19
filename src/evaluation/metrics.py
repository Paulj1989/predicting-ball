# src/evaluation/metrics.py

import numpy as np
import pandas as pd
from typing import Union, Dict


def calculate_rps(
    predictions: Union[pd.DataFrame, np.ndarray], actuals: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate Ranked Probability Score (RPS).

    Normalised for 3 outcomes by dividing by (K-1) = 2.
    """
    # convert inputs to standard format
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame predictions must have columns {required_cols}")
        predictions = predictions.reset_index(drop=True)
        ordered_predictions = predictions[required_cols].values
    elif isinstance(predictions, np.ndarray):
        if predictions.ndim != 2 or predictions.shape[1] != 3:
            raise ValueError(
                f"Array predictions must be shape (n, 3), got {predictions.shape}"
            )
        ordered_predictions = predictions
    else:
        raise TypeError(
            f"predictions must be DataFrame or ndarray, got {type(predictions)}"
        )

    # convert actuals to standard format
    if isinstance(actuals, pd.Series):
        actuals = actuals.reset_index(drop=True).values
    elif not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # verify lengths match
    if len(ordered_predictions) != len(actuals):
        raise ValueError(
            f"Length mismatch: {len(ordered_predictions)} predictions vs {len(actuals)} actuals"
        )

    # handle both string outcomes ('H', 'D', 'A') and integer outcomes (0, 1, 2)
    if actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        try:
            actuals_idx = np.array([outcome_map[a] for a in actuals])
        except KeyError as e:
            raise ValueError(f"Unknown outcome in actuals: {e}")
    else:
        actuals_idx = actuals.astype(int)
        if not all(a in [0, 1, 2] for a in actuals_idx):
            raise ValueError("Integer actuals must be 0 (H), 1 (D), or 2 (A)")

    n_outcomes = ordered_predictions.shape[1]
    rps_scores = []

    for i in range(len(ordered_predictions)):
        # cumulative predicted probabilities
        pred_probs = ordered_predictions[i]
        cum_pred = np.cumsum(pred_probs)

        # cumulative actual probabilities (one-hot encoded then cumulative)
        actual_idx = actuals_idx[i]
        actual_one_hot = np.zeros(n_outcomes)
        actual_one_hot[actual_idx] = 1.0
        cum_actual = np.cumsum(actual_one_hot)

        # rps: sum of squared differences, normalised by (K-1)
        rps = np.sum((cum_pred - cum_actual) ** 2) / (n_outcomes - 1)
        rps_scores.append(rps)

    return float(np.mean(rps_scores))


def calculate_brier_score(
    predictions: Union[pd.DataFrame, np.ndarray], actuals: Union[pd.Series, np.ndarray]
) -> float:
    """Calculate Brier Score (mean squared error of probabilities)"""
    # convert inputs to standard format
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame predictions must have columns {required_cols}")
        predictions = predictions.reset_index(drop=True)
        pred_array = predictions[required_cols].values
    elif isinstance(predictions, np.ndarray):
        if predictions.ndim != 2 or predictions.shape[1] != 3:
            raise ValueError(
                f"Array predictions must be shape (n, 3), got {predictions.shape}"
            )
        pred_array = predictions
    else:
        raise TypeError(
            f"predictions must be DataFrame or ndarray, got {type(predictions)}"
        )

    # convert actuals to standard format
    if isinstance(actuals, pd.Series):
        actuals = actuals.reset_index(drop=True).values
    elif not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # verify lengths match
    if len(pred_array) != len(actuals):
        raise ValueError(
            f"Length mismatch: {len(pred_array)} predictions vs {len(actuals)} actuals"
        )

    # handle both string and integer outcomes
    if actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        try:
            actuals_idx = np.array([outcome_map[a] for a in actuals])
        except KeyError as e:
            raise ValueError(f"Unknown outcome in actuals: {e}")
    else:
        actuals_idx = actuals.astype(int)
        if not all(a in [0, 1, 2] for a in actuals_idx):
            raise ValueError("Integer actuals must be 0 (H), 1 (D), or 2 (A)")

    # calculate brier score
    brier_scores = []

    for i in range(len(pred_array)):
        pred_probs = pred_array[i]
        actual_idx = actuals_idx[i]
        actual_one_hot = np.array([1.0 if j == actual_idx else 0.0 for j in range(3)])

        # brier score is sum of squared errors, divided by two
        # following Kruppa et al (2014))
        brier = np.sum((pred_probs - actual_one_hot) ** 2) / 2
        brier_scores.append(brier)

    return float(np.mean(brier_scores))


def calculate_log_loss(
    predictions: Union[pd.DataFrame, np.ndarray],
    actuals: Union[pd.Series, np.ndarray],
    eps: float = 1e-15,
) -> float:
    """Calculate logarithmic loss (cross-entropy)"""
    # convert inputs to standard format
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame predictions must have columns {required_cols}")
        predictions = predictions.reset_index(drop=True)
        pred_array = predictions[required_cols].values
    elif isinstance(predictions, np.ndarray):
        if predictions.ndim != 2 or predictions.shape[1] != 3:
            raise ValueError(
                f"Array predictions must be shape (n, 3), got {predictions.shape}"
            )
        pred_array = predictions
    else:
        raise TypeError(
            f"predictions must be DataFrame or ndarray, got {type(predictions)}"
        )

    # convert actuals to standard format
    if isinstance(actuals, pd.Series):
        actuals = actuals.reset_index(drop=True).values
    elif not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # verify lengths match
    if len(pred_array) != len(actuals):
        raise ValueError(
            f"Length mismatch: {len(pred_array)} predictions vs {len(actuals)} actuals"
        )

    # clip probabilities to avoid log(0)
    pred_array = np.clip(pred_array, eps, 1 - eps)

    # handle both string and integer outcomes
    if actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        try:
            actuals_idx = np.array([outcome_map[a] for a in actuals])
        except KeyError as e:
            raise ValueError(f"Unknown outcome in actuals: {e}")
    else:
        actuals_idx = actuals.astype(int)
        if not all(a in [0, 1, 2] for a in actuals_idx):
            raise ValueError("Integer actuals must be 0 (H), 1 (D), or 2 (A)")

    # calculate log loss
    log_losses = []

    for i in range(len(pred_array)):
        actual_idx = actuals_idx[i]
        log_loss = -np.log(pred_array[i, actual_idx])
        log_losses.append(log_loss)

    return float(np.mean(log_losses))


def calculate_accuracy(
    predictions: Union[pd.DataFrame, np.ndarray], actuals: Union[pd.Series, np.ndarray]
) -> float:
    """Calculate classification accuracy"""
    # convert inputs to standard format
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame predictions must have columns {required_cols}")
        predictions = predictions.reset_index(drop=True)
        pred_array = predictions[required_cols].values
    elif isinstance(predictions, np.ndarray):
        if predictions.ndim != 2 or predictions.shape[1] != 3:
            raise ValueError(
                f"Array predictions must be shape (n, 3), got {predictions.shape}"
            )
        pred_array = predictions
    else:
        raise TypeError(
            f"predictions must be DataFrame or ndarray, got {type(predictions)}"
        )

    # convert actuals to standard format
    if isinstance(actuals, pd.Series):
        actuals = actuals.reset_index(drop=True).values
    elif not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # verify lengths match
    if len(pred_array) != len(actuals):
        raise ValueError(
            f"Length mismatch: {len(pred_array)} predictions vs {len(actuals)} actuals"
        )

    # get predicted class (highest probability)
    pred_classes = pred_array.argmax(axis=1)

    # handle both string and integer outcomes
    if actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        try:
            actuals_idx = np.array([outcome_map[a] for a in actuals])
        except KeyError as e:
            raise ValueError(f"Unknown outcome in actuals: {e}")
    else:
        actuals_idx = actuals.astype(int)
        if not all(a in [0, 1, 2] for a in actuals_idx):
            raise ValueError("Integer actuals must be 0 (H), 1 (D), or 2 (A)")

    # calculate accuracy
    correct = (pred_classes == actuals_idx).sum()
    accuracy = correct / len(actuals_idx)

    return float(accuracy)


def evaluate_model_comprehensive(params: Dict, test_data: pd.DataFrame) -> tuple:
    """Comprehensive model evaluation"""
    from ..simulation.predictions import predict_match_probabilities

    # generate predictions for all test matches
    predictions_list = []

    for idx, row in test_data.iterrows():
        probs = predict_match_probabilities(params, row)
        predictions_list.append(probs)

    predictions = pd.DataFrame(predictions_list)
    actuals = test_data["result"].reset_index(drop=True)

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
) -> tuple[float, float, float]:
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
