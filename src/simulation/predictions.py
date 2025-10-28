# src/simulation/predictions.py

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Optional, Dict, Any

from ..models.poisson import calculate_lambdas, calculate_lambdas_single
from ..models.dixon_coles import calculate_match_probabilities_dixon_coles


def get_next_round_fixtures(
    current_season: pd.DataFrame, matchday_window_days: int = 4
) -> Optional[pd.DataFrame]:
    """Get next round of unplayed fixtures"""
    future_fixtures = current_season[current_season["is_played"] == False].copy()

    if len(future_fixtures) == 0:
        return None

    future_fixtures["date"] = pd.to_datetime(future_fixtures["date"])
    earliest_date = future_fixtures["date"].min()
    matchday_window = pd.Timedelta(days=matchday_window_days)

    next_fixtures = future_fixtures[
        future_fixtures["date"] <= (earliest_date + matchday_window)
    ]

    return next_fixtures.sort_values("date")


def predict_single_match(
    home_team: str,
    away_team: str,
    params: Dict[str, Any],
    home_log_odds_ratio: float = 0.0,
    home_form_w5: float = 0.0,
    away_form_w5: float = 0.0,
    max_goals: int = 8,
    use_dixon_coles: bool = True,
) -> Dict[str, Any]:
    """Generate predictions for a single match"""

    lambda_h, lambda_a = calculate_lambdas_single(
        home_team=home_team,
        away_team=away_team,
        params=params,
        home_log_odds_ratio=home_log_odds_ratio,
        home_form_w5=home_form_w5,
        away_form_w5=away_form_w5,
    )

    # get rho parameter (default to -0.13 if not fitted)
    rho = params.get("rho", -0.13)

    if use_dixon_coles:
        # use dixon-coles corrected probabilities
        home_win_prob, draw_prob, away_win_prob, score_probs = (
            calculate_match_probabilities_dixon_coles(
                lambda_h, lambda_a, rho=rho, max_goals=max_goals
            )
        )
    else:
        # use independent poisson
        home_win_prob = draw_prob = away_win_prob = 0
        score_probs = {}

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                score_probs[(h, a)] = p

                if h > a:
                    home_win_prob += p
                elif h == a:
                    draw_prob += p
                else:
                    away_win_prob += p

        # normalise probabilities
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total

    # get most likely score
    most_likely = max(score_probs.items(), key=lambda x: x[1])
    most_likely_score = f"{most_likely[0][0]}-{most_likely[0][1]}"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "expected_goals_home": lambda_h,
        "expected_goals_away": lambda_a,
        "home_win": home_win_prob,
        "draw": draw_prob,
        "away_win": away_win_prob,
        "most_likely_score": most_likely_score,
        "score_probabilities": score_probs,
    }


def predict_match_probabilities(
    params: Dict[str, Any],
    match_data: pd.Series,
    max_goals: int = 8,
    use_dixon_coles: bool = True,
) -> Dict[str, float]:
    """Predict outcome probabilities for a single match"""
    # extract match information
    home_team = match_data["home_team"]
    away_team = match_data["away_team"]

    # get features (with safe defaults)
    home_log_odds_ratio = match_data.get("home_log_odds_ratio", 0.0)
    if pd.isna(home_log_odds_ratio):
        home_log_odds_ratio = 0.0

    home_form_w5 = match_data.get("home_form_w5", 0.0)
    if pd.isna(home_form_w5):
        home_form_w5 = 0.0

    away_form_w5 = match_data.get("away_form_w5", 0.0)
    if pd.isna(away_form_w5):
        away_form_w5 = 0.0

    # call the existing prediction function
    prediction = predict_single_match(
        home_team=home_team,
        away_team=away_team,
        params=params,
        home_log_odds_ratio=home_log_odds_ratio,
        home_form_w5=home_form_w5,
        away_form_w5=away_form_w5,
        max_goals=max_goals,
        use_dixon_coles=use_dixon_coles,
    )

    # return just the probability fields
    return {
        "home_win": prediction["home_win"],
        "draw": prediction["draw"],
        "away_win": prediction["away_win"],
    }


def predict_next_fixtures(
    fixtures: pd.DataFrame,
    params: Dict[str, Any],
    calibrators: Optional[Any] = None,
    use_dixon_coles: bool = True,
) -> Optional[pd.DataFrame]:
    """Generate predictions for multiple fixtures"""
    if fixtures is None or len(fixtures) == 0:
        return None

    predictions = []

    for idx, match in fixtures.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        # get odds if available
        home_log_odds_ratio = match.get("home_log_odds_ratio", 0)
        if pd.isna(home_log_odds_ratio):
            home_log_odds_ratio = 0

        home_form_w5 = match.get("home_form_w5", 0.0)
        if pd.isna(home_form_w5):
            home_form_w5 = 0.0

        away_form_w5 = match.get("away_form_w5", 0.0)
        if pd.isna(away_form_w5):
            away_form_w5 = 0.0

        # generate prediction
        pred = predict_single_match(
            home_team,
            away_team,
            params,
            home_log_odds_ratio=home_log_odds_ratio,
            home_form_w5=home_form_w5,
            away_form_w5=away_form_w5,
            use_dixon_coles=use_dixon_coles,
        )

        predictions.append(pred)

    # create dataframe (exclude score_probabilities for cleaner output)
    predictions_df = pd.DataFrame(
        [
            {k: v for k, v in p.items() if k != "score_probabilities"}
            for p in predictions
        ]
    )

    # apply calibration if provided
    if calibrators is not None:
        from ..models.calibration import apply_calibration

        # extract probability columns
        prob_cols = ["home_win", "draw", "away_win"]
        if all(col in predictions_df.columns for col in prob_cols):
            probs = predictions_df[prob_cols].values

            # apply calibration (handles both standard and outcome-specific)
            calibrated_probs = apply_calibration(probs, calibrators)

            # update dataframe with calibrated probabilities
            predictions_df["home_win"] = calibrated_probs[:, 0]
            predictions_df["draw"] = calibrated_probs[:, 1]
            predictions_df["away_win"] = calibrated_probs[:, 2]

    return predictions_df
