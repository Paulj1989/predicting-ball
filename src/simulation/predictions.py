# src/simulation/predictions.py

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Optional, Dict, Any

from ..models.poisson import calculate_lambdas


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
    home_log_odds: float = 0.0,
    home_pens_att: float = 0.0,
    away_pens_att: float = 0.0,
    max_goals: int = 8,
) -> Dict[str, Any]:
    """Generate predictions for a single match"""
    # get team parameters
    att_h = params["attack"].get(home_team, 0)
    def_h = params["defense"].get(home_team, 0)
    att_a = params["attack"].get(away_team, 0)
    def_a = params["defense"].get(away_team, 0)

    # calculate expected goals
    lambda_h = np.exp(
        att_h
        + def_a
        + params["home_adv"]
        + params["beta_odds"] * home_log_odds
        + params.get("beta_penalty", 0.0) * home_pens_att
    )
    lambda_a = np.exp(
        att_a
        + def_h
        - params["beta_odds"] * home_log_odds
        + params.get("beta_penalty", 0.0) * away_pens_att
    )

    lambda_h = np.clip(lambda_h, 0.1, 10.0)
    lambda_a = np.clip(lambda_a, 0.1, 10.0)

    # calculate outcome probabilities and score distribution
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

    # get most likely score
    most_likely = max(score_probs.items(), key=lambda x: x[1])
    most_likely_score = f"{most_likely[0][0]}-{most_likely[0][1]}"

    # normalise probabilities
    total = home_win_prob + draw_prob + away_win_prob

    return {
        "home_team": home_team,
        "away_team": away_team,
        "expected_goals_home": lambda_h,
        "expected_goals_away": lambda_a,
        "home_win": home_win_prob / total,
        "draw": draw_prob / total,
        "away_win": away_win_prob / total,
        "most_likely_score": most_likely_score,
        "score_probabilities": score_probs,
    }


def predict_next_fixtures(
    fixtures: pd.DataFrame, params: Dict[str, Any], calibrators: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Generate predictions for multiple fixtures.

    Applies temperature scaling if calibrators are provided and contain
    a temperature parameter.
    """
    if fixtures is None or len(fixtures) == 0:
        return None

    predictions = []

    for idx, match in fixtures.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        # get odds if available
        home_log_odds = match.get("home_log_odds", 0)
        if pd.isna(home_log_odds):
            home_log_odds = 0

        # get penalty data if available
        home_pens_att = match.get("home_pens_att", 0)
        away_pens_att = match.get("away_pens_att", 0)
        if pd.isna(home_pens_att):
            home_pens_att = 0
        if pd.isna(away_pens_att):
            away_pens_att = 0

        # generate prediction
        pred = predict_single_match(
            home_team,
            away_team,
            params,
            home_log_odds=home_log_odds,
            home_pens_att=home_pens_att,
            away_pens_att=away_pens_att,
        )

        # apply temperature scaling if available
        if calibrators is not None and "temperature" in calibrators:
            from ..models.calibration import apply_temperature_scaling

            temperature = calibrators["temperature"]
            probs = np.array([[pred["home_win"], pred["draw"], pred["away_win"]]])
            calibrated = apply_temperature_scaling(probs, temperature)

            pred["home_win"] = calibrated[0, 0]
            pred["draw"] = calibrated[0, 1]
            pred["away_win"] = calibrated[0, 2]

        predictions.append(pred)

    # create dataframe (exclude score_probabilities for cleaner output)
    predictions_df = pd.DataFrame(
        [
            {k: v for k, v in p.items() if k != "score_probabilities"}
            for p in predictions
        ]
    )

    return predictions_df
