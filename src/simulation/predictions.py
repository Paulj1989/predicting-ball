# src/simulation/predictions.py

from typing import Any

import pandas as pd
from scipy.stats import poisson

from ..models.dixon_coles import calculate_match_probabilities_dixon_coles
from ..models.poisson import calculate_lambdas_single


def get_next_round_fixtures(
    current_season: pd.DataFrame,
    full_matchday_size: int = 9,
    rearranged_threshold: int = 5,
) -> pd.DataFrame | None:
    """Get next round of unplayed fixtures, including rearranged games.

    Uses the matchweek column to identify the upcoming round and any
    rearranged games from earlier matchweeks.

    Logic:
    1. Find the next matchweek (lowest matchweek with a full set of unplayed games)
    2. Get all games from that matchweek
    3. Find any rearranged games (earlier matchweeks) occurring before/during it
    4. If rearranged games > threshold, show only rearranged games as their own round
    5. Otherwise, combine rearranged games with the main matchweek
    """
    # filter to unplayed fixtures
    if "is_played" in current_season.columns:
        future_fixtures = current_season[not current_season["is_played"]].copy()
    else:
        # fall back to checking for null goals
        future_fixtures = current_season[current_season["home_goals"].isna()].copy()

    if len(future_fixtures) == 0:
        return None

    future_fixtures["date"] = pd.to_datetime(future_fixtures["date"])
    future_fixtures = future_fixtures.sort_values(["date", "matchweek"])

    # check if matchweek column exists
    if "matchweek" not in future_fixtures.columns:
        # fall back to date-based logic
        return _get_next_fixtures_by_date(future_fixtures, full_matchday_size)

    # count games per matchweek
    matchweek_counts = future_fixtures.groupby("matchweek").size()

    # find the next full matchweek (lowest matchweek with full_matchday_size games)
    full_matchweeks = matchweek_counts[matchweek_counts >= full_matchday_size]

    if len(full_matchweeks) == 0:
        # no full matchweeks remaining, return all remaining fixtures
        return future_fixtures.sort_values(["date", "matchweek"]).reset_index(drop=True)

    next_matchweek = full_matchweeks.index.min()

    # get fixtures for the next matchweek
    matchweek_fixtures = future_fixtures[future_fixtures["matchweek"] == next_matchweek]
    matchweek_end_date = matchweek_fixtures["date"].max()

    # find rearranged games: earlier matchweeks occurring up to the end of next matchweek
    rearranged = future_fixtures[
        (future_fixtures["matchweek"] < next_matchweek)
        & (future_fixtures["date"] <= matchweek_end_date)
    ]

    # if many rearranged games, show them as their own round
    if len(rearranged) > rearranged_threshold:
        return rearranged.sort_values(["date", "matchweek"]).reset_index(drop=True)

    # combine rearranged games with matchweek fixtures
    combined = pd.concat([rearranged, matchweek_fixtures], ignore_index=True)
    return combined.sort_values(["date", "matchweek"]).reset_index(drop=True)


def _get_next_fixtures_by_date(
    future_fixtures: pd.DataFrame,
    min_matchday_size: int,
) -> pd.DataFrame | None:
    """Fallback: get next fixtures by date when matchweek is unavailable."""
    # group by date and count fixtures per date
    date_counts = future_fixtures.groupby(future_fixtures["date"].dt.date).size()

    # find the first date that looks like a real matchday
    matchday_date = None
    for date, count in date_counts.items():
        if count >= min_matchday_size:
            matchday_date = date
            break

    # if no date has enough fixtures, fall back to earliest date
    if matchday_date is None:
        matchday_date = date_counts.index[0]

    return future_fixtures[
        future_fixtures["date"].dt.date == matchday_date
    ].reset_index(drop=True)


def get_all_future_fixtures(
    current_season: pd.DataFrame,
) -> pd.DataFrame | None:
    """Get all unplayed fixtures for the current season"""
    future_fixtures = current_season[not current_season["is_played"]].copy()

    if len(future_fixtures) == 0:
        return None

    future_fixtures["date"] = pd.to_datetime(future_fixtures["date"])
    future_fixtures = future_fixtures.sort_values("date")

    return future_fixtures.reset_index(drop=True)


def predict_single_match(
    home_team: str,
    away_team: str,
    params: dict[str, Any],
    home_log_odds_ratio: float = 0.0,
    home_npxgd_w5: float = 0.0,
    away_npxgd_w5: float = 0.0,
    max_goals: int = 8,
    use_dixon_coles: bool = True,
) -> dict[str, Any]:
    """Generate predictions for a single match"""

    lambda_h, lambda_a = calculate_lambdas_single(
        home_team=home_team,
        away_team=away_team,
        params=params,
        home_log_odds_ratio=home_log_odds_ratio,
        home_npxgd_w5=home_npxgd_w5,
        away_npxgd_w5=away_npxgd_w5,
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
    params: dict[str, Any],
    match_data: pd.Series,
    max_goals: int = 8,
    use_dixon_coles: bool = True,
) -> dict[str, float]:
    """Predict outcome probabilities for a single match"""
    # extract match information
    home_team = match_data["home_team"]
    away_team = match_data["away_team"]

    # get features (with safe defaults)
    home_log_odds_ratio = match_data.get("home_log_odds_ratio", 0.0)
    if pd.isna(home_log_odds_ratio):
        home_log_odds_ratio = 0.0

    home_npxgd_w5 = match_data.get("home_npxgd_w5", 0.0)
    if pd.isna(home_npxgd_w5):
        home_npxgd_w5 = 0.0

    away_npxgd_w5 = match_data.get("away_npxgd_w5", 0.0)
    if pd.isna(away_npxgd_w5):
        away_npxgd_w5 = 0.0

    # call the existing prediction function
    prediction = predict_single_match(
        home_team=home_team,
        away_team=away_team,
        params=params,
        home_log_odds_ratio=home_log_odds_ratio,
        home_npxgd_w5=home_npxgd_w5,
        away_npxgd_w5=away_npxgd_w5,
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
    params: dict[str, Any],
    calibrators: Any | None = None,
    use_dixon_coles: bool = True,
) -> pd.DataFrame | None:
    """Generate predictions for multiple fixtures"""
    if fixtures is None or len(fixtures) == 0:
        return None

    predictions = []

    for _idx, match in fixtures.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]

        # get odds if available
        home_log_odds_ratio = match.get("home_log_odds_ratio", 0)
        if pd.isna(home_log_odds_ratio):
            home_log_odds_ratio = 0

        home_npxgd_w5 = match.get("home_npxgd_w5", 0.0)
        if pd.isna(home_npxgd_w5):
            home_npxgd_w5 = 0.0

        away_npxgd_w5 = match.get("away_npxgd_w5", 0.0)
        if pd.isna(away_npxgd_w5):
            away_npxgd_w5 = 0.0

        # generate prediction
        pred = predict_single_match(
            home_team,
            away_team,
            params,
            home_log_odds_ratio=home_log_odds_ratio,
            home_npxgd_w5=home_npxgd_w5,
            away_npxgd_w5=away_npxgd_w5,
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
