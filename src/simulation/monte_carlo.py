# src/simulation/monte_carlo.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List

from .sampling import sample_goals_calibrated
from ..models.poisson import calculate_lambdas


def get_current_standings(played_matches: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Calculate current league standings from played matches"""
    if len(played_matches) == 0:
        return {}

    teams = set(played_matches["home_team"].unique()) | set(
        played_matches["away_team"].unique()
    )

    standings = {}

    for team in teams:
        # home games
        home_games = played_matches[played_matches["home_team"] == team]
        home_points = home_gf = home_ga = 0

        for _, game in home_games.iterrows():
            home_gf += game["home_goals"]
            home_ga += game["away_goals"]
            if game["home_goals"] > game["away_goals"]:
                home_points += 3
            elif game["home_goals"] == game["away_goals"]:
                home_points += 1

        # away games
        away_games = played_matches[played_matches["away_team"] == team]
        away_points = away_gf = away_ga = 0

        for _, game in away_games.iterrows():
            away_gf += game["away_goals"]
            away_ga += game["home_goals"]
            if game["away_goals"] > game["home_goals"]:
                away_points += 3
            elif game["away_goals"] == game["home_goals"]:
                away_points += 1

        standings[team] = {
            "points": home_points + away_points,
            "goals_for": home_gf + away_gf,
            "goals_against": home_ga + away_ga,
            "goal_diff": (home_gf + away_gf) - (home_ga + away_ga),
            "games_played": len(home_games) + len(away_games),
        }

    return standings


def simulate_remaining_season_calibrated(
    future_fixtures: pd.DataFrame,
    bootstrap_params: List[Dict[str, Any]],
    current_standings: Dict[str, Dict[str, int]],
    n_simulations: int = 100000,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Simulate remaining fixtures using calibrated goal distributions.

    Uses bootstrap parameter samples to account for parameter uncertainty,
    and calibrated sampling to account for goal variance.
    """
    if seed is not None:
        np.random.seed(seed)

    if not bootstrap_params:
        return None, None

    # get dispersion factor
    dispersion_factor = bootstrap_params[0].get("dispersion_factor", 1.0)

    print(f"\nSimulating with dispersion factor: {dispersion_factor:.3f}")
    if dispersion_factor > 1.2:
        print("Using negative binomial distribution (overdispersed)")
    else:
        print("Using Poisson distribution")

    # setup teams
    teams_in_standings = set(current_standings.keys())
    teams_in_fixtures = set(future_fixtures["home_team"].unique()) | set(
        future_fixtures["away_team"].unique()
    )
    all_teams = sorted(teams_in_standings | teams_in_fixtures)

    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    # initialise with current standings
    base_points = np.zeros(n_teams)
    base_gf = np.zeros(n_teams)
    base_ga = np.zeros(n_teams)

    for team, stats in current_standings.items():
        idx = team_to_idx[team]
        base_points[idx] = stats["points"]
        base_gf[idx] = stats["goals_for"]
        base_ga[idx] = stats["goals_against"]

    if len(future_fixtures) > 0:
        # extract match info
        home_idx = np.array(
            [team_to_idx[t] for t in future_fixtures["home_team"]], dtype=int
        )
        away_idx = np.array(
            [team_to_idx[t] for t in future_fixtures["away_team"]], dtype=int
        )

        n_matches = len(future_fixtures)

        # initialise results storage
        results = {
            "points": np.zeros((n_simulations, n_teams)),
            "goals_for": np.zeros((n_simulations, n_teams)),
            "goals_against": np.zeros((n_simulations, n_teams)),
            "positions": np.zeros((n_simulations, n_teams), dtype=int),
        }

        # simulation loop
        for s in range(n_simulations):
            if s % 1000 == 0:
                print(f"\rSimulation {s}/{n_simulations}", end="")

            # sample parameters from bootstrap
            params = bootstrap_params[np.random.randint(len(bootstrap_params))]

            # create temporary df for vectorised calculation
            temp_df = future_fixtures[["home_team", "away_team"]].copy()
            if "home_log_odds" in future_fixtures.columns:
                temp_df["home_log_odds"] = future_fixtures["home_log_odds"].fillna(0)
            else:
                temp_df["home_log_odds"] = 0.0

            lambda_home, lambda_away = calculate_lambdas(temp_df, params)

            # sample goals with calibrated distribution
            hg = sample_goals_calibrated(lambda_home, dispersion_factor, size=1)
            ag = sample_goals_calibrated(lambda_away, dispersion_factor, size=1)

            # calculate points
            hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            ap = np.where(ag > hg, 3, np.where(ag == hg, 1, 0))

            # update standings
            sim_points = base_points.copy()
            sim_gf = base_gf.copy()
            sim_ga = base_ga.copy()

            np.add.at(sim_points, home_idx, hp)
            np.add.at(sim_points, away_idx, ap)
            np.add.at(sim_gf, home_idx, hg)
            np.add.at(sim_gf, away_idx, ag)
            np.add.at(sim_ga, home_idx, ag)
            np.add.at(sim_ga, away_idx, hg)

            # store results
            results["points"][s] = sim_points
            results["goals_for"][s] = sim_gf
            results["goals_against"][s] = sim_ga

            # calculate final positions (sort by points, then gd)
            goal_diff = sim_gf - sim_ga
            order = np.lexsort((-goal_diff, -sim_points))
            for pos, idx in enumerate(order):
                results["positions"][s, idx] = pos + 1

        print("\nSimulation complete!")
    else:
        # no future fixtures: just return current standings
        results = {
            "points": np.tile(base_points, (n_simulations, 1)),
            "goals_for": np.tile(base_gf, (n_simulations, 1)),
            "goals_against": np.tile(base_ga, (n_simulations, 1)),
            "positions": np.zeros((n_simulations, n_teams), dtype=int),
        }

    return results, all_teams


def create_final_summary(
    results: Dict[str, np.ndarray],
    params: Dict[str, Any],
    teams: List[str],
    current_standings: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """Create final summary table from simulation results"""
    n_simulations = len(results["points"])
    n_teams = len(teams)

    rows = []
    for i, team in enumerate(teams):
        mean_points = results["points"][:, i].mean()
        mean_gf = results["goals_for"][:, i].mean()
        mean_ga = results["goals_against"][:, i].mean()
        mean_gd = mean_gf - mean_ga

        # calculate probabilities
        pos_counts = np.bincount(results["positions"][:, i] - 1, minlength=n_teams)
        title_prob = pos_counts[0] / n_simulations
        ucl_prob = pos_counts[:4].sum() / n_simulations
        relegation_prob = pos_counts[-2:].sum() / n_simulations

        rows.append(
            {
                "team": team,
                "projected_points": mean_points,
                "projected_gd": mean_gd,
                "overall_rating": params["overall_rating"].get(team, np.nan),
                "attack_rating": params["attack_rating"].get(team, np.nan),
                "defense_rating": params["defense_rating"].get(team, np.nan),
                "title_prob": title_prob,
                "ucl_prob": ucl_prob,
                "relegation_prob": relegation_prob,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("projected_points", ascending=False)
        .reset_index(drop=True)
    )
