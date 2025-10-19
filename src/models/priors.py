# src/models/priors.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


def calculate_home_advantage_prior(
    historic_data: pd.DataFrame, use_actual_goals: bool = True, verbose: bool = True
) -> Tuple[float, float]:
    """
    Calculate home advantage prior from historical data.

    Measures the average boost in performance when playing at home by comparing
    home vs away goals. The prior is on log scale (additive to log-lambda).

    The calculation:
        ratio = avg_home_goals / avg_away_goals
        home_adv_prior = log(ratio)

    This represents the multiplicative boost teams get at home.
    """
    if use_actual_goals:
        home_metric = historic_data["home_goals"].dropna().mean()
        away_metric = historic_data["away_goals"].dropna().mean()
        metric_name = "actual goals"
    else:
        home_metric = historic_data["home_goals_weighted"].dropna().mean()
        away_metric = historic_data["away_goals_weighted"].dropna().mean()
        metric_name = "weighted performance"

    # home advantage ratio (multiplicative on goals scale)
    ratio = home_metric / away_metric

    # convert to log scale (additive on model scale)
    home_adv_prior = np.log(ratio)

    if verbose:
        print("\nHome Advantage Prior from Historical Data:")
        print(f"  Metric: {metric_name}")
        print(f"  Average home {metric_name}: {home_metric:.3f}")
        print(f"  Average away {metric_name}: {away_metric:.3f}")
        print(f"  Home advantage ratio: {ratio:.3f} ({(ratio - 1) * 100:.1f}% boost)")
        print(f"  Home advantage (log scale): {home_adv_prior:.3f}")

    # estimate uncertainty from season-to-season variation
    seasons_home_adv = []
    for season in sorted(historic_data["season_end_year"].unique()):
        season_data = historic_data[historic_data["season_end_year"] == season]

        if use_actual_goals:
            h_metric = season_data["home_goals"].dropna().mean()
            a_metric = season_data["away_goals"].dropna().mean()
        else:
            h_metric = season_data["home_goals_weighted"].dropna().mean()
            a_metric = season_data["away_goals_weighted"].dropna().mean()

        if a_metric > 0 and h_metric > 0:
            seasons_home_adv.append(np.log(h_metric / a_metric))

    # standard deviation across seasons captures uncertainty
    home_adv_std = np.std(seasons_home_adv) if len(seasons_home_adv) > 1 else 0.05

    if verbose:
        print(f"  Seasons analysed: {len(seasons_home_adv)}")
        print(
            f"  Range across seasons: [{min(seasons_home_adv):.3f}, {max(seasons_home_adv):.3f}]"
        )
        print(f"  Standard deviation: {home_adv_std:.3f}")
        print(f"  Prior: Normal({home_adv_prior:.3f}, {home_adv_std:.3f})")

        # interpret in practical terms
        lower_bound = np.exp(home_adv_prior - 2 * home_adv_std)
        upper_bound = np.exp(home_adv_prior + 2 * home_adv_std)
        print(
            f"  95% credible interval: {(lower_bound - 1) * 100:.1f}% to {(upper_bound - 1) * 100:.1f}% boost"
        )

    return home_adv_prior, home_adv_std


def identify_promoted_teams(
    historic_data: pd.DataFrame, current_season: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Identify teams in current season not in previous season"""
    print("\n" + "=" * 60)
    print("PROMOTED TEAM IDENTIFICATION")
    print("=" * 60)

    last_season_year = historic_data["season_end_year"].max()
    last_season_data = historic_data[
        historic_data["season_end_year"] == last_season_year
    ]

    print(f"Last historic season: {last_season_year}")
    print(f"Last season matches: {len(last_season_data)}")

    last_season_teams = set(last_season_data[["home_team", "away_team"]].values.ravel())

    print(f"Teams in {last_season_year} season ({len(last_season_teams)}):")
    for team in sorted(last_season_teams):
        print(f"  - {team}")

    current_season_year = (
        current_season["season_end_year"].iloc[0]
        if len(current_season) > 0
        else last_season_year + 1
    )
    current_teams = set(current_season[["home_team", "away_team"]].values.ravel())

    print(f"\nTeams in {current_season_year} season ({len(current_teams)}):")
    for team in sorted(current_teams):
        print(f"  - {team}")

    promoted = current_teams - last_season_teams
    relegated = last_season_teams - current_teams

    print("\n" + "=" * 60)
    if relegated:
        print(f"Relegated from {last_season_year}: {', '.join(sorted(relegated))}")
    print(f"Promoted to {current_season_year}: {len(promoted)} teams")
    print("=" * 60)

    if len(promoted) == 0:
        print("\nNo promoted teams identified")
        return {}

    promoted_info = {}
    for team in promoted:
        team_rows = current_season[
            (current_season["home_team"] == team)
            | (current_season["away_team"] == team)
        ]

        if len(team_rows) > 0:
            home_val = team_rows[team_rows["home_team"] == team][
                "home_value_pct"
            ].mean()
            away_val = team_rows[team_rows["away_team"] == team][
                "away_value_pct"
            ].mean()
            squad_value_pct = np.nanmean([home_val, away_val])
        else:
            squad_value_pct = np.nan

        historic_appearances = historic_data[
            (historic_data["home_team"] == team) | (historic_data["away_team"] == team)
        ]

        promoted_info[team] = {
            "squad_value_pct": squad_value_pct,
            "is_promoted": True,
            "historic_matches": len(historic_appearances),
        }

        print(f"\n  {team}:")
        print(f"    Squad value: {squad_value_pct:.1f}%")
        if len(historic_appearances) > 0:
            seasons_played = sorted(historic_appearances["season_end_year"].unique())
            print(f"    Previous seasons: {', '.join(map(str, seasons_played))}")
            print("    Status: RETURNING")
        else:
            print("    Status: NEW")

    return promoted_info


def calculate_promoted_team_priors(
    df_train: pd.DataFrame,
    promoted_teams: Dict[str, Dict[str, Any]],
    current_season: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Calculate informed priors for promoted teams.

    Uses three sources of information:
    1. Historical league parameter distributions (quartiles)
    2. Squad values relative to league average
    3. Early season performance (if 3+ matches played)
    """
    print("\n" + "=" * 60)
    print("CALCULATING PROMOTED TEAM PRIORS")
    print("=" * 60)

    # calculate home advantage prior from historical data
    home_adv_prior, home_adv_std = calculate_home_advantage_prior(
        df_train, use_actual_goals=True, verbose=True
    )

    # remove promoted teams from historic data for baseline model
    historic_only = df_train.copy()
    for team in promoted_teams.keys():
        historic_only = historic_only[
            (historic_only["home_team"] != team) & (historic_only["away_team"] != team)
        ]

    if len(historic_only) < 100:
        print("  Warning: Insufficient historic data, using neutral priors")
        return (
            {
                team: {"attack_prior": -0.3, "defense_prior": 0.3}
                for team in promoted_teams
            },
            home_adv_prior,
            home_adv_std,
        )

    # fit baseline model to get league parameter distributions
    # import here to avoid circular dependency
    from .poisson import fit_poisson_model

    temp_hyperparams = {
        "time_decay": 0.001,
        "lambda_reg": 0.3,
        "prior_decay_rate": 10.0,
    }

    base_model = fit_poisson_model(
        historic_only,
        temp_hyperparams,
        home_adv_prior=home_adv_prior,
        home_adv_std=home_adv_std,
        n_random_starts=1,
        verbose=False,
    )

    if not base_model:
        print("  Warning: Could not fit baseline model, using neutral priors")
        return (
            {
                team: {"attack_prior": -0.3, "defense_prior": 0.3}
                for team in promoted_teams
            },
            home_adv_prior,
            home_adv_std,
        )

    # get parameter distributions
    all_attacks = np.array(list(base_model["attack"].values()))
    all_defenses = np.array(list(base_model["defense"].values()))

    attack_25 = np.percentile(all_attacks, 25)
    attack_50 = np.percentile(all_attacks, 50)
    attack_75 = np.percentile(all_attacks, 75)

    defense_25 = np.percentile(all_defenses, 25)
    defense_50 = np.percentile(all_defenses, 50)
    defense_75 = np.percentile(all_defenses, 75)

    print("\nLeague parameter distributions:")
    print(f"  Attack: 25th={attack_25:.3f}, 50th={attack_50:.3f}, 75th={attack_75:.3f}")
    print(
        f"  Defense: 25th={defense_25:.3f}, 50th={defense_50:.3f}, 75th={defense_75:.3f}"
    )

    league_avg_value = df_train["home_value_pct"].mean()

    priors = {}

    for team, info in promoted_teams.items():
        squad_value = info.get("squad_value_pct", np.nan)

        if not np.isnan(squad_value):
            relative_strength = squad_value / league_avg_value

            # map squad value to prior based on quartiles
            if relative_strength < 0.8:
                # weak team: bottom quartile
                attack_prior = attack_25
                defense_prior = defense_75
            elif relative_strength < 1.2:
                # average team: interpolate between 25th and 50th percentile
                blend = (relative_strength - 0.8) / 0.4
                attack_prior = attack_25 + blend * (attack_50 - attack_25)
                defense_prior = defense_75 + blend * (defense_50 - defense_75)
            else:
                # strong team: interpolate between 50th and 75th percentile
                blend = min((relative_strength - 1.2) / 0.8, 1.0)
                attack_prior = attack_50 + blend * (attack_75 - attack_50)
                defense_prior = defense_50 + blend * (defense_25 - defense_50)

            print(f"\n  {team}:")
            print(
                f"    Squad value: {squad_value:.1f}% (league avg: {league_avg_value:.1f}%)"
            )
            print(f"    Relative strength: {relative_strength:.2f}x")
            print(f"    Attack prior: {attack_prior:.3f}")
            print(f"    Defense prior: {defense_prior:.3f}")
        else:
            # no squad value: use bottom quartile (conservative)
            attack_prior = attack_25
            defense_prior = defense_75
            print(f"\n  {team}: No squad value, using bottom quartile")

        # adjust with early season performance if available
        team_matches = current_season[
            (
                (current_season["home_team"] == team)
                | (current_season["away_team"] == team)
            )
            & (current_season["is_played"] == True)
        ]

        if len(team_matches) >= 3:
            home_matches = team_matches[team_matches["home_team"] == team]
            away_matches = team_matches[team_matches["away_team"] == team]

            goals_scored = (
                home_matches["home_goals"].sum() + away_matches["away_goals"].sum()
            ) / len(team_matches)

            goals_conceded = (
                home_matches["away_goals"].sum() + away_matches["home_goals"].sum()
            ) / len(team_matches)

            # convert to rating scale (rough approximation)
            performance_attack = (goals_scored - 1.5) * 0.3
            performance_defense = (goals_conceded - 1.5) * 0.3

            # blend: 70% prior, 30% performance (small sample)
            attack_prior = 0.7 * attack_prior + 0.3 * performance_attack
            defense_prior = 0.7 * defense_prior + 0.3 * performance_defense

            print(f"    Adjusted with {len(team_matches)} early matches")
            print(f"    Final attack prior: {attack_prior:.3f}")
            print(f"    Final defense prior: {defense_prior:.3f}")

        priors[team] = {"attack_prior": attack_prior, "defense_prior": defense_prior}

    return priors, home_adv_prior, home_adv_std
