# src/models/priors.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional


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
        metric_name = "Goals"
    else:
        home_metric = historic_data["home_goals_weighted"].dropna().mean()
        away_metric = historic_data["away_goals_weighted"].dropna().mean()
        metric_name = "npxG/npG"

    # home advantage ratio (multiplicative on goals scale)
    ratio = home_metric / away_metric

    # convert to log scale (additive on model scale)
    home_adv_prior = np.log(ratio)

    if verbose:
        print("\nHome Advantage Prior from Historical Data:")
        print(f"  Average Home {metric_name}: {home_metric:.3f}")
        print(f"  Average Away {metric_name}: {away_metric:.3f}")
        print(f"  Home Advantage Ratio: {ratio:.3f} ({(ratio - 1) * 100:.1f}% boost)")
        print(f"  Home Advantage (Log Scale): {home_adv_prior:.3f}")

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
        print(f"  Standard Deviation: {home_adv_std:.3f}")

        # interpret in practical terms
        lower_bound = np.exp(home_adv_prior - 2 * home_adv_std)
        upper_bound = np.exp(home_adv_prior + 2 * home_adv_std)
        print(
            f"  95% Credible Interval: {(lower_bound - 1) * 100:.1f}% - {(upper_bound - 1) * 100:.1f}%"
        )

    return home_adv_prior, home_adv_std


def calculate_squad_value_priors(
    df_train: pd.DataFrame,
    all_teams: list,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate attack/defense priors from squad values for all teams.

    Maps relative squad value to expected attack/defense strength using
    piecewise linear interpolation between quartiles.
    """
    # extract squad values
    squad_values = {}
    for team in all_teams:
        team_rows = df_train[
            (df_train["home_team"] == team) | (df_train["away_team"] == team)
        ]
        if len(team_rows) > 0:
            home_val = team_rows[team_rows["home_team"] == team][
                "home_value_pct"
            ].mean()
            away_val = team_rows[team_rows["away_team"] == team][
                "away_value_pct"
            ].mean()
            squad_values[team] = np.nanmean([home_val, away_val])
        else:
            squad_values[team] = np.nan

    # calculate league statistics
    valid_values = [v for v in squad_values.values() if not np.isnan(v)]
    if len(valid_values) == 0:
        # no squad values available
        return {team: {"attack_prior": 0.0, "defense_prior": 0.0} for team in all_teams}

    q25, q50, q75 = np.percentile(valid_values, [25, 50, 75])
    league_avg = np.mean(valid_values)

    if verbose:
        print("\nSquad Value Distribution:")
        print(f"  25th percentile: {q25:.1f}%")
        print(f"  50th percentile: {q50:.1f}%")
        print(f"  75th percentile: {q75:.1f}%")
        print(f"  League average: {league_avg:.1f}%")

    # prior strength ranges (map to attack/defense parameters)
    attack_range = (-0.4, 0.0, 0.4)  # 25th, 50th, 75th percentile
    defense_range = (0.3, 0.0, -0.3)  # 25th (weak), 50th, 75th (strong)

    # map squad values to priors
    priors = {}
    for team in all_teams:
        sv = squad_values[team]

        if np.isnan(sv):
            priors[team] = {"attack_prior": 0.0, "defense_prior": 0.0}
            continue

        relative = sv / league_avg

        # piecewise linear mapping
        if relative < 0.8:
            attack_prior = attack_range[0]
            defense_prior = defense_range[0]
        elif relative < 1.2:
            blend = (relative - 0.8) / 0.4
            attack_prior = attack_range[0] + blend * (attack_range[1] - attack_range[0])
            defense_prior = defense_range[0] + blend * (
                defense_range[1] - defense_range[0]
            )
        else:
            blend = min((relative - 1.2) / 0.8, 1.0)
            attack_prior = attack_range[1] + blend * (attack_range[2] - attack_range[1])
            defense_prior = defense_range[1] + blend * (
                defense_range[2] - defense_range[1]
            )

        priors[team] = {
            "attack_prior": attack_prior,
            "defense_prior": defense_prior,
            "squad_value_pct": sv,
            "relative_strength": relative,
        }

    if verbose:
        print("\nSquad Value → Prior Mapping (selected teams):")
        sorted_teams = sorted(
            all_teams,
            key=lambda t: squad_values[t] if not np.isnan(squad_values[t]) else 0,
            reverse=True,
        )
        for i, team in enumerate(sorted_teams[:3] + sorted_teams[-3:]):
            if i == 3:
                print("  ...")
            p = priors[team]
            if not np.isnan(p.get("squad_value_pct", np.nan)):
                print(
                    f"  {team}: {p['squad_value_pct']:.1f}% → attack={p['attack_prior']:.3f}, defense={p['defense_prior']:.3f}"
                )

    return priors


def calculate_seasonal_priors(
    df_train: pd.DataFrame,
    all_teams: list,
    previous_params: Optional[Dict[str, Any]] = None,
    blend_weight_prev: float = 0.7,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Blend previous season parameters with squad value priors.

    For returning teams: 70% previous season + 30% squad value.
    For promoted teams: 100% squad value.
    """
    # get squad value priors
    squad_priors = calculate_squad_value_priors(df_train, all_teams, verbose=False)

    if previous_params is None:
        # no previous season: use squad values only
        if verbose:
            print("  No previous season params, using squad values only")
        return squad_priors

    # blend with previous season
    blended_priors = {}
    prev_attack = previous_params.get("attack", {})
    prev_defense = previous_params.get("defense", {})

    for team in all_teams:
        sp = squad_priors.get(team, {"attack_prior": 0.0, "defense_prior": 0.0})

        # check if team existed previous season
        if team in prev_attack and team in prev_defense:
            # blend: 70% previous season, 30% squad value
            attack_prior = (
                blend_weight_prev * prev_attack[team]
                + (1 - blend_weight_prev) * sp["attack_prior"]
            )
            defense_prior = (
                blend_weight_prev * prev_defense[team]
                + (1 - blend_weight_prev) * sp["defense_prior"]
            )
            source = "blended"
        else:
            # new team: squad value only
            attack_prior = sp["attack_prior"]
            defense_prior = sp["defense_prior"]
            source = "squad_value"

        blended_priors[team] = {
            "attack_prior": attack_prior,
            "defense_prior": defense_prior,
            "source": source,
        }

    if verbose:
        n_blended = sum(
            1 for p in blended_priors.values() if p.get("source") == "blended"
        )
        n_squad = len(blended_priors) - n_blended
        print(
            f"\nSeasonal Prior Blending ({blend_weight_prev:.0%} previous / {1 - blend_weight_prev:.0%} squad):"
        )
        print(f"  {n_blended} teams: blended from previous season")
        print(f"  {n_squad} teams: squad value only (new/promoted)")

    return blended_priors


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

    last_season_teams = set(last_season_data[["home_team", "away_team"]].values.ravel())

    current_teams = set(current_season[["home_team", "away_team"]].values.ravel())

    promoted = current_teams - last_season_teams

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
    previous_season_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """Calculate informed priors for promoted teams"""
    print("\n" + "=" * 60)
    print("CALCULATING PROMOTED TEAM PRIORS")
    print("=" * 60)

    # calculate home advantage prior
    home_adv_prior, home_adv_std = calculate_home_advantage_prior(
        df_train, use_actual_goals=True, verbose=True
    )

    if len(promoted_teams) == 0:
        return {}, home_adv_prior, home_adv_std

    # get all teams
    all_teams = sorted(pd.unique(df_train[["home_team", "away_team"]].values.ravel()))
    for team in promoted_teams.keys():
        if team not in all_teams:
            all_teams.append(team)
    all_teams = sorted(all_teams)

    # calculate squad value priors
    squad_priors = calculate_squad_value_priors(df_train, all_teams, verbose=True)

    print("\nPromoted team priors from squad values:")
    priors = {}

    for team in promoted_teams.keys():
        if team not in squad_priors:
            print(f"\n  {team}: No squad value data, using conservative prior")
            priors[team] = {"attack_prior": -0.3, "defense_prior": 0.3}
            continue

        sp = squad_priors[team]
        attack_prior = sp["attack_prior"]
        defense_prior = sp["defense_prior"]

        print(f"\n  {team}:")
        if "squad_value_pct" in sp:
            print(f"    Squad value: {sp['squad_value_pct']:.1f}%")
            print(f"    Relative strength: {sp['relative_strength']:.2f}x")
        print(f"    Attack prior: {attack_prior:.3f}")
        print(f"    Defense prior: {defense_prior:.3f}")

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

            # convert to rating scale
            performance_attack = (goals_scored - 1.5) * 0.3
            performance_defense = (goals_conceded - 1.5) * 0.3

            # blend: 70% prior, 30% performance
            attack_prior = 0.7 * attack_prior + 0.3 * performance_attack
            defense_prior = 0.7 * defense_prior + 0.3 * performance_defense

            print(f"    Adjusted with {len(team_matches)} early matches")
            print(f"    Final attack prior: {attack_prior:.3f}")
            print(f"    Final defense prior: {defense_prior:.3f}")

        priors[team] = {"attack_prior": attack_prior, "defense_prior": defense_prior}

    # add previous season params as metadata
    if previous_season_params is not None:
        priors["_previous_season_params"] = previous_season_params

    return priors, home_adv_prior, home_adv_std
