# src/models/priors.py

from typing import Any

import numpy as np
import pandas as pd

# blend weight constants for prior calculations — same formula applied to all teams
ELO_WEIGHT = 2 / 3
SQUAD_WEIGHT = 1 / 3


def _get_team_metric(df: pd.DataFrame, team: str, metric_col: str) -> float:
    """Helper function to calculate average metric for a team from home/away columns"""
    team_rows = df[(df["home_team"] == team) | (df["away_team"] == team)]

    if len(team_rows) == 0:
        return np.nan

    # get values when team is home
    home_col = f"home_{metric_col}" if "home_" not in metric_col else metric_col
    away_col = f"away_{metric_col}" if "away_" not in metric_col else metric_col

    home_val = team_rows[team_rows["home_team"] == team][home_col].mean()
    away_val = team_rows[team_rows["away_team"] == team][away_col].mean()

    # if both are NaN (e.g. metric column is entirely missing), return NaN cleanly
    if pd.isna(home_val) and pd.isna(away_val):
        return np.nan
    return np.nanmean([home_val, away_val])


def calculate_home_advantage_prior(
    historic_data: pd.DataFrame, use_actual_goals: bool = True, verbose: bool = True
) -> tuple[float, float]:
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

    # safety check for division by zero
    if away_metric <= 0 or home_metric <= 0:
        raise ValueError(
            f"Invalid metric values: home={home_metric:.3f}, away={away_metric:.3f}. "
            "Both must be positive for home advantage calculation."
        )

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
) -> dict[str, dict[str, float]]:
    """
    Calculate attack/defense priors from squad values for all teams.

    Maps relative squad value to expected attack/defense strength using
    piecewise linear interpolation between quartiles.
    """
    # extract squad values using helper function
    squad_values = {team: _get_team_metric(df_train, team, "value_pct") for team in all_teams}

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
            defense_prior = defense_range[0] + blend * (defense_range[1] - defense_range[0])
        else:
            blend = min((relative - 1.2) / 0.8, 1.0)
            attack_prior = attack_range[1] + blend * (attack_range[2] - attack_range[1])
            defense_prior = defense_range[1] + blend * (defense_range[2] - defense_range[1])

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


def calculate_elo_priors(
    df_train: pd.DataFrame,
    all_teams: list,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Calculate attack/defense priors from Elo ratings for all teams"""
    # extract elo ratings using helper function
    elo_ratings = {team: _get_team_metric(df_train, team, "elo") for team in all_teams}

    # calculate league statistics
    valid_elos = [e for e in elo_ratings.values() if not np.isnan(e)]
    if len(valid_elos) == 0:
        # no elo ratings available
        if verbose:
            print("\nNo Elo ratings available - returning neutral priors")
        return {team: {"attack_prior": 0.0, "defense_prior": 0.0} for team in all_teams}

    league_avg_elo = np.mean(valid_elos)
    league_std_elo = np.std(valid_elos)

    if verbose:
        print("\nElo Rating Distribution:")
        print(f"  League average: {league_avg_elo:.1f}")
        print(f"  League std dev: {league_std_elo:.1f}")
        print(f"  Min: {np.min(valid_elos):.1f}, Max: {np.max(valid_elos):.1f}")

    # map elo to attack/defense priors
    priors = {}
    for team in all_teams:
        elo = elo_ratings[team]

        if np.isnan(elo):
            priors[team] = {"attack_prior": 0.0, "defense_prior": 0.0}
            continue

        # z-score normalisation
        z_score = (elo - league_avg_elo) / league_std_elo

        # map to parameter space
        # 1 std dev (~150 elo points) = 0.25 parameter units
        attack_prior = z_score * 0.25
        defense_prior = -z_score * 0.25  # inverted: higher elo = better defense = lower param

        priors[team] = {
            "attack_prior": attack_prior,
            "defense_prior": defense_prior,
            "elo_rating": elo,
            "elo_z_score": z_score,
        }

    if verbose:
        print("\nElo → Prior Mapping (selected teams):")
        sorted_teams = sorted(
            all_teams,
            key=lambda t: elo_ratings[t] if not np.isnan(elo_ratings[t]) else 0,
            reverse=True,
        )
        for i, team in enumerate(sorted_teams[:3] + sorted_teams[-3:]):
            if i == 3:
                print("  ...")
            p = priors[team]
            if not np.isnan(p.get("elo_rating", np.nan)):
                print(
                    f"  {team}: Elo={p['elo_rating']:.0f} (z={p['elo_z_score']:+.2f}) → "
                    f"attack={p['attack_prior']:+.3f}, defense={p['defense_prior']:+.3f}"
                )

    return priors


def calculate_all_team_priors(
    df_train: pd.DataFrame,
    all_teams: list,
    promoted_teams: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Calculate priors for all teams using 2/3 Elo + 1/3 squad value blend.

    All teams use the same formula regardless of whether they are promoted or
    returning. The prior is based entirely on external signals (Elo ratings and
    squad values) to avoid double-counting training data.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("CALCULATING PRIORS FOR ALL TEAMS")
        print("=" * 60)

    # calculate component priors
    squad_priors = calculate_squad_value_priors(df_train, all_teams, verbose=False)
    elo_priors = calculate_elo_priors(df_train, all_teams, verbose=False)

    # check elo availability
    has_elo = any(
        not np.isnan(elo_priors.get(team, {}).get("elo_rating", np.nan)) for team in all_teams
    )

    all_priors = {}
    promoted_team_names = set(promoted_teams.keys())

    for team in all_teams:
        sp = squad_priors.get(team, {"attack_prior": 0.0, "defense_prior": 0.0})
        ep = elo_priors.get(team, {"attack_prior": 0.0, "defense_prior": 0.0})
        elo_available = not np.isnan(ep.get("elo_rating", np.nan))
        is_promoted = team in promoted_team_names

        if elo_available:
            attack_prior = ELO_WEIGHT * ep["attack_prior"] + SQUAD_WEIGHT * sp["attack_prior"]
            defense_prior = (
                ELO_WEIGHT * ep["defense_prior"] + SQUAD_WEIGHT * sp["defense_prior"]
            )
            source = "elo_squad"
        else:
            attack_prior = sp["attack_prior"]
            defense_prior = sp["defense_prior"]
            source = "squad_only"

        all_priors[team] = {
            "attack_prior": attack_prior,
            "defense_prior": defense_prior,
            "source": source,
            "is_promoted": is_promoted,
        }

    if verbose:
        print(f"\n  Formula: {ELO_WEIGHT:.0%} Elo + {SQUAD_WEIGHT:.0%} squad value")
        print(f"  Elo data available: {'Yes' if has_elo else 'No'}")
        print(f"  Promoted teams: {len(promoted_team_names)}")

        if promoted_team_names:
            print("\nPromoted teams:")
            for team in promoted_team_names:
                if team in all_priors:
                    p = all_priors[team]
                    print(
                        f"  {team}: attack={p['attack_prior']:+.3f}, defense={p['defense_prior']:+.3f} ({p['source']})"
                    )

        print("\nReturning teams (sample):")
        returning = [t for t in all_teams if t not in promoted_team_names]
        for team in returning[:3]:
            p = all_priors[team]
            print(
                f"  {team}: attack={p['attack_prior']:+.3f}, defense={p['defense_prior']:+.3f} ({p['source']})"
            )

    return all_priors


def identify_promoted_teams(
    historic_data: pd.DataFrame, current_season: pd.DataFrame, verbose: bool = True
) -> dict[str, dict[str, Any]]:
    """Identify teams in current season not in previous season"""
    if verbose:
        print("\n" + "=" * 60)
        print("PROMOTED TEAM IDENTIFICATION")
        print("=" * 60)

    last_season_year = historic_data["season_end_year"].max()
    last_season_data = historic_data[historic_data["season_end_year"] == last_season_year]

    last_season_teams = set(last_season_data[["home_team", "away_team"]].values.ravel())

    current_teams = set(current_season[["home_team", "away_team"]].values.ravel())

    promoted = current_teams - last_season_teams

    if len(promoted) == 0:
        if verbose:
            print("\nNo promoted teams identified")
        return {}

    promoted_info = {}
    for team in promoted:
        squad_value_pct = _get_team_metric(current_season, team, "value_pct")

        historic_appearances = historic_data[
            (historic_data["home_team"] == team) | (historic_data["away_team"] == team)
        ]

        promoted_info[team] = {
            "squad_value_pct": squad_value_pct,
            "is_promoted": True,
            "historic_matches": len(historic_appearances),
        }

        if verbose:
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
    promoted_teams: dict[str, dict[str, Any]],
    current_season: pd.DataFrame,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """
    Calculate priors for ALL teams using Elo + squad value blend.

    Calculates home advantage prior and delegates to calculate_all_team_priors
    to compute attack/defense priors for each team using external signals only.
    """
    # calculate home advantage prior
    home_adv_prior, home_adv_std = calculate_home_advantage_prior(
        df_train, use_actual_goals=True, verbose=verbose
    )

    # get all teams
    all_teams = sorted(
        pd.unique(
            pd.concat([df_train, current_season])[["home_team", "away_team"]].values.ravel()
        )
    )

    # calculate priors for all teams
    all_priors = calculate_all_team_priors(
        df_train,
        all_teams,
        promoted_teams,
        verbose=verbose,
    )

    return all_priors, home_adv_prior, home_adv_std
