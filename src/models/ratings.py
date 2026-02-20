# src/models/ratings.py

from typing import Any

import numpy as np


def create_interpretable_ratings(
    params: dict[str, Any],
    reference_teams: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Transform ratings to interpretable Z-score scales (approx [-1, 1]).

    0 = Average, + = Above, - = Below, +-1 equivalent to ~3 SDs.

    reference_teams: if provided, mean and std are computed over only those
    teams (e.g. the current season's 18 clubs) so the normalisation reflects
    the current competitive context rather than all historically trained teams.
    Ratings are still returned for every team in params["teams"].
    """
    teams = params["teams"]

    # extract raw parameters for all teams
    attack_raw = np.array([params["attack"][t] for t in teams])
    defense_raw = np.array([-params["defense"][t] for t in teams])  # flip: positive = good

    # compute mean/std over the reference population (current teams if provided)
    if reference_teams is not None:
        ref_idx = [i for i, t in enumerate(teams) if t in set(reference_teams)]
        att_mean, att_std = attack_raw[ref_idx].mean(), attack_raw[ref_idx].std()
        def_mean, def_std = defense_raw[ref_idx].mean(), defense_raw[ref_idx].std()
    else:
        att_mean, att_std = attack_raw.mean(), attack_raw.std()
        def_mean, def_std = defense_raw.mean(), defense_raw.std()

    # normalise to z-scores then scale to approx [-1, 1]; clips extreme outliers
    # when std=0 (all teams identical), z-scores are 0 — everyone is average
    attack_z = (attack_raw - att_mean) / att_std if att_std > 0 else np.zeros_like(attack_raw)
    defense_z = (
        (defense_raw - def_mean) / def_std if def_std > 0 else np.zeros_like(defense_raw)
    )
    attack_scaled = np.clip(attack_z / 3, -1, 1)
    defense_scaled = np.clip(defense_z / 3, -1, 1)

    # overall rating (equal weight to attack and defense)
    overall = (attack_scaled + defense_scaled) / 2

    return {
        "attack": {t: float(attack_scaled[i]) for i, t in enumerate(teams)},
        "defense": {t: float(defense_scaled[i]) for i, t in enumerate(teams)},
        "overall": {t: float(overall[i]) for i, t in enumerate(teams)},
    }


def add_interpretable_ratings_to_params(
    params: dict[str, Any],
    reference_teams: list[str] | None = None,
) -> dict[str, Any]:
    """
    Add interpretable ratings to params dict (in-place and return).

    reference_teams: passed through to create_interpretable_ratings.
    """
    ratings = create_interpretable_ratings(params, reference_teams=reference_teams)

    params["attack_rating"] = ratings["attack"]
    params["defense_rating"] = ratings["defense"]
    params["overall_rating"] = ratings["overall"]

    return params
