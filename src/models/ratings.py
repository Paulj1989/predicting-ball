# src/models/ratings.py

from typing import Any

import numpy as np


def create_interpretable_ratings(params: dict[str, Any]) -> dict[str, dict[str, float]]:
    """
    Transform ratings to interpretable Z-score scales (approx [-1, 1]).

    0 = Average, + = Above, - = Below, +-1 equivalent to ~3 SDs.
    """
    teams = params["teams"]

    # extract raw parameters
    attack_raw = np.array([params["attack"][t] for t in teams])
    defense_raw = np.array([params["defense"][t] for t in teams])

    # flip defense sign so positive = good defense
    defense_raw = -defense_raw

    # normalise to z-scores (mean=0, std=1)
    attack_z = (attack_raw - attack_raw.mean()) / attack_raw.std()
    defense_z = (defense_raw - defense_raw.mean()) / defense_raw.std()

    # scale to approximately [-1, 1] by dividing by 3
    # clips extreme outliers (>3 sd)
    attack_scaled = np.clip(attack_z / 3, -1, 1)
    defense_scaled = np.clip(defense_z / 3, -1, 1)

    # overall rating (equal weight to attack and defense)
    overall = (attack_scaled + defense_scaled) / 2

    return {
        "attack": {t: float(attack_scaled[i]) for i, t in enumerate(teams)},
        "defense": {t: float(defense_scaled[i]) for i, t in enumerate(teams)},
        "overall": {t: float(overall[i]) for i, t in enumerate(teams)},
    }


def add_interpretable_ratings_to_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Add interpretable ratings to params dict (in-place and return).
    """
    ratings = create_interpretable_ratings(params)

    params["attack_rating"] = ratings["attack"]
    params["defense_rating"] = ratings["defense"]
    params["overall_rating"] = ratings["overall"]

    return params
