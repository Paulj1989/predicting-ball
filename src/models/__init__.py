# src/models/__init__.py

from .calibration import (
    apply_temperature_scaling,
    calibrate_dispersion_for_coverage,
    calibrate_model_comprehensively,
    fit_temperature_scaler,
)
from .hyperparameters import (
    get_default_hyperparameters,
    optimise_hyperparameters,
)
from .poisson import (
    calculate_lambdas,
    calculate_lambdas_single,
    fit_baseline_strengths,
    fit_feature_coefficients,
    fit_poisson_model_two_stage,
)
from .priors import (
    calculate_all_team_priors,
    calculate_elo_priors,
    calculate_home_advantage_prior,
    calculate_promoted_team_priors,
    calculate_squad_value_priors,
    identify_promoted_teams,
)
from .ratings import (
    add_interpretable_ratings_to_params,
    create_interpretable_ratings,
)

__all__ = [
    "add_interpretable_ratings_to_params",
    "apply_temperature_scaling",
    "calculate_all_team_priors",
    "calculate_elo_priors",
    # priors
    "calculate_home_advantage_prior",
    "calculate_lambdas",
    "calculate_lambdas_single",
    "calculate_promoted_team_priors",
    "calculate_squad_value_priors",
    "calibrate_dispersion_for_coverage",
    "calibrate_model_comprehensively",
    # ratings
    "create_interpretable_ratings",
    # core model
    "fit_baseline_strengths",
    "fit_feature_coefficients",
    "fit_poisson_model_two_stage",
    # calibration (temperature scaling)
    "fit_temperature_scaler",
    "get_default_hyperparameters",
    "identify_promoted_teams",
    # hyperparameters
    "optimise_hyperparameters",
]
