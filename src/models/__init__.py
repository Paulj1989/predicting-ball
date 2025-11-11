# src/models/__init__.py

from .poisson import (
    fit_baseline_strengths,
    fit_feature_coefficients,
    fit_poisson_model_two_stage,
    calculate_lambdas,
    calculate_lambdas_single,
)

from .priors import (
    calculate_home_advantage_prior,
    calculate_promoted_team_priors,
    identify_promoted_teams,
    calculate_squad_value_priors,
    calculate_elo_priors,
    calculate_all_team_priors,
)

from .calibration import (
    fit_temperature_scaler,
    apply_temperature_scaling,
    calibrate_dispersion_for_coverage,
    calibrate_model_comprehensively,
)

from .hyperparameters import (
    optimise_hyperparameters,
    get_default_hyperparameters,
)

from .ratings import (
    create_interpretable_ratings,
    add_interpretable_ratings_to_params,
)

__all__ = [
    # core model
    "fit_baseline_strengths",
    "fit_feature_coefficients",
    "fit_poisson_model_two_stage",
    "calculate_lambdas",
    "calculate_lambdas_single",
    # priors
    "calculate_home_advantage_prior",
    "calculate_promoted_team_priors",
    "identify_promoted_teams",
    "calculate_squad_value_priors",
    "calculate_elo_priors",
    "calculate_all_team_priors",
    # calibration (temperature scaling)
    "fit_temperature_scaler",
    "apply_temperature_scaling",
    "calibrate_dispersion_for_coverage",
    "calibrate_model_comprehensively",
    # hyperparameters
    "optimise_hyperparameters",
    "get_default_hyperparameters",
    # ratings
    "create_interpretable_ratings",
    "add_interpretable_ratings_to_params",
]
