# src/models/__init__.py

from .poisson import (
    fit_baseline_strengths,
    fit_feature_coefficients,
    fit_poisson_model_two_stage,
    calculate_lambdas,
    calculate_lambdas_single
)

from .priors import (
    calculate_home_advantage_prior,
    calculate_promoted_team_priors,
    identify_promoted_teams,
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
    # calibration (temperature scaling)
    "fit_temperature_scaler",
    "apply_temperature_scaling",
    "calibrate_dispersion_for_coverage",
    "calibrate_model_comprehensively",
    # hyperparameters
    "optimise_hyperparameters",
    "get_default_hyperparameters",
]
