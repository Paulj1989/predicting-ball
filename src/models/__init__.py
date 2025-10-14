# src/models/__init__.py

from .poisson import (
    fit_poisson_model,
    calculate_lambdas,
)

from .priors import (
    calculate_home_advantage_prior,
    calculate_promoted_team_priors,
    identify_promoted_teams,
)

from .calibration import (
    fit_isotonic_calibrator,
    apply_calibration,
    calibrate_dispersion_for_coverage,
    calibrate_model_comprehensively,
)

from .hyperparameters import (
    optimise_hyperparameters,
    get_default_hyperparameters,
)

__all__ = [
    # core model
    "fit_poisson_model",
    "calculate_lambdas",
    # priors
    "calculate_home_advantage_prior",
    "calculate_promoted_team_priors",
    "identify_promoted_teams",
    # calibration
    "fit_isotonic_calibrator",
    "apply_calibration",
    "calibrate_dispersion_for_coverage",
    "calibrate_model_comprehensively",
    # hyperparameters
    "optimise_hyperparameters",
    "get_default_hyperparameters",
]
