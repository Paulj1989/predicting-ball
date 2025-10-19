# src/simulation/__init__.py

from .bootstrap import (
    parametric_bootstrap_with_residuals,
    plot_parameter_diagnostics,
)

from .monte_carlo import (
    simulate_remaining_season_calibrated,
    get_current_standings,
    create_final_summary,
)

from .sampling import (
    sample_goals_calibrated,
    sample_match_outcome,
)

from .predictions import (
    predict_next_fixtures,
    predict_single_match,
    predict_match_probabilities,
    get_next_round_fixtures,
)

__all__ = [
    # bootstrap
    "parametric_bootstrap_with_residuals",
    "plot_parameter_diagnostics",
    # monte carlo
    "simulate_remaining_season_calibrated",
    "get_current_standings",
    "create_final_summary",
    # sampling
    "sample_goals_calibrated",
    "sample_match_outcome",
    # predictions
    "predict_next_fixtures",
    "predict_single_match",
    "predict_match_probabilities",
    "get_next_round_fixtures",
]
