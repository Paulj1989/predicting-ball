# src/evaluation/__init__.py

from .baselines import (
    evaluate_implied_odds_baseline,
    evaluate_odds_only_model,
)
from .calibration_plots import (
    calculate_expected_calibration_error,
    create_calibration_report,
    plot_calibration_comparison,
    plot_calibration_curve,
)
from .coverage import (
    diagnose_bootstrap_lambda_distribution,
    run_coverage_test,
    test_base_poisson_coverage,
)
from .metrics import (
    calculate_brier_score,
    calculate_log_loss,
    calculate_rps,
    evaluate_model_comprehensive,
)

__all__ = [
    "calculate_brier_score",
    "calculate_expected_calibration_error",
    "calculate_log_loss",
    # metrics
    "calculate_rps",
    "create_calibration_report",
    "diagnose_bootstrap_lambda_distribution",
    # baselines
    "evaluate_implied_odds_baseline",
    "evaluate_model_comprehensive",
    "evaluate_odds_only_model",
    "plot_calibration_comparison",
    # calibration
    "plot_calibration_curve",
    # coverage
    "run_coverage_test",
    "test_base_poisson_coverage",
]
