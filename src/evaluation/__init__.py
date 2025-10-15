# src/evaluation/__init__.py

from .metrics import (
    calculate_rps,
    calculate_brier_score,
    calculate_log_loss,
    evaluate_model_comprehensive,
)

from .baselines import (
    evaluate_implied_odds_baseline,
    evaluate_odds_only_model,
)

from .coverage import (
    run_coverage_test,
    test_base_poisson_coverage,
    diagnose_bootstrap_lambda_distribution,
)

from .calibration_plots import (
    plot_calibration_curve,
    plot_calibration_comparison,
    calculate_expected_calibration_error,
    create_calibration_report,
)

__all__ = [
    # metrics
    "calculate_rps",
    "calculate_brier_score",
    "calculate_log_loss",
    "evaluate_model_comprehensive",
    # baselines
    "evaluate_implied_odds_baseline",
    "evaluate_odds_only_model",
    # coverage
    "run_coverage_test",
    "test_base_poisson_coverage",
    "diagnose_bootstrap_lambda_distribution",
    # calibration
    "plot_calibration_curve",
    "plot_calibration_comparison",
    "calculate_expected_calibration_error",
    "create_calibration_report",
]
