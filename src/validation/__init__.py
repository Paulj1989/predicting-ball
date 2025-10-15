# src/validation/__init__.py

from .splits import (
    TimeSeriesSplit,
    create_train_test_split,
    create_calibration_split,
)

from .backtest import (
    backtest_single_season,
    backtest_multiple_seasons,
    run_rolling_validation,
)

from .diagnostics import (
    analyse_prediction_errors,
    analyse_performance_by_team,
    analyse_performance_by_odds,
    create_validation_report,
)

__all__ = [
    # splits
    "TimeSeriesSplit",
    "create_train_test_split",
    "create_calibration_split",
    # backtesting
    "backtest_single_season",
    "backtest_multiple_seasons",
    "run_rolling_validation",
    # diagnostics
    "analyse_prediction_errors",
    "analyse_performance_by_team",
    "analyse_performance_by_odds",
    "create_validation_report",
]
