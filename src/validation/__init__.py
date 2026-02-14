# src/validation/__init__.py

from .backtest import (
    backtest_multiple_seasons,
    backtest_single_season,
    run_rolling_validation,
)
from .diagnostics import (
    analyse_performance_by_odds,
    analyse_performance_by_team,
    analyse_prediction_errors,
    create_validation_report,
)
from .splits import (
    TimeSeriesSplit,
    create_calibration_split,
    create_train_test_split,
)

__all__ = [
    # splits
    "TimeSeriesSplit",
    "analyse_performance_by_odds",
    "analyse_performance_by_team",
    # diagnostics
    "analyse_prediction_errors",
    "backtest_multiple_seasons",
    # backtesting
    "backtest_single_season",
    "create_calibration_split",
    "create_train_test_split",
    "create_validation_report",
    "run_rolling_validation",
]
