# tests/conftest.py
"""Shared pytest fixtures and configuration."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_predictions():
    """Sample predictions DataFrame for testing metrics."""
    return pd.DataFrame(
        {
            "home_win": [0.5, 0.3, 0.6, 0.4, 0.7],
            "draw": [0.25, 0.35, 0.25, 0.3, 0.2],
            "away_win": [0.25, 0.35, 0.15, 0.3, 0.1],
        }
    )


@pytest.fixture
def sample_actuals():
    """Sample actual outcomes for testing metrics."""
    return np.array(["H", "D", "H", "A", "H"])


@pytest.fixture
def sample_match_data():
    """Sample match data for model testing."""
    return pd.DataFrame(
        {
            "home_team": ["Bayern", "Dortmund", "Leipzig"],
            "away_team": ["Dortmund", "Leipzig", "Bayern"],
            "home_goals": [2, 1, 0],
            "away_goals": [1, 1, 2],
            "home_xg": [2.1, 1.3, 0.8],
            "away_xg": [0.9, 1.2, 1.9],
            "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        }
    )


@pytest.fixture
def sample_team_params():
    """Sample team parameters for prediction testing."""
    return {
        "team_attack": {
            "Bayern": 0.3,
            "Dortmund": 0.2,
            "Leipzig": 0.15,
            "Union Berlin": -0.1,
        },
        "team_defense": {
            "Bayern": -0.2,
            "Dortmund": 0.0,
            "Leipzig": 0.05,
            "Union Berlin": 0.1,
        },
        "home_advantage": 0.25,
        "rho": -0.13,
    }
