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


@pytest.fixture
def sample_model_params():
    """Full model params dict as returned by fit_poisson_model_two_stage."""
    teams = ["Bayern", "Dortmund", "Frankfurt", "Freiburg", "Leipzig"]
    return {
        "attack": {
            "Bayern": 0.35,
            "Dortmund": 0.20,
            "Frankfurt": 0.05,
            "Freiburg": -0.10,
            "Leipzig": 0.15,
        },
        "defense": {
            "Bayern": -0.25,
            "Dortmund": -0.05,
            "Frankfurt": 0.10,
            "Freiburg": 0.15,
            "Leipzig": -0.10,
        },
        "home_adv": 0.27,
        "rho": -0.13,
        "odds_blend_weight": 0.85,
        "beta_form": 0.08,
        "dispersion_factor": 1.15,
        "teams": teams,
        "success": True,
        "nll": -500.0,
        "attack_rating": {t: 0.0 for t in teams},
        "defense_rating": {t: 0.0 for t in teams},
        "overall_rating": {t: 0.0 for t in teams},
    }


@pytest.fixture
def sample_training_data():
    """Training data DataFrame with required columns for model fitting."""
    np.random.seed(42)
    teams = ["Bayern", "Dortmund", "Frankfurt", "Freiburg", "Leipzig"]
    n_matches = 40
    rows = []
    for i in range(n_matches):
        home = teams[i % len(teams)]
        away = teams[(i + 1 + i // len(teams)) % len(teams)]
        if home == away:
            away = teams[(i + 2) % len(teams)]
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_goals_weighted": home_goals + np.random.normal(0, 0.1),
                "away_goals_weighted": away_goals + np.random.normal(0, 0.1),
                "date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=i * 7),
                "season_end_year": 2025,
                "matchweek": (i // 5) + 1,
                "home_log_odds_ratio": np.random.normal(0.1, 0.3),
                "odds_home_prob": np.random.uniform(0.3, 0.5),
                "odds_draw_prob": np.random.uniform(0.2, 0.35),
                "odds_away_prob": np.random.uniform(0.2, 0.4),
                "home_npxgd_w5": np.random.normal(0.0, 0.5),
                "away_npxgd_w5": np.random.normal(0.0, 0.5),
                "home_elo": np.random.normal(1500, 100),
                "away_elo": np.random.normal(1500, 100),
                "home_value_pct": np.random.uniform(5, 25),
                "away_value_pct": np.random.uniform(5, 25),
                "is_played": True,
                "result": "H"
                if home_goals > away_goals
                else ("A" if away_goals > home_goals else "D"),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_historic_data():
    """Multi-season historical data for priors testing."""
    np.random.seed(123)
    teams_s1 = ["Bayern", "Dortmund", "Frankfurt", "Freiburg", "Leipzig"]
    teams_s2 = ["Bayern", "Dortmund", "Frankfurt", "Freiburg", "Holstein Kiel"]
    rows = []

    for season, teams in [(2024, teams_s1), (2025, teams_s2)]:
        for i in range(20):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            if home == away:
                away = teams[(i + 2) % len(teams)]
            hg = np.random.poisson(1.5)
            ag = np.random.poisson(1.2)
            rows.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_goals_weighted": hg + np.random.normal(0, 0.1),
                    "away_goals_weighted": ag + np.random.normal(0, 0.1),
                    "date": pd.Timestamp(f"{season - 1}-08-01") + pd.Timedelta(days=i * 7),
                    "season_end_year": season,
                    "home_elo": np.random.normal(1500, 100),
                    "away_elo": np.random.normal(1500, 100),
                    "home_value_pct": np.random.uniform(5, 25),
                    "away_value_pct": np.random.uniform(5, 25),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_bootstrap_params(sample_model_params):
    """List of bootstrap parameter dicts simulating bootstrap output."""
    np.random.seed(99)
    params_list = [sample_model_params]
    for _ in range(9):
        p = sample_model_params.copy()
        p["attack"] = {
            t: v + np.random.normal(0, 0.05) for t, v in sample_model_params["attack"].items()
        }
        p["defense"] = {
            t: v + np.random.normal(0, 0.05) for t, v in sample_model_params["defense"].items()
        }
        p["home_adv"] = sample_model_params["home_adv"] + np.random.normal(0, 0.02)
        p["odds_blend_weight"] = np.clip(
            sample_model_params["odds_blend_weight"] + np.random.normal(0, 0.05), 0, 1
        )
        p["beta_form"] = sample_model_params["beta_form"] + np.random.normal(0, 0.02)
        params_list.append(p)
    return params_list
