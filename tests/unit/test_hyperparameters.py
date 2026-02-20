# tests/unit/test_hyperparameters.py
"""Tests for hyperparameter module."""

import pytest

from src.models.hyperparameters import get_default_hyperparameters, optimise_hyperparameters


class TestGetDefaultHyperparameters:
    """Tests for default hyperparameter values."""

    def test_returns_dict(self):
        """Should return a dict."""
        result = get_default_hyperparameters()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Should contain all required hyperparameter keys."""
        result = get_default_hyperparameters()
        assert "time_decay" in result
        assert "lambda_reg" in result
        assert "prior_decay_rate" in result
        # rho is fitted via MLE in stage 1, not a hyperparameter
        assert "rho" not in result

    def test_time_decay_positive(self):
        """Time decay should be a small positive number."""
        result = get_default_hyperparameters()
        assert 0 < result["time_decay"] < 0.1

    def test_lambda_reg_positive(self):
        """Regularisation weight should be positive."""
        result = get_default_hyperparameters()
        assert result["lambda_reg"] > 0

    def test_prior_decay_rate_positive(self):
        """Prior decay rate default should be a positive number."""
        result = get_default_hyperparameters()
        assert result["prior_decay_rate"] > 0

    def test_values_are_floats(self):
        """All values should be floats."""
        result = get_default_hyperparameters()
        for v in result.values():
            assert isinstance(v, float)

    def test_has_xg_weight(self):
        """Default hyperparameters should include xg_weight."""
        result = get_default_hyperparameters()
        assert "xg_weight" in result

    def test_xg_weight_in_range(self):
        """Default xg_weight should be between 0.5 and 1.0."""
        result = get_default_hyperparameters()
        assert 0.5 <= result["xg_weight"] <= 1.0


class TestOptimiseHyperparameters:
    """Smoke tests for hyperparameter optimisation."""

    @pytest.fixture
    def large_training_data(self):
        """Larger dataset needed for time-series CV (test_size=153 per fold)."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        teams = [
            "Bayern",
            "Dortmund",
            "Frankfurt",
            "Freiburg",
            "Leipzig",
            "Leverkusen",
            "Gladbach",
            "Wolfsburg",
            "Stuttgart",
            "Mainz",
        ]
        n_matches = 1000
        rows = []
        for i in range(n_matches):
            home = teams[i % len(teams)]
            away = teams[(i + 1 + i // len(teams)) % len(teams)]
            if home == away:
                away = teams[(i + 2) % len(teams)]
            hg = np.random.poisson(1.5)
            ag = np.random.poisson(1.2)
            d = pd.Timestamp("2022-08-01") + pd.Timedelta(days=i)
            # season ends in the year the second half of the season falls in
            season_end_year = d.year + 1 if d.month >= 7 else d.year
            rows.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_goals_weighted": hg + np.random.normal(0, 0.1),
                    "away_goals_weighted": ag + np.random.normal(0, 0.1),
                    "date": d,
                    "season_end_year": season_end_year,
                    "home_npxgd_w5": np.random.normal(0.0, 0.5),
                    "away_npxgd_w5": np.random.normal(0.0, 0.5),
                    "odds_home_prob": np.random.uniform(0.3, 0.5),
                    "odds_draw_prob": np.random.uniform(0.2, 0.35),
                    "odds_away_prob": np.random.uniform(0.2, 0.4),
                    "home_npxg": np.random.uniform(0.5, 2.5),
                    "away_npxg": np.random.uniform(0.3, 2.0),
                    "home_npg": hg,
                    "away_npg": ag,
                    "result": "H" if hg > ag else ("A" if ag > hg else "D"),
                }
            )
        return pd.DataFrame(rows)

    @pytest.mark.slow
    def test_returns_dict_with_expected_keys(self, large_training_data):
        """Should return dict with hyperparameter keys (2 trials only)."""
        result = optimise_hyperparameters(
            large_training_data, n_trials=2, n_jobs=1, verbose=False
        )
        assert isinstance(result, dict)
        assert "time_decay" in result
        assert "lambda_reg" in result
        assert "prior_decay_rate" in result
        # rho is fitted by MLE, not returned as a hyperparameter
        assert "rho" not in result

    @pytest.mark.slow
    def test_values_in_search_bounds(self, large_training_data):
        """Returned values should be within search bounds."""
        result = optimise_hyperparameters(
            large_training_data, n_trials=2, n_jobs=1, verbose=False
        )
        assert 0.001 <= result["time_decay"] <= 0.005
        assert 0.1 <= result["lambda_reg"] <= 0.5
        assert 5.0 <= result["prior_decay_rate"] <= 15.0

    @pytest.mark.slow
    def test_verbose_output(self, large_training_data):
        """Verbose mode should print optimisation info."""
        result = optimise_hyperparameters(
            large_training_data, n_trials=2, n_jobs=1, verbose=True
        )
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_xg_weight_returned(self, large_training_data):
        """Optimised params should include xg_weight."""
        result = optimise_hyperparameters(
            large_training_data, n_trials=2, n_jobs=1, verbose=False
        )
        assert "xg_weight" in result
        assert 0.5 <= result["xg_weight"] <= 1.0
