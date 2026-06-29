import pandas as pd

from app.main import _detect_season_state


def _make_proj(matches_played, points, gd=None):
    """Build a minimal projections DataFrame for detection tests."""
    teams = [f"Team{i}" for i in range(len(matches_played))]
    return pd.DataFrame(
        {
            "team": teams,
            "matches_played": matches_played,
            "current_points": points,
            "current_gd": gd if gd is not None else [0] * len(teams),
        }
    )


def test_season_in_progress_not_detected_as_over():
    df = _make_proj([20, 20, 19], [45, 38, 35])
    state = _detect_season_state(df)
    assert not state.is_over
    assert state.champion is None


def test_season_complete_detected():
    df = _make_proj([34, 34, 34], [82, 65, 55])
    state = _detect_season_state(df)
    assert state.is_over
    assert state.champion == "Team0"


def test_season_complete_champion_tiebreak_by_gd():
    # two teams level on points — higher GD wins
    df = _make_proj([34, 34], [80, 80], gd=[30, 45])
    state = _detect_season_state(df)
    assert state.is_over
    assert state.champion == "Team1"


def test_missing_matches_played_column_defaults_to_not_over():
    df = pd.DataFrame({"team": ["Bayern", "Dortmund"]})
    state = _detect_season_state(df)
    assert not state.is_over
    assert state.champion is None


def test_one_team_short_of_34_means_season_not_over():
    df = _make_proj([34, 34, 33], [80, 65, 60])
    state = _detect_season_state(df)
    assert not state.is_over
