import pandas as pd

from app.pages.projections import _build_final_standings_display


def _make_proj(**kwargs):
    defaults = {
        "team": ["Bayern", "Dortmund", "Leverkusen"],
        "current_points": [82, 65, 60],
        "current_gd": [55, 30, 20],
        "matches_played": [34, 34, 34],
        "projected_points": [82, 65, 60],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_final_standings_sorted_by_points_descending():
    result = _build_final_standings_display(_make_proj())
    assert list(result["Team"]) == ["Bayern", "Dortmund", "Leverkusen"]


def test_final_standings_has_expected_columns():
    result = _build_final_standings_display(_make_proj())
    assert set(result.columns) == {"Team", "Points", "GD", "Played"}


def test_final_standings_points_are_integers():
    result = _build_final_standings_display(_make_proj())
    assert result["Points"].dtype in ["int32", "int64"]


def test_final_standings_projected_points_not_included():
    result = _build_final_standings_display(_make_proj())
    assert "projected_points" not in result.columns
    assert "Projected" not in " ".join(result.columns)


def test_final_standings_tiebreak_by_gd():
    df = pd.DataFrame(
        {
            "team": ["A", "B"],
            "current_points": [80, 80],
            "current_gd": [20, 40],
            "matches_played": [34, 34],
        }
    )
    result = _build_final_standings_display(df)
    assert result.iloc[0]["Team"] == "B"


def test_final_standings_missing_optional_columns_handled():
    # only team and current_points present
    df = pd.DataFrame(
        {
            "team": ["Bayern", "Dortmund"],
            "current_points": [82, 65],
        }
    )
    result = _build_final_standings_display(df)
    assert "Team" in result.columns
    assert "Points" in result.columns


def test_final_standings_index_is_one_based():
    result = _build_final_standings_display(_make_proj())
    assert list(result.index) == [1, 2, 3]
