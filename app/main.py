# app/main.py

import sys
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_js_eval import streamlit_js_eval

from app.components import render_footer, umami_tracker
from app.pages import about, fixtures, projections, team_strengths
from app.styles.custom_css import apply_custom_styles


@dataclass
class SeasonState:
    """Captures whether the current season is complete and who won"""

    is_over: bool
    champion: str | None


def _detect_season_state(proj_data: pd.DataFrame) -> SeasonState:
    """Determine whether the season is complete from the projections parquet"""
    if "matches_played" not in proj_data.columns:
        return SeasonState(is_over=False, champion=None)

    # all 18 teams must have played all 34 matchweeks
    if proj_data["matches_played"].min() < 34:
        return SeasonState(is_over=False, champion=None)

    champion = None
    if "current_points" in proj_data.columns and "current_gd" in proj_data.columns:
        top = proj_data.sort_values(
            ["current_points", "current_gd"], ascending=[False, False]
        ).iloc[0]
        champion = str(top["team"])

    return SeasonState(is_over=True, champion=champion)


# DO Spaces public URL base
DO_SPACES_BASE_URL = "https://ball-bucket.lon1.digitaloceanspaces.com/serving"


def _detect_user_timezone():
    """Detect user timezone from browser and cache in session state"""
    if "user_timezone" in st.session_state:
        return

    try:
        tz_name = streamlit_js_eval(
            js_expressions="Intl.DateTimeFormat().resolvedOptions().timeZone",
            key="user_timezone_js",
        )
        if tz_name:
            st.session_state.user_timezone = ZoneInfo(tz_name)
    except Exception:
        pass


@st.cache_data(ttl="6h")
def load_predictions():
    """Load predictions from DO Spaces (public Parquet files)"""
    try:
        proj = pd.read_parquet(f"{DO_SPACES_BASE_URL}/latest_buli_projections.parquet")
        fix = pd.read_parquet(f"{DO_SPACES_BASE_URL}/latest_buli_matches.parquet")
        return proj, fix
    except Exception as e:
        st.error(f"Failed to load predictions from DO Spaces: {e}")
        return None, None


def main():
    """Main application entry point"""

    # detect timezone early (before visible content) to avoid layout issues
    _detect_user_timezone()

    # inject analytics tracker
    umami_tracker()

    # apply custom styling
    apply_custom_styles()

    # load data
    proj_data, fix_data = load_predictions()

    if proj_data is None:
        st.stop()

    season_state = _detect_season_state(proj_data)

    st.title("predicting-ball")

    if season_state.is_over:
        if season_state.champion:
            st.info(
                f"The Bundesliga season is complete. {season_state.champion} are champions! "
                "Predictions will return for the next season."
            )
        else:
            st.info(
                "The Bundesliga season is complete. Predictions will return for the next season."
            )

    # create tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Season Projections",
            "Team Strengths",
            "Fixture Predictions",
            "About the Model",
        ]
    )

    # render each page in its tab
    with tab1:
        projections.render(proj_data, season_state)

    with tab2:
        team_strengths.render(None, proj_data, season_state)

    with tab3:
        fixtures.render(fix_data, season_state)

    with tab4:
        about.render()

    # add footer
    render_footer()


if __name__ == "__main__":
    main()
