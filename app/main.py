# app/main.py

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.styles.custom_css import apply_custom_styles
from app.pages import projections, team_strengths, fixtures, about
from app.components import render_footer, umami_tracker

# DO Spaces public URL base
DO_SPACES_BASE_URL = "https://ball-bucket.lon1.digitaloceanspaces.com/serving"


@st.cache_data
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

    # inject analytics tracker
    umami_tracker()

    # apply custom styling
    apply_custom_styles()

    # load data
    proj_data, fix_data = load_predictions()

    if proj_data is None:
        st.stop()

    st.title("predicting-ball")

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
        projections.render(proj_data)

    with tab2:
        team_strengths.render(None, proj_data)

    with tab3:
        fixtures.render(fix_data)

    with tab4:
        about.render()

    # add footer
    render_footer()


if __name__ == "__main__":
    main()
