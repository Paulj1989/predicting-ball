# app/main.py

import streamlit as st

# page configuration
st.set_page_config(
    page_title="Predicting Ball",
    page_icon="⚽",
    layout="wide",
)

# noqa: E402 (module level import not at top of file)
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.styles.custom_css import apply_custom_styles  # noqa: E402
from app.pages import projections, team_strengths, fixtures, about  # noqa: E402
from app.components import footer  # noqa: E402
from src.io.model_io import load_model  # noqa: E402


@st.cache_data
def load_predictions():
    """Load pre-generated predictions from CSV files"""
    try:
        proj = pd.read_csv("outputs/predictions/season_projections.csv")
        fix = pd.read_csv("outputs/predictions/next_fixtures.csv")
        return proj, fix
    except FileNotFoundError:
        st.error(
            "Prediction files not found. Please run generate_predictions.py first."
        )
        return None, None


@st.cache_resource
def load_trained_model():
    """Load the trained model for team ratings"""
    try:
        model = load_model("outputs/models/production_model.pkl")
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Some visualisations may be unavailable.")
        return None


def main():
    """Main application entry point"""
    # apply custom styling
    apply_custom_styles()

    # load data
    proj_data, fix_data = load_predictions()
    model = load_trained_model()

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
        team_strengths.render(model, proj_data)

    with tab3:
        fixtures.render(fix_data)

    with tab4:
        about.render()

    # add footer
    footer.render()


if __name__ == "__main__":
    main()
