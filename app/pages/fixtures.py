# app/pages/fixtures.py

import streamlit as st
import pandas as pd
import altair as alt
from app.components.probability_bar import create_probability_bar


def render(fixtures):
    """Display fixture predictions page"""
    st.markdown(
        '<h2 style="font-size: 1.8rem; text-align: center;">Upcoming Fixture Predictions</h2>',
        unsafe_allow_html=True,
    )

    if fixtures is None or len(fixtures) == 0:
        st.info("No upcoming fixtures available")
        return

    st.markdown(
        '<div class="sub-header">Next matchday predictions with outcome probabilities</div>',
        unsafe_allow_html=True,
    )

    _render_match_cards(fixtures)
    st.subheader("Matchday Overview")
    _render_matchday_chart(fixtures)


def _render_match_cards(fixtures):
    """Render individual match prediction cards"""
    for idx, match in fixtures.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.markdown(f"### {match['home_team']}")
                st.markdown(f"**xG:** {match['expected_goals_home']:.2f}")

            with col2:
                st.markdown("### VS")

            with col3:
                st.markdown(f"### {match['away_team']}")
                st.markdown(f"**xG:** {match['expected_goals_away']:.2f}")

            # probability bars
            st.markdown("**Match Outcome Probabilities:**")
            prob_col1, prob_col2, prob_col3 = st.columns(3)

            with prob_col1:
                st.markdown("**Home Win**")
                st.markdown(
                    create_probability_bar(match["home_win"], "#026E99"),
                    unsafe_allow_html=True,
                )

            with prob_col2:
                st.markdown("**Draw**")
                st.markdown(
                    create_probability_bar(match["draw"], "#FFA600"),
                    unsafe_allow_html=True,
                )

            with prob_col3:
                st.markdown("**Away Win**")
                st.markdown(
                    create_probability_bar(match["away_win"], "#D93649"),
                    unsafe_allow_html=True,
                )

            st.markdown("---")


def _render_matchday_chart(fixtures):
    """Render stacked bar chart of all match probabilities"""
    fixtures_viz = fixtures.copy()
    fixtures_viz["match"] = (
        fixtures_viz["home_team"] + " vs " + fixtures_viz["away_team"]
    )

    # create probability data
    prob_data = []
    for _, row in fixtures_viz.iterrows():
        prob_data.append(
            {
                "match": row["match"],
                "outcome": "Home Win",
                "probability": row["home_win"],
            }
        )
        prob_data.append(
            {"match": row["match"], "outcome": "Draw", "probability": row["draw"]}
        )
        prob_data.append(
            {
                "match": row["match"],
                "outcome": "Away Win",
                "probability": row["away_win"],
            }
        )

    prob_df = pd.DataFrame(prob_data)

    chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "probability:Q",
                title="Probability",
                axis=alt.Axis(format="%"),
                stack="normalize",
            ),
            y=alt.Y("match:N", title=None, sort=None),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=["Home Win", "Draw", "Away Win"],
                    range=["#026E99", "#FFA600", "#D93649"],
                ),
                legend=alt.Legend(title="Outcome"),
            ),
            tooltip=[
                alt.Tooltip("match:N", title="Match"),
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)
