# app/pages/fixtures.py

import streamlit as st
import pandas as pd
import altair as alt
from app.components.probability_bar import create_probability_bar


def render(fixtures):
    """Display fixture predictions page - shows next matchday only"""
    st.markdown(
        '<h2 style="font-size: 1.8rem; text-align: center;">Upcoming Fixture Predictions</h2>',
        unsafe_allow_html=True,
    )

    if fixtures is None or len(fixtures) == 0:
        st.info("No upcoming fixtures available")
        return

    # filter to next round only
    if "is_next_round" in fixtures.columns:
        next_fixtures = fixtures[fixtures["is_next_round"] == True].copy()
        if len(next_fixtures) == 0:
            st.info("No upcoming fixtures available")
            return
    else:
        next_fixtures = fixtures.copy()

    # sort fixtures by date, kickoff_time (if available), then home_team
    next_fixtures = _sort_fixtures(next_fixtures)

    st.markdown(
        '<div class="sub-header">Next matchday predictions with outcome probabilities</div>',
        unsafe_allow_html=True,
    )

    _render_match_cards(next_fixtures)
    st.subheader("Matchday Overview")
    _render_matchday_chart(next_fixtures)


def _sort_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Sort fixtures by date, kickoff_time (if available), then home_team.

    Handles missing kickoff_time gracefully.
    """
    df = fixtures.copy()

    # ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # build sort columns list
    sort_cols = []
    if "date" in df.columns:
        sort_cols.append("date")
    if "kickoff_time" in df.columns and df["kickoff_time"].notna().any():
        sort_cols.append("kickoff_time")
    if "home_team" in df.columns:
        sort_cols.append("home_team")

    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    return df.reset_index(drop=True)


def _render_match_cards(fixtures):
    """Render individual match prediction cards"""
    # group by date if available for better visual organization
    if "date" in fixtures.columns:
        fixtures = fixtures.copy()
        fixtures["date_str"] = pd.to_datetime(fixtures["date"]).dt.strftime(
            "%A, %d %B %Y"
        )
        dates = fixtures["date_str"].unique()

        for date_str in dates:
            st.markdown(
                f'<div class="date-header">{date_str}</div>',
                unsafe_allow_html=True,
            )
            date_fixtures = fixtures[fixtures["date_str"] == date_str]
            for idx, match in date_fixtures.iterrows():
                _render_single_match(match)
    else:
        for idx, match in fixtures.iterrows():
            _render_single_match(match)


def _render_single_match(match):
    """Render a single match card"""
    with st.container():
        # show kickoff time if available
        kickoff_time = match.get("kickoff_time")
        if kickoff_time and pd.notna(kickoff_time):
            st.markdown(f"**{kickoff_time}**")

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

    # preserve match order from sorted fixtures
    match_order = fixtures_viz["match"].tolist()

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
            y=alt.Y("match:N", title=None, sort=match_order),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=["Home Win", "Draw", "Away Win"],
                    range=["#026E99", "#FFA600", "#D93649"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("match:N", title="Match"),
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=max(400, len(fixtures) * 40))
    )

    st.altair_chart(chart, use_container_width=True)
