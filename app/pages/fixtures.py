# app/pages/fixtures.py

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from zoneinfo import ZoneInfo
from app.components.probability_bar import create_probability_bar


# kickoff times in database are stored in UTC
SOURCE_TIMEZONE = ZoneInfo("UTC")


def _get_user_timezone() -> ZoneInfo:
    """Get user timezone from session state, fallback to UTC"""
    return st.session_state.get("user_timezone", SOURCE_TIMEZONE)


def _convert_kickoff_time(date_val, time_val, target_tz: ZoneInfo) -> str | None:
    """
    Convert kickoff time from source timezone to target timezone.

    Returns formatted time string (HH:MM TZ) or None if conversion fails.
    """
    if not time_val or pd.isna(time_val):
        return None

    try:
        # parse date
        if isinstance(date_val, str):
            date_obj = datetime.strptime(date_val, "%Y-%m-%d").date()
        else:
            date_obj = pd.to_datetime(date_val).date()

        # parse time (handles both HH:MM and HH:MM:SS)
        time_str = str(time_val)
        if len(time_str.split(":")) == 2:
            time_obj = datetime.strptime(time_str, "%H:%M").time()
        else:
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()

        # combine and localize to source timezone
        dt_source = datetime.combine(date_obj, time_obj)
        dt_source = dt_source.replace(tzinfo=SOURCE_TIMEZONE)

        # convert to target timezone
        dt_target = dt_source.astimezone(target_tz)

        # get timezone abbreviation
        tz_abbrev = dt_target.strftime("%Z")

        return f"{dt_target.strftime('%H:%M')} {tz_abbrev}"
    except Exception:
        # fallback: just strip seconds if present
        time_str = str(time_val)
        if ":" in time_str:
            parts = time_str.split(":")
            return f"{parts[0]}:{parts[1]}"
        return None


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

    user_tz = _get_user_timezone()

    _render_match_cards(next_fixtures, user_tz)
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


def _render_match_cards(fixtures, user_tz: ZoneInfo):
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
                _render_single_match(match, user_tz)
    else:
        for idx, match in fixtures.iterrows():
            _render_single_match(match, user_tz)


def _render_single_match(match, user_tz: ZoneInfo):
    """Render a single match card"""
    with st.container():
        # convert and format kickoff time
        kickoff_display = _convert_kickoff_time(
            match.get("date"), match.get("kickoff_time"), user_tz
        )

        # show kickoff time aligned right if available
        if kickoff_display:
            st.markdown(
                f'<div style="text-align: right; margin-bottom: 0.5rem;"><strong>{kickoff_display}</strong></div>',
                unsafe_allow_html=True,
            )

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
