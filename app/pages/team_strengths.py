# app/pages/team_strengths.py

import streamlit as st
import pandas as pd
import altair as alt


def render(model, projections):
    """Display team analysis page with rating visualisations"""
    st.markdown(
        '<div class="main-header">Team Strength Ratings</div>', unsafe_allow_html=True
    )

    if model is None:
        st.error("Model data not available")
        return

    params = model["params"]

    # team selector
    teams = sorted(projections["team"].unique())
    selected_team = st.selectbox("Select a team:", teams)

    _render_team_metrics(selected_team, projections)
    st.markdown("---")
    _render_comparison_charts(selected_team, teams, params)
    st.markdown("---")
    _render_top_performers(teams, params)


def _render_team_metrics(selected_team, projections):
    """Render team overview metrics"""
    st.subheader(f"{selected_team} - Team Metrics")

    col1, col2, col3, col4 = st.columns(4)

    team_data = projections[projections["team"] == selected_team].iloc[0]

    with col1:
        st.metric("Overall Rating", f"{team_data['overall_rating']:.2f}")
    with col2:
        st.metric("Attack Rating", f"{team_data['attack_rating']:.2f}")
    with col3:
        st.metric("Defense Rating", f"{team_data['defense_rating']:.2f}")
    with col4:
        st.metric("Projected Points", f"{team_data['projected_points']:.0f}")


def _render_comparison_charts(selected_team, teams, params):
    """Render league-wide comparison charts"""
    col1, col2 = st.columns(2)

    # prepare ratings data
    ratings_df = pd.DataFrame(
        {
            "team": teams,
            "attack": [params["attack"].get(t, 0) for t in teams],
            "defense": [params["defense_scaled"].get(t, 0) for t in teams],
            "overall": [params["overall"].get(t, 0) for t in teams],
        }
    )
    ratings_df["is_selected"] = ratings_df["team"] == selected_team

    with col1:
        st.subheader("Attack vs Defense Ratings")

        scatter = (
            alt.Chart(ratings_df)
            .mark_circle(size=200)
            .encode(
                x=alt.X(
                    "attack:Q",
                    title="Attack Rating",
                    scale=alt.Scale(domain=[-0.8, 0.8]),
                ),
                y=alt.Y(
                    "defense:Q",
                    title="Defense Rating",
                    scale=alt.Scale(domain=[-0.8, 0.8]),
                ),
                color=alt.condition(
                    alt.datum.is_selected == True,
                    alt.value("#D93649"),
                    alt.value("#026E99"),
                ),
                opacity=alt.condition(
                    alt.datum.is_selected == True, alt.value(1.0), alt.value(0.6)
                ),
                size=alt.condition(
                    alt.datum.is_selected == True, alt.value(400), alt.value(200)
                ),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("attack:Q", title="Attack", format=".3f"),
                    alt.Tooltip("defense:Q", title="Defense", format=".3f"),
                    alt.Tooltip("overall:Q", title="Overall", format=".3f"),
                ],
            )
            .properties(height=450)
        )

        # add quadrant lines
        h_line = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(strokeDash=[5, 5])
            .encode(y="y")
        )
        v_line = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(strokeDash=[5, 5])
            .encode(x="x")
        )

        st.altair_chart(scatter + h_line + v_line, use_container_width=True)

    with col2:
        st.subheader("Overall Ratings Distribution")

        ratings_sorted = ratings_df.sort_values("overall", ascending=False)

        bars = (
            alt.Chart(ratings_sorted)
            .mark_bar()
            .encode(
                x=alt.X("overall:Q", title="Overall Rating"),
                y=alt.Y("team:N", title=None, sort="-x"),
                color=alt.condition(
                    alt.datum.is_selected == True,
                    alt.value("#D93649"),
                    alt.value("#026E99"),
                ),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("overall:Q", title="Overall Rating", format=".3f"),
                ],
            )
            .properties(height=450)
        )

        st.altair_chart(bars, use_container_width=True)


def _render_top_performers(teams, params):
    """Render top performers table"""
    st.subheader("Top Performers by Category")

    ratings_df = pd.DataFrame(
        {
            "team": teams,
            "attack": [params["attack"].get(t, 0) for t in teams],
            "defense": [params["defense_scaled"].get(t, 0) for t in teams],
            "overall": [params["overall"].get(t, 0) for t in teams],
        }
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Best Attack**")
        top_attack = ratings_df.nlargest(5, "attack")[["team", "attack"]]
        top_attack.columns = ["Team", "Rating"]
        st.dataframe(top_attack, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Best Defense**")
        top_defense = ratings_df.nlargest(5, "defense")[["team", "defense"]]
        top_defense.columns = ["Team", "Rating"]
        st.dataframe(top_defense, hide_index=True, use_container_width=True)

    with col3:
        st.markdown("**Best Overall**")
        top_overall = ratings_df.nlargest(5, "overall")[["team", "overall"]]
        top_overall.columns = ["Team", "Rating"]
        st.dataframe(top_overall, hide_index=True, use_container_width=True)
