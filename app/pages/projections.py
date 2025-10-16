# app/pages/projections.py

import streamlit as st
import pandas as pd
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
import matplotlib.colors as mcolors


def render(projections):
    """Display the season projections page"""
    st.markdown(
        '<div class="main-header">Bundesliga 2025/26 Season Projections</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Based on 100,000 Monte Carlo simulations</div>',
        unsafe_allow_html=True,
    )

    _render_metrics(projections)
    st.markdown("---")
    _render_standings_table(projections)
    st.markdown("---")
    _render_charts(projections)


def _render_metrics(projections):
    """Render top-level metrics"""
    col1, col2, col3 = st.columns(3)

    top_team = projections.iloc[0]
    with col1:
        st.metric(
            "Title Favorite",
            top_team["team"],
            f"{top_team['title_prob'] * 100:.1f}% probability",
        )

    with col2:
        top_points = projections.iloc[0]["projected_points"]
        st.metric("Projected Winner Points", f"{top_points:.0f}", "points")

    with col3:
        st.metric(
            "Relegation Battle",
            f"{len(projections[projections['relegation_prob'] > 0.05])} teams",
            "with >5% chance",
        )


def _render_standings_table(projections):
    """Render interactive standings table"""
    st.subheader("Projected League Standings")

    # prepare data for display
    display_df = projections[
        [
            "team",
            "projected_points",
            "projected_gd",
            "title_prob",
            "ucl_prob",
            "relegation_prob",
        ]
    ].copy()

    display_df.columns = [
        "Team",
        "Points",
        "Goal Difference",
        "Title %",
        "UCL %",
        "Relegation %",
    ]
    display_df["Points"] = display_df["Points"].round(0).astype(int)
    display_df["Goal Difference"] = display_df["Goal Difference"].round(0).astype(int)
    display_df["Title %"] = (display_df["Title %"] * 100).round(1)
    display_df["UCL %"] = (display_df["UCL %"] * 100).round(1)
    display_df["Relegation %"] = (display_df["Relegation %"] * 100).round(1)

    # create aggrid with custom styling
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(
        groupable=False,
        value=True,
        enableRowGroup=False,
        editable=False,
        filterable=False,
        resizable=False,
    )

    # disable pagination
    gb.configure_pagination(enabled=False)

    # configure columns with flex sizing
    gb.configure_column("Team", pinned="left", flex=2)
    gb.configure_column("Points", flex=1, type=["numericColumn"])
    gb.configure_column(
        "Goal Difference", flex=1, type=["numericColumn"]
    )
    gb.configure_column(
        "Title %", flex=1, type=["numericColumn"]
    )
    gb.configure_column(
        "UCL %", flex=1, type=["numericColumn"]
    )
    gb.configure_column(
        "Relegation %", flex=1, type=["numericColumn"]
    )

    # cell styling for probabilities
    cell_style_jscode = JsCode("""
    function(params) {
        const value = params.value;
        const column = params.colDef.field;

        // Helper to interpolate between white and target color
        function interpolateColor(value, targetColor) {
            const intensity = value / 100;  // 0 to 1
            const rgb = {
                r: parseInt(targetColor.slice(1,3), 16),
                g: parseInt(targetColor.slice(3,5), 16),
                b: parseInt(targetColor.slice(5,7), 16)
            };
            const finalR = Math.round(255 + (rgb.r - 255) * intensity);
            const finalG = Math.round(255 + (rgb.g - 255) * intensity);
            const finalB = Math.round(255 + (rgb.b - 255) * intensity);
            return `rgb(${finalR}, ${finalG}, ${finalB})`;
        }

        if (column === 'Title %') {
            return {'backgroundColor': interpolateColor(value, '#026E99')};
        } else if (column === 'UCL %') {
            return {'backgroundColor': interpolateColor(value, '#FFA600')};
        } else if (column === 'Relegation %') {
            return {'backgroundColor': interpolateColor(value, '#D93649')};
        }

        return {};
    }
    """)

    gb.configure_column("Title %", cellStyle=cell_style_jscode, flex=1)
    gb.configure_column("UCL %", cellStyle=cell_style_jscode, flex=1)
    gb.configure_column("Relegation %", cellStyle=cell_style_jscode, flex=1)

    gridOptions = gb.build()
    gridOptions["domLayout"] = "autoHeight"
    gridOptions["suppressRowHoverHighlight"] = True

    AgGrid(
        display_df,
        gridOptions=gridOptions,
        height=668,
        theme="streamlit",
        update_on=["SELECTION_CHANGED"],
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
    )


def _render_charts(projections):
    """Render title and relegation probability charts"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Meisterschale Race")

        title_contenders = projections.nlargest(8, "title_prob")

        title_chart = (
            alt.Chart(title_contenders)
            .mark_bar()
            .encode(
                x=alt.X(
                    "title_prob:Q",
                    title="Meisterschale Race",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("team:N", title=None, sort="-x"),
                color=alt.Color(
                    "title_prob:Q",
                    scale=alt.Scale(
                        range=["#b3d9e6", "#026E99"]
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("title_prob:Q", title="Probability", format=".2%"),
                    alt.Tooltip(
                        "projected_points:Q", title="Projected Points", format=".0f"
                    ),
                ],
            )
            .properties(height=400)
            .configure_axis(labelFontSize=12, titleFontSize=14)
        )

        st.altair_chart(title_chart, use_container_width=True)

    with col2:
        st.subheader("Relegation Battle")

        relegation_candidates = projections.nlargest(8, "relegation_prob")

        rel_chart = (
            alt.Chart(relegation_candidates)
            .mark_bar(color="#D93649")
            .encode(
                x=alt.X(
                    "relegation_prob:Q",
                    title="Relegation Probability",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("team:N", title=None, sort="-x"),
                opacity=alt.Opacity("relegation_prob:Q", legend=None),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("relegation_prob:Q", title="Probability", format=".2%"),
                    alt.Tooltip(
                        "projected_points:Q", title="Projected Points", format=".0f"
                    ),
                ],
            )
            .properties(height=400)
            .configure_axis(labelFontSize=12, titleFontSize=14)
        )

        st.altair_chart(rel_chart, use_container_width=True)
