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
        '<h2 style="font-size: 1.8rem; text-align: center;">Bundesliga 2025/26 Season Projections</h2>',
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

    # prepare data for display - now including ratings
    display_df = projections[
        [
            "team",
            "projected_points",
            "projected_gd",
            "overall_rating",
            "attack_rating",
            "defense_rating",
            "title_prob",
            "ucl_prob",
            "relegation_prob",
        ]
    ].copy()

    display_df.columns = [
        "Team",
        "Points",
        "Goal Difference",
        "Overall",
        "Attack",
        "Defense",
        "Title %",
        "UCL %",
        "Relegation %",
    ]

    display_df["Points"] = display_df["Points"].round(0).astype(int)
    display_df["Goal Difference"] = display_df["Goal Difference"].round(0).astype(int)
    display_df["Overall"] = display_df["Overall"].round(2)
    display_df["Attack"] = display_df["Attack"].round(2)
    display_df["Defense"] = display_df["Defense"].round(2)
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
    gb.configure_column("Goal Difference", flex=1.2, type=["numericColumn"])
    gb.configure_column("Overall", flex=1, type=["numericColumn"])
    gb.configure_column("Attack", flex=1, type=["numericColumn"])
    gb.configure_column("Defense", flex=1, type=["numericColumn"])
    gb.configure_column("Title %", flex=1, type=["numericColumn"])
    gb.configure_column("UCL %", flex=1, type=["numericColumn"])
    gb.configure_column("Relegation %", flex=1.2, type=["numericColumn"])

    # Enhanced cell styling for probabilities with white text on dark backgrounds
    cell_style_jscode = JsCode("""
    function(params) {
        const value = params.value;
        const column = params.colDef.field;

        // Helper to interpolate between white and target color
        function interpolateColor(value, targetColor, threshold) {
            // Apply shading only if value is above threshold
            if (value <= threshold) {
                return {bg: 'rgb(255, 255, 255)', text: '#000000'};
            }

            // Scale the intensity based on the threshold
            const intensity = Math.min((value - threshold) / (100 - threshold), 1);

            const rgb = {
                r: parseInt(targetColor.slice(1,3), 16),
                g: parseInt(targetColor.slice(3,5), 16),
                b: parseInt(targetColor.slice(5,7), 16)
            };

            const finalR = Math.round(255 + (rgb.r - 255) * intensity);
            const finalG = Math.round(255 + (rgb.g - 255) * intensity);
            const finalB = Math.round(255 + (rgb.b - 255) * intensity);

            // Calculate luminance to determine text color
            const luminance = (0.299 * finalR + 0.587 * finalG + 0.114 * finalB) / 255;
            const textColor = luminance > 0.5 ? '#000000' : '#FFFFFF';

            return {bg: `rgb(${finalR}, ${finalG}, ${finalB})`, text: textColor};
        }

        let colors = null;

        if (column === 'Title %') {
            colors = interpolateColor(value, '#026E99', 1);  // Shade if >1%
        } else if (column === 'UCL %') {
            colors = interpolateColor(value, '#FFA600', 1);  // Shade if >1%
        } else if (column === 'Relegation %') {
            colors = interpolateColor(value, '#D93649', 1);  // Shade if >1%
        }

        if (colors) {
            return {
                'backgroundColor': colors.bg,
                'color': colors.text
            };
        }

        return {};
    }
    """)

    # Apply the cell style to probability columns
    gb.configure_column("Title %", cellStyle=cell_style_jscode)
    gb.configure_column("UCL %", cellStyle=cell_style_jscode)
    gb.configure_column("Relegation %", cellStyle=cell_style_jscode)

    # Add header styling for visual grouping (since AgGrid doesn't support spanners)
    # We can at least style the headers to show grouping
    header_style = {
        "Overall": {"backgroundColor": "#f0f0f0"},
        "Attack": {"backgroundColor": "#f0f0f0"},
        "Defense": {"backgroundColor": "#f0f0f0"},
        "Title %": {"backgroundColor": "#e6f3f7"},
        "UCL %": {"backgroundColor": "#fff4e6"},
        "Relegation %": {"backgroundColor": "#fde8eb"},
    }

    for col, style in header_style.items():
        gb.configure_column(col, headerStyle=style)

    gridOptions = gb.build()
    gridOptions["domLayout"] = "autoHeight"
    gridOptions["suppressRowHoverHighlight"] = True

    st.caption(
        "Columns: **Projections** (Points, Goal Difference) | **Ratings** (Overall, Attack, Defense) | **Probabilities** (Title, UCL, Relegation)"
    )

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
                    title="Probability",
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
                    title="Probability",
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
