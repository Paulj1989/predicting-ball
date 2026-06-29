# app/pages/projections.py

import altair as alt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode


def _build_final_standings_display(projections: pd.DataFrame) -> pd.DataFrame:
    """Build a sorted display DataFrame for the end-of-season final standings table"""
    cols = ["team", "current_points", "current_gd", "matches_played"]
    available = [c for c in cols if c in projections.columns]
    df = projections[available].copy()

    sort_cols = [c for c in ["current_points", "current_gd"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    # 1-based index so the dataframe shows positions 1-18
    df.index = range(1, len(df) + 1)

    rename = {
        "team": "Team",
        "current_points": "Points",
        "current_gd": "GD",
        "matches_played": "Played",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "Points" in df.columns:
        df["Points"] = df["Points"].round(0).astype(int)
    if "GD" in df.columns:
        df["GD"] = df["GD"].round(0).astype(int)

    return df


def _render_final_metrics(projections, season_state):
    """Render end-of-season summary metrics"""
    standings = projections.copy()
    sort_cols = [c for c in ["current_points", "current_gd"] if c in standings.columns]
    if sort_cols:
        standings = standings.sort_values(sort_cols, ascending=False)

    winner = standings.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Champions", season_state.champion or str(winner["team"]))

    with col2:
        final_pts = int(winner["current_points"]) if "current_points" in winner else "—"
        st.metric("Final Points", final_pts, "points")

    with col3:
        # bottom 3: 2 auto-relegated + 1 relegation playoff
        st.metric("Relegation Zone", "3 teams", "bottom 3")


def _render_final_standings_table(projections):
    """Render the actual final league table using current standings"""
    st.subheader("Final League Table")
    display_df = _build_final_standings_display(projections)
    st.dataframe(display_df, use_container_width=True)


def render(projections, season_state=None):
    """Display the season projections page"""
    if season_state is not None and season_state.is_over:
        st.markdown(
            '<h2 style="font-size: 1.8rem; text-align: center;">Bundesliga Final Standings</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Final league positions</div>',
            unsafe_allow_html=True,
        )
        _render_final_metrics(projections, season_state)
        st.markdown("---")
        _render_final_standings_table(projections)
        return

    st.markdown(
        '<h2 style="font-size: 1.8rem; text-align: center;">Bundesliga 2025/26 Season Projections</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Based on 10,000 Monte Carlo simulations</div>',
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

    is_mobile = False

    try:
        from streamlit_js_eval import streamlit_js_eval

        # get screen width using javascript
        screen_width = streamlit_js_eval(
            js_expressions="window.innerWidth",
            key="screen_width_" + str(hash(str(projections))),
        )

        if screen_width is not None:
            is_mobile = screen_width < 768
        else:
            is_mobile = False

    except ImportError:
        # alternative detection using container test
        _test_col1, _test_col2, _test_col3, _test_col4, _test_col5 = st.columns(5)
        is_mobile = False

    # select columns based on view mode
    if is_mobile:
        display_df = projections[
            [
                "team",
                "projected_points",
                "title_prob",
                "ucl_prob",
                "relegation_prob",
            ]
        ].copy()

        display_df.columns = [
            "Team",
            "Pts",
            "Title %",
            "UCL %",
            "Rel %",
        ]
    else:
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
            "GD",
            "Overall",
            "Attack",
            "Defense",
            "Title %",
            "UCL %",
            "Relegation %",
        ]

    # format columns
    if is_mobile:
        display_df["Pts"] = display_df["Pts"].round(0).astype(int)
    else:
        display_df["Points"] = display_df["Points"].round(0).astype(int)
        display_df["GD"] = display_df["GD"].round(0).astype(int)
        display_df["Overall"] = display_df["Overall"].round(2)
        display_df["Attack"] = display_df["Attack"].round(2)
        display_df["Defense"] = display_df["Defense"].round(2)

    display_df["Title %"] = (display_df["Title %"] * 100).round(1)
    display_df["UCL %"] = (display_df["UCL %"] * 100).round(1)
    display_df[display_df.columns[-1]] = (display_df[display_df.columns[-1]] * 100).round(1)

    # create aggrid with custom styling
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(
        enableRowGroup=False,
        editable=False,
        filter=False,
        resizable=False,
    )

    # disable pagination
    gb.configure_pagination(enabled=False)

    # configure columns based on view mode
    if is_mobile:
        gb.configure_column("Team", pinned="left", flex=2, maxWidth=120)
        gb.configure_column("Pts", flex=1, maxWidth=60, type=["numericColumn"])
        gb.configure_column("Title %", flex=1, maxWidth=70, type=["numericColumn"])
        gb.configure_column("UCL %", flex=1, maxWidth=70, type=["numericColumn"])
        gb.configure_column("Rel %", flex=1, maxWidth=70, type=["numericColumn"])
    else:
        gb.configure_column("Team", pinned="left", flex=3, minWidth=165)
        gb.configure_column("Points", flex=1, type=["numericColumn"])
        gb.configure_column("GD", flex=0.8, type=["numericColumn"])
        gb.configure_column("Overall", flex=1, type=["numericColumn"])
        gb.configure_column("Attack", flex=1, type=["numericColumn"])
        gb.configure_column("Defense", flex=1, type=["numericColumn"])
        gb.configure_column("Title %", flex=1, type=["numericColumn"])
        gb.configure_column("UCL %", flex=1, type=["numericColumn"])
        gb.configure_column("Relegation %", flex=1.5, minWidth=100, type=["numericColumn"])

    # enhanced cell styling for probabilities
    cell_style_jscode = JsCode("""
    function(params) {
        const value = params.value;
        const column = params.colDef.field;

        function interpolateColor(value, targetColor, threshold) {
            if (value <= threshold) {
                return {bg: 'rgb(255, 255, 255)', text: '#000000'};
            }

            const intensity = Math.min((value - threshold) / (100 - threshold), 1);

            const rgb = {
                r: parseInt(targetColor.slice(1,3), 16),
                g: parseInt(targetColor.slice(3,5), 16),
                b: parseInt(targetColor.slice(5,7), 16)
            };

            const finalR = Math.round(255 + (rgb.r - 255) * intensity);
            const finalG = Math.round(255 + (rgb.g - 255) * intensity);
            const finalB = Math.round(255 + (rgb.b - 255) * intensity);

            const luminance = (0.299 * finalR + 0.587 * finalG + 0.114 * finalB) / 255;
            const textColor = luminance > 0.5 ? '#000000' : '#FFFFFF';

            return {bg: `rgb(${finalR}, ${finalG}, ${finalB})`, text: textColor};
        }

        let colors = null;

        if (column === 'Title %') {
            colors = interpolateColor(value, '#026E99', 1);
        } else if (column === 'UCL %') {
            colors = interpolateColor(value, '#FFA600', 1);
        } else if (column === 'Relegation %' || column === 'Rel %') {
            colors = interpolateColor(value, '#D93649', 1);
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

    # apply the cell style to probability columns
    gb.configure_column("Title %", cellStyle=cell_style_jscode)
    gb.configure_column("UCL %", cellStyle=cell_style_jscode)
    if is_mobile:
        gb.configure_column("Rel %", cellStyle=cell_style_jscode)
    else:
        gb.configure_column("Relegation %", cellStyle=cell_style_jscode)

        # add header styling for visual grouping on desktop
        header_style = {
            "Overall": {"backgroundColor": "#f0f0f0"},
            "Attack": {"backgroundColor": "#f0f0f0"},
            "Defense": {"backgroundColor": "#f0f0f0"},
            "Title %": {"backgroundColor": "#e6f3f7"},
            "UCL %": {"backgroundColor": "#fff4e6"},
            "Relegation %": {"backgroundColor": "#fde8eb"},
        }

        for col, style in header_style.items():
            if col in display_df.columns:
                gb.configure_column(col, headerStyle=style)

    gridOptions = gb.build()
    gridOptions["suppressRowHoverHighlight"] = True

    if is_mobile:
        st.caption(
            "**Columns:** Projections (Points) | Probabilities (Title, UCL, Relegation)"
        )
    else:
        st.caption(
            "**Columns:** Projections (Points, GD) | Ratings (Overall, Attack, Defense) | Probabilities (Title, UCL, Relegation)"
        )

    AgGrid(
        display_df,
        gridOptions=gridOptions,
        height=None,  # type: ignore[arg-type]  # library accepts None to enable autoHeight
        theme="streamlit",
        update_on=["SELECTION_CHANGED"],
        allow_unsafe_jscode=True,
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
                    stack=None,
                ),
                y=alt.Y("team:N", title=None, sort="-x"),
                color=alt.Color(
                    "title_prob:Q",
                    scale=alt.Scale(range=["#b3d9e6", "#026E99"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("title_prob:Q", title="Probability", format=".2%"),
                    alt.Tooltip("projected_points:Q", title="Projected Points", format=".0f"),
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
            .mark_bar()
            .encode(
                x=alt.X(
                    "relegation_prob:Q",
                    title="Probability",
                    axis=alt.Axis(format="%"),
                    stack=None,
                ),
                y=alt.Y("team:N", title=None, sort="-x"),
                color=alt.Color(
                    "relegation_prob:Q",
                    scale=alt.Scale(range=["#f9b4bc", "#D93649"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("relegation_prob:Q", title="Probability", format=".2%"),
                    alt.Tooltip("projected_points:Q", title="Projected Points", format=".0f"),
                ],
            )
            .properties(height=400)
            .configure_axis(labelFontSize=12, titleFontSize=14)
        )

        st.altair_chart(rel_chart, use_container_width=True)
