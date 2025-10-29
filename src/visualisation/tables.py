# src/visualisation/tables.py

import numpy as np
import pandas as pd
from great_tables import GT, style, loc
from typing import Optional


def format_probability(prob: float) -> str:
    """Format probability as percentage string"""
    if prob >= 0.995:
        return ">99%"
    elif prob <= 0.001:
        return "<0.1%"
    elif prob < 0.01:
        return f"{prob * 100:.1f}%"
    else:
        return f"{prob * 100:.1f}%"


def create_standings_table(
    summary: pd.DataFrame,
    save_path: Optional[str] = None,
    table_width: str = "1200px",
) -> GT:
    """Create formatted league standings table using great_tables"""
    # prepare data
    table_data = summary.copy()

    # select and rename columns for display
    display_data = pd.DataFrame(
        {
            "Team": table_data["team"],
            "Points": table_data["projected_points"].round(0),
            "Goal Difference": table_data["projected_gd"].round(1),
            "Overall": table_data["overall_rating"].round(2),
            "Attack": table_data["attack_rating"].round(2),
            "Defense": table_data["defense_rating"].round(2),
            "Meisterschale": table_data["title_prob"],
            "Champions League": table_data["ucl_prob"],
            "Relegation": table_data["relegation_prob"],
        }
    )

    # create table
    tbl = (
        GT(display_data, rowname_col="Team")
        .tab_spanner(label="Team Ratings", columns=["Overall", "Attack", "Defense"])
        .tab_spanner(
            label="Simulated Probabilities",
            columns=["Meisterschale", "Champions League", "Relegation"],
        )
        .fmt_integer(columns=["Points", "Goal Difference"])
        .fmt_number(columns=["Overall", "Attack", "Defense"], decimals=2)
        .fmt_percent(
            columns=["Meisterschale", "Champions League", "Relegation"],
            decimals=1,
            drop_trailing_zeros=True,
        )
        .cols_align(align="center")
        .tab_style(style=[style.text(weight="bold")], locations=loc.stub())
        .tab_header(
            title="Bundesliga 2025/26 Season Projections",
            subtitle="10,000 Monte Carlo simulations using calibrated Poisson regression with Dixon-Coles correction and parametric bootstrapping.",
        )
        .tab_options(
            table_width=table_width,
            table_font_size="16px",
            data_row_padding="10px",
            column_labels_padding="10px",
            heading_padding="10px",
        )
    )

    if save_path:
        tbl.save(save_path)
        print(f"✓ Table saved: {save_path}")

    return tbl


def create_next_fixtures_table(
    predictions: pd.DataFrame,
    save_path: Optional[str] = None,
    table_width: str = "1000px",
) -> GT:
    """Create formatted table for next fixtures using great_tables"""
    # prepare data
    table_data = predictions.copy()

    # create display dataframe
    display_data = pd.DataFrame(
        {
            "Home Team": table_data["home_team"],
            "Away Team": table_data["away_team"],
            "xG Home": table_data["expected_goals_home"].round(2),
            "xG Away": table_data["expected_goals_away"].round(2),
            "Home Win": table_data["home_win"],
            "Draw": table_data["draw"],
            "Away Win": table_data["away_win"],
        }
    )

    # create table
    tbl = (
        GT(display_data)
        .tab_spanner(label="Expected Goals", columns=["xG Home", "xG Away"])
        .tab_spanner(
            label="Match Outcome Probabilities",
            columns=["Home Win", "Draw", "Away Win"],
        )
        .fmt_number(columns=["xG Home", "xG Away"], decimals=2)
        .fmt_percent(
            columns=["Home Win", "Draw", "Away Win"],
            decimals=1,
            drop_trailing_zeros=True,
        )
        .cols_align(
            align="center",
            columns=[
                "xG Home",
                "xG Away",
                "Home Win",
                "Draw",
                "Away Win",
            ],
        )
        .cols_align(align="left", columns=["Home Team", "Away Team"])
        .tab_header(
            title="Next Bundesliga Matchday Predictions",
            subtitle="Predicted probabilities using Poisson model fitted on match-level performance, squad values, and betting odds.",
        )
        .tab_options(
            table_width=table_width,
            table_font_size="16px",
            data_row_padding="10px",
            column_labels_padding="10px",
            heading_padding="10px",
        )
    )

    if save_path:
        tbl.save(save_path)
        print(f"✓ Fixtures table saved: {save_path}")

    return tbl


def create_comparison_table(
    models_dict: dict,
    save_path: Optional[str] = None,
    table_width: str = "800px",
) -> GT:
    """Create model comparison table using great_tables"""
    # create dataframe from dict
    rows = []
    for model_name, metrics in models_dict.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # identify metric columns (all except 'Model')
    metric_cols = [col for col in df.columns if col != "Model"]

    # create table
    tbl = (
        GT(df, rowname_col="Model")
        .fmt_number(columns=metric_cols, decimals=4)
        .cols_align(align="center", columns=metric_cols)
        .tab_style(style=[style.text(weight="bold")], locations=loc.stub())
        .tab_header(
            title="Model Comparison",
            subtitle="Performance metrics across different modelling approaches.",
        )
        .tab_options(
            table_width=table_width,
            table_font_size="16px",
            data_row_padding="10px",
            column_labels_padding="10px",
            heading_padding="10px",
        )
    )

    if save_path:
        tbl.save(save_path)
        print(f"✓ Comparison table saved: {save_path}")

    return tbl


def create_team_ratings_table(
    params: dict,
    top_n: int = 18,
    save_path: Optional[str] = None,
    table_width: str = "800px",
) -> GT:
    """Create table showing team ratings (attack, defense, overall)"""
    # extract ratings
    teams = params.get("teams", [])
    attack_rating = params.get("attack_rating", {})
    defense_rating = params.get("defense_rating", {})
    overall_rating = params.get("overall_rating", {})

    # create dataframe
    ratings_data = []
    for team in teams:
        ratings_data.append(
            {
                "Team": team,
                "Overall": overall_rating.get(team, np.nan),
                "Attack": attack_rating.get(team, np.nan),
                "Defense": defense_rating.get(team, np.nan),
            }
        )

    df = pd.DataFrame(ratings_data)

    # sort by overall rating
    df = df.sort_values("Overall", ascending=False).head(top_n).reset_index(drop=True)

    # create table
    tbl = (
        GT(df, rowname_col="Team")
        .fmt_number(columns=["Overall", "Attack", "Defense"], decimals=2)
        .cols_align(align="center")
        .tab_style(style=[style.text(weight="bold")], locations=loc.stub())
        .tab_header(
            title="Team Strength Ratings",
            subtitle=f"Top {top_n} teams by overall rating from Dixon-Coles model.",
        )
        .tab_options(
            table_width=table_width,
            table_font_size="16px",
            data_row_padding="10px",
            column_labels_padding="10px",
            heading_padding="10px",
        )
    )

    if save_path:
        tbl.save(save_path)
        print(f"✓ Team ratings table saved: {save_path}")

    return tbl


def create_validation_summary_table(
    validation_results: list,
    save_path: Optional[str] = None,
    table_width: str = "1000px",
) -> GT:
    """Create validation summary table showing per-season metrics"""
    # extract metrics by season
    rows = []
    for result in validation_results:
        season = result.get("season", "Unknown")
        metrics = result.get("metrics", {})

        rows.append(
            {
                "Season": f"{season - 1}/{season}",
                "RPS": metrics.get("rps", np.nan),
                "Brier Score": metrics.get("brier_score", np.nan),
                "Log Loss": metrics.get("log_loss", np.nan),
                "Accuracy": metrics.get("accuracy", np.nan),
            }
        )

    df = pd.DataFrame(rows)

    # create table
    tbl = (
        GT(df)
        .fmt_number(columns=["RPS", "Brier Score", "Log Loss"], decimals=4)
        .fmt_percent(columns=["Accuracy"], decimals=1)
        .cols_align(align="center")
        .cols_align(align="left", columns=["Season"])
        .tab_header(
            title="Model Validation Results",
            subtitle="Performance metrics from backtesting on historical seasons.",
        )
        .tab_options(
            table_width=table_width,
            table_font_size="16px",
            data_row_padding="10px",
            column_labels_padding="10px",
            heading_padding="10px",
        )
    )

    if save_path:
        tbl.save(save_path)
        print(f"✓ Validation summary saved: {save_path}")

    return tbl
