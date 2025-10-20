# src/utils/season_utils.py

from datetime import datetime


def determine_current_season() -> int:
    """Determine current Bundesliga season based on current date"""
    current_date = datetime.now()
    current_month = current_date.month

    # season runs august to may
    return current_date.year + 1 if current_month >= 8 else current_date.year


def format_season_string(season_end_year: int) -> str:
    """Format season end year as human-readable string"""
    start_year = season_end_year - 1
    return f"{start_year}/{str(season_end_year)[-2:]}"


def convert_to_odds_season_format(season_end_year: int) -> str:
    """Convert season end year to football-data.co.uk season format"""
    start_year = season_end_year - 1
    return f"{str(start_year)[-2:]}{str(season_end_year)[-2:]}"
