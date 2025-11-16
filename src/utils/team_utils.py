# src/utils/team_mapper.py

from typing import Dict, List, Optional
import pandas as pd


# canonical team names (standardised format)
CANONICAL_NAMES = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich Town",
    "Leeds United",
    "Leicester City",
    "Liverpool",
    "Luton Town",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Norwich City",
    "Nottingham Forest",
    "Sheffield United",
    "Southampton",
    "Sunderland",
    "Tottenham",
    "Watford",
    "West Brom",
    "West Ham",
    "Wolves",
}


# comprehensive mapping from all known variations to canonical names
TEAM_NAME_MAPPING: Dict[str, str] = {
    # arsenal
    "Arsenal": "Arsenal",
    "Arsenal FC": "Arsenal",
    # aston villa
    "Aston Villa": "Aston Villa",
    "Aston Villa FC": "Aston Villa",
    # bournemouth
    "Bournemouth": "Bournemouth",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth FC": "Bournemouth",
    # brentford
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",
    # brighton (many variations from fbref/odds)
    "Brighton": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton & HA": "Brighton",
    "Brighton Hove": "Brighton",
    # burnley
    "Burnley": "Burnley",
    "Burnley FC": "Burnley",
    # chelsea
    "Chelsea": "Chelsea",
    "Chelsea FC": "Chelsea",
    # crystal palace
    "Crystal Palace": "Crystal Palace",
    "Crystal Palace FC": "Crystal Palace",
    # everton
    "Everton": "Everton",
    "Everton FC": "Everton",
    # fulham
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",
    # ipswich town
    "Ipswich Town": "Ipswich Town",
    "Ipswich": "Ipswich Town",
    # leeds united
    "Leeds United": "Leeds United",
    "Leeds": "Leeds United",
    "Leeds Utd": "Leeds United",
    # leicester city
    "Leicester City": "Leicester City",
    "Leicester": "Leicester City",
    "Leicester FC": "Leicester City",
    # liverpool
    "Liverpool": "Liverpool",
    "Liverpool FC": "Liverpool",
    # luton town
    "Luton Town": "Luton Town",
    "Luton": "Luton Town",
    # manchester city
    "Manchester City": "Manchester City",
    "Man City": "Manchester City",
    "Manchester City FC": "Manchester City",
    # manchester united (many variations)
    "Manchester United": "Manchester United",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Manchester United FC": "Manchester United",
    # newcastle united
    "Newcastle United": "Newcastle United",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle FC": "Newcastle United",
    # norwich city
    "Norwich City": "Norwich City",
    "Norwich": "Norwich City",
    # nottingham forest (many variations)
    "Nottingham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Nott'ham Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Notts Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    # sheffield united
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Sheffield": "Sheffield United",
    # southampton
    "Southampton": "Southampton",
    "Southampton FC": "Southampton",
    # sunderland
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",
    # tottenham (many variations)
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "Spurs": "Tottenham",
    "Tottenham FC": "Tottenham",
    # watford
    "Watford": "Watford",
    "Watford FC": "Watford",
    # west bromwich albion
    "West Brom": "West Brom",
    "West Bromwich Albion": "West Brom",
    "West Bromwich": "West Brom",
    # west ham united
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
    "West Ham FC": "West Ham",
    # wolverhampton wanderers (many variations)
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
    "Wolverhampton": "Wolves",
    "Wolves FC": "Wolves",
    "Wolverhampton FC": "Wolves",
}


def standardise_team_name(team_name: str) -> str:
    """Convert any team name variation to canonical form"""
    if pd.isna(team_name):
        return team_name

    return TEAM_NAME_MAPPING.get(str(team_name).strip(), str(team_name).strip())


def standardise_dataframe(
    df: pd.DataFrame, team_columns: List[str], inplace: bool = False
) -> pd.DataFrame:
    """Standardise team names in specified columns of a dataframe"""
    if not inplace:
        df = df.copy()

    for col in team_columns:
        if col in df.columns:
            df[col] = df[col].apply(standardise_team_name)

    return df


# fbref team ids (stable across seasons, used in fbref urls)
FBREF_TEAM_IDS = {
    "Arsenal": "18bb7c10",
    "Aston Villa": "8602292d",
    "Bournemouth": "4ba7cbea",
    "Brentford": "cd051869",
    "Brighton": "d07537b9",
    "Burnley": "943e8050",
    "Chelsea": "cff3d9bb",
    "Crystal Palace": "47c64c55",
    "Everton": "d3fd31cc",
    "Fulham": "fd962109",
    "Ipswich Town": "cfb41daa",
    "Leeds United": "5bfb9659",
    "Leicester City": "a2d435b3",
    "Liverpool": "822bd0ba",
    "Luton Town": "e297cd13",
    "Manchester City": "b8fd03ef",
    "Manchester United": "19538871",
    "Newcastle United": "b2b47a98",
    "Norwich City": "1c781004",
    "Nottingham Forest": "e4a775cb",
    "Sheffield United": "1df6b87e",
    "Southampton": "33c895d4",
    "Sunderland": "4dcdcea3",
    "Tottenham": "361ca564",
    "Watford": "2abfe087",
    "West Brom": "60c6b05f",
    "West Ham": "7c21e445",
    "Wolves": "8cec06e1",
}


# fbref url names (team names as they appear in fbref urls)
# maps canonical names to url-safe versions
FBREF_URL_NAMES = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston-Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton-and-Hove-Albion",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal-Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich-Town",
    "Leeds United": "Leeds-United",
    "Leicester City": "Leicester-City",
    "Liverpool": "Liverpool",
    "Luton Town": "Luton-Town",
    "Manchester City": "Manchester-City",
    "Manchester United": "Manchester-United",
    "Newcastle United": "Newcastle-United",
    "Norwich City": "Norwich-City",
    "Nottingham Forest": "Nottingham-Forest",
    "Sheffield United": "Sheffield-United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham-Hotspur",
    "Watford": "Watford",
    "West Brom": "West-Bromwich-Albion",
    "West Ham": "West-Ham-United",
    "Wolves": "Wolverhampton-Wanderers",
}


# season-specific team lists (season_end_year -> list of canonical team names)
PREMIER_LEAGUE_SEASONS = {
    2021: [
        "Manchester City",
        "Manchester United",
        "Liverpool",
        "Chelsea",
        "Leicester City",
        "West Ham",
        "Tottenham",
        "Arsenal",
        "Leeds United",
        "Everton",
        "Aston Villa",
        "Newcastle United",
        "Wolves",
        "Crystal Palace",
        "Southampton",
        "Brighton",
        "Burnley",
        "Fulham",
        "West Brom",
        "Sheffield United",
    ],
    2022: [
        "Manchester City",
        "Liverpool",
        "Chelsea",
        "Tottenham",
        "Arsenal",
        "Manchester United",
        "West Ham",
        "Leicester City",
        "Brighton",
        "Wolves",
        "Newcastle United",
        "Crystal Palace",
        "Brentford",
        "Aston Villa",
        "Southampton",
        "Everton",
        "Leeds United",
        "Burnley",
        "Watford",
        "Norwich City",
    ],
    2023: [
        "Manchester City",
        "Arsenal",
        "Manchester United",
        "Newcastle United",
        "Liverpool",
        "Brighton",
        "Aston Villa",
        "Tottenham",
        "Brentford",
        "Fulham",
        "Crystal Palace",
        "Chelsea",
        "Wolves",
        "West Ham",
        "Bournemouth",
        "Nottingham Forest",
        "Everton",
        "Leicester City",
        "Leeds United",
        "Southampton",
    ],
    2024: [
        "Manchester City",
        "Arsenal",
        "Liverpool",
        "Aston Villa",
        "Tottenham",
        "Chelsea",
        "Newcastle United",
        "Manchester United",
        "West Ham",
        "Crystal Palace",
        "Brighton",
        "Bournemouth",
        "Fulham",
        "Wolves",
        "Everton",
        "Brentford",
        "Nottingham Forest",
        "Luton Town",
        "Burnley",
        "Sheffield United",
    ],
    2025: [
        "Manchester City",
        "Arsenal",
        "Liverpool",
        "Chelsea",
        "Newcastle United",
        "Tottenham",
        "Aston Villa",
        "Manchester United",
        "West Ham",
        "Brighton",
        "Bournemouth",
        "Fulham",
        "Wolves",
        "Crystal Palace",
        "Everton",
        "Brentford",
        "Nottingham Forest",
        "Ipswich Town",
        "Southampton",
        "Leicester City",
    ],
    2026: [
        "Manchester City",
        "Arsenal",
        "Liverpool",
        "Chelsea",
        "Newcastle United",
        "Tottenham",
        "Aston Villa",
        "Manchester United",
        "West Ham",
        "Brighton",
        "Bournemouth",
        "Fulham",
        "Wolves",
        "Crystal Palace",
        "Everton",
        "Brentford",
        "Nottingham Forest",
        "Leeds United",
        "Burnley",
        "Sunderland",
    ],
}


def get_fbref_team_id(team_name: str) -> Optional[str]:
    """Get the FBref team ID for a given team name"""
    # standardise the team name first
    canonical_name = standardise_team_name(team_name)
    return FBREF_TEAM_IDS.get(canonical_name)


def get_fbref_url_name(team_name: str) -> str:
    """Get the correct team name for FBref URLs"""
    # standardise the team name first
    canonical_name = standardise_team_name(team_name)

    # return the fbref url version if it exists
    if canonical_name in FBREF_URL_NAMES:
        return FBREF_URL_NAMES[canonical_name]

    # fallback: just replace spaces with dashes
    return canonical_name.replace(" ", "-")


def get_premier_league_teams_for_season(season_end_year: int) -> List[str]:
    """Get the list of teams in the Premier League for a specific season"""
    if season_end_year in PREMIER_LEAGUE_SEASONS:
        return PREMIER_LEAGUE_SEASONS[season_end_year].copy()

    # fallback: return all teams with fbref ids
    return list(FBREF_TEAM_IDS.keys())


def get_fbref_team_ids_for_season(season_end_year: int) -> Dict[str, str]:
    """Get team ID mappings for a specific Premier League season"""
    teams = get_premier_league_teams_for_season(season_end_year)
    return {team: FBREF_TEAM_IDS[team] for team in teams if team in FBREF_TEAM_IDS}
