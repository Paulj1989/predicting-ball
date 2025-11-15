# src/utils/team_mapper.py

from typing import Dict, List, Optional
import pandas as pd


# canonical team names (standardised format)
CANONICAL_NAMES = {
    "Bayern Munich",
    "Borussia Dortmund",
    "RB Leipzig",
    "Bayer Leverkusen",
    "Borussia Mönchengladbach",
    "Eintracht Frankfurt",
    "Union Berlin",
    "Wolfsburg",
    "Stuttgart",
    "Werder Bremen",
    "Hoffenheim",
    "Mainz",
    "Augsburg",
    "Freiburg",
    "FC Köln",
    "Schalke 04",
    "Bochum",
    "Arminia Bielefeld",
    "Greuther Fürth",
    "Holstein Kiel",
    "Heidenheim",
    "St. Pauli",
    "Darmstadt",
    "Hertha BSC",
    "FC Nürnberg",
    "Hannover 96",
    "Hamburger SV",
    "SC Paderborn 07",
    "Fortuna Düsseldorf",
}


# comprehensive mapping from all known variations to canonical names
TEAM_NAME_MAPPING: Dict[str, str] = {
    "Bayern": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "Bayern Munchen": "Bayern Munich",
    "FC Bayern München": "Bayern Munich",
    "FC Bayern Munich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund",
    "Borussia Dortmund": "Borussia Dortmund",
    "BVB": "Borussia Dortmund",
    "Leipzig": "RB Leipzig",
    "RB Leipzig": "RB Leipzig",
    "Leverkusen": "Bayer Leverkusen",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "Bayer 04": "Bayer Leverkusen",
    "M'Gladbach": "Borussia Mönchengladbach",
    "Mönchengladbach": "Borussia Mönchengladbach",
    "Moenchengladbach": "Borussia Mönchengladbach",
    "Monchengladbach": "Borussia Mönchengladbach",
    "Gladbach": "Borussia Mönchengladbach",
    "M'gladbach": "Borussia Mönchengladbach",
    "Bor. Mönchengladbach": "Borussia Mönchengladbach",
    "Borussia Mönchengladbach": "Borussia Mönchengladbach",
    "Borussia Monchengladbach": "Borussia Mönchengladbach",
    "Borussia M'gladbach": "Borussia Mönchengladbach",
    "Frankfurt": "Eintracht Frankfurt",
    "Eint Frankfurt": "Eintracht Frankfurt",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Union Berlin": "Union Berlin",
    "1. FC Union Berlin": "Union Berlin",
    "FC Union Berlin": "Union Berlin",
    "1.FC Union Berlin": "Union Berlin",
    "Wolfsburg": "Wolfsburg",
    "VfL Wolfsburg": "Wolfsburg",
    "Stuttgart": "Stuttgart",
    "VfB Stuttgart": "Stuttgart",
    "Bremen": "Werder Bremen",
    "Werder Bremen": "Werder Bremen",
    "SV Werder Bremen": "Werder Bremen",
    "Werder": "Werder Bremen",
    "Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1899 Hoffenheim": "Hoffenheim",
    "Mainz": "Mainz",
    "Mainz 05": "Mainz",
    "FSV Mainz 05": "Mainz",
    "1. FSV Mainz 05": "Mainz",
    "1.FSV Mainz 05": "Mainz",
    "Augsburg": "Augsburg",
    "FC Augsburg": "Augsburg",
    "Freiburg": "Freiburg",
    "SC Freiburg": "Freiburg",
    "Köln": "FC Köln",
    "Koeln": "FC Köln",
    "Koln": "FC Köln",
    "FC Köln": "FC Köln",
    "FC Koln": "FC Köln",
    "1. FC Köln": "FC Köln",
    "1.FC Köln": "FC Köln",
    "Cologne": "FC Köln",
    "Schalke": "Schalke 04",
    "Schalke 04": "Schalke 04",
    "FC Schalke 04": "Schalke 04",
    "Bochum": "Bochum",
    "VfL Bochum": "Bochum",
    "Arminia": "Arminia Bielefeld",
    "Bielefeld": "Arminia Bielefeld",
    "Arminia Bielefeld": "Arminia Bielefeld",
    "DSC Arminia Bielefeld": "Arminia Bielefeld",
    "Fürth": "Greuther Fürth",
    "Furth": "Greuther Fürth",
    "Fuerth": "Greuther Fürth",
    "Greuther Fürth": "Greuther Fürth",
    "Greuther Furth": "Greuther Fürth",
    "SpVgg Greuther Fürth": "Greuther Fürth",
    "Kiel": "Holstein Kiel",
    "Holstein": "Holstein Kiel",
    "Holstein Kiel": "Holstein Kiel",
    "Heidenheim": "Heidenheim",
    "1. FC Heidenheim": "Heidenheim",
    "1.FC Heidenheim": "Heidenheim",
    "1.FC Heidenheim 1846": "Heidenheim",
    "FC Heidenheim": "Heidenheim",
    "St. Pauli": "St. Pauli",
    "St Pauli": "St. Pauli",
    "FC St. Pauli": "St. Pauli",
    "Darmstadt": "Darmstadt",
    "Darmstadt 98": "Darmstadt",
    "SV Darmstadt 98": "Darmstadt",
    "Hertha": "Hertha BSC",
    "Hertha BSC": "Hertha BSC",
    "Hertha Berlin": "Hertha BSC",
    "Nürnberg": "FC Nürnberg",
    "Nurnberg": "FC Nürnberg",
    "Nuernberg": "FC Nürnberg",
    "1. FC Nürnberg": "FC Nürnberg",
    "1.FC Nürnberg": "FC Nürnberg",
    "FC Nürnberg": "FC Nürnberg",
    "Hannover": "Hannover 96",
    "Hannover 96": "Hannover 96",
    "Hamburg": "Hamburger SV",
    "Hamburger SV": "Hamburger SV",
    "HSV": "Hamburger SV",
    "Paderborn": "SC Paderborn 07",
    "SC Paderborn": "SC Paderborn 07",
    "SC Paderborn 07": "SC Paderborn 07",
    "Düsseldorf": "Fortuna Düsseldorf",
    "Dusseldorf": "Fortuna Düsseldorf",
    "Duesseldorf": "Fortuna Düsseldorf",
    "Fortuna Düsseldorf": "Fortuna Düsseldorf",
    "Fortuna Dusseldorf": "Fortuna Düsseldorf",
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
    "Bayern Munich": "054efa67",
    "Borussia Dortmund": "add600ae",
    "RB Leipzig": "acbb6a5b",
    "Union Berlin": "7a41008f",
    "Freiburg": "a486e511",
    "Bayer Leverkusen": "c7a9f859",
    "Eintracht Frankfurt": "f0ac8ee6",
    "Wolfsburg": "4eaa11d7",
    "Mainz": "a224b06a",
    "Borussia Mönchengladbach": "32f3ee20",
    "FC Köln": "bc357bf7",
    "Hoffenheim": "033ea6b8",
    "Werder Bremen": "62add3bf",
    "Bochum": "b42c6323",
    "Augsburg": "0cdc4311",
    "Stuttgart": "598bc722",
    "Hertha BSC": "2818f8bc",
    "Schalke 04": "c539e393",
    "St. Pauli": "54864664",
    "Holstein Kiel": "2ac661d9",
    "Heidenheim": "18d9d2a7",
    "Darmstadt": "6a6967fc",
    "Greuther Fürth": "12192a4c",
    "Arminia Bielefeld": "247c4b67",
    "Hamburger SV": "26790c6a",
}


# fbref url names (team names as they appear in fbref urls)
# maps canonical names to url-safe versions
FBREF_URL_NAMES = {
    "Bayern Munich": "Bayern-Munich",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB-Leipzig",
    "Union Berlin": "Union-Berlin",
    "Freiburg": "Freiburg",
    "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Eint-Frankfurt",
    "Wolfsburg": "Wolfsburg",
    "Mainz": "Mainz-05",
    "Borussia Mönchengladbach": "Gladbach",
    "FC Köln": "Köln",
    "Hoffenheim": "Hoffenheim",
    "Werder Bremen": "Werder-Bremen",
    "Bochum": "Bochum",
    "Augsburg": "Augsburg",
    "Stuttgart": "Stuttgart",
    "Hertha BSC": "Hertha-BSC",
    "Schalke 04": "Schalke-04",
    "St. Pauli": "St-Pauli",
    "Holstein Kiel": "Holstein-Kiel",
    "Heidenheim": "Heidenheim",
    "Darmstadt": "Darmstadt-98",
    "Greuther Fürth": "Greuther-Fürth",
    "Arminia Bielefeld": "Arminia",
    "Hamburger SV": "Hamburger-SV",
}


# season-specific team lists (season_end_year -> list of canonical team names)
BUNDESLIGA_SEASONS = {
    2021: [
        "Bayern Munich",
        "Borussia Dortmund",
        "RB Leipzig",
        "Wolfsburg",
        "Eintracht Frankfurt",
        "Bayer Leverkusen",
        "Union Berlin",
        "Borussia Mönchengladbach",
        "Stuttgart",
        "Freiburg",
        "Hoffenheim",
        "Mainz",
        "Augsburg",
        "FC Köln",
        "Werder Bremen",
        "Hertha BSC",
        "Schalke 04",
        "Arminia Bielefeld",
    ],
    2022: [
        "Bayern Munich",
        "Borussia Dortmund",
        "RB Leipzig",
        "Bayer Leverkusen",
        "Union Berlin",
        "Freiburg",
        "FC Köln",
        "Mainz",
        "Hoffenheim",
        "Borussia Mönchengladbach",
        "Eintracht Frankfurt",
        "Bochum",
        "Wolfsburg",
        "Stuttgart",
        "Augsburg",
        "Hertha BSC",
        "Greuther Fürth",
        "Arminia Bielefeld",
    ],
    2023: [
        "Bayern Munich",
        "Borussia Dortmund",
        "RB Leipzig",
        "Union Berlin",
        "Freiburg",
        "Bayer Leverkusen",
        "Eintracht Frankfurt",
        "Wolfsburg",
        "Mainz",
        "Borussia Mönchengladbach",
        "FC Köln",
        "Hoffenheim",
        "Werder Bremen",
        "Bochum",
        "Augsburg",
        "Stuttgart",
        "Hertha BSC",
        "Schalke 04",
    ],
    2024: [
        "Bayern Munich",
        "Borussia Dortmund",
        "RB Leipzig",
        "Stuttgart",
        "Bayer Leverkusen",
        "Eintracht Frankfurt",
        "Hoffenheim",
        "Werder Bremen",
        "Freiburg",
        "Augsburg",
        "Heidenheim",
        "Mainz",
        "Wolfsburg",
        "Borussia Mönchengladbach",
        "Bochum",
        "Union Berlin",
        "FC Köln",
        "Darmstadt",
    ],
    2025: [
        "Bayern Munich",
        "Borussia Dortmund",
        "RB Leipzig",
        "Union Berlin",
        "Freiburg",
        "Bayer Leverkusen",
        "Eintracht Frankfurt",
        "Wolfsburg",
        "Mainz",
        "Borussia Mönchengladbach",
        "Hoffenheim",
        "Werder Bremen",
        "Bochum",
        "Augsburg",
        "Stuttgart",
        "Heidenheim",
        "St. Pauli",
        "Holstein Kiel",
    ],
    2026: [
        "Bayern Munich",
        "Borussia Dortmund",
        "RB Leipzig",
        "Union Berlin",
        "Freiburg",
        "Bayer Leverkusen",
        "Eintracht Frankfurt",
        "Wolfsburg",
        "Mainz",
        "Borussia Mönchengladbach",
        "Hoffenheim",
        "Werder Bremen",
        "Augsburg",
        "Stuttgart",
        "Heidenheim",
        "St. Pauli",
        "Hamburger SV",
        "FC Köln",
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


def get_bundesliga_teams_for_season(season_end_year: int) -> List[str]:
    """Get the list of teams in the Bundesliga for a specific season"""
    if season_end_year in BUNDESLIGA_SEASONS:
        return BUNDESLIGA_SEASONS[season_end_year].copy()

    # fallback: return all teams with fbref ids
    return list(FBREF_TEAM_IDS.keys())


def get_fbref_team_ids_for_season(season_end_year: int) -> Dict[str, str]:
    """Get team ID mappings for a specific Bundesliga season"""
    teams = get_bundesliga_teams_for_season(season_end_year)
    return {team: FBREF_TEAM_IDS[team] for team in teams if team in FBREF_TEAM_IDS}
