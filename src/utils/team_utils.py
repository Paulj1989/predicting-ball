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
