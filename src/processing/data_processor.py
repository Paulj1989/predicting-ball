# src/processing/data_processor.py

import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
import logging
from typing import Dict, List, Tuple


class DataProcessor:
    """
    Data integration layer for combining raw data sources.

    Responsibilities:
    - Merge data from multiple sources (FBRef, Transfermarkt, Odds)
    - Standardise team names across sources
    - Basic data quality checks and deduplication
    """

    def __init__(self, db_path: str = "data/club_football.duckdb"):
        """Initialise data processor"""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.team_mappings = self._initialise_team_mappings()

    def _initialise_team_mappings(self) -> Dict[str, str]:
        """Initialise comprehensive team name mappings"""
        return {

            "Bayern": "Bayern Munich",
            "Bayern Munich": "Bayern Munich",
            "FC Bayern München": "Bayern Munich",

            "Dortmund": "Borussia Dortmund",
            "Borussia Dortmund": "Borussia Dortmund",
            "BVB": "Borussia Dortmund",

            "Leipzig": "RB Leipzig",
            "RB Leipzig": "RB Leipzig",

            "Leverkusen": "Bayer Leverkusen",
            "Bayer Leverkusen": "Bayer Leverkusen",
            "Bayer 04 Leverkusen": "Bayer Leverkusen",

            "M'Gladbach": "Borussia Mönchengladbach",
            "Mönchengladbach": "Borussia Mönchengladbach",
            "Gladbach": "Borussia Mönchengladbach",
            "Bor. Mönchengladbach": "Borussia Mönchengladbach",
            "Monchengladbach": "Borussia Mönchengladbach",
            "M'gladbach": "Borussia Mönchengladbach",
            "Borussia Mönchengladbach": "Borussia Mönchengladbach",

            "Frankfurt": "Eintracht Frankfurt",
            "Eint Frankfurt": "Eintracht Frankfurt",
            "Eintracht Frankfurt": "Eintracht Frankfurt",
            "Ein Frankfurt": "Eintracht Frankfurt",

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

            "Hoffenheim": "Hoffenheim",
            "TSG Hoffenheim": "Hoffenheim",
            "TSG 1899 Hoffenheim": "Hoffenheim",

            "Mainz": "Mainz",
            "Mainz 05": "Mainz",
            "1. FSV Mainz 05": "Mainz",
            "1.FSV Mainz 05": "Mainz",

            "Augsburg": "Augsburg",
            "FC Augsburg": "Augsburg",

            "Freiburg": "Freiburg",
            "SC Freiburg": "Freiburg",

            "Köln": "FC Köln",
            "FC Köln": "FC Köln",
            "1. FC Köln": "FC Köln",
            "1.FC Köln": "FC Köln",
            "Cologne": "FC Köln",
            "FC Koln": "FC Köln",

            "Schalke": "Schalke 04",
            "Schalke 04": "Schalke 04",
            "FC Schalke 04": "Schalke 04",

            "Bochum": "Bochum",
            "VfL Bochum": "Bochum",

            "Arminia": "Arminia Bielefeld",
            "Bielefeld": "Arminia Bielefeld",
            "Arminia Bielefeld": "Arminia Bielefeld",

            "Fürth": "Greuther Fürth",
            "Greuther Fürth": "Greuther Fürth",
            "SpVgg Greuther Fürth": "Greuther Fürth",
            "Greuther Furth": "Greuther Fürth",

            "Kiel": "Holstein Kiel",
            "Holstein Kiel": "Holstein Kiel",

            "Heidenheim": "Heidenheim",
            "1. FC Heidenheim": "Heidenheim",
            "1.FC Heidenheim 1846": "Heidenheim",

            "St. Pauli": "St. Pauli",
            "St Pauli": "St. Pauli",
            "FC St. Pauli": "St. Pauli",

            "Darmstadt": "Darmstadt",
            "Darmstadt 98": "Darmstadt",
            "SV Darmstadt 98": "Darmstadt",

            "Hertha": "Hertha BSC",
            "Hertha BSC": "Hertha BSC",

            "Nurnberg": "FC Nürnberg",
            "Hannover": "Hannover 96",
            "Hamburg": "Hamburger SV",
            "Paderborn": "SC Paderborn 07",
            "Fortuna Dusseldorf": "Fortuna Düsseldorf",
        }

    def connect_db(self):
        """Establish connection to DuckDB database"""
        self.conn = duckdb.connect(self.db_path)

    def close_db(self):
        """Close database connection if open"""
        if self.conn:
            self.conn.close()

    def standardise_team_names(
        self, df: pd.DataFrame, team_columns: List[str]
    ) -> pd.DataFrame:
        """Standardise team names across datasets using mapping dictionary"""
        df = df.copy()

        for col in team_columns:
            if col in df.columns:
                df[col] = df[col].map(lambda x: self.team_mappings.get(x, x))

        return df

    def process_transfermarkt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Transfermarkt squad values data"""
        # standardise team names
        df = self.standardise_team_names(df, ["team"])

        # fill missing values with sensible defaults
        df["squad_size"] = df["squad_size"].fillna(25)
        df["avg_age"] = df["avg_age"].fillna(
            df.groupby("season_end_year")["avg_age"].transform("mean")
        )

        self.logger.info(f"Processed {len(df)} Transfermarkt records")
        return df

    def process_fbref_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process FBRef match data"""
        # standardise team names
        df = self.standardise_team_names(df, ["home_team", "away_team"])

        # identify played matches
        played_mask = df["home_goals"].notna() & df["away_goals"].notna()

        if played_mask.any():
            # Add basic result indicator (needed for merging/filtering)
            df.loc[played_mask, "goal_difference"] = (
                df.loc[played_mask, "home_goals"] - df.loc[played_mask, "away_goals"]
            )

            df.loc[played_mask, "result"] = "D"
            df.loc[played_mask & (df["goal_difference"] > 0), "result"] = "H"
            df.loc[played_mask & (df["goal_difference"] < 0), "result"] = "A"

        self.logger.info(
            f"Processed {len(df)} FBRef records ({played_mask.sum()} played)"
        )
        return df

    def calculate_implied_probabilities(
        self, home_odds: float, draw_odds: float, away_odds: float
    ) -> Dict[str, float]:
        """
        Calculate normalised implied probabilities from decimal odds.
        Removes bookmaker overround by normalising probabilities to sum to 1.
        """
        if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
            return {"home_prob": None, "draw_prob": None, "away_prob": None}

        # convert odds to raw probabilities
        raw_home = 1 / home_odds
        raw_draw = 1 / draw_odds
        raw_away = 1 / away_odds

        # normalise to remove bookmaker margin
        total = raw_home + raw_draw + raw_away

        return {
            "home_prob": raw_home / total,
            "draw_prob": raw_draw / total,
            "away_prob": raw_away / total,
        }

    def process_historical_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process historical betting odds from football-data.co.uk.

        Operations:
        - Standardise team names
        - Calculate implied probabilities (Bet365 odds)
        - Rename columns for consistency
        """
        # standardise team names
        df = self.standardise_team_names(df, ["HomeTeam", "AwayTeam"])

        # convert date to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # use bet365 odds as primary (most complete coverage)
        odds_cols = ["B365H", "B365D", "B365A"]
        if all(col in df.columns for col in odds_cols):
            # calculate implied probabilities
            implied_probs = df.apply(
                lambda row: self.calculate_implied_probabilities(
                    row["B365H"], row["B365D"], row["B365A"]
                ),
                axis=1,
            )

            df["odds_home_prob"] = [p["home_prob"] for p in implied_probs]
            df["odds_draw_prob"] = [p["draw_prob"] for p in implied_probs]
            df["odds_away_prob"] = [p["away_prob"] for p in implied_probs]

            # rename for consistency
            df["home_odds"] = df["B365H"]
            df["draw_odds"] = df["B365D"]
            df["away_odds"] = df["B365A"]

            self.logger.info(f"Processed {len(df)} historical odds records")
        else:
            self.logger.warning("Bet365 odds columns not found")

        return df

    def process_upcoming_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process upcoming match odds from The Odds API"""
        # standardise team names
        df = self.standardise_team_names(df, ["home_team", "away_team"])

        # convert timestamp
        df["commence_time"] = pd.to_datetime(df["commence_time"])

        # rename for consistency with historical odds
        if "consensus_home_odds" in df.columns:
            df["home_odds"] = df["consensus_home_odds"]
            df["draw_odds"] = df["consensus_draw_odds"]
            df["away_odds"] = df["consensus_away_odds"]

            # probabilities already calculated by scraper
            df["odds_home_prob"] = df["implied_home_prob"]
            df["odds_draw_prob"] = df["implied_draw_prob"]
            df["odds_away_prob"] = df["implied_away_prob"]

        self.logger.info(f"Processed {len(df)} upcoming odds records")
        return df

    def create_integrated_dataset(self) -> pd.DataFrame:
        """Create integrated dataset by merging all data sources"""
        self.connect_db()

        try:
            # load all data sources
            self.logger.info("Loading data from database")
            matches = self._load_matches()
            squad_values = self._load_squad_values()
            historical_odds, has_historical = self._load_historical_odds()
            upcoming_odds, has_upcoming = self._load_upcoming_odds()

            # process each source (standardisation only)
            self.logger.info("Processing data sources")
            matches = self.process_fbref_data(matches)
            squad_values = self.process_transfermarkt_data(squad_values)

            if has_historical:
                historical_odds = self.process_historical_odds(historical_odds)
            if has_upcoming:
                upcoming_odds = self.process_upcoming_odds(upcoming_odds)

            # prepare for merging
            matches["date"] = pd.to_datetime(matches["date"])

            # merge squad values
            matches = self._merge_squad_values(matches, squad_values)

            # merge odds
            if has_historical:
                matches = self._merge_historical_odds(matches, historical_odds)
            if has_upcoming:
                matches = self._merge_upcoming_odds(matches, upcoming_odds)

            # clean and save
            matches = self._deduplicate_and_save(matches)

            # log summary
            self._log_integration_summary(matches)

            return matches

        finally:
            self.close_db()

    def _load_matches(self) -> pd.DataFrame:
        """Load match data from database"""
        return self.conn.execute("SELECT * FROM raw.match_logs_fbref").df()

    def _load_squad_values(self) -> pd.DataFrame:
        """Load squad values from database"""
        return self.conn.execute("SELECT * FROM raw.squad_values_tm").df()

    def _load_historical_odds(self) -> Tuple[pd.DataFrame, bool]:
        """Load historical odds from database"""
        try:
            df = self.conn.execute("SELECT * FROM raw.historical_odds").df()
            return df, True
        except:
            self.logger.warning("No historical odds available")
            return pd.DataFrame(), False

    def _load_upcoming_odds(self) -> Tuple[pd.DataFrame, bool]:
        """Load upcoming odds from database"""
        try:
            df = self.conn.execute("SELECT * FROM raw.upcoming_odds").df()
            return df, True
        except:
            self.logger.warning("No upcoming odds available")
            return pd.DataFrame(), False

    def _merge_squad_values(
        self, matches: pd.DataFrame, squad_values: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge squad values for home and away teams"""
        # merge home team values
        matches = matches.merge(
            squad_values[
                ["season_end_year", "team", "total_value_eur", "squad_size", "avg_age"]
            ],
            left_on=["season_end_year", "home_team"],
            right_on=["season_end_year", "team"],
            how="left",
            suffixes=("", "_home"),
        ).drop(columns=["team"], errors="ignore")

        matches = matches.rename(
            columns={
                "total_value_eur": "home_value",
                "squad_size": "home_squad_size",
                "avg_age": "home_avg_age",
            }
        )

        # merge away team values
        matches = matches.merge(
            squad_values[
                ["season_end_year", "team", "total_value_eur", "squad_size", "avg_age"]
            ],
            left_on=["season_end_year", "away_team"],
            right_on=["season_end_year", "team"],
            how="left",
            suffixes=("", "_away"),
        ).drop(columns=["team"], errors="ignore")

        matches = matches.rename(
            columns={
                "total_value_eur": "away_value",
                "squad_size": "away_squad_size",
                "avg_age": "away_avg_age",
            }
        )

        return matches

    def _merge_historical_odds(
        self, matches: pd.DataFrame, historical_odds: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge historical odds data"""
        matches = matches.merge(
            historical_odds[
                [
                    "Date",
                    "HomeTeam",
                    "AwayTeam",
                    "home_odds",
                    "draw_odds",
                    "away_odds",
                    "odds_home_prob",
                    "odds_draw_prob",
                    "odds_away_prob",
                ]
            ],
            left_on=["date", "home_team", "away_team"],
            right_on=["Date", "HomeTeam", "AwayTeam"],
            how="left",
        )
        matches = matches.drop(
            columns=["Date", "HomeTeam", "AwayTeam"], errors="ignore"
        )

        coverage = matches["home_odds"].notna().mean()
        self.logger.info(f"Historical odds coverage: {coverage:.1%}")

        return matches

    def _merge_upcoming_odds(
        self, matches: pd.DataFrame, upcoming_odds: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge upcoming odds data (fills gaps in historical odds)"""
        # match on date
        matches["match_date"] = matches["date"].dt.date
        upcoming_odds["match_date"] = upcoming_odds["commence_time"].dt.date

        # merge upcoming odds
        matches = matches.merge(
            upcoming_odds[
                [
                    "match_date",
                    "home_team",
                    "away_team",
                    "home_odds",
                    "draw_odds",
                    "away_odds",
                    "odds_home_prob",
                    "odds_draw_prob",
                    "odds_away_prob",
                ]
            ],
            on=["match_date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_upcoming"),
        )

        # fill gaps in historical odds with upcoming odds
        odds_cols = [
            "home_odds",
            "draw_odds",
            "away_odds",
            "odds_home_prob",
            "odds_draw_prob",
            "odds_away_prob",
        ]

        for col in odds_cols:
            if f"{col}_upcoming" in matches.columns:
                matches[col] = matches[col].fillna(matches[f"{col}_upcoming"])
                matches = matches.drop(columns=[f"{col}_upcoming"])

        matches = matches.drop(columns=["match_date"], errors="ignore")

        upcoming_coverage = (
            matches[matches["home_goals"].isna()]["home_odds"].notna().mean()
        )
        self.logger.info(f"Upcoming odds coverage: {upcoming_coverage:.1%}")

        return matches

    def _deduplicate_and_save(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and save to database"""
        before = len(matches)
        matches = matches.sort_values(
            ["date", "home_team", "away_team", "home_goals"], na_position="first"
        )
        matches = matches.drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="first"
        )
        after = len(matches)

        if before != after:
            self.logger.warning(f"Removed {before - after} duplicate rows")

        # reset index
        matches = matches.reset_index(drop=True)

        # save to database
        self.conn.register("integrated_data", matches)
        self.conn.execute("""
            DROP TABLE IF EXISTS processed.integrated_data;
            CREATE TABLE processed.integrated_data AS
            SELECT * FROM integrated_data
        """)

        return matches

    def _log_integration_summary(self, matches: pd.DataFrame):
        """Log summary of integrated data"""
        self.logger.info(
            f"Created {len(matches)} match records with {len(matches.columns)} columns"
        )

        self.logger.info("Data coverage:")
        self.logger.info(f"  Squad values: {matches['home_value'].notna().mean():.1%}")

        if "home_odds" in matches.columns:
            self.logger.info(
                f"  Betting odds: {matches['home_odds'].notna().mean():.1%}"
            )

    def integrate_all_data(self):
        """Main data integration pipeline entry point"""
        self.logger.info("=" * 60)
        self.logger.info("INTEGRATING DATA SOURCES")
        self.logger.info("=" * 60)

        integrated_data = self.create_integrated_dataset()

        self.logger.info("=" * 60)
        self.logger.info("INTEGRATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info("Saved to: processed.integrated_data")
        self.logger.info(f"Total records: {len(integrated_data)}")

        return integrated_data
