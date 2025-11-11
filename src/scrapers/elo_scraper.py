# src/scrapers/elo_scraper.py

import pandas as pd
import requests
from datetime import datetime
from typing import Optional, List, Dict
import logging
from io import StringIO

from .base_scraper import BaseScraper
from ..utils.team_utils import standardise_team_name, standardise_dataframe


class EloScraper(BaseScraper):
    """Scraper for Club Elo ratings of German teams"""

    def __init__(self, min_delay: float = 2.0, max_delay: float = 5.0):
        """Initialise Elo scraper"""
        super().__init__(min_delay, max_delay)
        self.base_url = "http://api.clubelo.com"

    def _fetch_elo_data(
        self, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch raw Elo data from Club Elo API"""

        # api expects date in the url path, not as query parameters
        if from_date:
            url = f"{self.base_url}/{from_date}"
        else:
            url = f"{self.base_url}"

        self.respect_rate_limit()

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # api returns csv format
            df = pd.read_csv(StringIO(response.text))

            # standardise column names
            df.columns = df.columns.str.lower()

            if "from" in df.columns:
                df.rename(columns={"from": "date"}, inplace=True)

            # if to_date is specified and different from from_date, filter the results
            if to_date and from_date and to_date != from_date:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    from_ts = pd.to_datetime(from_date)
                    to_ts = pd.to_datetime(to_date)
                    df = df[(df["date"] >= from_ts) & (df["date"] <= to_ts)]

            return df

        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch Elo data: {e}")
            return pd.DataFrame()

    def _filter_german_teams(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for teams in Germany"""
        if df.empty:
            return df

        if "country" not in df.columns:
            self.logger.warning("No country column found in Elo data")
            return df

        german_teams = df.loc[df["country"] == "GER"].copy()
        self.logger.info(f"Filtered to {len(german_teams)} German team records")

        return german_teams

    def scrape_season(
        self, season_end_year: int, reference_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Scrape Elo ratings for start of season"""
        season_start_year = season_end_year - 1

        # default to july 1st of season start year
        if reference_date is None:
            reference_date = f"{season_start_year}-07-01"

        self.logger.info(
            f"Scraping Elo ratings for {season_start_year}/{season_end_year} "
            f"(reference: {reference_date})"
        )

        # fetch data for specific date (api allows date range)
        df = self._fetch_elo_data(from_date=reference_date, to_date=reference_date)

        if df.empty:
            self.logger.warning(f"No Elo data found for {reference_date}")
            return pd.DataFrame()

        # filter for german teams
        german_df = self._filter_german_teams(df)

        if german_df.empty:
            self.logger.warning("No German teams found in Elo data")
            return pd.DataFrame()

        # add season information
        german_df["season_end_year"] = season_end_year
        german_df["season_start_year"] = season_start_year
        german_df["reference_date"] = reference_date

        # standardise team names using centralised mapper
        if "club" in german_df.columns:
            german_df["team"] = german_df["club"].apply(standardise_team_name)
            german_df["team_original_elo"] = german_df["club"]

        # select and order relevant columns
        output_cols = [
            "season_end_year",
            "season_start_year",
            "reference_date",
            "team",
            "team_original_elo",
            "elo",
        ]

        german_df = german_df[output_cols].copy()

        self.logger.info(f"Retrieved Elo ratings for {len(german_df)} German teams")

        return german_df

    def scrape_multiple_seasons(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Scrape Elo ratings for multiple seasons"""
        all_data = []

        for year in range(start_year, end_year + 1):
            season_data = self.scrape_season(year)

            if not season_data.empty:
                all_data.append(season_data)
                self.logger.info(
                    f"✓ Season {year - 1}/{year}: {len(season_data)} teams"
                )
            else:
                self.logger.warning(f"✗ Season {year - 1}/{year}: no data")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self.logger.info(
                f"Completed: {len(combined)} total records across "
                f"{len(all_data)} seasons"
            )
            return combined

        self.logger.warning("No Elo data retrieved")
        return pd.DataFrame()

    def match_teams_to_league(
        self, elo_df: pd.DataFrame, league_teams: List[str]
    ) -> pd.DataFrame:
        """Match Elo ratings to specific teams in the league"""
        if elo_df.empty:
            return elo_df

        # standardise league team names using centralised mapper
        standardised_league = [standardise_team_name(t) for t in league_teams]

        # filter elo data to league teams
        matched = elo_df.loc[elo_df["team"].isin(standardised_league)].copy()

        self.logger.info(f"Matched {len(matched)} of {len(league_teams)} league teams")

        # warn about unmatched teams
        unmatched = set(standardised_league) - set(matched["team"])
        if unmatched:
            self.logger.warning(f"Unmatched teams: {sorted(unmatched)}")

        return matched
