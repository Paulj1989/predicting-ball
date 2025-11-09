# src/scraper/odds_scraper.py

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Optional
import time
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class OddsScraper:
    """Scraper for historical and current Bundesliga betting odds"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialise scraper with API key"""
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required: set ODDS_API_KEY environment variable or pass api_key parameter"
            )

        self.historical_base_url = "https://www.football-data.co.uk"
        self.api_base_url = "https://api.the-odds-api.com/v4"
        self.bundesliga_page = f"{self.historical_base_url}/germanym.php"

    def get_historical_odds(self, seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """Scrape historical odds from football-data.co.uk"""
        logger.info("Fetching historical odds from football-data.co.uk")

        # fetch main page to discover available seasons
        try:
            response = requests.get(self.bundesliga_page, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch main page: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, "html.parser")

        # find all bundesliga csv links
        csv_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # pattern: /mmz4281/YYYY/D1.csv (where YYYY is 4-digit season code)
            if "D1.csv" in href and "mmz4281" in href:
                # construct full url
                full_url = (
                    f"{self.historical_base_url}/{href.lstrip('/')}"
                    if not href.startswith("http")
                    else href
                )

                # extract season code
                parts = href.split("/")
                for i, part in enumerate(parts):
                    if part == "mmz4281" and i + 1 < len(parts):
                        season = parts[i + 1]
                        if season.isdigit() and len(season) == 4:
                            csv_links.append({"season": season, "url": full_url})
                        break

        # remove duplicates
        csv_links = [dict(t) for t in {tuple(d.items()) for d in csv_links}]

        logger.info(f"Found {len(csv_links)} historical seasons")

        # fallback: construct urls directly if discovery fails
        if not csv_links and seasons:
            logger.info("Using direct URL construction")
            for season in seasons:
                url = f"{self.historical_base_url}/mmz4281/{season}/D1.csv"
                csv_links.append({"season": season, "url": url})

        # filter by requested seasons
        if seasons:
            csv_links = [link for link in csv_links if link["season"] in seasons]
            logger.info(f"Filtered to {len(csv_links)} requested seasons")

        # download and combine all csv files
        all_data = []
        for link_info in csv_links:
            try:
                logger.info(f"  Downloading {link_info['season']}")
                df = pd.read_csv(link_info["url"], encoding="latin1")
                df["Season"] = link_info["season"]
                all_data.append(df)
                time.sleep(0.5)  # polite delay
            except Exception as e:
                logger.warning(f"  Failed {link_info['season']}: {e}")

        if not all_data:
            logger.warning("No historical data retrieved")
            return pd.DataFrame()

        # combine and standardise
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)

        if "Date" in combined_df.columns:
            combined_df["Date"] = pd.to_datetime(
                combined_df["Date"], format="%d/%m/%Y", errors="coerce"
            )

        logger.info(f"Retrieved {len(combined_df)} historical matches")
        return combined_df

    def get_upcoming_odds(
        self,
        markets: List[str] = ["h2h"],
        regions: str = "eu",
        odds_format: str = "decimal",
        calculate_consensus: bool = True,
    ) -> pd.DataFrame:
        """Fetch upcoming bundesliga odds from The Odds API"""
        logger.info("Fetching upcoming odds from The Odds API")

        url = f"{self.api_base_url}/sports/soccer_germany_bundesliga/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # log api usage
            remaining = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            logger.info(f"API usage - remaining: {remaining}, used: {used}")

            data = response.json()

            if not data:
                logger.info("No upcoming matches found")
                return pd.DataFrame()

            # parse response into structured format
            matches = []
            for game in data:
                match_info = {
                    "match_id": game["id"],
                    "commence_time": pd.to_datetime(game["commence_time"]),
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                }

                # extract odds from each bookmaker
                for bookmaker in game.get("bookmakers", []):
                    bookmaker_key = bookmaker["key"]

                    for market in bookmaker.get("markets", []):
                        market_key = market["key"]

                        # map outcomes to odds
                        outcomes = {
                            outcome["name"]: outcome["price"]
                            for outcome in market["outcomes"]
                        }

                        # add bookmaker odds to match
                        match_info[f"{bookmaker_key}_{market_key}_home"] = outcomes.get(
                            game["home_team"]
                        )
                        match_info[f"{bookmaker_key}_{market_key}_draw"] = outcomes.get(
                            "Draw"
                        )
                        match_info[f"{bookmaker_key}_{market_key}_away"] = outcomes.get(
                            game["away_team"]
                        )

                matches.append(match_info)

            df = pd.DataFrame(matches)
            logger.info(f"Retrieved {len(df)} upcoming matches")

            # automatically calculate consensus odds
            if calculate_consensus and not df.empty:
                df = self.calculate_consensus_odds(df)

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return pd.DataFrame()

    def calculate_consensus_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate consensus odds by averaging across all bookmakers"""
        if df.empty:
            return df

        df = df.copy()

        # identify bookmaker odds columns
        home_cols = [col for col in df.columns if col.endswith("_h2h_home")]
        draw_cols = [col for col in df.columns if col.endswith("_h2h_draw")]
        away_cols = [col for col in df.columns if col.endswith("_h2h_away")]

        if not home_cols or not draw_cols or not away_cols:
            logger.warning("No bookmaker odds found for consensus calculation")
            return df

        # average across bookmakers (raw odds only)
        df["consensus_home_odds"] = df[home_cols].mean(axis=1)
        df["consensus_draw_odds"] = df[draw_cols].mean(axis=1)
        df["consensus_away_odds"] = df[away_cols].mean(axis=1)

        logger.info("Consensus odds calculated")
        return df
