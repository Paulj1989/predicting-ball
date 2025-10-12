# src/scrapers/transfermarkt_scraper.py

import time
import random
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, List, Dict
import logging

from .base_scraper import BaseScraper


class TransfermarktScraper(BaseScraper):
    """Transfermarkt squad values scraper"""

    def __init__(self, min_delay: float = 3.0, max_delay: float = 7.0):
        """Initialise Transfermarkt scraper"""
        super().__init__(min_delay, max_delay)
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        ]
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Generate request headers with random user agent"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def _parse_value(self, value_str: str) -> Optional[float]:
        """Convert Transfermarkt value strings to numeric"""
        if not value_str or value_str == "-":
            return None

        clean = re.sub(r"[€£$,\s]", "", value_str)

        # handle billions
        if "bn" in clean.lower():
            number = re.sub(r"bn.*", "", clean.lower())
            try:
                return float(number) * 1e9
            except ValueError:
                return None

        # handle millions
        if clean.lower().endswith("m"):
            number = clean[:-1]
            try:
                return float(number) * 1e6
            except ValueError:
                return None

        # handle thousands
        if clean.lower().endswith("k"):
            number = clean[:-1]
            try:
                return float(number) * 1e3
            except ValueError:
                return None

        # handle plain numbers
        try:
            return float(clean)
        except ValueError:
            return None

    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch and parse a page with retries"""
        for attempt in range(max_retries):
            try:
                self.respect_rate_limit()

                response = self.session.get(
                    url, headers=self._get_headers(), timeout=30, allow_redirects=True
                )

                if response.status_code == 200:
                    return BeautifulSoup(response.content, "html.parser")

                if response.status_code == 429:
                    # exponential backoff for rate limiting
                    wait_time = (2**attempt) * 10
                    self.logger.warning(f"Rate limited - waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    self.logger.warning(f"http {response.status_code}")

            except requests.RequestException as e:
                self.logger.error(f"Request failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))

        self.logger.error(f"Failed to fetch after {max_retries} attempts")
        return None

    def scrape_season(self, season_end_year: int) -> pd.DataFrame:
        """Scrape squad values for a single season"""
        season_start = season_end_year - 1
        url = f"https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1/plus/?saison_id={season_start}"

        self.logger.info(f"Scraping season {season_start}/{season_end_year}")

        soup = self.fetch_page(url)
        if not soup:
            return pd.DataFrame()

        tables = soup.find_all("table", {"class": "items"})
        if not tables:
            self.logger.warning("No data tables found")
            return pd.DataFrame()

        # try each table until we find one with complete data
        for table in tables:
            data = self._parse_table(table, season_end_year)
            if data and len(data) >= 15:  # bundesliga has 18 teams
                df = pd.DataFrame(data)
                self.logger.info(f"Scraped {len(df)} teams")
                return df

        self.logger.warning("Incomplete team data found")
        return pd.DataFrame()

    def _parse_table(self, table, season_end_year: int) -> List[Dict]:
        """Parse a table for team data"""
        data = []

        tbody = table.find("tbody")
        if not tbody:
            return data

        rows = tbody.find_all("tr")

        for row in rows:
            try:
                cells = row.find_all(["td", "th"])
                if len(cells) < 5:
                    continue

                # extract team name (cell 1)
                team_name = None
                if len(cells) > 1:
                    team_cell = cells[1]
                    team_link = team_cell.find("a")
                    team_name = (
                        team_link.get_text(strip=True)
                        if team_link
                        else team_cell.get_text(strip=True)
                    )
                    team_name = team_name.strip()
                    if not team_name or team_name.isdigit():
                        team_name = None

                # extract squad size (cell 2)
                squad_size = None
                if len(cells) > 2:
                    text = cells[2].get_text(strip=True)
                    if text.isdigit():
                        squad_size = int(text)

                # extract average age (cell 3)
                avg_age = None
                if len(cells) > 3:
                    text = cells[3].get_text(strip=True)
                    if re.match(r"^\d+[.,]\d+$", text):
                        avg_age = float(text.replace(",", "."))

                # extract total value (usually last cell with € and m/bn/k)
                total_value_raw = None
                for i in range(len(cells) - 1, 3, -1):
                    text = cells[i].get_text(strip=True)
                    if "€" in text and any(x in text.lower() for x in ["m", "bn", "k"]):
                        total_value_raw = text
                        break

                # validate and add to results
                if team_name and total_value_raw:
                    parsed_value = self._parse_value(total_value_raw)

                    # sanity check: values should be > €1m for bundesliga teams
                    if parsed_value and parsed_value > 1_000_000:
                        data.append(
                            {
                                "season_end_year": season_end_year,
                                "team": team_name,
                                "squad_size": squad_size,
                                "avg_age": avg_age,
                                "total_value_raw": total_value_raw,
                                "total_value_eur": parsed_value,
                                "scrape_date": datetime.now().date(),
                            }
                        )

            except Exception as e:
                self.logger.debug(f"Error parsing row: {e}")
                continue

        return data

    def scrape_multiple_seasons(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Scrape squad values for multiple seasons"""
        all_data = []

        for year in range(start_year, end_year + 1):
            self.logger.info(f"Scraping season {year - 1}/{year}")

            season_data = self.scrape_season(year)

            if not season_data.empty:
                all_data.append(season_data)
                self.logger.info(f"✓ Season {year}: {len(season_data)} teams")
            else:
                self.logger.warning(f"✗ Season {year}: no data")

            # polite delay between seasons
            if year < end_year:
                delay = random.uniform(5, 10)
                time.sleep(delay)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Completed: {len(combined)} total records")
            return combined

        self.logger.warning("No data retrieved")
        return pd.DataFrame()
