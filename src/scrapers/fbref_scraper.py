# src/scrapers/fbref_scraper.py

import time
import random
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging
from typing import Optional, Dict, List
import re

from .base_scraper import BaseScraper


class FBRefScraper(BaseScraper):
    """FBRef match logs scraper using Selenium"""

    def __init__(
        self, headless: bool = True, min_delay: float = 3.0, max_delay: float = 7.0
    ):
        """Initialise fbref scraper with selenium webdriver"""
        super().__init__(min_delay, max_delay)
        self.headless = headless
        self.driver = None
        self.team_id_cache = {}

    def setup_driver(self):
        """Configure and initialise chrome webdriver"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(options=chrome_options)

        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
            },
        )

        self.logger.info("Chrome driver initialised")

    def _extract_team_id_from_url(self, url: str) -> Optional[str]:
        """Extract FBRef Team ID from URL"""
        match = re.search(r"/squads/([a-f0-9]+)/", url)
        return match.group(1) if match else None

    def _extract_team_ids_from_fixtures(self, match_table) -> Dict[str, str]:
        """Extract Team IDs from fixtures table"""
        team_ids = {}

        try:
            team_links = match_table.find_elements(
                By.CSS_SELECTOR, 'a[href*="/squads/"]'
            )

            for link in team_links:
                try:
                    team_name = link.text.strip()
                    url = link.get_attribute("href")
                    team_id = self._extract_team_id_from_url(url)

                    if team_name and team_id and team_name not in team_ids:
                        team_ids[team_name] = team_id
                        self.logger.debug(f"Found: {team_name} -> {team_id}")
                except:
                    continue

            self.logger.info(f"Extracted {len(team_ids)} Team IDs")

        except Exception as e:
            self.logger.error(f"Error extracting Team IDs: {e}")

        return team_ids

    def scrape_season(self, season_end_year: int) -> pd.DataFrame:
        """
        Scrape complete match logs and statistics for a Bundesliga season.

        Process:
        1. Load fixtures page
        2. Extract Team IDs from match table
        3. Parse all match results
        4. Scrape team-level statistics for each team
        5. Merge team stats to match data
        """
        season_string = f"{season_end_year - 1}-{season_end_year}"
        url = f"https://fbref.com/en/comps/20/{season_string}/schedule/{season_string}-Bundesliga-Scores-and-Fixtures"

        self.logger.info(f"Scraping season {season_string}")

        try:
            self.driver.get(url)
            time.sleep(random.uniform(2, 4))

            # find main fixtures table (should have 300+ rows for bundesliga)
            match_table = self._find_match_table()
            if not match_table:
                self.logger.error("Match table not found")
                return pd.DataFrame()

            # extract team ids for later stat scraping
            team_ids = self._extract_team_ids_from_fixtures(match_table)

            # parse all match rows
            matches = self._parse_all_matches(match_table, season_end_year)
            if not matches:
                self.logger.warning("No matches parsed")
                return pd.DataFrame()

            matches_df = pd.DataFrame(matches)
            self.logger.info(f"Parsed {len(matches)} matches")

            # filter team ids to only bundesliga teams (in case page had extra links)
            teams_in_matches = set(matches_df["home_team"].unique()) | set(
                matches_df["away_team"].unique()
            )
            team_ids = {
                team: tid for team, tid in team_ids.items() if team in teams_in_matches
            }

            self.logger.info(f"Filtered to {len(team_ids)} Bundesliga teams")
            self.team_id_cache[season_end_year] = team_ids

            # scrape detailed team statistics
            if team_ids:
                team_stats = self._scrape_all_team_stats(
                    season_end_year, season_string, team_ids
                )

                if not team_stats.empty:
                    matches_df = self._merge_team_stats_to_matches(
                        matches_df, team_stats
                    )
            else:
                self.logger.warning("No Team IDs - skipping advanced stats")

            return matches_df

        except Exception as e:
            self.logger.error(f"Error scraping season {season_end_year}: {e}")
            return pd.DataFrame()

    def _find_match_table(self):
        """Find the main fixtures table on the page"""
        tables = self.driver.find_elements(By.TAG_NAME, "table")

        for table in tables:
            try:
                rows = table.find_elements(By.TAG_NAME, "tr")
                # bundesliga has 306 matches per season
                if len(rows) > 100:
                    self.logger.debug(f"Found match table with {len(rows)} rows")
                    return table
            except:
                continue

        return None

    def _parse_all_matches(self, match_table, season_end_year: int) -> List[Dict]:
        """Parse all match rows from fixtures table"""
        matches = []

        tbody = (
            match_table.find_element(By.TAG_NAME, "tbody")
            if match_table.find_elements(By.TAG_NAME, "tbody")
            else match_table
        )
        rows = tbody.find_elements(By.TAG_NAME, "tr")

        for row in rows:
            match_data = self._parse_match_row(row, season_end_year)
            if match_data:
                matches.append(match_data)

        return matches

    def _scrape_all_team_stats(
        self, season_end_year: int, season_string: str, team_ids: Dict[str, str]
    ) -> pd.DataFrame:
        """Scrape shooting, possession, and misc stats for all teams"""
        all_team_stats = []

        for idx, (team_name, team_id) in enumerate(team_ids.items()):
            self.logger.info(f"Scraping {team_name} ({idx + 1}/{len(team_ids)})")

            # scrape each stat type
            shooting_stats = self._scrape_team_shooting(
                team_id, season_string, team_name
            )
            self.respect_rate_limit()

            possession_stats = self._scrape_team_possession(
                team_id, season_string, team_name
            )
            self.respect_rate_limit()

            misc_stats = self._scrape_team_misc(team_id, season_string, team_name)

            # merge all stat types for this team
            team_match_stats = self._merge_stat_types(
                shooting_stats, possession_stats, misc_stats
            )

            if not team_match_stats.empty:
                team_match_stats["season_end_year"] = season_end_year
                all_team_stats.append(team_match_stats)

            # polite delay between teams
            if idx < len(team_ids) - 1:
                self.respect_rate_limit()

        if all_team_stats:
            combined = pd.concat(all_team_stats, ignore_index=True)
            self.logger.info(f"Scraped {len(combined)} total stat records")
            return combined

        return pd.DataFrame()

    def _merge_stat_types(
        self,
        shooting_stats: pd.DataFrame,
        possession_stats: pd.DataFrame,
        misc_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge shooting, possession, and misc stats for a single team"""
        result = pd.DataFrame()

        # start with first non-empty dataframe
        if not shooting_stats.empty:
            result = shooting_stats
        elif not possession_stats.empty:
            result = possession_stats
        elif not misc_stats.empty:
            result = misc_stats
        else:
            return result

        # merge additional stat types
        for stats in [possession_stats, misc_stats]:
            if not stats.empty and not result.empty:
                result = result.merge(
                    stats, on=["date", "team", "opponent"], how="outer"
                )
            elif not stats.empty:
                result = stats

        return result

    def _scrape_team_shooting(
        self, team_id: str, season_string: str, team_name: str
    ) -> pd.DataFrame:
        """Scrape shooting statistics for a team"""
        team_url_name = team_name.replace(" ", "-")
        url = f"https://fbref.com/en/squads/{team_id}/{season_string}/matchlogs/c20/shooting/{team_url_name}-Match-Logs-Bundesliga"

        try:
            self.driver.get(url)
            time.sleep(random.uniform(2, 3))

            table = self.driver.find_element(By.ID, "matchlogs_for")
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            shooting_data = []

            for row in rows:
                try:
                    # extract date and opponent
                    match_data = self._extract_row_data(row)
                    date_str = match_data.get("date")
                    opponent = match_data.get("opponent")

                    if not date_str or not opponent:
                        continue

                    match_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                    # extract shooting stats
                    shooting_row = {
                        "date": match_date,
                        "team": team_name,
                        "opponent": opponent,
                        "shots": self._safe_float(match_data.get("shots")),
                        "shots_on_target": self._safe_float(
                            match_data.get("shots_on_target")
                        ),
                        "avg_shot_distance": self._safe_float(
                            match_data.get("average_shot_distance")
                        ),
                        "shots_free_kicks": self._safe_float(
                            match_data.get("shots_free_kicks")
                        ),
                        "npxg": self._safe_float(match_data.get("npxg")),
                        "pens_made": self._safe_float(match_data.get("pens_made")),
                        "pens_att": self._safe_float(match_data.get("pens_att")),
                        "goals_per_shot": self._safe_float(
                            match_data.get("goals_per_shot")
                        ),
                        "npxg_per_shot": self._safe_float(
                            match_data.get("npxg_per_shot")
                        ),
                    }

                    shooting_data.append(shooting_row)

                except Exception as e:
                    self.logger.debug(f"Error parsing shooting row: {e}")
                    continue

            if shooting_data:
                df = pd.DataFrame(shooting_data)
                self.logger.debug(f"  {len(df)} shooting records")
                return df

        except NoSuchElementException:
            self.logger.debug(f"  No shooting table for {team_name}")
        except Exception as e:
            self.logger.error(f"  Error scraping shooting for {team_name}: {e}")

        return pd.DataFrame()

    def _scrape_team_possession(
        self, team_id: str, season_string: str, team_name: str
    ) -> pd.DataFrame:
        """Scrape possession statistics for a team"""
        team_url_name = team_name.replace(" ", "-")
        url = f"https://fbref.com/en/squads/{team_id}/{season_string}/matchlogs/c20/possession/{team_url_name}-Match-Logs-Bundesliga"

        try:
            self.driver.get(url)
            time.sleep(random.uniform(2, 3))

            table = self.driver.find_element(By.ID, "matchlogs_for")
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            possession_data = []

            for row in rows:
                try:
                    match_data = self._extract_row_data(row)
                    date_str = match_data.get("date")
                    opponent = match_data.get("opponent")

                    if not date_str or not opponent:
                        continue

                    match_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                    possession_row = {
                        "date": match_date,
                        "team": team_name,
                        "opponent": opponent,
                        "possession": self._safe_float(match_data.get("possession")),
                        "touches": self._safe_float(match_data.get("touches")),
                        "touches_att_pen_area": self._safe_float(
                            match_data.get("touches_att_pen_area")
                        ),
                        "touches_att_3rd": self._safe_float(
                            match_data.get("touches_att_3rd")
                        ),
                    }

                    possession_data.append(possession_row)

                except Exception as e:
                    self.logger.debug(f"Error parsing possession row: {e}")
                    continue

            if possession_data:
                df = pd.DataFrame(possession_data)
                self.logger.debug(f"  {len(df)} possession records")
                return df

        except NoSuchElementException:
            self.logger.debug(f"  No possession table for {team_name}")
        except Exception as e:
            self.logger.error(f"  Error scraping possession for {team_name}: {e}")

        return pd.DataFrame()

    def _scrape_team_misc(
        self, team_id: str, season_string: str, team_name: str
    ) -> pd.DataFrame:
        """Scrape miscellaneous statistics for a team"""
        team_url_name = team_name.replace(" ", "-")
        url = f"https://fbref.com/en/squads/{team_id}/{season_string}/matchlogs/c20/misc/{team_url_name}-Match-Logs-Bundesliga"

        try:
            self.driver.get(url)
            time.sleep(random.uniform(2, 3))

            table = self.driver.find_element(By.ID, "matchlogs_for")
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            misc_data = []

            for row in rows:
                try:
                    match_data = self._extract_row_data(row)
                    date_str = match_data.get("date")
                    opponent = match_data.get("opponent")

                    if not date_str or not opponent:
                        continue

                    match_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                    misc_row = {
                        "date": match_date,
                        "team": team_name,
                        "opponent": opponent,
                        "cards_yellow": self._safe_float(
                            match_data.get("cards_yellow")
                        ),
                        "cards_red": self._safe_float(match_data.get("cards_red")),
                        "fouls": self._safe_float(match_data.get("fouls")),
                        "offsides": self._safe_float(match_data.get("offsides")),
                    }

                    misc_data.append(misc_row)

                except Exception as e:
                    self.logger.debug(f"Error parsing misc row: {e}")
                    continue

            if misc_data:
                df = pd.DataFrame(misc_data)
                self.logger.debug(f"  {len(df)} misc records")
                return df

        except NoSuchElementException:
            self.logger.debug(f"  No misc table for {team_name}")
        except Exception as e:
            self.logger.error(f"  Error scraping misc for {team_name}: {e}")

        return pd.DataFrame()

    def _extract_row_data(self, row) -> Dict[str, str]:
        """
        Extract all data-stat attributes from a table row.

        Combines th cells (typically date) and td cells (stats) into
        a single dictionary keyed by data-stat attribute.
        """
        match_data = {}

        # extract from both th (header) and td (data) cells
        th_cells = row.find_elements(By.TAG_NAME, "th")
        td_cells = row.find_elements(By.TAG_NAME, "td")

        for cell in th_cells + td_cells:
            stat = cell.get_attribute("data-stat")
            if stat:
                match_data[stat] = cell.text.strip()

        return match_data

    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float, returning none on failure"""
        if not value or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _merge_team_stats_to_matches(
        self, matches_df: pd.DataFrame, team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge team-level statistics to match-level data"""
        self.logger.info("Merging team stats to matches")

        # ensure date types match
        matches_df["date"] = pd.to_datetime(matches_df["date"]).dt.date
        team_stats["date"] = pd.to_datetime(team_stats["date"]).dt.date

        # remove duplicates from team stats
        before_dedup = len(team_stats)
        team_stats = team_stats.drop_duplicates(subset=["date", "team", "opponent"])
        after_dedup = len(team_stats)
        if before_dedup != after_dedup:
            self.logger.warning(
                f"Removed {before_dedup - after_dedup} duplicate team stat rows"
            )

        # define all stat columns to rename
        stat_cols = [
            "shots",
            "shots_on_target",
            "avg_shot_distance",
            "shots_free_kicks",
            "npxg",
            "pens_made",
            "pens_att",
            "goals_per_shot",
            "npxg_per_shot",
            "possession",
            "touches",
            "touches_att_pen_area",
            "touches_att_3rd",
            "cards_yellow",
            "cards_red",
            "fouls",
            "offsides",
        ]

        # merge home team stats
        matches_df = matches_df.merge(
            team_stats,
            left_on=["date", "home_team", "away_team"],
            right_on=["date", "team", "opponent"],
            how="left",
            suffixes=("", "_home_drop"),
        )

        # rename to home_* prefix
        for col in stat_cols:
            if col in matches_df.columns:
                matches_df = matches_df.rename(columns={col: f"home_{col}"})

        # cleanup temporary columns
        matches_df = matches_df.drop(columns=["team", "opponent"], errors="ignore")
        drop_cols = [c for c in matches_df.columns if "_home_drop" in c]
        matches_df = matches_df.drop(columns=drop_cols, errors="ignore")

        # merge away team stats
        matches_df = matches_df.merge(
            team_stats,
            left_on=["date", "away_team", "home_team"],
            right_on=["date", "team", "opponent"],
            how="left",
            suffixes=("", "_away_drop"),
        )

        # rename to away_* prefix
        for col in stat_cols:
            if col in matches_df.columns:
                matches_df = matches_df.rename(columns={col: f"away_{col}"})

        # cleanup temporary columns
        matches_df = matches_df.drop(columns=["team", "opponent"], errors="ignore")
        drop_cols = [c for c in matches_df.columns if "_away_drop" in c]
        matches_df = matches_df.drop(columns=drop_cols, errors="ignore")

        # final deduplication safety check
        before_final = len(matches_df)
        matches_df = matches_df.drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="first"
        )
        after_final = len(matches_df)
        if before_final != after_final:
            self.logger.warning(
                f"Removed {before_final - after_final} duplicate match rows"
            )

        self.logger.info(f"Merge complete - {matches_df.shape}")
        return matches_df

    def _parse_match_row(self, row, season_end_year: int) -> Optional[Dict]:
        """Parse a single match row from fixtures table"""
        try:
            cells = row.find_elements(By.TAG_NAME, "td")
            if not cells:
                cells = row.find_elements(By.TAG_NAME, "th")

            if len(cells) < 8:
                return None

            # skip month separator rows
            first_text = cells[0].text.strip() if cells else ""
            month_names = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            if any(month in first_text for month in month_names):
                return None

            # extract all cell data
            match_data = {}
            for cell in cells:
                try:
                    stat = cell.get_attribute("data-stat")
                    if stat:
                        match_data[stat] = cell.text.strip()
                except:
                    continue

            # parse gameweek
            week = None
            if "gameweek" in match_data and match_data["gameweek"].isdigit():
                week = int(match_data["gameweek"])

            # parse date
            match_date = None
            if "date" in match_data and match_data["date"]:
                try:
                    match_date = datetime.strptime(
                        match_data["date"], "%Y-%m-%d"
                    ).date()
                except:
                    pass

            # extract team names
            home_team = match_data.get("home_team") or match_data.get("squad_a")
            away_team = match_data.get("away_team") or match_data.get("squad_b")

            # validate team names
            invalid_names = {"Home", "Away", "Team", "Opponent", "", None}
            if (
                not home_team
                or not away_team
                or home_team in invalid_names
                or away_team in invalid_names
            ):
                return None

            # parse score
            home_goals, away_goals = self._parse_score(match_data.get("score"))

            # parse xg
            home_xg = self._parse_xg(match_data, ["home_xg", "xg_a"])
            away_xg = self._parse_xg(match_data, ["away_xg", "xg_b"])

            return {
                "season_end_year": season_end_year,
                "season": f"{season_end_year - 1}/{season_end_year}",
                "week": week,
                "date": match_date,
                "home_team": home_team,
                "home_xg": home_xg,
                "home_goals": home_goals,
                "away_team": away_team,
                "away_xg": away_xg,
                "away_goals": away_goals,
            }

        except Exception as e:
            self.logger.debug(f"Error parsing row: {e}")
            return None

    def _parse_score(self, score_str: str) -> tuple:
        """Parse score string into home and away goals"""
        if not score_str:
            return None, None

        for separator in ["–", "-", "—"]:
            if separator in score_str:
                parts = score_str.split(separator)
                if len(parts) == 2:
                    try:
                        home_goals = int(parts[0].strip())
                        away_goals = int(parts[1].strip())
                        return home_goals, away_goals
                    except ValueError:
                        pass

        return None, None

    def _parse_xg(self, match_data: Dict, field_names: List[str]) -> Optional[float]:
        """Parse xG value from match data"""
        for field in field_names:
            if field in match_data:
                try:
                    return float(match_data[field])
                except (ValueError, TypeError):
                    pass
        return None

    def scrape_multiple_seasons(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Scrape match data for multiple Bundesliga seasons"""
        self.setup_driver()
        all_data = []

        try:
            for year in range(start_year, end_year + 1):
                self.logger.info(f"Scraping season {year - 1}/{year}")

                season_data = self.scrape_season(year)

                if not season_data.empty:
                    all_data.append(season_data)
                    self.logger.info(f"✓ Season {year}: {len(season_data)} matches")
                else:
                    self.logger.warning(f"✗ Season {year}: no data")

                # polite delay between seasons
                if year < end_year:
                    delay = random.uniform(5, 10)
                    time.sleep(delay)

        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("Browser closed")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Completed: {len(combined)} total matches")
            return combined

        self.logger.warning("No data retrieved")
        return pd.DataFrame()
