# src/scrapers/fbref_scraper.py

import time
import random
from datetime import datetime
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import logging
from typing import Optional, Dict, List

from .base_scraper import BaseScraper
from ..utils.team_utils import get_fbref_team_ids_for_season, get_fbref_url_name


class FBRefScraper(BaseScraper):
    """
    FBRef scraper using shooting logs for all competitions.

    This scraper:
    - Uses only shooting logs (which include goals, npxG, and penalties)
    - Scrapes all competitions (Bundesliga, DFB-Pokal, European competitions)
    - Extracts full team schedules for calculating rest days and consecutive away games
    - Merges team perspectives into match-level records
    """

    def __init__(
        self, headless: bool = True, min_delay: float = 5.0, max_delay: float = 10.0
    ):
        """Initialise FBRef scraper with Selenium WebDriver"""
        super().__init__(min_delay, max_delay)
        self.headless = headless
        self.driver = None

    def setup_driver(self):
        """
        Configure and initialise undetected Chrome WebDriver.

        Uses undetected_chromedriver for better cloudflare bypass.
        Supports both headless and headed modes.
        """
        chrome_options = uc.ChromeOptions()

        # configure headless mode (default for automated pipelines)
        if self.headless:
            chrome_options.add_argument("--headless=new")
            self.logger.info("Using headless mode")
        else:
            self.logger.info("Using headed mode (visible browser)")

        # core arguments
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

        # additional arguments
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-site-isolation-trials")
        chrome_options.add_argument(
            "--disable-features=IsolateOrigins,site-per-process"
        )
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--disable-gpu")

        # realistic user agent (updated to match current chrome)
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )

        # browser preferences
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # initialise undetected chrome
        self.driver = uc.Chrome(
            options=chrome_options,
            use_subprocess=True,
            version_main=None,
            driver_executable_path=None,
            headless=self.headless,
        )

        # override navigator.webdriver property
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                window.chrome = {
                    runtime: {}
                };
            """
            },
        )

        mode = "headless" if self.headless else "headed"
        self.logger.info(f"Undetected Chrome driver initialised ({mode} mode)")

    def scrape_season(self, season_end_year: int) -> pd.DataFrame:
        """
        Scrape complete match data for a Bundesliga season (all competitions).

        Process:
        1. Load team IDs for Bundesliga teams
        2. For each team, scrape shooting logs (all competitions)
        3. Calculate schedule-based features (rest days, consecutive away games)
        4. Merge team perspectives into match-level records
        """
        season_string = f"{season_end_year - 1}-{season_end_year}"
        self.logger.info(f"Scraping season {season_string} - all competitions")

        # get bundesliga team ids
        team_ids = get_fbref_team_ids_for_season(season_end_year)
        self.logger.info(
            f"Retrieved {len(team_ids)} teams for season {season_end_year}"
        )
        self.logger.debug(
            f"Teams: {', '.join(list(team_ids.keys())[:5])}... (showing first 5)"
        )

        # scrape shooting logs for all teams
        all_team_data = []
        failed_teams = []

        for idx, (team_name, team_id) in enumerate(team_ids.items()):
            self.logger.info(f"Scraping {team_name} ({idx + 1}/{len(team_ids)})")

            team_shooting = self._scrape_team_shooting_logs(
                team_id, season_string, team_name
            )

            if not team_shooting.empty:
                team_shooting["season_end_year"] = season_end_year
                all_team_data.append(team_shooting)
            else:
                # track failed teams for retry
                failed_teams.append((team_name, team_id))

            # polite delay between teams
            if idx < len(team_ids) - 1:
                self.respect_rate_limit()

        # retry failed teams once with longer delay
        # this handles transient network issues or temporary rate limiting
        if failed_teams:
            self.logger.info(
                f"Retrying {len(failed_teams)} failed teams with longer delays"
            )

            for retry_num, (team_name, team_id) in enumerate(failed_teams):
                # extra polite delay before retry (15-20 seconds)
                # this is longer than normal rate limiting to avoid triggering defences
                retry_delay = random.uniform(15, 20)
                self.logger.info(f"  Waiting {retry_delay:.1f}s before retry...")
                time.sleep(retry_delay)

                self.logger.info(
                    f"  Retry: {team_name} ({retry_num + 1}/{len(failed_teams)})"
                )

                team_shooting = self._scrape_team_shooting_logs(
                    team_id, season_string, team_name
                )

                if not team_shooting.empty:
                    team_shooting["season_end_year"] = season_end_year
                    all_team_data.append(team_shooting)
                    self.logger.info(f"    ✓ Retry successful for {team_name}")
                else:
                    self.logger.warning(
                        f"    ✗ Retry failed for {team_name} - check team ID/URL"
                    )

                # apply rate limiting between retries
                if retry_num < len(failed_teams) - 1:
                    self.respect_rate_limit()

        if not all_team_data:
            self.logger.warning("No data retrieved")
            return pd.DataFrame()

        # combine all team data
        combined_team_data = pd.concat(all_team_data, ignore_index=True)
        self.logger.info(f"Retrieved {len(combined_team_data)} team match records")

        # calculate schedule-based features
        combined_team_data = self._calculate_schedule_features(combined_team_data)

        # convert team-level data to match-level data
        matches_df = self._create_match_level_data(combined_team_data, season_end_year)

        self.logger.info(f"Created {len(matches_df)} match records")
        return matches_df

    def _scrape_team_shooting_logs(
        self, team_id: str, season_string: str, team_name: str
    ) -> pd.DataFrame:
        """Scrape shooting log data for a team (all competitions)"""
        team_url_name = get_fbref_url_name(team_name)
        url = f"https://fbref.com/en/squads/{team_id}/{season_string}/matchlogs/all_comps/shooting/{team_url_name}-Match-Logs-All-Competitions"

        try:
            self.logger.debug(f"  Fetching: {url}")
            self.driver.get(url)

            # wait for cloudflare verification (longer in headless mode)
            wait_time = (
                random.uniform(10, 15) if self.headless else random.uniform(8, 12)
            )
            self.logger.debug(
                f"  Waiting {wait_time:.1f}s for Cloudflare verification..."
            )
            time.sleep(wait_time)

            try:
                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight/2);"
                )
                time.sleep(random.uniform(0.5, 1.5))
                self.driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(random.uniform(0.5, 1.0))
            except Exception:
                pass

            cookie_selectors = [
                "button.qc-cmp2-summary-buttons button",
                ".qc-cmp2-summary-buttons button",
                "button[mode='primary']",
                "button.fc-cta-consent",
                "button.fc-button.fc-cta-consent",
            ]

            for selector in cookie_selectors:
                try:
                    cookie_button = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    cookie_button.click()
                    self.logger.debug(
                        f"  Cookie consent dismissed using selector: {selector}"
                    )
                    time.sleep(0.5)  # brief pause after clicking
                    break
                except (TimeoutException, Exception):
                    continue

            # wait for the shooting logs table to load (up to 20 seconds)
            # use visibility instead of just presence for more reliable detection
            try:
                wait = WebDriverWait(self.driver, 20)
                table = wait.until(
                    EC.visibility_of_element_located((By.ID, "matchlogs_for"))
                )
                self.logger.debug("  Table found and visible")
            except TimeoutException:
                # table not found within timeout - log available tables for debugging
                all_tables = self.driver.find_elements(By.TAG_NAME, "table")
                table_ids = [
                    t.get_attribute("id") for t in all_tables if t.get_attribute("id")
                ]
                self.logger.warning(
                    "  ✗ Table 'matchlogs_for' not visible after 20s wait"
                )
                self.logger.debug(f"    Available tables: {table_ids}")

                # save page source for debugging
                if self.logger.level <= 10:  # DEBUG level
                    page_source = self.driver.page_source
                    self.logger.debug(f"    Page title: {self.driver.title}")
                    self.logger.debug(f"    Page source length: {len(page_source)}")

                return pd.DataFrame()

            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            self.logger.debug(f"  Found {len(rows)} rows in table")

            shooting_data = []
            skipped_rows = 0

            for row in rows:
                try:
                    match_data = self._extract_row_data(row)

                    # debug: log first row's field names
                    if len(shooting_data) == 0 and match_data:
                        self.logger.debug(
                            f"  Sample fields from first row: {list(match_data.keys())[:10]}"
                        )

                    date_str = match_data.get("date")
                    opponent = match_data.get("opponent")

                    if not date_str or not opponent:
                        skipped_rows += 1
                        continue

                    match_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                    # extract all relevant shooting stats and schedule info
                    # using fbref's actual data-stat field names from the page
                    shooting_row = {
                        "date": match_date,
                        "team": team_name,
                        "opponent": opponent,
                        "venue": match_data.get("venue", ""),
                        "competition": match_data.get("comp", ""),
                        "result": match_data.get("result", ""),
                        # goals and xg
                        "goals": self._safe_int(match_data.get("goals_for")),
                        "goals_against": self._safe_int(
                            match_data.get("goals_against")
                        ),
                        "npxg": self._safe_float(match_data.get("npxg")),
                        # penalties
                        "pens_made": self._safe_int(match_data.get("pens_made")),
                        "pens_att": self._safe_int(match_data.get("pens_att")),
                        # other shooting stats
                        "shots": self._safe_int(match_data.get("shots")),
                        "shots_on_target": self._safe_int(
                            match_data.get("shots_on_target")
                        ),
                        "avg_shot_distance": self._safe_float(
                            match_data.get("average_shot_distance")
                        ),
                        "shots_free_kicks": self._safe_int(
                            match_data.get("shots_free_kicks")
                        ),
                    }

                    shooting_data.append(shooting_row)

                except Exception as e:
                    self.logger.debug(f"Error parsing row for {team_name}: {e}")
                    skipped_rows += 1
                    continue

            if skipped_rows > 0:
                self.logger.debug(f"  Skipped {skipped_rows} rows (headers/empty)")

            if shooting_data:
                df = pd.DataFrame(shooting_data)
                self.logger.info(f"  ✓ {len(df)} match records")
                return df
            else:
                self.logger.warning("  ✗ No valid match data extracted")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"  ✗ Error: {e}")
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
        """Safely convert string to float, returning None on failure"""
        if not value or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_int(self, value: str) -> Optional[int]:
        """Safely convert string to int, returning None on failure"""
        if not value or value == "":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _calculate_schedule_features(self, team_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate schedule-based features for each team.

        Features calculated:
        - rest_days: Number of days since last match
        - consecutive_away_games: Number of consecutive away games (including current)
        """
        if team_data.empty:
            return team_data

        self.logger.info("Calculating schedule features")

        # sort by team and date
        team_data = team_data.sort_values(["team", "date"]).reset_index(drop=True)

        # initialise feature columns
        team_data["rest_days"] = None
        team_data["consecutive_away_games"] = 0

        # calculate features for each team
        for team in team_data["team"].unique():
            team_mask = team_data["team"] == team
            team_matches = team_data[team_mask].copy()

            # calculate rest days
            rest_days = []
            prev_date = None

            for date in team_matches["date"]:
                if prev_date is None:
                    rest_days.append(None)  # first match has no previous match
                else:
                    days = (date - prev_date).days
                    rest_days.append(days)
                prev_date = date

            team_data.loc[team_mask, "rest_days"] = rest_days

            # calculate consecutive away games
            consecutive_away = []
            away_count = 0

            for venue in team_matches["venue"]:
                if venue == "Away":
                    away_count += 1
                    consecutive_away.append(away_count)
                else:
                    away_count = 0
                    consecutive_away.append(0)

            team_data.loc[team_mask, "consecutive_away_games"] = consecutive_away

        self.logger.info("Schedule features calculated")
        return team_data

    def _create_match_level_data(
        self, team_data: pd.DataFrame, season_end_year: int
    ) -> pd.DataFrame:
        """
        Convert team-level data to match-level data.

        Each match has two team perspectives (home and away).
        This method merges them into single match records.
        """
        if team_data.empty:
            return pd.DataFrame()

        self.logger.info("Converting to match-level data")

        matches = []
        processed_matches = set()

        # ensure date is date type
        team_data["date"] = pd.to_datetime(team_data["date"]).dt.date

        for _, row in team_data.iterrows():
            date = row["date"]
            team = row["team"]
            opponent = row["opponent"]
            venue = row["venue"]

            # create unique match identifier
            teams_sorted = tuple(sorted([team, opponent]))
            match_id = (date, teams_sorted[0], teams_sorted[1])

            if match_id in processed_matches:
                continue
            processed_matches.add(match_id)

            # determine home and away teams
            if venue == "Home":
                home_team = team
                away_team = opponent
                home_data = row

                # find away team's data for this match
                away_data = team_data[
                    (team_data["date"] == date)
                    & (team_data["team"] == opponent)
                    & (team_data["opponent"] == team)
                ]
                away_data = away_data.iloc[0] if len(away_data) > 0 else None

            elif venue == "Away":
                home_team = opponent
                away_team = team
                away_data = row

                # find home team's data for this match
                home_data = team_data[
                    (team_data["date"] == date)
                    & (team_data["team"] == opponent)
                    & (team_data["opponent"] == team)
                ]
                home_data = home_data.iloc[0] if len(home_data) > 0 else None

            else:
                # neutral venue or unknown - skip or use alphabetical order
                self.logger.debug(
                    f"Skipping neutral/unknown venue match: {team} vs {opponent}"
                )
                continue

            # build match record
            match_record = {
                "season_end_year": season_end_year,
                "season": f"{season_end_year - 1}/{season_end_year}",
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
            }

            # add competition info (use home team's data, should be same)
            if home_data is not None:
                match_record["competition"] = home_data.get("competition", "")
            elif away_data is not None:
                match_record["competition"] = away_data.get("competition", "")

            # add home team stats
            if home_data is not None:
                for col in [
                    "goals",
                    "goals_against",
                    "npxg",
                    "pens_made",
                    "pens_att",
                    "shots",
                    "shots_on_target",
                    "avg_shot_distance",
                    "shots_free_kicks",
                    "rest_days",
                    "consecutive_away_games",
                ]:
                    match_record[f"home_{col}"] = home_data.get(col)

            # add away team stats
            if away_data is not None:
                for col in [
                    "goals",
                    "goals_against",
                    "npxg",
                    "pens_made",
                    "pens_att",
                    "shots",
                    "shots_on_target",
                    "avg_shot_distance",
                    "shots_free_kicks",
                    "rest_days",
                    "consecutive_away_games",
                ]:
                    match_record[f"away_{col}"] = away_data.get(col)

            matches.append(match_record)

        if not matches:
            return pd.DataFrame()

        matches_df = pd.DataFrame(matches)

        # sort by date
        matches_df = matches_df.sort_values("date").reset_index(drop=True)

        # final deduplication check
        before_dedup = len(matches_df)
        matches_df = matches_df.drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="first"
        )
        after_dedup = len(matches_df)

        if before_dedup != after_dedup:
            self.logger.warning(
                f"Removed {before_dedup - after_dedup} duplicate match rows"
            )

        return matches_df

    def scrape_multiple_seasons(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Scrape match data for multiple Bundesliga seasons (all competitions)"""
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
