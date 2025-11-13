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
from ..utils.team_utils import (
    get_fbref_team_ids_for_season,
    get_fbref_url_name,
    standardise_team_name,
)


class FBRefScraper(BaseScraper):
    """
    FBRef scraper with multilevel data architecture for Bundesliga predictions.

    Data Architecture:
    1. Bundesliga Schedule (authoritative source for fixtures we're predicting)
    - Match dates, home/away teams, final scores
    - Single source of truth for Bundesliga fixtures

    2. Complete Team Activity Log (all competitions)
    - Used ONLY for calculating schedule features
    - Includes: Bundesliga, DFB-Pokal, Champions League, Europa League
    - Critical: rest days and consecutive away games must account for ALL matches

    3. Bundesliga Shooting Statistics
    - Performance metrics for Bundesliga matches only
    - npxg, shots, shots_on_target, penalties, etc.
    - Merged into Bundesliga schedule
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

        # give browser time to fully initialise (especially important in headed mode)
        time.sleep(1)

        # override navigator.webdriver property
        # wrap in try-except to handle timing issues in headed mode
        try:
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
        except Exception as e:
            # CDP command may fail in some environments, log but continue
            self.logger.warning(f"Could not set CDP commands: {e}")
            self.logger.warning("Continuing without navigator property overrides")

        mode = "headless" if self.headless else "headed"
        self.logger.info(f"Undetected Chrome driver initialised ({mode} mode)")

    def scrape_season(self, season_end_year: int) -> pd.DataFrame:
        """
        Scrape complete match data for a Bundesliga season.

        Revised process with correct schedule feature calculation:
        1. Scrape Bundesliga schedule
        2. Scrape ALL team activity (all competitions) for schedule feature calculation
        3. Calculate schedule features using ALL team activity
        4. Merge Bundesliga shooting statistics into the schedule
        """
        season_string = f"{season_end_year - 1}-{season_end_year}"
        self.logger.info(f"Scraping season {season_string}")

        # step 1: scrape bundesliga schedule (authoritative source for fixtures)
        bundesliga_schedule = self._scrape_bundesliga_schedule(season_end_year)

        if bundesliga_schedule.empty:
            self.logger.warning("Failed to retrieve Bundesliga schedule")
            return pd.DataFrame()

        # polite delay after schedule scrape
        self.respect_rate_limit()

        # step 2: scrape ALL team activity (all competitions)
        team_ids = get_fbref_team_ids_for_season(season_end_year)
        self.logger.info(
            f"Scraping complete activity logs for {len(team_ids)} teams (all competitions)"
        )

        all_team_activity = []
        failed_teams = []

        for idx, (team_name, team_id) in enumerate(team_ids.items()):
            self.logger.info(f"Scraping {team_name} ({idx + 1}/{len(team_ids)})")

            team_data = self._scrape_team_shooting_logs(
                team_id, season_string, team_name
            )

            if not team_data.empty:
                # Keep ALL competitions - critical for schedule features
                all_team_activity.append(team_data)
            else:
                # track failed teams for retry
                failed_teams.append((team_name, team_id))

            # polite delay between teams
            if idx < len(team_ids) - 1:
                self.respect_rate_limit()

        # retry failed teams once with longer delay
        if failed_teams:
            self.logger.info(
                f"Retrying {len(failed_teams)} failed teams with longer delays"
            )

            for retry_num, (team_name, team_id) in enumerate(failed_teams):
                retry_delay = random.uniform(15, 20)
                self.logger.info(f"  Waiting {retry_delay:.1f}s before retry...")
                time.sleep(retry_delay)

                self.logger.info(
                    f"  Retry: {team_name} ({retry_num + 1}/{len(failed_teams)})"
                )

                team_data = self._scrape_team_shooting_logs(
                    team_id, season_string, team_name
                )

                if not team_data.empty:
                    all_team_activity.append(team_data)
                    self.logger.info(f"    ✓ Retry successful for {team_name}")
                else:
                    self.logger.warning(f"    ✗ Retry failed for {team_name}")

                # apply rate limiting between retries
                if retry_num < len(failed_teams) - 1:
                    self.respect_rate_limit()

        if not all_team_activity:
            self.logger.warning("No team activity data retrieved")
            return pd.DataFrame()

        # combine all team activity (ALL competitions)
        combined_team_activity = pd.concat(all_team_activity, ignore_index=True)
        self.logger.info(
            f"Retrieved {len(combined_team_activity)} team match records across all competitions"
        )

        # step 3: calculate schedule features using ALL team activity
        bundesliga_schedule = self._calculate_schedule_features_from_all_activity(
            bundesliga_schedule, combined_team_activity
        )

        # step 4: merge Bundesliga shooting statistics
        bundesliga_shooting = combined_team_activity[
            combined_team_activity["competition"] == "Bundesliga"
        ].copy()

        if not bundesliga_shooting.empty:
            self.logger.info(
                f"Merging shooting statistics from {len(bundesliga_shooting)} Bundesliga team records"
            )
            bundesliga_schedule = self._merge_shooting_stats_into_schedule(
                bundesliga_schedule, bundesliga_shooting
            )
        else:
            self.logger.warning(
                "No Bundesliga shooting statistics found - statistics will be missing"
            )

        # add season information
        bundesliga_schedule["season_end_year"] = season_end_year
        bundesliga_schedule["season"] = f"{season_end_year - 1}/{season_end_year}"

        self.logger.info(f"Completed: {len(bundesliga_schedule)} Bundesliga matches")
        return bundesliga_schedule

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

    def _scrape_bundesliga_schedule(self, season_end_year: int) -> pd.DataFrame:
        """Scrape Bundesliga schedule from the competition schedule page"""
        season_string = f"{season_end_year - 1}-{season_end_year}"

        # construct schedule url (competition ID 20 = Bundesliga)
        url = f"https://fbref.com/en/comps/20/{season_string}/schedule/{season_string}-Bundesliga-Scores-and-Fixtures"

        self.logger.info(f"Scraping Bundesliga schedule for {season_string}")

        try:
            self.logger.debug(f"  Fetching: {url}")
            self.driver.get(url)

            # wait for cloudflare verification
            wait_time = (
                random.uniform(10, 15) if self.headless else random.uniform(8, 12)
            )
            self.logger.debug(
                f"  Waiting {wait_time:.1f}s for Cloudflare verification..."
            )
            time.sleep(wait_time)

            # handle cookie consent
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
                    time.sleep(0.5)
                    break
                except (TimeoutException, Exception):
                    continue

            # wait for the schedule table to load
            # try multiple possible table IDs (FBRef uses different IDs for different pages)
            table = None
            possible_table_ids = [
                "sched_all",
                "schedule",
                f"sched_{season_end_year - 1}_{season_end_year}_20_1",
            ]

            for table_id in possible_table_ids:
                try:
                    wait = WebDriverWait(self.driver, 5)
                    table = wait.until(
                        EC.visibility_of_element_located((By.ID, table_id))
                    )
                    self.logger.debug(f"  Found table with ID: {table_id}")
                    break
                except TimeoutException:
                    continue

            if table is None:
                # if specific IDs don't work, try finding any table with 'sched' in the ID
                self.logger.debug("  Trying to find schedule table by partial ID match")
                all_tables = self.driver.find_elements(By.TAG_NAME, "table")
                for t in all_tables:
                    table_id = t.get_attribute("id")
                    if table_id and "sched" in table_id.lower():
                        table = t
                        self.logger.debug(f"  Found table with ID: {table_id}")
                        break

            if table is None:
                all_tables = self.driver.find_elements(By.TAG_NAME, "table")
                table_ids = [
                    t.get_attribute("id") for t in all_tables if t.get_attribute("id")
                ]
                self.logger.warning("  ✗ Schedule table not found")
                self.logger.debug(f"    Available tables: {table_ids}")
                return pd.DataFrame()

            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            self.logger.debug(f"  Found {len(rows)} rows in schedule table")

            matches = []
            skipped_rows = 0

            for row in rows:
                try:
                    row_data = self._extract_row_data(row)

                    # debug: log first row's field names
                    if len(matches) == 0 and row_data:
                        self.logger.debug(
                            f"  Sample fields from first row: {list(row_data.keys())[:15]}"
                        )

                    # extract key fields
                    date_str = row_data.get("date")
                    home_team = row_data.get("home_team") or row_data.get("squad_a")
                    away_team = row_data.get("away_team") or row_data.get("squad_b")

                    # skip if essential fields are missing
                    if not date_str or not home_team or not away_team:
                        skipped_rows += 1
                        continue

                    # parse date
                    try:
                        match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        # try alternative date format
                        skipped_rows += 1
                        continue

                    # extract scores (if match has been played)
                    score_str = row_data.get("score")
                    home_goals = None
                    away_goals = None

                    if score_str and "–" in score_str:
                        try:
                            parts = score_str.split("–")
                            home_goals = int(parts[0].strip())
                            away_goals = int(parts[1].strip())
                        except (ValueError, IndexError):
                            pass

                    match_record = {
                        "date": match_date,
                        "home_team": standardise_team_name(home_team.strip()),
                        "away_team": standardise_team_name(away_team.strip()),
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "competition": "Bundesliga",
                        "venue": row_data.get("venue", ""),
                        "match_report": row_data.get("match_report", ""),
                    }

                    matches.append(match_record)

                except Exception as e:
                    self.logger.debug(f"  Error parsing row: {e}")
                    skipped_rows += 1
                    continue

            if skipped_rows > 0:
                self.logger.debug(f"  Skipped {skipped_rows} rows (headers/empty)")

            if matches:
                df = pd.DataFrame(matches)
                self.logger.info(f"  ✓ {len(df)} Bundesliga matches")
                return df
            else:
                self.logger.warning("  ✗ No valid match data extracted from schedule")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"  ✗ Error scraping schedule: {e}")
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

    def _calculate_schedule_features_from_all_activity(
        self, bundesliga_matches: pd.DataFrame, all_team_activity: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate schedule features for Bundesliga matches"""
        if bundesliga_matches.empty or all_team_activity.empty:
            return bundesliga_matches

        self.logger.info(
            "Calculating schedule features from complete team activity (all competitions)"
        )

        # ensure date columns are date type
        bundesliga_matches["date"] = pd.to_datetime(bundesliga_matches["date"]).dt.date
        all_team_activity["date"] = pd.to_datetime(all_team_activity["date"]).dt.date

        # build a team activity lookup (all competitions, all venues)
        team_activity_log = []
        for _, row in all_team_activity.iterrows():
            # each team match becomes an activity record
            team_activity_log.append(
                {
                    "date": row["date"],
                    "team": row["team"],
                    "venue": row["venue"],  # "Home" or "Away"
                    "competition": row["competition"],
                }
            )

        team_activity_df = pd.DataFrame(team_activity_log)
        team_activity_df = team_activity_df.sort_values(["team", "date"]).reset_index(
            drop=True
        )

        # initialise feature columns
        bundesliga_matches["home_rest_days"] = None
        bundesliga_matches["home_consecutive_away_games"] = 0
        bundesliga_matches["away_rest_days"] = None
        bundesliga_matches["away_consecutive_away_games"] = 0

        # calculate features for each Bundesliga match
        for idx, match in bundesliga_matches.iterrows():
            match_date = match["date"]
            home_team = match["home_team"]
            away_team = match["away_team"]

            # calculate home team features
            home_activity = team_activity_df[
                (team_activity_df["team"] == home_team)
                & (team_activity_df["date"] < match_date)
            ].sort_values("date")

            if len(home_activity) > 0:
                # rest days since last match (any competition)
                last_match = home_activity.iloc[-1]
                rest_days = (match_date - last_match["date"]).days
                bundesliga_matches.at[idx, "home_rest_days"] = rest_days

                # count consecutive away games BEFORE this match
                consecutive_away = 0
                for _, prev_match in home_activity.iloc[::-1].iterrows():
                    if prev_match["venue"] == "Away":
                        consecutive_away += 1
                    else:
                        break  # stop at first home game
                bundesliga_matches.at[idx, "home_consecutive_away_games"] = (
                    consecutive_away
                )

            # calculate away team features
            away_activity = team_activity_df[
                (team_activity_df["team"] == away_team)
                & (team_activity_df["date"] < match_date)
            ].sort_values("date")

            if len(away_activity) > 0:
                # rest days since last match (any competition)
                last_match = away_activity.iloc[-1]
                rest_days = (match_date - last_match["date"]).days
                bundesliga_matches.at[idx, "away_rest_days"] = rest_days

                # count consecutive away games BEFORE this match
                consecutive_away = 0
                for _, prev_match in away_activity.iloc[::-1].iterrows():
                    if prev_match["venue"] == "Away":
                        consecutive_away += 1
                    else:
                        break  # stop at first home game
                bundesliga_matches.at[idx, "away_consecutive_away_games"] = (
                    consecutive_away
                )

        self.logger.info("Schedule features calculated from complete activity")
        return bundesliga_matches


    def _merge_shooting_stats_into_schedule(
        self, schedule_df: pd.DataFrame, shooting_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge shooting statistics into schedule-based match data"""
        if schedule_df.empty:
            return schedule_df

        self.logger.info("Merging shooting statistics into schedule")

        # ensure date columns are the same type
        schedule_df["date"] = pd.to_datetime(schedule_df["date"]).dt.date
        shooting_data["date"] = pd.to_datetime(shooting_data["date"]).dt.date

        # for each match in schedule, find matching shooting stats for both teams
        for idx, row in schedule_df.iterrows():
            date = row["date"]
            home_team = row["home_team"]
            away_team = row["away_team"]

            # find home team's shooting stats
            home_shooting = shooting_data[
                (shooting_data["date"] == date)
                & (shooting_data["team"] == home_team)
                & (shooting_data["opponent"] == away_team)
            ]

            if len(home_shooting) > 0:
                home_stats = home_shooting.iloc[0]
                # merge shooting stats for home team
                for col in [
                    "npxg",
                    "pens_made",
                    "pens_att",
                    "shots",
                    "shots_on_target",
                    "avg_shot_distance",
                    "shots_free_kicks",
                ]:
                    if col in home_stats:
                        schedule_df.at[idx, f"home_{col}"] = home_stats[col]

            # find away team's shooting stats
            away_shooting = shooting_data[
                (shooting_data["date"] == date)
                & (shooting_data["team"] == away_team)
                & (shooting_data["opponent"] == home_team)
            ]

            if len(away_shooting) > 0:
                away_stats = away_shooting.iloc[0]
                # merge shooting stats for away team
                for col in [
                    "npxg",
                    "pens_made",
                    "pens_att",
                    "shots",
                    "shots_on_target",
                    "avg_shot_distance",
                    "shots_free_kicks",
                ]:
                    if col in away_stats:
                        schedule_df.at[idx, f"away_{col}"] = away_stats[col]

        self.logger.info("Shooting statistics merged")
        return schedule_df

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
