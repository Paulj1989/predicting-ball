# scripts/run_data_pipeline.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import duckdb
import pandas as pd

from src.processing.data_processor import DataProcessor
from src.scrapers.fbref_scraper import FBRefScraper
from src.scrapers.transfermarkt_scraper import TransfermarktScraper
from src.scrapers.odds_scraper import OddsScraper


def setup_logging() -> logging.Logger:
    """Configure logging for the pipeline"""
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


logger = setup_logging()


def setup_database(db_path: str = "data/club_football.duckdb"):
    """
    Initialise database with required schemas.

    Creates the following schemas:
    - raw: scraped data
    - processed: integrated data (post-merging)
    - features: feature-engineered data (post-preparation)
    - predictions: model predictions
    - metadata: pipeline metadata
    """
    conn = duckdb.connect(db_path)

    try:
        # create all required schemas
        schemas = ["raw", "processed", "features", "predictions", "metadata"]
        for schema in schemas:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        logger.info("Database schemas initialised")
    finally:
        conn.close()


def determine_current_season() -> int:
    """Determine current Bundesliga season"""
    current_date = datetime.now()
    current_month = current_date.month
    return current_date.year + 1 if current_month >= 8 else current_date.year


def get_seasons_to_scrape(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    start_year: int,
    end_year: int,
    force_rescrape: bool = False,
) -> Tuple[List[int], int]:
    """Determine which seasons need scraping based on existing data"""
    current_season = determine_current_season()

    if force_rescrape:
        logger.info(
            f"Force rescrape enabled - will scrape all seasons {start_year}-{end_year}"
        )
        return list(range(start_year, end_year + 1)), current_season

    # map source to table name
    table_map = {
        "fbref": "raw.match_logs_fbref",
        "transfermarkt": "raw.squad_values_tm",
        "odds": "raw.historical_odds",
    }

    if source not in table_map:
        return list(range(start_year, end_year + 1)), current_season

    table_name = table_map[source]
    table_short_name = table_name.split(".")[1]

    try:
        # check if table exists
        table_exists = (
            conn.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'raw' AND table_name = '{table_short_name}'
        """).fetchone()[0]
            > 0
        )

        if not table_exists:
            logger.info(f"Table {table_name} doesn't exist - will scrape all seasons")
            return list(range(start_year, end_year + 1)), current_season

        # get seasons already in database
        season_col = "season_end_year" if source != "odds" else "Season"

        existing_seasons = conn.execute(f"""
            SELECT DISTINCT {season_col} as season_end_year, COUNT(*) as record_count
            FROM {table_name}
            GROUP BY {season_col}
            ORDER BY {season_col}
        """).df()

        if existing_seasons.empty:
            logger.info(f"No data in {table_name} - will scrape all seasons")
            return list(range(start_year, end_year + 1)), current_season

        # determine which seasons need scraping
        seasons_to_scrape = _identify_missing_seasons(
            existing_seasons, source, start_year, end_year, current_season
        )

        if seasons_to_scrape:
            logger.info(
                f"Will scrape {len(seasons_to_scrape)} seasons for {source}: {seasons_to_scrape}"
            )
        else:
            logger.info(f"All seasons already scraped for {source}")

        return seasons_to_scrape, current_season

    except Exception as e:
        logger.error(f"Error checking existing data for {source}: {e}")
        logger.info("Defaulting to scraping all seasons")
        return list(range(start_year, end_year + 1)), current_season


def _identify_missing_seasons(
    existing_seasons: pd.DataFrame,
    source: str,
    start_year: int,
    end_year: int,
    current_season: int,
) -> List[int]:
    """Identify which seasons are missing or need updating"""
    seasons_to_scrape = []

    for year in range(start_year, end_year + 1):
        # convert year to season string for odds (e.g., 2024 -> '2324')
        if source == "odds":
            season_str = f"{str(year - 1)[-2:]}{str(year)[-2:]}"
            season_exists = season_str in existing_seasons["season_end_year"].values
        else:
            season_exists = year in existing_seasons["season_end_year"].values

        if year == current_season:
            # always update current season
            seasons_to_scrape.append(year)
            logger.info(f"Season {year} is current season - will update")
        elif not season_exists:
            # season not in database
            seasons_to_scrape.append(year)
            logger.info(f"Season {year} not in database - will scrape")
        else:
            # season exists - log and skip
            season_key = season_str if source == "odds" else year
            season_data = existing_seasons[
                existing_seasons["season_end_year"] == season_key
            ]
            logger.info(
                f"Season {year} already scraped with {season_data['record_count'].iloc[0]} records - skipping"
            )

    return seasons_to_scrape


def scrape_transfermarkt(
    conn: duckdb.DuckDBPyConnection, seasons: List[int], force_rescrape: bool = False
):
    """Scrape Transfermarkt squad values"""
    logger.info("=" * 60)
    logger.info("TRANSFERMARKT SQUAD VALUES")
    logger.info("=" * 60)

    if not seasons:
        logger.info("Transfermarkt data up to date - skipping")
        return

    scraper = TransfermarktScraper()

    # load existing data if any
    existing_data = _load_existing_data(conn, "raw.squad_values_tm", "Transfermarkt")

    # scrape new seasons
    new_data = scraper.scrape_multiple_seasons(min(seasons), max(seasons))

    if not new_data.empty:
        # combine with existing data
        combined_data = _combine_with_existing(
            existing_data, new_data, seasons, "season_end_year"
        )

        # save to database
        _save_to_database(conn, combined_data, "raw.squad_values_tm", "squad values")
    else:
        logger.warning("No new Transfermarkt data retrieved")


def scrape_fbref(
    conn: duckdb.DuckDBPyConnection, seasons: List[int], force_rescrape: bool = False
):
    """Scrape FBRef match logs and advanced statistics"""
    logger.info("=" * 60)
    logger.info("FBREF MATCH LOGS + ADVANCED STATS")
    logger.info("=" * 60)

    if not seasons:
        logger.info("FBRef data up to date - skipping")
        return

    scraper = FBRefScraper(headless=True)

    # load existing data
    existing_data = _load_existing_data(conn, "raw.match_logs_fbref", "FBRef")

    # scrape new seasons
    new_data = scraper.scrape_multiple_seasons(min(seasons), max(seasons))

    if not new_data.empty:
        # combine with existing data
        combined_data = _combine_with_existing(
            existing_data, new_data, seasons, "season_end_year"
        )

        # save to database
        _save_to_database(conn, combined_data, "raw.match_logs_fbref", "match records")
    else:
        logger.warning("No new FBRef data retrieved")


def scrape_odds(
    conn: duckdb.DuckDBPyConnection, seasons: List[int], force_rescrape: bool = False
):
    """Scrape betting odds (historical and upcoming)"""
    logger.info("=" * 60)
    logger.info("BETTING ODDS (Historical + Upcoming)")
    logger.info("=" * 60)

    try:
        scraper = OddsScraper()

        # scrape historical odds
        _scrape_historical_odds(conn, scraper, seasons, force_rescrape)

        # scrape upcoming odds (always fetch for current fixtures)
        _scrape_upcoming_odds(conn, scraper)

    except ValueError as e:
        logger.error(f"Odds scraper initialisation failed: {e}")
        logger.error("Make sure ODDS_API_KEY is set in your .env file")
    except Exception as e:
        logger.error(f"Odds scraping failed: {e}")
        logger.error("Continuing with pipeline without odds data")


def _scrape_historical_odds(
    conn: duckdb.DuckDBPyConnection,
    scraper: OddsScraper,
    seasons: List[int],
    force_rescrape: bool,
):
    """Scrape historical odds from football-data.co.uk"""
    logger.info("\nHistorical Odds (football-data.co.uk):")

    if not seasons and not force_rescrape:
        logger.info("Historical odds data up to date - skipping")
        return

    # convert year format for odds scraper (e.g., 2024 -> '2324')
    season_strings = [f"{str(year - 1)[-2:]}{str(year)[-2:]}" for year in seasons]
    logger.info(f"Scraping historical odds for seasons: {season_strings}")

    # load existing data
    existing_odds = _load_existing_data(conn, "raw.historical_odds", "historical odds")

    # scrape new seasons
    new_odds = scraper.get_historical_odds(seasons=season_strings)

    if not new_odds.empty:
        # combine with existing
        if not existing_odds.empty:
            # remove old data for seasons being updated
            existing_odds = existing_odds[~existing_odds["Season"].isin(season_strings)]
            combined_odds = pd.concat([existing_odds, new_odds], ignore_index=True)
            logger.info(
                f"Combined {len(existing_odds)} existing + {len(new_odds)} new = {len(combined_odds)} total"
            )
        else:
            combined_odds = new_odds

        # save to database
        _save_to_database(
            conn, combined_odds, "raw.historical_odds", "historical odds records"
        )
    else:
        logger.warning("No new historical odds retrieved")


def _scrape_upcoming_odds(conn: duckdb.DuckDBPyConnection, scraper: OddsScraper):
    """Scrape upcoming odds from The Odds API."""
    logger.info("\nUpcoming Odds (The Odds API):")

    upcoming_odds = scraper.get_upcoming_odds()

    if not upcoming_odds.empty:
        # calculate consensus odds
        upcoming_odds = scraper.calculate_consensus_odds(upcoming_odds)

        # add timestamp
        upcoming_odds["fetch_timestamp"] = datetime.now()

        # save to database
        _save_to_database(conn, upcoming_odds, "raw.upcoming_odds", "upcoming matches")

        # show next fixtures
        _display_upcoming_fixtures(upcoming_odds)
    else:
        logger.info("No upcoming matches with odds available")


def _display_upcoming_fixtures(upcoming_odds: pd.DataFrame, num_fixtures: int = 5):
    """Display next upcoming fixtures"""
    if len(upcoming_odds) > 0:
        logger.info(f"\nNext {num_fixtures} fixtures:")
        next_fixtures = upcoming_odds.nsmallest(num_fixtures, "commence_time")

        for _, match in next_fixtures.iterrows():
            logger.info(
                f"  {match['commence_time'].strftime('%Y-%m-%d %H:%M')}: "
                f"{match['home_team']} vs {match['away_team']} "
                f"(Odds: {match['consensus_home_odds']:.2f} / "
                f"{match['consensus_draw_odds']:.2f} / "
                f"{match['consensus_away_odds']:.2f})"
            )


def _load_existing_data(
    conn: duckdb.DuckDBPyConnection, table_name: str, description: str
) -> pd.DataFrame:
    """Load existing data from database table"""
    try:
        data = conn.execute(f"SELECT * FROM {table_name}").df()
        logger.info(f"Loaded {len(data)} existing {description} records")
        return data
    except:
        logger.info(f"No existing {description} data")
        return pd.DataFrame()


def _combine_with_existing(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    seasons_to_update: List[int],
    season_column: str,
) -> pd.DataFrame:
    """
    Combine new scraped data with existing data.

    Removes old data for seasons being updated and concatenates.
    Handles column mismatches by adding missing columns with NA.
    """
    if existing_data.empty:
        return new_data

    # remove old data for seasons being updated
    existing_data = existing_data[~existing_data[season_column].isin(seasons_to_update)]

    # align columns before concatenation
    all_columns = list(set(existing_data.columns) | set(new_data.columns))

    for col in all_columns:
        if col not in existing_data.columns:
            existing_data[col] = pd.NA
        if col not in new_data.columns:
            new_data[col] = pd.NA

    # concatenate
    combined = pd.concat([existing_data, new_data], ignore_index=True)

    logger.info(
        f"Combined {len(existing_data)} existing + {len(new_data)} new = {len(combined)} total"
    )

    return combined


def _save_to_database(
    conn: duckdb.DuckDBPyConnection,
    data: pd.DataFrame,
    table_name: str,
    description: str,
):
    """Save DataFrame to database table"""
    temp_name = "temp_data"
    conn.register(temp_name, data)
    conn.execute(f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} AS
        SELECT * FROM {temp_name}
    """)
    logger.info(f"Saved {len(data)} total {description}")


def run_scrapers(
    start_year: int, end_year: Optional[int] = None, force_rescrape: bool = False
):
    """Execute all scrapers with incremental updates"""
    if end_year is None:
        end_year = determine_current_season()

    logger.info(f"Checking data coverage for seasons {start_year}-{end_year}")

    conn = duckdb.connect("data/club_football.duckdb")

    try:
        # determine which seasons need scraping for each source
        tm_seasons, current_season = get_seasons_to_scrape(
            conn, "transfermarkt", start_year, end_year, force_rescrape
        )
        fb_seasons, _ = get_seasons_to_scrape(
            conn, "fbref", start_year, end_year, force_rescrape
        )
        odds_seasons, _ = get_seasons_to_scrape(
            conn, "odds", start_year, end_year, force_rescrape
        )

        # run each scraper
        scrape_transfermarkt(conn, tm_seasons, force_rescrape)
        scrape_fbref(conn, fb_seasons, force_rescrape)
        scrape_odds(conn, odds_seasons, force_rescrape)

    finally:
        conn.close()


def integrate_data():
    """Integrate all scraped data sources"""
    logger.info("=" * 60)
    logger.info("INTEGRATING DATA SOURCES")
    logger.info("=" * 60)

    processor = DataProcessor()
    processor.integrate_all_data()


def display_pipeline_summary():
    """Display summary of available data after pipeline completion"""
    logger.info("\nData Available:")

    conn = duckdb.connect("data/club_football.duckdb")

    try:
        # raw data tables
        _display_table_count(
            conn, "raw.squad_values_tm", "Squad values (Transfermarkt)"
        )
        _display_table_count(conn, "raw.match_logs_fbref", "Match results & xG (FBRef)")

        # odds tables
        try:
            hist_odds = conn.execute(
                "SELECT COUNT(*) FROM raw.historical_odds"
            ).fetchone()[0]
            upcoming_odds = conn.execute(
                "SELECT COUNT(*) FROM raw.upcoming_odds"
            ).fetchone()[0]
            logger.info(
                f"  Betting odds: {hist_odds} historical, {upcoming_odds} upcoming"
            )
        except:
            logger.info("  Betting odds: Not available")

        # integrated data
        _display_table_count(conn, "processed.integrated_data", "\n  Integrated data")

    finally:
        conn.close()


def _display_table_count(
    conn: duckdb.DuckDBPyConnection, table_name: str, description: str
):
    """Display record count for a table"""
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info(f"  {description}: {count} records")
    except:
        logger.info(f"  {description}: Not available")


def main(
    start_year: int = 2021, end_year: Optional[int] = None, force_rescrape: bool = False
):
    """Execute complete data pipeline"""
    logger.info("=" * 80)
    logger.info("BUNDESLIGA DATA PIPELINE")
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Mode: {'FORCE RESCRAPE' if force_rescrape else 'INCREMENTAL UPDATE'}")
    logger.info("=" * 80)

    try:
        # step 1: database setup
        logger.info("\n[Step 1/3] Setting up database")
        setup_database()

        # step 2: data collection
        logger.info("\n[Step 2/3] Running scrapers")
        run_scrapers(start_year, end_year, force_rescrape)

        # step 3: data integration
        logger.info("\n[Step 3/3] Integrating data sources")
        integrate_data()

        # display summary
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Finished at: {datetime.now()}")
        logger.info("=" * 80)

        display_pipeline_summary()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bundesliga data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run incremental update for current season
  python scripts/run_data_pipeline.py

  # Scrape specific seasons
  python scripts/run_data_pipeline.py --start-year 2020 --end-year 2024

  # Force rescrape all data
  python scripts/run_data_pipeline.py --force-rescrape
        """,
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year for scraping (default: 2021)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for scraping (default: current season)",
    )
    parser.add_argument(
        "--force-rescrape",
        action="store_true",
        help="Force rescrape all data, ignoring existing data",
    )

    args = parser.parse_args()

    # create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    main(
        start_year=args.start_year,
        end_year=args.end_year,
        force_rescrape=args.force_rescrape,
    )
