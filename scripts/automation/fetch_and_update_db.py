#!/usr/bin/env python3
"""
Fetch and Update Database
==========================

Fetch latest match data and update DuckDB database.

Usage:
    python scripts/automation/fetch_and_update_db.py [--dry-run]
"""

import duckdb
import pandas as pd
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# import existing data fetching functions
from src.scrapers.fbref_scraper import FBRefScraper
from src.scrapers.transfermarkt_scraper import TransfermarktScraper
from src.scrapers.odds_scraper import OddsScraper
from src.utils import determine_current_season, format_season_string
from src.processing.data_processor import DataProcessor

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fetch latest data and update database"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Run without making changes to database"
    )

    return parser.parse_args()


def align_dataframe_columns(new_df, table_name, conn):
    """
    Align new dataframe columns with existing table schema.
    Adds missing columns with NULL, removes extra columns.
    """
    try:
        # get existing table columns
        existing_cols = conn.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = '{table_name.split(".")[-1]}'
            ORDER BY ordinal_position
        """).fetchdf()

        if existing_cols.empty:
            # table doesn't exist yet, return as-is
            return new_df

        existing_col_list = existing_cols["column_name"].tolist()
        new_col_list = new_df.columns.tolist()

        # find differences
        missing_in_new = set(existing_col_list) - set(new_col_list)
        extra_in_new = set(new_col_list) - set(existing_col_list)

        if missing_in_new:
            print(f"   Adding {len(missing_in_new)} missing columns with NULL values")
            for col in missing_in_new:
                new_df[col] = pd.NA

        if extra_in_new:
            print(f"   Removing {len(extra_in_new)} extra columns not in table")
            new_df = new_df.drop(columns=list(extra_in_new))

        # reorder columns to match table schema
        new_df = new_df[existing_col_list]

        return new_df

    except Exception as e:
        print(f"   Warning: Could not align columns: {e}")
        return new_df


def update_database(dry_run=False):
    """Fetch latest data and update database"""

    db_path = "data/club_football.duckdb"

    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        print("   Run download_database.py first")
        sys.exit(1)

    print("=" * 70)
    print("UPDATING DATABASE WITH LATEST DATA")
    print("=" * 70)
    if dry_run:
        print("DRY RUN MODE - no changes will be saved")
        print("=" * 70)

    conn = duckdb.connect(db_path if not dry_run else ":memory:")

    if dry_run:
        conn.execute(f"ATTACH '{db_path}' AS prod (READ_ONLY)")
        try:
            # copy schema structure for dry run
            tables_to_copy = ["match_logs_fbref", "squad_values_tm"]
            for table in tables_to_copy:
                try:
                    conn.execute(f"""
                        CREATE TABLE raw.{table} AS
                        SELECT * FROM prod.raw.{table} LIMIT 0
                    """)
                except:
                    pass
        except Exception as e:
            print(f"   Could not setup dry run: {e}")
        conn.execute("DETACH prod")

    try:
        # determine current season automatically
        current_season_year = determine_current_season()

        # fetch latest FBRef match results
        print("\n1. Fetching latest Bundesliga matches...")
        print(
            f"   Season: {format_season_string(current_season_year)} (season_end_year={current_season_year})"
        )

        fbref_scraper = FBRefScraper(headless=True)
        latest_matches = fbref_scraper.scrape_multiple_seasons(
            current_season_year, current_season_year
        )

        if not latest_matches.empty:
            print(f"   Found {len(latest_matches)} matches for current season")

            # count played vs unplayed
            played_count = latest_matches["home_goals"].notna().sum()
            upcoming_count = latest_matches["home_goals"].isna().sum()
            print(f"   Played: {played_count}, Upcoming: {upcoming_count}")

            # align columns with existing table
            latest_matches = align_dataframe_columns(
                latest_matches, "raw.match_logs_fbref", conn
            )

            if not dry_run:
                # delete existing data for current season
                conn.execute(
                    """
                    DELETE FROM raw.match_logs_fbref
                    WHERE season_end_year = ?
                """,
                    [current_season_year],
                )

                # insert new data
                conn.register("new_matches", latest_matches)
                conn.execute("""
                    INSERT INTO raw.match_logs_fbref
                    SELECT * FROM new_matches
                """)
                print(f"   Updated {len(latest_matches)} matches")
            else:
                print(f"   Would update {len(latest_matches)} matches")
        else:
            print("   No new matches found")

        # fetch latest squad values
        print("\n2. Fetching latest squad values...")
        tm_scraper = TransfermarktScraper()
        squad_values = tm_scraper.scrape_multiple_seasons(
            current_season_year, current_season_year
        )

        if not squad_values.empty:
            print(f"   Found values for {len(squad_values)} teams")

            # align columns
            squad_values = align_dataframe_columns(
                squad_values, "raw.squad_values_tm", conn
            )

            if not dry_run:
                conn.execute(
                    """
                    DELETE FROM raw.squad_values_tm
                    WHERE season_end_year = ?
                """,
                    [current_season_year],
                )

                conn.register("new_values", squad_values)
                conn.execute("""
                    INSERT INTO raw.squad_values_tm
                    SELECT * FROM new_values
                """)
                print("   Squad values updated")
            else:
                print(f"   Would update {len(squad_values)} squad values")

        # fetch upcoming odds
        print("\n3. Fetching upcoming odds...")
        try:
            odds_scraper = OddsScraper()
            upcoming_odds = odds_scraper.get_upcoming_odds()

            if not upcoming_odds.empty:
                upcoming_odds = odds_scraper.calculate_consensus_odds(upcoming_odds)
                upcoming_odds["fetch_timestamp"] = datetime.now()

                print(f"   Found odds for {len(upcoming_odds)} fixtures")
                if not dry_run:
                    conn.register("upcoming", upcoming_odds)
                    conn.execute("""
                        DROP TABLE IF EXISTS raw.upcoming_odds;
                        CREATE TABLE raw.upcoming_odds AS SELECT * FROM upcoming
                    """)
                    print("   Upcoming odds updated")
                else:
                    print(f"   Would update {len(upcoming_odds)} odds")
            else:
                print("   No upcoming fixtures with odds")
        except ValueError as e:
            print(f"   Could not fetch odds: {e}")
            print("   Continuing without odds data")
        except Exception as e:
            print(f"   Error fetching odds: {e}")
            print("   Continuing without odds data")

        # show summary
        if not dry_run:
            try:
                match_count = conn.execute(
                    "SELECT COUNT(*) FROM raw.match_logs_fbref"
                ).fetchone()[0]
                played_in_db = conn.execute("""
                    SELECT COUNT(*) FROM raw.match_logs_fbref
                    WHERE home_goals IS NOT NULL
                """).fetchone()[0]
                upcoming_in_db = conn.execute(
                    """
                    SELECT COUNT(*) FROM raw.match_logs_fbref
                    WHERE home_goals IS NULL AND season_end_year = ?
                """,
                    [current_season_year],
                ).fetchone()[0]

                print("\n   Database summary:")
                print(f"   Total matches: {match_count}")
                print(f"   Played matches: {played_in_db}")
                print(
                    f"   Upcoming fixtures (season {current_season_year}): {upcoming_in_db}"
                )
            except:
                pass

        conn.close()

        # integrate data sources to update processed tables
        if not dry_run:
            print("\n4. Integrating data sources...")

            processor = DataProcessor(db_path=db_path)
            processor.integrate_all_data()

        print("\n" + "=" * 70)
        print("DATABASE UPDATE COMPLETE")
        print("=" * 70)

        if dry_run:
            print("DRY RUN - no changes were saved")

        return True

    except Exception as e:
        print(f"\nError updating database: {e}")
        import traceback

        traceback.print_exc()
        if "conn" in locals():
            conn.close()
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    update_database(dry_run=args.dry_run)
