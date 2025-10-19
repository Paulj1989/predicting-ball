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
        conn.execute(f"ATTACH '{db_path}' AS prod")
        try:
            conn.execute(
                "CREATE TABLE match_logs_fbref AS SELECT * FROM prod.raw.match_logs_fbref LIMIT 0"
            )
        except:
            print("   Could not create temp table structure")
        conn.execute("DETACH prod")

    try:
        # fetch latest fbref match results
        print("\n1. Fetching latest Bundesliga matches...")
        current_season_year = 2025

        fbref_scraper = FBRefScraper(headless=True)
        latest_matches = fbref_scraper.scrape_multiple_seasons(
            current_season_year, current_season_year
        )

        if not latest_matches.empty:
            print(f"   Found {len(latest_matches)} matches for current season")

            if not dry_run:
                # update database
                conn.execute(
                    """
                    DELETE FROM raw.match_logs_fbref
                    WHERE season_end_year = ?
                """,
                    [current_season_year],
                )

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

        # show summary
        if not dry_run:
            match_count = conn.execute(
                "SELECT COUNT(*) FROM raw.match_logs_fbref"
            ).fetchone()[0]
            print(f"\n   Total matches in database: {match_count}")

        print("\n" + "=" * 70)
        print("DATABASE UPDATE COMPLETE")
        print("=" * 70)

        if dry_run:
            print("DRY RUN - no changes were saved")

        conn.close()
        return True

    except Exception as e:
        print(f"\nError updating database: {e}")
        import traceback

        traceback.print_exc()
        conn.close()
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    update_database(dry_run=args.dry_run)
