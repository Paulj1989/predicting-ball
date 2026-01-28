#!/usr/bin/env python3
"""Validate that weekly update completed successfully"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import duckdb
import pickle
import pandas as pd


def validate_update():
    """Run validation checks on updated outputs"""

    errors = []
    warnings = []

    print("=" * 70)
    print("VALIDATING WEEKLY UPDATE")
    print("=" * 70)

    # Check database
    print("\n1. Checking database...")
    db_path = Path("data/pb.duckdb")

    if not db_path.exists():
        errors.append("Database file not found")
    else:
        con = duckdb.connect(str(db_path))

        try:
            # Check we have recent matches
            latest_match = con.execute("""
                SELECT MAX(date)::DATE FROM models.match_features
            """).fetchone()[0]

            if latest_match:
                days_old = (datetime.now().date() - latest_match).days

                if days_old > 14:
                    warnings.append(f"Latest match is {days_old} days old")
                else:
                    print(f"   Latest match from {days_old} days ago")
            else:
                warnings.append("No matches found in database")
        except Exception as e:
            errors.append(f"Could not query matches: {e}")

        con.close()

    # Check model
    print("\n2. Checking model...")
    model_path = Path("outputs/models/production_model.pkl")

    if not model_path.exists():
        errors.append("Model file not found")
    else:
        model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)

        if model_age > timedelta(days=8):
            warnings.append(f"Model is {model_age.days} days old")
        else:
            print(
                f"   Model updated {model_age.days} days, {model_age.seconds // 3600} hours ago"
            )

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("   Model loads successfully")
        except Exception as e:
            errors.append(f"Model failed to load: {e}")

    # Check predictions
    print("\n3. Checking predictions...")
    pred_path = Path("outputs/predictions/next_fixtures.csv")

    if not pred_path.exists():
        errors.append("Predictions file not found")
    else:
        try:
            preds = pd.read_csv(pred_path)

            if len(preds) == 0:
                warnings.append("No upcoming fixtures in predictions")
            else:
                print(f"   {len(preds)} fixtures predicted")

            # Validate probabilities sum to 1
            if "home_win" in preds.columns:
                prob_sum = preds["home_win"] + preds["draw"] + preds["away_win"]

                if not prob_sum.between(0.99, 1.01).all():
                    errors.append("Prediction probabilities don't sum to 1")
                else:
                    print("   Probabilities valid")
        except Exception as e:
            errors.append(f"Error reading predictions: {e}")

    # Check simulations
    print("\n4. Checking simulations...")
    sim_path = Path("outputs/predictions/season_projections.csv")

    if not sim_path.exists():
        errors.append("Season projections file not found")
    else:
        try:
            sims = pd.read_csv(sim_path)

            if len(sims) != 18:
                errors.append(f"Expected 18 teams, got {len(sims)}")
            else:
                print(f"   Season projections has 18 teams")
        except Exception as e:
            errors.append(f"Error reading projections: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for error in errors:
            print(f"   - {error}")

    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"   - {warning}")

    if not errors and not warnings:
        print("\nAll checks passed")
        return 0
    elif not errors:
        print("\nWarnings found but no critical errors")
        return 0
    else:
        print("\nValidation failed")
        return 1


if __name__ == "__main__":
    sys.exit(validate_update())
