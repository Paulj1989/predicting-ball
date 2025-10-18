#!/usr/bin/env python3
"""Upload DuckDB database to Digital Ocean Spaces"""

import boto3
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def upload_database():
    """Upload database to DO Spaces with backup"""

    spaces_key = os.getenv("DO_SPACES_KEY")
    spaces_secret = os.getenv("DO_SPACES_SECRET")
    space_name = os.getenv("DO_SPACE_NAME", "ball-bucket")
    region = os.getenv("DO_SPACE_REGION", "lon1")

    if not spaces_key or not spaces_secret:
        print("Error: DO_SPACES_KEY and DO_SPACES_SECRET must be set")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=spaces_key,
        aws_secret_access_key=spaces_secret,
    )

    local_path = "data/club_football.duckdb"
    remote_path = "database/club_football.duckdb"
    backup_path = f"database/backups/club_football_{datetime.now().strftime('%Y%m%d_%H%M%S')}.duckdb"

    if not Path(local_path).exists():
        print(f"Error: Database not found at {local_path}")
        sys.exit(1)

    try:
        file_size = Path(local_path).stat().st_size / (1024 * 1024)

        # Backup existing database
        print(f"Creating backup at {backup_path}...")
        try:
            s3.copy_object(
                Bucket=space_name,
                CopySource={"Bucket": space_name, "Key": remote_path},
                Key=backup_path,
            )
            print("Backup created")
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print("No existing database to backup (first upload)")
            else:
                raise

        # Upload new version
        print(
            f"Uploading database to {space_name}/{remote_path} ({file_size:.2f} MB)..."
        )

        s3.upload_file(
            local_path, space_name, remote_path, ExtraArgs={"ACL": "private"}
        )

        print("Database uploaded successfully")

        # Clean up old backups (keep last 10)
        print("Cleaning up old backups...")
        try:
            backups = s3.list_objects_v2(Bucket=space_name, Prefix="database/backups/")

            if backups.get("Contents"):
                backup_files = sorted(
                    backups["Contents"], key=lambda x: x["LastModified"], reverse=True
                )

                for backup_file in backup_files[10:]:
                    s3.delete_object(Bucket=space_name, Key=backup_file["Key"])
                    print(f"   Deleted old backup: {backup_file['Key']}")

                print(f"Kept {min(len(backup_files), 10)} most recent backups")
        except Exception as e:
            print(f"Could not clean up backups: {e}")

        print("Upload complete")
        return True

    except Exception as e:
        print(f"Error uploading database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    upload_database()
