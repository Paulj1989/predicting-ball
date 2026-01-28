#!/usr/bin/env python3
"""Download DuckDB database from Digital Ocean Spaces"""

import boto3
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def download_database():
    """Download database from DO Spaces"""

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

    remote_path = "database/pb.duckdb"
    local_path = "data/pb.duckdb"

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading database from {space_name}/{remote_path}...")

        try:
            s3.head_object(Bucket=space_name, Key=remote_path)
        except s3.exceptions.ClientError:
            print(f"Error: Database not found at {space_name}/{remote_path}")
            print("   Have you uploaded the initial database?")
            print("   Run: python scripts/initial_upload.py")
            sys.exit(1)

        s3.download_file(space_name, remote_path, local_path)

        file_size = Path(local_path).stat().st_size / (1024 * 1024)
        print(f"Database downloaded successfully ({file_size:.2f} MB)")
        return True

    except Exception as e:
        print(f"Error downloading database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_database()
