#!/usr/bin/env python3
"""Download DuckDB database and model artifacts from Digital Ocean Spaces"""

import argparse
import boto3
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_s3_client():
    """Create S3 client for DO Spaces"""
    spaces_key = os.getenv("DO_SPACES_KEY")
    spaces_secret = os.getenv("DO_SPACES_SECRET")
    region = os.getenv("DO_SPACE_REGION", "lon1")

    if not spaces_key or not spaces_secret:
        print("Error: DO_SPACES_KEY and DO_SPACES_SECRET must be set")
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=spaces_key,
        aws_secret_access_key=spaces_secret,
    )


def download_file(s3, space_name: str, remote_path: str, local_path: str) -> bool:
    """Download a single file from DO Spaces"""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        s3.head_object(Bucket=space_name, Key=remote_path)
    except s3.exceptions.ClientError:
        return False

    s3.download_file(space_name, remote_path, local_path)
    return True


def download_database(s3=None):
    """Download database from DO Spaces"""
    space_name = os.getenv("DO_SPACE_NAME", "ball-bucket")

    if s3 is None:
        s3 = get_s3_client()

    remote_path = "database/pb.duckdb"
    local_path = "data/pb.duckdb"

    print(f"Downloading database from {space_name}/{remote_path}...")

    if not download_file(s3, space_name, remote_path, local_path):
        print(f"Error: Database not found at {space_name}/{remote_path}")
        print("   Have you uploaded the initial database?")
        sys.exit(1)

    file_size = Path(local_path).stat().st_size / (1024 * 1024)
    print(f"Database downloaded successfully ({file_size:.2f} MB)")
    return True


def download_model(s3=None):
    """Download model and calibrator from DO Spaces"""
    space_name = os.getenv("DO_SPACE_NAME", "ball-bucket")

    if s3 is None:
        s3 = get_s3_client()

    files = [
        ("serving/buli_model.pkl", "outputs/models/buli_model.pkl"),
        ("serving/buli_calibrators.pkl", "outputs/models/buli_calibrators.pkl"),
    ]

    downloaded = 0
    for remote_path, local_path in files:
        print(f"Downloading {remote_path}...")

        if download_file(s3, space_name, remote_path, local_path):
            file_size = Path(local_path).stat().st_size / 1024
            print(f"   Downloaded ({file_size:.1f} KB)")
            downloaded += 1
        else:
            print(f"   Not found (may not exist yet)")

    if downloaded > 0:
        print(f"Model artifacts downloaded: {downloaded} file(s)")
    else:
        print("No model artifacts found in DO Spaces")

    return downloaded > 0


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Download database and model artifacts from DO Spaces"
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Download only model artifacts (skip database)",
    )
    parser.add_argument(
        "--include-model",
        action="store_true",
        help="Also download model artifacts alongside database",
    )

    args = parser.parse_args()

    s3 = get_s3_client()

    if args.model_only:
        download_model(s3)
    else:
        download_database(s3)
        if args.include_model:
            print()
            download_model(s3)


if __name__ == "__main__":
    main()
