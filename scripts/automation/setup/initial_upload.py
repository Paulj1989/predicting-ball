#!/usr/bin/env python3
"""One-time script to upload your existing database to DO Spaces"""

import boto3
import os
import sys
from pathlib import Path


def initial_upload():
    """Upload existing database to DO Spaces for the first time"""

    print("=" * 70)
    print("INITIAL DATABASE UPLOAD TO DIGITAL OCEAN SPACES")
    print("=" * 70)

    spaces_key = input("\nDO Spaces Access Key: ").strip()
    if not spaces_key:
        print("Error: Access key is required")
        sys.exit(1)

    spaces_secret = input("DO Spaces Secret Key: ").strip()
    if not spaces_secret:
        print("Error: Secret key is required")
        sys.exit(1)

    space_name = input("Space Name (default: ball-bucket): ").strip() or "ball-bucket"
    region = input("Region (default: lon1): ").strip() or "lon1"

    local_path = input("Path to your DuckDB file: ").strip()
    if not local_path:
        print("Error: Database path is required")
        sys.exit(1)

    if not Path(local_path).exists():
        print(f"Error: File not found: {local_path}")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=spaces_key,
        aws_secret_access_key=spaces_secret,
    )

    # Check if space exists
    try:
        s3.head_bucket(Bucket=space_name)
        print(f"\nSpace '{space_name}' exists")
    except s3.exceptions.ClientError:
        print(f"\nCreating space '{space_name}'...")
        try:
            s3.create_bucket(Bucket=space_name)
            print("Space created")
        except Exception as e:
            print(f"Error: Could not create space: {e}")
            print("   Please create the space manually in DO dashboard")
            sys.exit(1)

    remote_path = "database/club_football.duckdb"
    file_size = Path(local_path).stat().st_size / (1024 * 1024)

    print(f"\nUploading {local_path} ({file_size:.2f} MB)...")
    print(f"   Destination: {space_name}/{remote_path}")

    try:
        s3.upload_file(
            local_path, space_name, remote_path, ExtraArgs={"ACL": "private"}
        )

        print("\n" + "=" * 70)
        print("UPLOAD COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("\n1. Add these secrets to GitHub repository:")
        print(
            "   Repository → Settings → Secrets and variables → Actions → New repository secret"
        )
        print(f"\n   Name: DO_SPACES_KEY")
        print(f"   Value: {spaces_key}")
        print(f"\n   Name: DO_SPACES_SECRET")
        print(f"   Value: {spaces_secret}")
        print(f"\n   Name: DO_SPACE_NAME")
        print(f"   Value: {space_name}")
        print(f"\n   Name: DO_SPACE_REGION")
        print(f"   Value: {region}")
        print(f"\n2. Test download:")
        print(f"   export DO_SPACES_KEY='{spaces_key}'")
        print(f"   export DO_SPACES_SECRET='{spaces_secret}'")
        print(f"   export DO_SPACE_NAME='{space_name}'")
        print(f"   export DO_SPACE_REGION='{region}'")
        print(f"   python scripts/download_database.py")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    initial_upload()
