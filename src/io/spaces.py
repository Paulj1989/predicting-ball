# src/io/spaces.py
"""
Digital Ocean Spaces integration for uploading and downloading files.

Provides utilities for:
- Uploading files and DataFrames to DO Spaces
- Downloading files from DO Spaces
- Generating public URLs for serving
"""

import io
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# path prefixes for DO Spaces
SERVING_PREFIX = "serving/"
INCOMING_PREFIX = "incoming/"
DATABASE_PREFIX = "database/"


def get_spaces_config() -> dict:
    """Get DO Spaces configuration from environment variables."""
    return {
        "key": os.getenv("DO_SPACES_KEY"),
        "secret": os.getenv("DO_SPACES_SECRET"),
        "space_name": os.getenv("DO_SPACE_NAME", "ball-bucket"),
        "region": os.getenv("DO_SPACE_REGION", "lon1"),
    }


def get_spaces_client():
    """Create a boto3 client configured for DO Spaces"""
    config = get_spaces_config()

    if not config["key"] or not config["secret"]:
        raise ValueError("DO_SPACES_KEY and DO_SPACES_SECRET must be set in environment")

    return boto3.client(
        "s3",
        endpoint_url=f"https://{config['region']}.digitaloceanspaces.com",
        aws_access_key_id=config["key"],
        aws_secret_access_key=config["secret"],
    )


def get_public_url(remote_key: str) -> str:
    """Generate the public URL for a file in DO Spaces"""
    config = get_spaces_config()
    return f"https://{config['space_name']}.{config['region']}.digitaloceanspaces.com/{remote_key}"


def upload_file(
    local_path: str | Path,
    remote_key: str,
    public: bool = False,
    content_type: str | None = None,
) -> str:
    """Upload a local file to DO Spaces"""
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    client = get_spaces_client()
    config = get_spaces_config()

    extra_args = {}
    if public:
        extra_args["ACL"] = "public-read"
    if content_type:
        extra_args["ContentType"] = content_type

    client.upload_file(
        str(local_path),
        config["space_name"],
        remote_key,
        ExtraArgs=extra_args if extra_args else None,
    )

    return remote_key


def upload_bytes(
    data: bytes,
    remote_key: str,
    public: bool = False,
    content_type: str | None = None,
) -> str:
    """Upload bytes directly to DO Spaces"""
    client = get_spaces_client()
    config = get_spaces_config()

    extra_args = {}
    if public:
        extra_args["ACL"] = "public-read"
    if content_type:
        extra_args["ContentType"] = content_type

    client.put_object(
        Bucket=config["space_name"],
        Key=remote_key,
        Body=data,
        **extra_args,
    )

    return remote_key


def upload_dataframe_as_parquet(
    df: Any,
    remote_key: str,
    public: bool = False,
) -> str:
    """Upload a pandas DataFrame as a Parquet file to DO Spaces"""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)

    return upload_bytes(
        buffer.getvalue(),
        remote_key,
        public=public,
        content_type="application/octet-stream",
    )


def upload_pickle(
    obj: Any,
    remote_key: str,
    public: bool = False,
) -> str:
    """Upload a Python object as a pickle file to DO Spaces"""
    import pickle

    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)

    return upload_bytes(
        buffer.getvalue(),
        remote_key,
        public=public,
        content_type="application/octet-stream",
    )


def download_file(remote_key: str, local_path: str | Path) -> Path:
    """Download a file from DO Spaces to local filesystem"""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_spaces_client()
    config = get_spaces_config()

    client.download_file(config["space_name"], remote_key, str(local_path))

    return local_path


def file_exists(remote_key: str) -> bool:
    """Check if a file exists in DO Spaces"""
    client = get_spaces_client()
    config = get_spaces_config()

    try:
        client.head_object(Bucket=config["space_name"], Key=remote_key)
        return True
    except ClientError:
        return False


def list_files(prefix: str = "") -> list[str]:
    """List files in DO Spaces under a given prefix"""
    client = get_spaces_client()
    config = get_spaces_config()

    response = client.list_objects_v2(Bucket=config["space_name"], Prefix=prefix)

    if "Contents" not in response:
        return []

    return [obj["Key"] for obj in response["Contents"]]
