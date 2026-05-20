import os
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def parse_s3_uri(uri: str) -> tuple:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def download_from_s3(
    s3_uri: str,
    local_dir: Optional[str] = None,
    max_files: Optional[int] = None,
) -> str:
    """
    Download images from an S3 prefix to a local directory.

    Args:
        s3_uri: S3 URI (e.g., s3://bucket/prefix)
        local_dir: Local directory to download into. If None, uses a temp dir.
        max_files: Maximum number of files to download (None = all)

    Returns:
        Path to the local directory containing downloaded files.
    """
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 is required for S3 support. Install it: pip install boto3")

    bucket, prefix = parse_s3_uri(s3_uri)

    if local_dir is None:
        local_dir = os.path.join(".", "TVPhotos")
    os.makedirs(local_dir, exist_ok=True)

    logger.info(f"Downloading from s3://{bucket}/{prefix} to {local_dir}")

    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=8 * 1024 * 1024,
    )
    paginator = s3.get_paginator("list_objects_v2")

    to_download = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = Path(key).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            filename = Path(key).name
            stem = Path(key).stem.lower()
            if "_tar" in stem:
                logger.debug(f"  Skipping target image: {filename}")
                continue

            local_path = os.path.join(local_dir, filename)

            if os.path.exists(local_path):
                logger.debug(f"  Skipping (exists): {filename}")
            else:
                to_download.append((key, local_path, filename))

            if max_files and (len(to_download) + sum(1 for _ in Path(local_dir).glob("*"))) >= max_files:
                break
        if max_files and (len(to_download) + sum(1 for _ in Path(local_dir).glob("*"))) >= max_files:
            logger.info(f"  Reached max_files limit ({max_files})")
            break

    def _download_one(item):
        key, local_path, filename = item
        logger.debug(f"  Downloading: {filename}")
        s3.download_file(bucket, key, local_path, Config=transfer_config)
        return filename

    max_workers = min(8, len(to_download)) if to_download else 1
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_one, item) for item in to_download]
        for fut in as_completed(futures):
            fut.result()
            downloaded += 1

    existing = sum(1 for p in Path(local_dir).iterdir() if p.is_file()) - downloaded
    logger.info(f"  Downloaded {downloaded} new, {existing} already existed in {local_dir}")
    return local_dir
