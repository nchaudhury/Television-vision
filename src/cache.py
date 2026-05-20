import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".cache"


def _file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file for cache keying."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cached_detections(image_path: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Optional[List[dict]]:
    """Return cached bounding boxes for an image, or None if not cached."""
    cache_file = _cache_path(image_path, cache_dir)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        logger.debug(f"  Cache hit for {os.path.basename(image_path)}")
        return data["boxes"]
    except (json.JSONDecodeError, KeyError):
        logger.debug(f"  Cache corrupt for {os.path.basename(image_path)}, ignoring")
        return None


def save_cached_detections(
    image_path: str, boxes: List[dict], cache_dir: str = DEFAULT_CACHE_DIR
) -> None:
    """Save bounding boxes to disk cache keyed by image hash."""
    cache_file = _cache_path(image_path, cache_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"file": os.path.basename(image_path), "boxes": boxes}, f)
    logger.debug(f"  Cached detection results for {os.path.basename(image_path)}")


def _cache_path(image_path: str, cache_dir: str) -> Path:
    img_hash = _file_hash(image_path)
    return Path(cache_dir) / f"{img_hash}.json"
