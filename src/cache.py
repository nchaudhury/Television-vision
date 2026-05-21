import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".cache"


def _file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file for cache keying."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _prompt_slug(prompt: str) -> str:
    """Return first 8 hex chars of SHA-256 of prompt string — stable short key."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def get_cached_detections(
    image_path: str,
    prompt: str = "",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[List[dict]]:
    """Return cached bounding boxes for an image+prompt pair, or None if not cached."""
    cache_file = _cache_path(image_path, prompt, cache_dir)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        logger.debug(f"  Cache hit for {os.path.basename(image_path)} / prompt={prompt!r}")
        return data["boxes"]
    except (json.JSONDecodeError, KeyError):
        logger.debug(f"  Cache corrupt for {os.path.basename(image_path)}, ignoring")
        return None


def save_cached_detections(
    image_path: str,
    boxes: List[dict],
    prompt: str = "",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> None:
    """Save bounding boxes to disk cache keyed by (image_hash, prompt)."""
    cache_file = _cache_path(image_path, prompt, cache_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"file": os.path.basename(image_path), "prompt": prompt, "boxes": boxes}, f)
    logger.debug(f"  Cached detection results for {os.path.basename(image_path)} / prompt={prompt!r}")


def _cache_path(image_path: str, prompt: str, cache_dir: str) -> Path:
    img_hash = _file_hash(image_path)
    slug = _prompt_slug(prompt)
    return Path(cache_dir) / f"{img_hash}_{slug}.json"


def get_cached_data(
    image_path: str,
    key: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[Any]:
    """Return cached arbitrary JSON data keyed by (image_hash, key), or None if not cached."""
    cache_file = _cache_path(image_path, key, cache_dir)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        logger.debug(f"  Cache hit [data] for {os.path.basename(image_path)} / key={key!r}")
        return data.get("data")
    except (json.JSONDecodeError, KeyError):
        logger.debug(f"  Cache corrupt [data] for {os.path.basename(image_path)}, ignoring")
        return None


def save_cached_data(
    image_path: str,
    data: Any,
    key: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> None:
    """Save arbitrary JSON-serializable data to disk cache keyed by (image_hash, key)."""
    cache_file = _cache_path(image_path, key, cache_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"file": os.path.basename(image_path), "key": key, "data": data}, f)
    logger.debug(f"  Cached data for {os.path.basename(image_path)} / key={key!r}")
