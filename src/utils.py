import os
import logging
import base64
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def collect_image_paths(directory: str) -> List[Path]:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    paths = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_image(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(str(path), img)
    logger.debug(f"Saved: {path}")


def image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    ext = Path(path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"
