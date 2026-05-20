import os
import ssl
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from .utils import image_to_data_uri
from .cache import get_cached_detections, save_cached_detections

logger = logging.getLogger(__name__)

load_dotenv()

FAL_API_URL = "https://fal.run/fal-ai/moondream3-preview/detect"


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_pixel_coords(self, img_width: int, img_height: int):
        return {
            "x_min": int(self.x_min * img_width),
            "y_min": int(self.y_min * img_height),
            "x_max": int(self.x_max * img_width),
            "y_max": int(self.y_max * img_height),
        }


def _get_http_client() -> httpx.Client:
    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        raise RuntimeError("FAL_KEY environment variable not set. Check your .env file.")
    return httpx.Client(
        verify=False,
        timeout=120.0,
        headers={
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        },
    )


def detect_tvs(image_path: str, max_retries: int = 3, use_cache: bool = True) -> List[BoundingBox]:
    if use_cache:
        cached = get_cached_detections(image_path)
        if cached is not None:
            return [BoundingBox(**b) for b in cached]

    data_uri = image_to_data_uri(image_path)

    prompt = "televisions, but the bounding box must be set to the 4 outside corners of the television"

    all_boxes: List[BoundingBox] = []

    with _get_http_client() as client:
        for attempt in range(max_retries):
            try:
                logger.debug(f"Detecting with prompt='{prompt}' (attempt {attempt + 1})")

                response = client.post(
                    FAL_API_URL,
                    json={
                        "image_url": data_uri,
                        "prompt": prompt,
                    },
                )
                response.raise_for_status()
                result = response.json()

                objects = result.get("objects", [])
                if objects:
                    logger.info(f"  prompt='{prompt}' found {len(objects)} object(s)")
                    for obj in objects:
                        box = BoundingBox(
                            x_min=obj["x_min"],
                            y_min=obj["y_min"],
                            x_max=obj["x_max"],
                            y_max=obj["y_max"],
                        )
                        all_boxes.append(box)
                break

            except Exception as e:
                logger.warning(f"  Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"  All retries exhausted for detection")

    deduplicated = _deduplicate_boxes(all_boxes)
    logger.info(f"  Total unique TVs detected: {len(deduplicated)}")

    if use_cache:
        boxes_dicts = [{"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max} for b in deduplicated]
        save_cached_detections(image_path, boxes_dicts)

    return deduplicated


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x_left = max(a.x_min, b.x_min)
    y_top = max(a.y_min, b.y_min)
    x_right = min(a.x_max, b.x_max)
    y_bottom = min(a.y_max, b.y_max)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_boxes(boxes: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
    if not boxes:
        return []

    kept: List[BoundingBox] = []
    for box in boxes:
        is_dup = False
        for existing in kept:
            if _iou(box, existing) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(box)
    return kept
