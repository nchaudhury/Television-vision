import io
import os
import ssl
import time
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple
from dataclasses import dataclass, field

import cv2
import httpx
import numpy as np
from PIL import Image
from dotenv import load_dotenv

if TYPE_CHECKING:
    pass  # numpy already imported above

from .utils import image_to_data_uri
from .cache import get_cached_detections, save_cached_detections

logger = logging.getLogger(__name__)

load_dotenv()

FAL_API_URL = "https://fal.run/fal-ai/moondream3-preview/detect"

# Detection prompts
_PROMPT_INITIAL = "television or monitor screen"
_PROMPT_SCREEN_REFINE = (
    "television display screen surface "
    "(front-facing screen only, excluding bezel frame shadow and wall)"
)
_PROMPT_BACK_OF_TV = (
    "ventilation slots, cable input ports, HDMI connectors, power socket, "
    "or mounting hardware on the rear panel of a television"
)
_PROMPT_MIRROR = "mirror or large glass surface reflecting the room"
_PROMPT_FOREGROUND = (
    "decorative items, figurines, plants, vases, lamps, electronics, toys, books, "
    "or objects placed directly in front of or overlapping the television screen"
)


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

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass
class TVDetection:
    bbox: BoundingBox
    is_mirror: bool = False
    foreground_bboxes: List[BoundingBox] = field(default_factory=list)
    is_refined: bool = True   # False = coarse-bbox fallback; use larger bezel
    quad: Optional[np.ndarray] = field(default=None)  # Pre-computed 4-corner quad from SAM2


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


def _call_detect(
    image_path: str,
    prompt: str,
    use_cache: bool = True,
    max_retries: int = 3,
    crop_bbox: Optional[BoundingBox] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> List[BoundingBox]:
    """
    Core detection call. Optionally crops image to crop_bbox before sending.
    If cropped, translates returned normalized coords back to full-image space.
    """
    # Encode crop coordinates into the prompt key so (image, crop, prompt) are all distinct cache entries.
    # The image_path itself must remain a real file path for _file_hash to read it.
    if crop_bbox is not None:
        cache_prompt = (
            f"{prompt}::crop={crop_bbox.x_min:.4f},{crop_bbox.y_min:.4f},"
            f"{crop_bbox.x_max:.4f},{crop_bbox.y_max:.4f}"
        )
    else:
        cache_prompt = prompt

    if use_cache:
        cached = get_cached_detections(str(image_path), prompt=cache_prompt)
        if cached is not None:
            return [BoundingBox(**b) for b in cached]

    # Build image data URI — either full image or cropped region
    if crop_bbox is not None:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        h_img, w_img = img.shape[:2]
        px = crop_bbox.to_pixel_coords(w_img, h_img)
        # Add 5% padding — just enough to catch edges, small enough to exclude shadows
        pad_x = int((px["x_max"] - px["x_min"]) * 0.05)
        pad_y = int((px["y_max"] - px["y_min"]) * 0.05)
        cx1 = max(0, px["x_min"] - pad_x)
        cy1 = max(0, px["y_min"] - pad_y)
        cx2 = min(w_img, px["x_max"] + pad_x)
        cy2 = min(h_img, px["y_max"] + pad_y)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return []
        # Encode crop as JPEG data URI
        success, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return []
        import base64
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        crop_origin_x = cx1
        crop_origin_y = cy1
    else:
        data_uri = image_to_data_uri(image_path)
        img_for_size = cv2.imread(str(image_path))
        if img_for_size is not None:
            h_img, w_img = img_for_size.shape[:2]
        else:
            h_img, w_img = image_height or 1, image_width or 1
        crop_w = w_img
        crop_h = h_img
        crop_origin_x = 0
        crop_origin_y = 0

    boxes: List[BoundingBox] = []
    with _get_http_client() as client:
        for attempt in range(max_retries):
            try:
                logger.debug(f"  API call prompt={prompt!r} (attempt {attempt + 1})")
                response = client.post(
                    FAL_API_URL,
                    json={"image_url": data_uri, "prompt": prompt},
                )
                response.raise_for_status()
                result = response.json()

                for obj in result.get("objects", []):
                    # Translate crop-relative normalized coords back to full-image normalized
                    full_x_min = (crop_origin_x + obj["x_min"] * crop_w) / w_img
                    full_y_min = (crop_origin_y + obj["y_min"] * crop_h) / h_img
                    full_x_max = (crop_origin_x + obj["x_max"] * crop_w) / w_img
                    full_y_max = (crop_origin_y + obj["y_max"] * crop_h) / h_img
                    boxes.append(BoundingBox(
                        x_min=max(0.0, min(1.0, full_x_min)),
                        y_min=max(0.0, min(1.0, full_y_min)),
                        x_max=max(0.0, min(1.0, full_x_max)),
                        y_max=max(0.0, min(1.0, full_y_max)),
                    ))
                logger.info(f"  Prompt={prompt!r} → {len(boxes)} object(s)")
                break

            except Exception as e:
                logger.warning(f"  Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"  All retries exhausted for prompt={prompt!r}")

    if use_cache:
        save_cached_detections(
            str(image_path),
            [{"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max} for b in boxes],
            prompt=cache_prompt,
        )

    return boxes


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


def _bbox_overlap_ratio(inner: BoundingBox, outer: BoundingBox) -> float:
    """Fraction of inner's area that overlaps with outer."""
    if inner.area == 0:
        return 0.0
    x_left = max(inner.x_min, outer.x_min)
    y_top = max(inner.y_min, outer.y_min)
    x_right = min(inner.x_max, outer.x_max)
    y_bottom = min(inner.y_max, outer.y_max)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return ((x_right - x_left) * (y_bottom - y_top)) / inner.area


def _deduplicate_boxes(boxes: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
    if not boxes:
        return []
    kept: List[BoundingBox] = []
    for box in boxes:
        if not any(_iou(box, existing) > iou_threshold for existing in kept):
            kept.append(box)
    return kept


def detect_tvs(image_path: str, max_retries: int = 3, use_cache: bool = True) -> List[BoundingBox]:
    """Legacy single-prompt detection. Kept for backward compatibility."""
    boxes = _call_detect(image_path, _PROMPT_INITIAL, use_cache=use_cache, max_retries=max_retries)
    deduped = _deduplicate_boxes(boxes)
    logger.info(f"  Total unique TVs detected: {len(deduped)}")
    return deduped


def detect_tvs_full(image_path: str, use_cache: bool = True) -> List[TVDetection]:
    """
    Full multi-step detection pipeline:
      1. Coarse TV detection
      2. Per-TV screen refinement (tighter bbox) + back-of-TV filtering
      3. Mirror region scan
      4. Per-TV foreground object detection
    Returns a list of TVDetection objects, one per confirmed front-facing TV.
    """
    # Step 1: Coarse detection
    coarse_boxes = _call_detect(image_path, _PROMPT_INITIAL, use_cache=use_cache)
    coarse_boxes = _deduplicate_boxes(coarse_boxes)
    if not coarse_boxes:
        logger.info("  No TVs detected in coarse scan.")
        return []

    logger.info(f"  Coarse detection: {len(coarse_boxes)} TV candidate(s)")

    # Step 2: Refine each bbox + back-of-TV filtering
    # Each entry is (bbox, is_refined)
    validated: List[tuple] = []
    for bbox in coarse_boxes:
        # 2a — Screen refinement: get tighter screen-only bbox
        refined = _call_detect(
            image_path,
            _PROMPT_SCREEN_REFINE,
            use_cache=use_cache,
            crop_bbox=bbox,
        )

        if refined:
            best = max(refined, key=lambda b: b.area)
            if best.area >= 0.0005:
                # 2b — Back-of-TV check: look for ports/vents on the refined region
                back_features = _call_detect(
                    image_path,
                    _PROMPT_BACK_OF_TV,
                    use_cache=use_cache,
                    crop_bbox=best,
                )
                if back_features:
                    logger.info(
                        f"  TV at ({bbox.x_min:.2f},{bbox.y_min:.2f}) discarded "
                        f"— back-of-TV features detected (ports/vents)"
                    )
                    continue
                logger.info(
                    f"  TV refined: ({best.x_min:.3f},{best.y_min:.3f})–"
                    f"({best.x_max:.3f},{best.y_max:.3f})"
                )
                validated.append((best, True))
                continue

        # Refinement returned nothing or too small — fallback to coarse bbox
        # Still check for back-of-TV features before keeping
        back_features = _call_detect(
            image_path,
            _PROMPT_BACK_OF_TV,
            use_cache=use_cache,
            crop_bbox=bbox,
        )
        if back_features:
            logger.info(
                f"  TV at ({bbox.x_min:.2f},{bbox.y_min:.2f}) discarded "
                f"— back-of-TV features on coarse bbox"
            )
            continue

        logger.info(
            f"  TV at ({bbox.x_min:.2f},{bbox.y_min:.2f}) using coarse bbox "
            f"(refinement found nothing) — will use larger bezel"
        )
        validated.append((bbox, False))

    if not validated:
        logger.info("  No front-facing TVs after refinement/validation.")
        return []

    # Step 3: Mirror scan — single call on full image
    mirror_regions = _call_detect(image_path, _PROMPT_MIRROR, use_cache=use_cache)
    logger.info(f"  Mirror scan: {len(mirror_regions)} mirror region(s) found")

    def _is_in_mirror(bbox: BoundingBox) -> bool:
        cx, cy = bbox.center()
        return any(
            m.x_min <= cx <= m.x_max and m.y_min <= cy <= m.y_max
            for m in mirror_regions
        )

    # Step 4: Per-TV foreground detection (one call covers all TVs in the scene)
    fg_raw = _call_detect(image_path, _PROMPT_FOREGROUND, use_cache=use_cache)

    tv_detections: List[TVDetection] = []
    for screen_bbox, is_refined in validated:
        is_mirror = _is_in_mirror(screen_bbox)
        if is_mirror:
            logger.info(f"  TV at ({screen_bbox.x_min:.2f},{screen_bbox.y_min:.2f}) is a mirror reflection")

        fg_filtered = [
            fb for fb in fg_raw
            if _bbox_overlap_ratio(fb, screen_bbox) > 0.03
        ]
        if fg_filtered:
            logger.info(f"  TV has {len(fg_filtered)} foreground object(s)")

        tv_detections.append(TVDetection(
            bbox=screen_bbox,
            is_mirror=is_mirror,
            foreground_bboxes=fg_filtered,
            is_refined=is_refined,
        ))

    logger.info(f"  detect_tvs_full: {len(tv_detections)} confirmed TV(s)")
    return tv_detections
