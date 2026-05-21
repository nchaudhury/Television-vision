"""
SAM2-based corner extraction via fal.ai.

Given a TV bounding box detected by Gemini, calls fal-ai/sam2/image with a box
prompt to get a pixel-level segmentation mask of the screen, then extracts a
precise 4-corner quadrilateral suitable for perspective warping.

Falls back to None on any failure so the caller can use refine_corners() instead.
"""
import base64
import logging
import os
import time
from typing import Optional

import cv2
import httpx
import numpy as np

from .detect import BoundingBox

logger = logging.getLogger(__name__)

SAM2_API_URL = "https://fal.run/fal-ai/sam2/image"


def _get_fal_client() -> httpx.Client:
    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        raise RuntimeError("FAL_KEY not set in environment.")
    return httpx.Client(
        verify=False,
        timeout=120.0,
        headers={
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        },
    )


def get_sam2_quad(
    image_path: str,
    bbox: BoundingBox,
    max_retries: int = 3,
) -> Optional[np.ndarray]:
    """
    Segment the TV screen using SAM2 and return a (4, 2) float32 quad
    in [TL, TR, BR, BL] order (pixel coords in original image space).

    Returns None on any failure so the caller falls back to OpenCV corner detection.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h_img, w_img = img.shape[:2]

    px = bbox.to_pixel_coords(w_img, h_img)

    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"

    payload = {
        "image_url": data_uri,
        "prompts": [
            {
                "type": "box",
                "x_min": float(px["x_min"]),
                "y_min": float(px["y_min"]),
                "x_max": float(px["x_max"]),
                "y_max": float(px["y_max"]),
            }
        ],
    }

    with _get_fal_client() as client:
        for attempt in range(max_retries):
            try:
                resp = client.post(SAM2_API_URL, json=payload)
                resp.raise_for_status()
                result = resp.json()
                logger.debug(f"  SAM2 response keys: {list(result.keys())}")

                quad = _extract_quad(result, w_img, h_img, client)
                if quad is not None:
                    logger.info(
                        f"  SAM2 quad: TL={quad[0].tolist()} TR={quad[1].tolist()} "
                        f"BR={quad[2].tolist()} BL={quad[3].tolist()}"
                    )
                    return quad

                logger.warning("  SAM2: could not extract quad from mask response")
                return None

            except Exception as e:
                logger.warning(f"  SAM2 attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    return None


def _extract_quad(
    result: dict,
    img_w: int,
    img_h: int,
    client: httpx.Client,
) -> Optional[np.ndarray]:
    """Parse SAM2 JSON response and return the best 4-corner quad."""
    masks = result.get("masks") or result.get("output") or []
    if not masks:
        logger.debug(f"  SAM2: no masks in response: {result}")
        return None

    # Pick highest-scoring mask
    def _score(m: dict) -> float:
        return float(m.get("score", m.get("iou_score", 0.0)))

    best = max(masks, key=_score)
    logger.info(f"  SAM2: best mask score={_score(best):.3f}, keys={list(best.keys())}")

    mask_img = _decode_mask(best, img_w, img_h, client)
    if mask_img is None:
        return None

    return _quad_from_mask(mask_img)


def _decode_mask(
    mask_data: dict,
    img_w: int,
    img_h: int,
    client: httpx.Client,
) -> Optional[np.ndarray]:
    """
    Decode a SAM2 mask entry to a binary uint8 numpy array.

    Uses recursive URI discovery so it works regardless of how fal.ai nests
    the mask URL in its response (the schema has changed over time).
    """
    logger.info(f"  SAM2 mask data keys: {list(mask_data.keys())}")

    # Recursively collect every string that looks like a URL or data-URI
    candidates: list = []

    def _collect(obj, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(obj, str):
            if obj.startswith("data:") or obj.startswith("http"):
                candidates.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item, depth + 1)

    _collect(mask_data)

    if not candidates:
        logger.warning(
            f"  SAM2: no URL / data-URI found in mask dict with keys "
            f"{list(mask_data.keys())} — cannot decode mask"
        )
        return None

    for uri in candidates:
        img_bytes = None
        if uri.startswith("data:"):
            try:
                _, b64 = uri.split(",", 1)
                img_bytes = base64.b64decode(b64)
            except Exception:
                continue
        elif uri.startswith("http"):
            try:
                dl = client.get(uri, timeout=30.0)
                dl.raise_for_status()
                img_bytes = dl.content
            except Exception as e:
                logger.debug(f"  SAM2 mask download failed: {e}")
                continue

        if img_bytes:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                return (mask > 127).astype(np.uint8) * 255

    return None


def _quad_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a precise 4-corner quadrilateral from a binary mask.

    Strategy:
      1. Find the largest external contour.
      2. Try approxPolyDP at decreasing epsilon to get exactly 4 vertices.
      3. Fall back to minAreaRect if we can't simplify to 4 points.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)

    # Try progressively coarser approximation until we get 4 corners
    for eps_frac in (0.02, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return _order_corners(corners)

    # Fallback: minimum area rectangle always gives exactly 4 corners
    rect = cv2.minAreaRect(hull)
    corners = cv2.boxPoints(rect).astype(np.float32)
    return _order_corners(corners)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners ordered as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR: largest x+y
    rect[1] = pts[np.argmin(d)]   # TR: smallest x-y
    rect[3] = pts[np.argmax(d)]   # BL: largest x-y
    return rect
