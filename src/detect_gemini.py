"""
TV detection using Gemini 3.5 Flash + SAM2 corner extraction.

Drop-in replacement for detect_tvs_full() with dramatically better accuracy:
  - Gemini 3.5 Flash understands context and never confuses paintings/artwork with TVs
  - SAM2 provides pixel-level screen segmentation for precise perspective quads
  - Single Gemini call per image handles detection + mirrors + foreground objects

Cache keys are namespaced separately from Moondream so both detectors can coexist.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .detect import BoundingBox, TVDetection
from .corners_sam2 import get_sam2_quad
from .corners import refine_corners
from .cache import get_cached_data, save_cached_data

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_CACHE_KEY = "gemini_detect_v1"

_DETECTION_PROMPT = """Analyze this real estate photograph. Find ALL flat-panel TV screens and monitors.

Return ONLY a valid JSON object — no markdown, no explanation — with this exact structure:
{
  "tvs": [
    {
      "bbox": [y_min, x_min, y_max, x_max],
      "is_mirror": false,
      "foreground_bboxes": [[y_min, x_min, y_max, x_max]]
    }
  ],
  "mirror_regions": [[y_min, x_min, y_max, x_max]]
}

COORDINATE FORMAT: Normalized 0.0–1.0. y_min = top / image_height, x_min = left / image_width.
bbox covers the SCREEN SURFACE ONLY — not the bezel, stand, shadow, or wall mount.

CRITICAL RULES:
- INCLUDE: flat-panel TV screens and computer monitors (on or off, any size), front-facing only
- EXCLUDE: paintings, artwork, photographs, picture frames, windows, decorative mirrors, posters, whiteboards, shelves
- EXCLUDE: the back of a TV or monitor (visible cable ports, HDMI connectors, ventilation slots, power sockets, or any rear-panel hardware facing the viewer)
- A dark/off TV screen IS a TV — include it only if the screen surface faces the viewer
- If a TV is reflected in a mirror: add one entry with is_mirror=true AND add the mirror to mirror_regions
- foreground_bboxes: objects physically overlapping the front of the screen surface
- If no TVs are found: {"tvs": [], "mirror_regions": []}
"""


def _gemini_call(image_path: str, max_retries: int = 3) -> Optional[dict]:
    """Send image to Gemini 3.5 Flash and return parsed JSON detection result."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "google-genai not installed. Run: pip install google-genai"
        )

    api_key = GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    client = genai.Client(api_key=api_key)

    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    _DETECTION_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )

            raw = response.text.strip()
            # Strip markdown code fences if model wraps in ```json ... ```
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            logger.info(
                f"  Gemini detected {len(result.get('tvs', []))} TV(s), "
                f"{len(result.get('mirror_regions', []))} mirror(s)"
            )
            return result

        except Exception as e:
            logger.warning(f"  Gemini attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def _parse_bbox(raw: list) -> Optional[BoundingBox]:
    """Parse [y_min, x_min, y_max, x_max] list (0–1) into BoundingBox."""
    if not raw or len(raw) < 4:
        return None
    try:
        y_min, x_min, y_max, x_max = [float(v) for v in raw[:4]]
        # Gemini occasionally returns 0–1000 scale instead of 0–1 (a known
        # model quirk). Detect and normalise before clamping.
        if max(y_min, x_min, y_max, x_max) > 1.5:
            y_min, x_min, y_max, x_max = (
                y_min / 1000.0, x_min / 1000.0,
                y_max / 1000.0, x_max / 1000.0,
            )
        # Swap if inverted
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        # Clamp
        bbox = BoundingBox(
            x_min=max(0.0, min(1.0, x_min)),
            y_min=max(0.0, min(1.0, y_min)),
            x_max=max(0.0, min(1.0, x_max)),
            y_max=max(0.0, min(1.0, y_max)),
        )
        if bbox.area < 0.0005:
            logger.warning(
                f"  Rejecting degenerate bbox (area={bbox.area:.6f}) "
                f"from raw={raw} — TV too small or coordinates wrong"
            )
            return None
        return bbox
    except (TypeError, ValueError):
        return None


def _serialize_detection(det: TVDetection) -> dict:
    """Serialize a TVDetection to a JSON-safe dict for caching."""
    return {
        "bbox": [det.bbox.y_min, det.bbox.x_min, det.bbox.y_max, det.bbox.x_max],
        "is_mirror": det.is_mirror,
        "is_refined": det.is_refined,
        "foreground_bboxes": [
            [fb.y_min, fb.x_min, fb.y_max, fb.x_max]
            for fb in det.foreground_bboxes
        ],
        "quad": det.quad.tolist() if det.quad is not None else None,
    }


def _deserialize_detection(d: dict) -> Optional[TVDetection]:
    """Reconstruct a TVDetection from a cached dict."""
    bbox = _parse_bbox(d.get("bbox", []))
    if bbox is None:
        return None
    fgs = [
        fb for fb in (
            _parse_bbox(raw) for raw in d.get("foreground_bboxes", [])
        )
        if fb is not None
    ]
    quad_raw = d.get("quad")
    quad = np.array(quad_raw, dtype=np.float32) if quad_raw is not None else None
    return TVDetection(
        bbox=bbox,
        is_mirror=bool(d.get("is_mirror", False)),
        is_refined=bool(d.get("is_refined", True)),
        foreground_bboxes=fgs,
        quad=quad,
    )


def detect_tvs_gemini(
    image_path: str,
    use_cache: bool = True,
    need_quad: bool = True,
) -> List[TVDetection]:
    """
    Full detection pipeline using Gemini 3.5 Flash + SAM2.

    Step 1: Gemini 3.5 Flash — single call detects all TVs, mirrors, foreground objects.
    Step 2: SAM2 — per-TV box-prompted segmentation to get precise 4-corner quads.
             Skipped when need_quad=False (e.g. --ai_edit, which never uses det.quad).

    Cache reads happen regardless of need_quad. Cache writes only happen when
    need_quad=True so a no-quad result never poisons the cache for future OpenCV runs.
    """
    # --- Cache check ---
    if use_cache:
        cached = get_cached_data(image_path, key=_CACHE_KEY)
        if cached is not None:
            detections = [
                d for d in (_deserialize_detection(entry) for entry in cached)
                if d is not None
            ]
            logger.info(
                f"  [cache] Gemini+SAM2: {len(detections)} TV(s) for "
                f"{Path(image_path).name}"
            )
            return detections

    # --- Step 1: Gemini detection ---
    raw = _gemini_call(image_path)
    if raw is None:
        logger.error("  Gemini call failed — returning no detections")
        return []

    tv_entries = raw.get("tvs", [])
    mirror_regions = [
        m for m in (_parse_bbox(r) for r in raw.get("mirror_regions", []))
        if m is not None
    ]

    if not tv_entries:
        logger.info("  Gemini: no TVs detected in scene")
        if use_cache:
            save_cached_data(image_path, [], key=_CACHE_KEY)
        return []

    # --- Step 2: SAM2 per-TV segmentation ---
    img = cv2.imread(str(image_path))
    img_h, img_w = img.shape[:2] if img is not None else (1, 1)

    detections: List[TVDetection] = []
    for entry in tv_entries:
        bbox = _parse_bbox(entry.get("bbox", []))
        if bbox is None:
            logger.warning(f"  Skipping TV entry with unusable bbox: {entry.get('bbox')}")
            continue

        is_mirror = bool(entry.get("is_mirror", False))

        # Foreground objects
        fg_bboxes = [
            fb for fb in (
                _parse_bbox(raw_fb) for raw_fb in entry.get("foreground_bboxes", [])
            )
            if fb is not None
        ]

        # SAM2 quad extraction — skipped when caller doesn't need corner coords
        quad = None
        if need_quad:
            try:
                quad = get_sam2_quad(image_path, bbox)
            except Exception as e:
                logger.warning(f"  SAM2 failed for TV at ({bbox.x_min:.2f},{bbox.y_min:.2f}): {e}")

            if quad is None:
                logger.info(
                    f"  SAM2 returned no quad for TV at ({bbox.x_min:.2f},{bbox.y_min:.2f}) "
                    f"— will use OpenCV fallback in run.py"
                )
            else:
                # Clamp SAM2 quad to bbox bounds (1% tolerance)
                if img is not None:
                    px = bbox.to_pixel_coords(img_w, img_h)
                    tol_x = (px["x_max"] - px["x_min"]) * 0.01
                    tol_y = (px["y_max"] - px["y_min"]) * 0.01
                    quad[:, 0] = np.clip(quad[:, 0], px["x_min"] - tol_x, px["x_max"] + tol_x)
                    quad[:, 1] = np.clip(quad[:, 1], px["y_min"] - tol_y, px["y_max"] + tol_y)

        detections.append(TVDetection(
            bbox=bbox,
            is_mirror=is_mirror,
            foreground_bboxes=fg_bboxes,
            is_refined=True,
            quad=quad,
        ))

        sam2_status = "skipped" if not need_quad else ("ok" if quad is not None else "fallback")
        logger.info(
            f"  TV: bbox=({bbox.x_min:.3f},{bbox.y_min:.3f})–({bbox.x_max:.3f},{bbox.y_max:.3f}) "
            f"mirror={is_mirror} fg={len(fg_bboxes)} sam2={sam2_status}"
        )

    # --- Cache result (only when quads were computed) ---
    if use_cache and need_quad:
        save_cached_data(
            image_path,
            [_serialize_detection(d) for d in detections],
            key=_CACHE_KEY,
        )

    logger.info(f"  detect_tvs_gemini: {len(detections)} confirmed TV(s)")
    return detections
