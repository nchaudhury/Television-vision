import logging
from typing import List, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .detect import BoundingBox, TVDetection

logger = logging.getLogger(__name__)


def composite_overlay(
    scene: np.ndarray,
    overlay: np.ndarray,
    quads: List[np.ndarray],
    feather_px: int = 5,
    brightness_match: bool = True,
    bezel_pct: float = 0.05,
    tv_detections: Optional[List] = None,
    per_tv_bezel: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Composite the overlay image onto each TV screen quad in the scene.

    Args:
        scene: Original photo (BGR, uint8)
        overlay: Image to place on TVs (BGR, uint8)
        quads: List of (4,2) float32 arrays — each quad's corners
               in [TL, TR, BR, BL] order, pixel coords
        feather_px: Pixels of Gaussian feathering at quad edges
        brightness_match: Adjust overlay brightness to match surroundings
        bezel_pct: Default bezel fraction for all TVs
        tv_detections: Optional list of TVDetection aligned with quads
        per_tv_bezel: Optional per-TV bezel overrides (aligned with quads)

    Returns:
        Composited image (BGR, uint8)
    """
    result = scene.copy()

    for i, quad in enumerate(quads):
        logger.debug(f"  Compositing TV #{i + 1}")
        det = tv_detections[i] if tv_detections and i < len(tv_detections) else None
        is_mirror = det.is_mirror if det else False
        fg_bboxes = det.foreground_bboxes if det else []
        this_bezel = per_tv_bezel[i] if per_tv_bezel and i < len(per_tv_bezel) else bezel_pct
        result = _composite_single(
            result, overlay, quad, feather_px, brightness_match, this_bezel,
            is_mirror=is_mirror,
            foreground_bboxes=fg_bboxes,
        )

    return result


def _shrink_quad(quad: np.ndarray, bezel_pct: float) -> np.ndarray:
    """Shrink a quad inward by bezel_pct toward its centroid."""
    centroid = quad.mean(axis=0)
    return (quad + (centroid - quad) * bezel_pct).astype(np.float32)


def _composite_single(
    scene: np.ndarray,
    overlay: np.ndarray,
    quad: np.ndarray,
    feather_px: int,
    brightness_match: bool,
    bezel_pct: float = 0.05,
    is_mirror: bool = False,
    foreground_bboxes: Optional[List] = None,
) -> np.ndarray:
    h_scene, w_scene = scene.shape[:2]
    h_ovl, w_ovl = overlay.shape[:2]

    src_corners = np.array([
        [0, 0],
        [w_ovl - 1, 0],
        [w_ovl - 1, h_ovl - 1],
        [0, h_ovl - 1],
    ], dtype=np.float32)

    dst_corners = _shrink_quad(quad, bezel_pct) if bezel_pct > 0 else quad.astype(np.float32)

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # Flip overlay horizontally for mirror reflections
    overlay_to_use = cv2.flip(overlay, 1) if is_mirror else overlay

    adjusted_overlay = overlay_to_use.copy()
    if brightness_match:
        adjusted_overlay = _match_brightness(scene, adjusted_overlay, quad)

    warped = cv2.warpPerspective(
        adjusted_overlay, M, (w_scene, h_scene),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask = np.zeros((h_scene, w_scene), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_corners.astype(np.int32), 255)

    if feather_px > 0:
        ksize = feather_px * 2 + 1
        mask_float = cv2.GaussianBlur(
            mask.astype(np.float32), (ksize, ksize), 0
        )
        mask_float = np.clip(mask_float, 0, 255)

        eroded = cv2.erode(mask, np.ones((feather_px, feather_px), np.uint8), iterations=1)
        mask_float[eroded > 0] = 255.0
        mask_3ch = np.stack([mask_float] * 3, axis=-1) / 255.0
    else:
        mask_3ch = np.stack([mask.astype(np.float32)] * 3, axis=-1) / 255.0

    result = scene.astype(np.float64)
    warped_f = warped.astype(np.float64)
    result = result * (1 - mask_3ch) + warped_f * mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Restore original scene pixels for foreground objects (things in front of TV)
    if foreground_bboxes:
        for fb in foreground_bboxes:
            fx1 = int(fb.x_min * w_scene)
            fy1 = int(fb.y_min * h_scene)
            fx2 = int(fb.x_max * w_scene)
            fy2 = int(fb.y_max * h_scene)
            # Clamp to image bounds
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(w_scene, fx2), min(h_scene, fy2)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            # Only restore pixels where the TV mask is active (inside TV region)
            region_mask = mask[fy1:fy2, fx1:fx2]
            orig_region = scene[fy1:fy2, fx1:fx2]
            result_region = result[fy1:fy2, fx1:fx2]
            # Where mask is set, restore original scene pixels
            restore = region_mask > 0
            result_region[restore] = orig_region[restore]
            result[fy1:fy2, fx1:fx2] = result_region

    return result


def _match_brightness(
    scene: np.ndarray,
    overlay: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    """
    Adjust overlay brightness to roughly match the ambient light around the TV
    by sampling pixels near the quad border.
    """
    h, w = scene.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = quad.astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    expand_px = 30
    kernel = np.ones((expand_px, expand_px), np.uint8)
    expanded = cv2.dilate(mask, kernel, iterations=1)
    border_mask = expanded - mask

    if border_mask.sum() == 0:
        return overlay

    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    border_pixels = scene_gray[border_mask > 0]
    ambient_brightness = np.mean(border_pixels)

    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    overlay_brightness = np.mean(overlay_gray)

    if overlay_brightness < 1:
        return overlay

    ratio = ambient_brightness / overlay_brightness
    ratio = np.clip(ratio, 0.4, 1.8)

    if abs(ratio - 1.0) < 0.1:
        return overlay

    adjusted = cv2.convertScaleAbs(overlay, alpha=ratio, beta=0)
    return adjusted
