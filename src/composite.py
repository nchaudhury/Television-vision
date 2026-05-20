import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def composite_overlay(
    scene: np.ndarray,
    overlay: np.ndarray,
    quads: List[np.ndarray],
    feather_px: int = 5,
    brightness_match: bool = True,
    bezel_pct: float = 0.05,
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
        bezel_pct: Fraction (0–1) to shrink each edge inward to simulate
                   a TV bezel. 0.05 = 5% inset on every side.

    Returns:
        Composited image (BGR, uint8)
    """
    result = scene.copy()

    for i, quad in enumerate(quads):
        logger.debug(f"  Compositing TV #{i + 1}")
        result = _composite_single(result, overlay, quad, feather_px, brightness_match, bezel_pct)

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

    adjusted_overlay = overlay.copy()
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
