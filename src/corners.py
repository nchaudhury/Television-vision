import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detect import BoundingBox

logger = logging.getLogger(__name__)


def refine_corners(
    image: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.0,
) -> np.ndarray:
    """
    Given a bounding box from Moondream3, find the precise 4 corners of the
    TV screen within that region using OpenCV edge/contour detection.

    Returns a (4, 2) numpy array of corners in order:
        [top-left, top-right, bottom-right, bottom-left]
    in full-image pixel coordinates.

    The returned quad is guaranteed to stay within the bbox bounds (+ 1% tolerance)
    so shadow and furniture edges cannot bleed into the compositing area.
    """
    h, w = image.shape[:2]
    px = bbox.to_pixel_coords(w, h)

    pad_x = int((px["x_max"] - px["x_min"]) * padding_ratio)
    pad_y = int((px["y_max"] - px["y_min"]) * padding_ratio)
    x1 = max(0, px["x_min"] - pad_x)
    y1 = max(0, px["y_min"] - pad_y)
    x2 = min(w, px["x_max"] + pad_x)
    y2 = min(h, px["y_max"] + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return _bbox_to_quad(px)

    bbox_quad = _bbox_to_quad(px)
    quad = _find_screen_quad(crop)

    if quad is not None:
        quad[:, 0] += x1
        quad[:, 1] += y1
        quad = _clamp_quad_to_bbox(quad, px)
        # If the found quad is implausibly larger than the bbox, discard it
        found_area = _quad_area(quad)
        bbox_area = _quad_area(bbox_quad)
        if found_area > bbox_area * 1.15:
            logger.debug("  Corners: found quad larger than bbox — using bbox rectangle")
            return bbox_quad
        return quad

    return bbox_quad


def _clamp_quad_to_bbox(quad: np.ndarray, px: dict) -> np.ndarray:
    """Clamp all quad corners to the bbox bounds + 1% tolerance."""
    tol_x = (px["x_max"] - px["x_min"]) * 0.01
    tol_y = (px["y_max"] - px["y_min"]) * 0.01
    clamped = quad.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], px["x_min"] - tol_x, px["x_max"] + tol_x)
    clamped[:, 1] = np.clip(clamped[:, 1], px["y_min"] - tol_y, px["y_max"] + tol_y)
    return clamped


def _quad_area(quad: np.ndarray) -> float:
    """Approximate area of a quad using the shoelace formula."""
    n = len(quad)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += quad[i][0] * quad[j][1]
        area -= quad[j][0] * quad[i][1]
    return abs(area) / 2.0


def _bbox_to_quad(px: dict) -> np.ndarray:
    """Fallback: use the bounding box itself as a rectangle quad."""
    return np.array([
        [px["x_min"], px["y_min"]],  # top-left
        [px["x_max"], px["y_min"]],  # top-right
        [px["x_max"], px["y_max"]],  # bottom-right
        [px["x_min"], px["y_max"]],  # bottom-left
    ], dtype=np.float32)


def _find_screen_quad(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt to find a 4-sided contour in the cropped region that represents
    the TV screen. Uses multiple strategies for robustness.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    min_area = ch * cw * 0.10  # 10% — more permissive than the old 15%

    quad = _try_edge_detection(gray, min_area)
    if quad is not None:
        return quad

    quad = _try_adaptive_threshold(gray, min_area)
    if quad is not None:
        return quad

    quad = _try_dark_screen(crop, min_area)
    if quad is not None:
        return quad

    quad = _try_color_segmentation(crop, min_area)
    if quad is not None:
        return quad

    logger.debug("  Corner refinement: no quad found, using bbox fallback")
    return None


def _try_edge_detection(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median_val = np.median(blurred)
    low = int(max(0, 0.5 * median_val))
    high = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(blurred, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _best_quad_from_contours(contours, min_area)


def _try_adaptive_threshold(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _best_quad_from_contours(contours, min_area)


def _try_dark_screen(crop: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """
    Detect large uniformly dark regions — useful for off (black) TV screens
    against lighter walls.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=3)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _best_quad_from_contours(contours, min_area)


def _try_color_segmentation(crop: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv, (0, 0, 0), (180, 80, 60))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_CLOSE, kernel, iterations=2)
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(low_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _best_quad_from_contours(contours, min_area)


def _corners_from_hull(hull_pts: np.ndarray) -> np.ndarray:
    """
    Extract 4 corner-like points from a convex hull by finding the hull vertex
    that projects furthest in each of the four diagonal directions (TL/TR/BR/BL).

    This naturally handles perspective-distorted screens (trapezoids) because it
    picks the four most "extreme" vertices rather than fitting an axis-aligned rect.
    """
    centroid = hull_pts.mean(axis=0)
    # Each direction is the unit diagonal toward that corner quadrant
    dirs = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]  # TL TR BR BL
    corners = []
    for dx, dy in dirs:
        scores = [
            dx * (p[0] - centroid[0]) + dy * (p[1] - centroid[1])
            for p in hull_pts
        ]
        corners.append(hull_pts[int(np.argmax(scores))])
    return _order_corners(np.array(corners, dtype=np.float32))


def _best_quad_from_contours(
    contours: List,
    min_area: float,
) -> Optional[np.ndarray]:
    """
    Find the best 4-corner quad from a set of contours.

    Strategy per contour (large enough ones only):
      1. Try approxPolyDP on the convex hull at progressively coarser epsilon
         until we get exactly 4 points.
      2. If no epsilon gives 4 points, fall back to _corners_from_hull which
         extracts 4 extreme diagonal hull vertices — handles perspective well.
    Picks the largest-area candidate.
    """
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        if len(hull_pts) < 4:
            continue

        peri = cv2.arcLength(hull, True)
        found = False
        for eps_frac in (0.02, 0.03, 0.05, 0.07, 0.10):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                corners = _order_corners(corners)
                if _is_reasonable_quad(corners):
                    candidates.append((area, corners))
                    found = True
                    break

        if not found:
            # Hull didn't simplify to 4 — use diagonal-extreme extraction
            corners = _corners_from_hull(hull_pts)
            if _is_reasonable_quad(corners):
                candidates.append((area, corners))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order corners as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]       # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]       # bottom-right: largest x+y
    rect[1] = pts[np.argmin(d)]       # top-right: smallest x-y
    rect[3] = pts[np.argmax(d)]       # bottom-left: largest x-y
    return rect


def _is_reasonable_quad(corners: np.ndarray) -> bool:
    """Check that the quad is roughly rectangular (not too skewed)."""
    def _dist(a, b):
        return np.linalg.norm(a - b)

    w_top = _dist(corners[0], corners[1])
    w_bot = _dist(corners[3], corners[2])
    h_left = _dist(corners[0], corners[3])
    h_right = _dist(corners[1], corners[2])

    if min(w_top, w_bot) < 10 or min(h_left, h_right) < 10:
        return False

    w_ratio = max(w_top, w_bot) / max(min(w_top, w_bot), 1)
    h_ratio = max(h_left, h_right) / max(min(h_left, h_right), 1)
    if w_ratio > 3.0 or h_ratio > 3.0:
        return False

    aspect = max(w_top, w_bot) / max(min(h_left, h_right), 1)
    if aspect > 10 or aspect < 0.1:
        return False

    return True
