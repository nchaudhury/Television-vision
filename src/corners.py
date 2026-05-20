import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detect import BoundingBox

logger = logging.getLogger(__name__)


def refine_corners(
    image: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.1,
) -> np.ndarray:
    """
    Given a bounding box from Moondream3, find the precise 4 corners of the
    TV screen within that region using OpenCV edge/contour detection.

    Returns a (4, 2) numpy array of corners in order:
        [top-left, top-right, bottom-right, bottom-left]
    in full-image pixel coordinates.
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

    quad = _find_screen_quad(crop)

    if quad is not None:
        quad[:, 0] += x1
        quad[:, 1] += y1
        return quad

    return _bbox_to_quad(px)


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
    min_area = ch * cw * 0.15

    quad = _try_edge_detection(gray, min_area)
    if quad is not None:
        return quad

    quad = _try_adaptive_threshold(gray, min_area)
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


def _try_color_segmentation(crop: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv, (0, 0, 0), (180, 80, 60))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_CLOSE, kernel, iterations=2)
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(low_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _best_quad_from_contours(contours, min_area)


def _best_quad_from_contours(
    contours: List,
    min_area: float,
) -> Optional[np.ndarray]:
    """Find the best 4-sided contour that's large enough and roughly rectangular."""
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = _order_corners(corners)
            if _is_reasonable_quad(corners):
                candidates.append((area, corners))

    if not candidates:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                corners = _order_corners(corners)
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
