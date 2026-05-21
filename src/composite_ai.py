"""
AI-powered TV screen compositing using fal-ai/nano-banana-2/edit.

Alternative to the OpenCV perspective-warp path: uploads the scene photo and
the overlay image to fal.ai, then lets the model naturally replace the TV
screen content with correct perspective, lighting, and reflections.

Activated by passing --ai_edit to run.py.
"""
import logging
from typing import List, Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


def composite_overlay_ai(
    scene_path: str,
    overlay_path: str,
    tv_detections: List,
    prompt: Optional[str] = None,
) -> np.ndarray:
    """
    Use fal-ai/nano-banana-2/edit to replace TV screen content with the overlay.

    Args:
        scene_path:     Local path to the scene photo.
        overlay_path:   Local path to the overlay image to place on the TV screen.
        tv_detections:  Detected TVDetection objects; used to build a location-aware
                        prompt when no custom prompt is provided.
        prompt:         Override the auto-generated prompt. Leave None to auto-build.

    Returns:
        BGR uint8 numpy array of the AI-edited scene (may differ in resolution
        from the input if the model adjusts aspect ratio).
    """
    import fal_client  # lazy — only needed when --ai_edit is active

    edit_prompt = prompt or _build_prompt(tv_detections)

    logger.info(f"  [AI edit] Uploading scene: {scene_path}")
    scene_url = fal_client.upload_file(scene_path)

    logger.info(f"  [AI edit] Uploading overlay: {overlay_path}")
    overlay_url = fal_client.upload_file(overlay_path)

    logger.info(f"  [AI edit] nano-banana-2/edit prompt: {edit_prompt!r}")
    result = fal_client.subscribe(
        "fal-ai/nano-banana-2/edit",
        arguments={
            "prompt": edit_prompt,
            "image_urls": [scene_url, overlay_url],
        },
        with_logs=False,
    )

    images = result.get("images", [])
    if not images:
        raise RuntimeError("fal-ai/nano-banana-2/edit returned no images")

    output_url = images[0]["url"]
    logger.info(f"  [AI edit] Downloading result from {output_url}")

    resp = httpx.get(output_url, timeout=120.0, follow_redirects=True)
    resp.raise_for_status()

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode fal-ai/nano-banana-2/edit output")

    logger.info(f"  [AI edit] Result image: {img.shape[1]}x{img.shape[0]}")
    return img


def _build_prompt(tv_detections: List) -> str:
    """Build a location-aware editing prompt from the detected TV positions."""
    count = len(tv_detections)

    if count == 1:
        det = tv_detections[0]
        cx, cy = det.bbox.center()
        location = f"at approximately {cx:.0%} across and {cy:.0%} down the image"
        screen_desc = f"the TV or monitor screen {location}"
    else:
        screen_desc = f"all {count} TV or monitor screens"

    return (
        f"Replace {screen_desc} with the reference image shown in the second photo. "
        "Warp the reference image to match the screen's perspective and shape exactly. "
        "Adjust brightness and color to match the ambient lighting of the room. "
        "Leave every other part of the scene completely unchanged."
    )
