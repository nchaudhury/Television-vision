## 1. ~~Back-of-TV Detections~~ ✅ **RESOLVED**

**Location:** `src/detect.py` — `detect_tvs_full()`; `src/detect_gemini.py` — `_DETECTION_PROMPT`

**Problem:** Moondream3 sometimes detects the back of a TV or monitor. The overlay was incorrectly pasted onto the TV housing instead of the screen.

**Solution (Moondream3):** A second Moondream3 call is made per TV using the cropped bbox region. If back-panel hardware is detected (`_PROMPT_BACK_OF_TV`) → detection discarded.

**Solution (Gemini):** The detection prompt explicitly excludes rear-facing panels: *"EXCLUDE: the back of a TV or monitor (visible cable ports, HDMI connectors, ventilation slots, power sockets, or any rear-panel hardware facing the viewer)"*. Only front-facing screen surfaces are included.

---

## 2. ~~Shadow-Inclusive Bounding Boxes / Skew~~ ✅ **RESOLVED**

**Location:** `src/detect.py` — Step 2; `src/detect_gemini.py` + SAM2

**Problem:** Axis-aligned bounding boxes included TV shadows and bezels, producing skewed overlays.

**Solution (Moondream3):** Screen refinement call targets screen surface only.
**Solution (Gemini):** Gemini returns screen-surface bbox; SAM2 then provides pixel-perfect mask → precise perspective-correct quad regardless of tilt or shadow.

---

## 3. ~~Foreground Object Occlusion~~ ✅ **RESOLVED**

**Location:** `src/detect.py` / `src/detect_gemini.py`; `src/composite.py`

**Problem:** Objects placed in front of the TV (figurines, plants, vases) were covered by the overlay.

**Solution:** Gemini (and Moondream3) detect foreground objects. After compositing, original scene pixels are restored within each foreground bbox clipped to the TV mask.

---

## 4. ~~Mirror Reflection Handling~~ ✅ **RESOLVED**

**Location:** `src/detect.py` / `src/detect_gemini.py`; `src/composite.py`

**Problem:** A TV reflected in a mirror received the same overlay as a direct-view TV.

**Solution:** Mirror regions detected. TVs whose bbox center falls inside a mirror → `is_mirror=True`. Compositing applies `cv2.flip(overlay, 1)` for mirror TVs. Gemini handles this in the same detection call at no extra cost.

---

## 5. ~~Moondream3 Artwork/Painting Misdetection~~ ✅ **RESOLVED**

**Location:** `src/detect_gemini.py`

**Problem:** Moondream3 frequently confused rectangular framed paintings, artwork, and decorative pieces with TV screens, producing incorrect composites on non-TV objects. Example: `168_src.jpg` had the overlay placed on a flower painting instead of the actual TV.

**Root cause:** Moondream3 is a small vision model without deep scene-level semantic understanding. It matches geometric shape (dark rectangle with border) rather than understanding what the object is.

**Solution:** Gemini 3.5 Flash (default detector) has deep semantic scene understanding. It explicitly distinguishes flat-panel displays from artwork, paintings, photos, and picture frames. The detection prompt includes explicit exclusion rules.

---

## 6. ~~Missed TV Detections~~ ✅ **RESOLVED**

**Location:** `src/detect_gemini.py`

**Problem:** Moondream3 missed TVs in several failure cases:
- `102_src.jpg`: large flat-panel TV at right edge of a wide commercial lobby scene
- `107_src.jpg`: TV above sofa in complex multi-element scene — detected but composite failed
- `11_src.jpg`: second TV/monitor visible through a doorway in a distant room
- `20_src.jpg`: side-wall TV at an angle not composited

**Root cause:** Moondream3 struggles with TVs that are small in frame, near image edges, in complex multi-object scenes, or at challenging perspective angles.

**Solution:** Gemini 3.5 Flash has far superior spatial reasoning and handles all of these cases reliably. A single call covers the full scene with rich contextual understanding.

---

## 7. ~~Incorrect Perspective / Wrong Bounding for Angled TVs~~ ✅ **RESOLVED**

**Location:** `src/corners_sam2.py`, `src/detect_gemini.py`

**Problem:** For TVs viewed at an angle (e.g. `21_src.jpg`, `20_src.jpg`), the overlay was placed with incorrect perspective because:
- Axis-aligned bounding boxes don't capture trapezoid screen shape
- OpenCV edge detection sometimes found the wrong contour (furniture, wall edges)

**Solution:** SAM2 (`fal-ai/sam2/image`) receives the TV's bounding box as a prompt and returns a pixel-level mask of the exact screen surface. The mask convex hull is simplified to a 4-corner quadrilateral that captures the true perspective shape. `minAreaRect` is used as fallback when approxPolyDP can't simplify cleanly to 4 points. OpenCV corner detection remains as a second-level fallback if SAM2 returns no mask.

---

## 17. ~~Geometric Warp Fails on Glare/Reflective Screens~~ ✅ **RESOLVED** (optional path)

**Location:** `src/composite_ai.py`; `run.py` — `--ai_edit`

**Problem:** The OpenCV perspective warp path relies on accurate 4-corner extraction. On highly reflective screens, screens with severe glare, or frames where SAM2 returns a noisy mask, corner extraction can produce a quad that doesn't precisely match the visible screen boundary. The result is a composited image where the overlay doesn't align naturally with the scene's lighting or reflections.

**Solution:** `--ai_edit` replaces the corner extraction + warp step with a single `fal-ai/nano-banana-2/edit` call. The model receives the scene photo and the overlay image and is prompted to replace the TV screen content. It handles perspective warping, lighting adaptation, and reflection blending internally without explicit corner coordinates.

**Implementation:**
- `src/composite_ai.py`: `composite_overlay_ai(scene_path, overlay_path, tv_detections, prompt)`
  - Auto-builds a location-aware prompt from each TV's normalized bbox center
  - Uploads both images via `fal_client.upload_file()`
  - Calls `fal-ai/nano-banana-2/edit` with `image_urls=[scene_url, overlay_url]`
  - Downloads and decodes result via `httpx` + `cv2.imdecode`
- `run.py`: `--ai_edit` flag routes compositing to `composite_overlay_ai()` instead of `composite.composite_overlay()`; `--ai_prompt` allows a custom prompt override
- Detection still runs first (Gemini or Moondream) — used to confirm TVs exist and to build the location prompt

**Trade-offs:** Non-deterministic; two extra fal.ai uploads per image; not cached; `--feather`/`--bezel`/`--no_brightness_match` have no effect in AI mode (handled by the model). Output resolution may differ from input.

---

## 8. No Batch API Support

**Problem:** API calls are per-image rather than batched.

**Status:** Gemini detector reduces calls from 4 to 2 per image. True batch API is not currently supported by these endpoints.

---

## 9. ~~Corner Refinement Strategy Overhead~~ ✅ **RESOLVED** (improved)

**Location:** `src/corners.py`

**Previous issue:** OpenCV corner refinement tried 3 strategies sequentially and still sometimes failed for angled or low-contrast screens. The last-resort fallback was a flat axis-aligned bounding box — if SAM2 also failed, perspective-distorted TVs received no perspective warp at all.

**Resolution (original):** SAM2 segmentation in the Gemini pipeline was intended to replace OpenCV corner detection. OpenCV strategies remain as a graceful fallback when SAM2 returns no mask.

**Resolution (improved — see Issue 16):** Because SAM2 mask decoding was failing silently (Issue 16), the OpenCV fallback was the actual code path for every image. It has been overhauled:

- **`_corners_from_hull`** (new): Instead of requiring `approxPolyDP` to collapse to exactly 4 points (which fails for perspective-distorted/trapezoidal screens), the function now projects each hull vertex onto the four diagonal directions (TL/TR/BR/BL) and picks the most extreme one per direction. This produces the true geometric corners of a trapezoid, not a rotated rectangle.
- **`_best_quad_from_contours`** (overhauled): Operates on the convex hull of each contour (not the raw contour), tries 5 epsilon values instead of 2, and falls back to `_corners_from_hull` when `approxPolyDP` never yields 4 points.
- **`_try_dark_screen`** (new strategy): Thresholds for large uniformly dark regions — targets off (black) TV screens on lighter walls. Inserted as strategy 3 (before `_try_color_segmentation`).
- **min_area threshold** lowered from 15% to 10% of the crop area.

---

## 10. ~~S3 Download Inefficiency~~ ✅ **RESOLVED**

Parallel downloads via `ThreadPoolExecutor` + `TransferConfig` with multipart downloads.

---

## 11. ~~Lack of Progress Persistence~~ ✅ **RESOLVED**

Checkpointing with `--resume` flag. Progress saved after each image.

---

## 12. ~~Limited Error Recovery~~ ✅ **RESOLVED**

Retry logic with exponential backoff at both API and image-processing levels.

---

## 13. ~~Duplicate API Calls~~ ✅ **RESOLVED**

Gemini detector: 1 + N calls per image (vs. 2 + 2N for Moondream3).

---

## 14. ~~Result Caching~~ ✅ **RESOLVED** (extended)

`cache.py` now exposes both `get/save_cached_detections` (bbox list format, for Moondream3) and `get/save_cached_data` (arbitrary JSON, for Gemini+SAM2). All results cached by `(image_hash, key_slug)`.

---

## 15. ~~Gemini 0–1000 Scale Coordinates → TVs Silently Rejected~~ ✅ **RESOLVED**

**Location:** `src/detect_gemini.py` — `_parse_bbox()`

**Problem:** Gemini occasionally returns bounding box coordinates in 0–1000 scale (a known model quirk) instead of the normalized 0–1 range requested by the prompt. After clamping with `min(1.0, x)`, all values collapse to `1.0`, producing a zero-area bbox that is immediately rejected. The log showed "Gemini detected N TV(s)" followed by "0 confirmed TV(s)" with no visible explanation (the rejection was logged at DEBUG level only).

**Root cause:** Models like Gemini sometimes output integer-grid coordinates as if the image is 1000×1000, ignoring the normalization instruction.

**Solution:** Before clamping, `_parse_bbox` now checks if `max(coords) > 1.5`. If so, all four values are divided by 1000 before further processing. The rejection log is also promoted from DEBUG to WARNING and now includes the raw coordinate values.

---

## 16. ~~SAM2 Mask Decode Always Fails (Hardcoded Key Lookup)~~ ✅ **RESOLVED**

**Location:** `src/corners_sam2.py` — `_decode_mask()`

**Problem:** `_decode_mask` searched for mask image data under a fixed list of keys (`mask`, `mask_image_url`, `url`, `image_url`). The fal.ai SAM2 API response uses different key nesting than expected, so the mask URL was never found. Every SAM2 call returned a mask object but `_decode_mask` returned `None`, causing `get_sam2_quad` to return `None` and fall through to the OpenCV fallback for 100% of images.

**Root cause:** The fal.ai SAM2 API response schema changed / differed from the hardcoded key list.

**Solution:** `_decode_mask` now uses a recursive traversal (`_collect`) that finds any string value resembling a URL (`http…`) or data-URI (`data:…`) anywhere in the response structure, regardless of key depth or naming. An INFO-level log now prints the top-level mask keys on every call so schema changes are immediately visible in normal (non-verbose) logs.

---

## Summary of All Optimizations

| Issue | Before | After | Status |
|-------|--------|-------|--------|
| Artwork misdetected as TV | Moondream3 confuses paintings/frames | Gemini 3.5 Flash semantic understanding | ✅ Fixed |
| Missed TVs (edge cases) | Moondream3 misses distant/angled/edge TVs | Gemini handles all scene types | ✅ Fixed |
| Incorrect perspective quad | Axis-aligned bbox + fragile OpenCV | SAM2 pixel mask → precise 4-corner quad | ✅ Fixed |
| Back-of-TV detection | No filtering | Screen refinement (Moondream3) / semantic (Gemini) | ✅ Fixed |
| Shadow-inclusive bbox | Bbox includes shadow | SAM2 mask is screen-surface only | ✅ Fixed |
| Foreground occlusion | Overlay covers foreground objects | Detected foreground bbox → pixel restoration | ✅ Fixed |
| Mirror reflections | Wrong-direction overlay | Mirror detection + horizontal flip | ✅ Fixed |
| API calls per image | 2 + 2N (Moondream3) | 1 + N (Gemini) | ✅ Reduced |
| Result caching | Per (image, prompt) bbox list | + generic JSON cache for Gemini results | ✅ Extended |
| Image processing | Sequential | Parallel (`--workers`) | ✅ Implemented |
| S3 downloads | Sequential | Parallel + multipart | ✅ Optimized |
| Progress persistence | None | Checkpointing + resume | ✅ Implemented |
| Error recovery | Basic | Retry with exponential backoff | ✅ Enhanced |
| Detector choice | Moondream3 only | `--detector gemini` (default) or `moondream` | ✅ Implemented |
| Gemini 0–1000 scale coords silently rejected | Area=0 after clamping → 0 confirmed TVs | Auto-detect scale (max>1.5) → divide by 1000 | ✅ Fixed |
| SAM2 mask decode always fails | Hardcoded key list missed actual response schema | Recursive URI discovery traverses full response | ✅ Fixed |
| OpenCV fallback: flat rect for angled TVs | `approxPolyDP` demanded exactly 4 pts; fell back to axis-aligned bbox | `_corners_from_hull` extracts diagonal-extreme vertices; handles trapezoids | ✅ Improved |
| Glare/reflective screens: poor warp alignment | Geometric warp needs accurate corners; fails on bright/reflective panels | `--ai_edit` uses nano-banana-2/edit; no corner coords needed | ✅ Implemented |
| Back-of-TV: Gemini had no explicit exclusion rule | Relied on implicit semantic understanding only | Explicit EXCLUDE rule added to `_DETECTION_PROMPT` for rear-panel hardware | ✅ Hardened |
