# TV Screen Replacement Pipeline - Code Flow Overview

## Project Purpose
Automatically detects TV screens in real estate photos and composites a provided image onto each detected screen with perspective-correct warping. The system supports two detection backends selectable via `--detector`.

## Tech Stack
- **Python 3.x**
- **google-genai** - Google Gemini 3.5 Flash for TV detection (default detector)
- **fal-client / httpx** - fal.ai API client: SAM2 segmentation + Moondream3 (legacy) + nano-banana-2/edit (AI compositing)
- **OpenCV (cv2)** - Perspective warping, corner detection fallback, image processing
- **Pillow (PIL)** - Image I/O
- **NumPy** - Numerical operations
- **boto3** - AWS S3 integration (optional)
- **python-dotenv** - Environment variable management
- **tqdm** - Progress bars

## Project Structure
```
run.py                # CLI entry point and orchestration
src/
  detect_gemini.py    # Gemini 3.5 Flash + SAM2 detection pipeline (default)
  detect.py           # Moondream3 multi-step detection (legacy)
  corners_sam2.py     # SAM2 box-prompt → mask → 4-corner quad
  corners.py          # OpenCV edge/contour corner refinement (SAM2 fallback)
  composite.py        # Perspective warp + blending + occlusion masking (default compositor)
  composite_ai.py     # fal-ai/nano-banana-2/edit AI compositing (--ai_edit)
  cache.py            # Disk-based caching: bbox lists + arbitrary JSON
  s3_loader.py        # S3 download support
  utils.py            # Image I/O, logging, data URI helpers
Input/input.png       # Overlay image
TVPhotos/             # Input real estate photos
Output/               # Processed results
.cache/               # Auto-generated detection cache (gitignored)
.env                  # FAL_KEY + GEMINI_API_KEY (required)
```

## High-Level Pipeline Flow

### 1. Entry Point (`run.py`)
The `main()` function:
- Parses CLI arguments including `--detector [gemini|moondream]` (default: gemini) and `--ai_edit`
- Validates inputs, loads the overlay image
- Downloads from S3 if `--s3_uri` provided
- Loads checkpoint if `--resume` set
- Processes images sequentially or in parallel (`--workers`)
- Saves checkpoint after each image

### 2. Image Processing (`process_image()` in `run.py`)
For each input image:
1. Load scene image via `utils.load_image()`
2. **Detection** (branched by `--detector`):
   - `gemini`: calls `detect_tvs_gemini()` → returns TVDetections with SAM2 quads pre-populated in `det.quad`
   - `moondream`: calls `detect_tvs_full()` → returns TVDetections with `det.quad = None`
3. **Compositing** (branched by `--ai_edit`):
   - **Default (OpenCV):** Corner extraction per TV (SAM2 quad or OpenCV fallback) → clamp to bbox → `composite_overlay()`
   - **AI (`--ai_edit`):** Skip corner extraction entirely → call `composite_overlay_ai()` which uploads scene + overlay to fal.ai and calls nano-banana-2/edit
4. Save result

---

## Gemini Detector Pipeline (`src/detect_gemini.py`)

### `detect_tvs_gemini(image_path, use_cache)` — main entry point

**Step 1 — Gemini 3.5 Flash detection (1 API call):**
- Sends full image + structured prompt to `gemini-3.5-flash`
- JSON response includes for each TV:
  - `bbox`: screen surface bounding box [y_min, x_min, y_max, x_max] normalized 0–1
  - `is_mirror`: whether it's a reflection
  - `foreground_bboxes`: objects in front of the screen
- Also returns `mirror_regions` for the scene
- `_parse_bbox` auto-detects and normalises 0–1000 scale coordinates (a known Gemini quirk where the model returns integer-grid coords instead of normalized floats); logs a WARNING with raw values when a bbox is rejected

**Step 2 — SAM2 segmentation per TV (`src/corners_sam2.py`):**
- Calls `fal-ai/sam2/image` with the TV's bounding box as a box prompt
- SAM2 returns a pixel-level mask of the screen surface
- Mask decoded via recursive URI discovery: `_decode_mask` traverses the full response object to find any `http…` URL or `data:…` data-URI, regardless of key naming or nesting depth (robust to API schema changes)
  1. Largest external contour extracted from binary mask
  2. Convex hull computed
  3. `approxPolyDP` at decreasing epsilon to get exactly 4 corners
  4. `minAreaRect` fallback if polygon simplification fails
  5. Corners ordered [TL, TR, BR, BL]
- Result stored in `TVDetection.quad` (None if SAM2 fails → OpenCV fallback)
- INFO log prints mask response keys on every call for schema-change visibility

**Caching:**
- Full Gemini+SAM2 result cached as JSON under key `"gemini_detect_v1"` via `cache.get_cached_data / save_cached_data`
- Quads serialized as nested float lists, reconstructed as numpy arrays on cache hit

---

## Moondream3 Detector Pipeline (legacy, `src/detect.py`)

### `detect_tvs_full(image_path, use_cache)` — legacy entry point

**Step 1 — Coarse detection:**
- Prompt: `"television or monitor screen"`

**Step 2 — Per-TV screen refinement + back-of-TV filtering:**
- Crops to coarse bbox, runs refinement prompt
- Discards detections where no display face found (back-of-TV)

**Step 3 — Mirror scan (single full-image call):**
- Prompt: `"mirror or large glass surface reflecting the room"`

**Step 4 — Per-TV foreground detection:**
- Prompt: `"decorative items, figurines, plants... in front of television"`

`TVDetection.quad` is always `None` for Moondream3 results → `refine_corners()` runs in `run.py`.

---

## Corner Refinement Fallback (`src/corners.py`)
`refine_corners(image, bbox)` — used when SAM2 returns no quad:
- Crops image region with 0% padding
- Tries four strategies in sequence (min contour area: 10% of crop):
  1. Canny edge detection + contour analysis
  2. Adaptive threshold + contours
  3. Dark-region threshold — targets off (black) TV screens on lighter backgrounds
  4. HSV low-saturation segmentation + contours
- **`_best_quad_from_contours`**: for each large-enough contour, computes its convex hull then tries `approxPolyDP` at 5 decreasing epsilons; if none yield exactly 4 points, falls back to **`_corners_from_hull`** which projects each hull vertex onto the four diagonal directions and picks the most-extreme per direction — this naturally extracts the actual corners of a perspective-distorted (trapezoidal) screen
- Validates quad geometry (aspect ratio, side-length symmetry)
- Last resort: axis-aligned bbox rectangle if all strategies fail

---

## Perspective Compositing (`src/composite.py`)
`composite_overlay(scene, overlay, quads, tv_detections=None)` — default compositor:

For each TV quad:
1. **Mirror flip** (`cv2.flip(overlay, 1)`) if `is_mirror=True`
2. **Shrink quad** by `bezel_pct` toward centroid
3. **Brightness match**: adjust overlay to ambient surroundings
4. **Perspective warp**: `cv2.warpPerspective()` onto scene canvas
5. **Feathered blend**: Gaussian-blurred mask for natural edges
6. **Foreground restoration**: restore original pixels for objects in front of TV

---

## AI Compositing (`src/composite_ai.py`)
`composite_overlay_ai(scene_path, overlay_path, tv_detections, prompt=None)` — activated by `--ai_edit`:

Replaces the entire OpenCV corner-extraction + warp pipeline with a single generative AI call:

1. **Prompt construction**: if no `--ai_prompt` is provided, auto-builds a location-aware prompt using each TV's normalized bbox center (e.g. "at approximately 60% across and 40% down the image"). Multi-TV scenes get a count-based prompt.
2. **Upload**: `fal_client.upload_file()` uploads both the scene photo and the overlay image to fal.ai storage, returning public URLs.
3. **Generate**: `fal_client.subscribe("fal-ai/nano-banana-2/edit", ...)` sends `image_urls=[scene_url, overlay_url]` + the prompt. The model handles perspective, lighting, and reflections.
4. **Download & decode**: result image URL fetched via `httpx`, decoded to a BGR numpy array via `cv2.imdecode`.

**Trade-offs vs OpenCV path:**
- No explicit corner coordinates required — robust to reflective/glare-heavy screens
- Non-deterministic output; output resolution may differ from input
- Two extra fal.ai uploads per image; not cached (detection results still are)
- `--feather`, `--bezel`, `--no_brightness_match` have no effect in AI mode (the model handles all of this internally)

---

## Detection Caching (`src/cache.py`)
Two cache interfaces:
- `get_cached_detections / save_cached_detections` — stores `List[dict]` of bboxes; used by Moondream3 pipeline
- `get_cached_data / save_cached_data` — stores arbitrary JSON-serializable data; used by Gemini pipeline

Cache key: SHA-256 of image file + SHA-256 prefix of prompt/key string.
Cache file: `.cache/{image_hash}_{key_slug}.json`

---

## Data Flow Summary

```
Input Image
    ↓
[utils.load_image] → numpy array (BGR)
    ↓
[--detector gemini (default)]           [--detector moondream (legacy)]
    ↓                                       ↓
[detect_gemini.detect_tvs_gemini]       [detect.detect_tvs_full]
    ├─ Gemini 3.5 Flash (1 call)            ├─ Moondream3: coarse detect
    │  └─ TVs + mirrors + foreground        ├─ Moondream3: per-TV refine
    │     (back-of-TV excluded by prompt)   │  (back-of-TV: _PROMPT_BACK_OF_TV)
    └─ SAM2 per TV (N calls)                ├─ Moondream3: mirror scan
       └─ Pixel mask → 4-corner quad        └─ Moondream3: foreground
    ↓                                       ↓
[TVDetection list]──────────────────────────┘
    ↓
    ├── [--ai_edit] ──────────────────────────────────────────────────────┐
    │                                                                      ↓
    │                                              [composite_ai.composite_overlay_ai]
    │                                                  ├─ Auto-build location prompt
    │                                                  ├─ fal_client.upload_file (scene)
    │                                                  ├─ fal_client.upload_file (overlay)
    │                                                  └─ nano-banana-2/edit → download result
    │
    └── [default OpenCV path]
            ↓
        [TVDetection.quad set?]
            ├─ Yes: use SAM2 quad directly
            └─ No:  corners.refine_corners (OpenCV)
            ↓
        [composite.composite_overlay]
            ├─ Mirror TVs: flip overlay
            ├─ Perspective warp + feathered blend
            └─ Foreground objects: restore original pixels
    ↓
[utils.save_image] → Output directory
```

## Configuration

Required environment variables (`.env`):
- `FAL_KEY` — fal.ai API key (SAM2 + Moondream3)
- `GEMINI_API_KEY` — Google Gemini API key (Gemini detector)
- `GEMINI_MODEL` — Gemini model ID (default: `gemini-3.5-flash`)

Optional for S3:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

## API Call Budget Per Image

### Gemini detector + OpenCV compositor (default)
| Call | Endpoint | Count |
|------|---------|-------|
| TV + mirror + foreground detection | `gemini-3.5-flash` | 1 |
| Screen segmentation | `fal-ai/sam2/image` | N per TV |
| **Total (1 TV)** | | **2** |

### Gemini detector + AI compositor (`--ai_edit`)
| Call | Endpoint | Count |
|------|---------|-------|
| TV + mirror + foreground detection | `gemini-3.5-flash` | 1 |
| Screen segmentation | `fal-ai/sam2/image` | N per TV |
| Upload scene image | fal.ai storage | 1 |
| Upload overlay image | fal.ai storage | 1 |
| AI screen replacement | `fal-ai/nano-banana-2/edit` | 1 |
| **Total (1 TV)** | | **5** |

Detection results cached. AI compositing uploads/generation are not cached.

### Moondream3 detector (legacy)
| Call | Endpoint | Count |
|------|---------|-------|
| Coarse detect | Moondream3 | 1 |
| Screen refine | Moondream3 | N |
| Mirror scan | Moondream3 | 1 |
| Foreground | Moondream3 | N |
| **Total (1 TV)** | | **4** |

All detection results cached. Re-runs skip detection entirely.
