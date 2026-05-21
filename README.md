# TV Screen Replacement Pipeline

Automatically detects TV screens in real estate photos and composites a provided image onto each detected screen with perspective-correct warping.

## Approach

### Default: Gemini 3.5 Flash + SAM2 (recommended)

1. **TV Detection** — [Gemini 3.5 Flash](https://ai.google.dev/) analyzes the full scene in a single API call, returning all TV/monitor bounding boxes. Unlike smaller models, Gemini reliably distinguishes flat-panel TVs from artwork, paintings, picture frames, and windows.

2. **Screen Segmentation** — For each detected TV, [SAM2 on fal.ai](https://fal.ai/models/fal-ai/sam2/image) segments the screen surface with a box prompt, returning a pixel-level mask. The mask is converted to a precise 4-corner quadrilateral that accounts for perspective, tilt, and partial occlusion.

3. **Mirror Detection** — Gemini identifies mirrors in the scene in the same detection call. TVs whose bounding box center falls inside a mirror region receive a horizontally flipped overlay.

4. **Foreground Occlusion** — Gemini returns objects visually in front of the screen. After compositing, original scene pixels are restored in those regions so foreground objects remain visible.

5. **Perspective Compositing** — The overlay image is warped onto each screen quad using `cv2.getPerspectiveTransform` + `cv2.warpPerspective`. Edge feathering and ambient brightness matching produce natural-looking results.

### Legacy: Moondream3 pipeline (`--detector moondream`)

Uses [fal.ai Moondream3](https://fal.ai/models/fal-ai/moondream3-preview/detect) for coarse detection + screen refinement + mirror scan + foreground detection (4 API calls per TV), followed by OpenCV edge/contour corner refinement.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — set FAL_KEY and GEMINI_API_KEY
```

## Usage

### Local images (Gemini detector, default)
```bash
python run.py --input_dir ./TVPhotos --output_dir ./Output --overlay ./Input/input.png
```

### Explicit detector selection
```bash
# Gemini 3.5 Flash + SAM2 (default, best quality)
python run.py --input_dir ./TVPhotos --output_dir ./Output --detector gemini

# Original Moondream3 pipeline (legacy fallback)
python run.py --input_dir ./TVPhotos --output_dir ./Output --detector moondream
```

### From S3
```bash
python run.py --s3_uri s3://autohdr-tv-test-project/images --output_dir ./Output --overlay ./Input/input.png
```

### Resume an interrupted run
```bash
python run.py --input_dir ./TVPhotos --output_dir ./Output --resume
```

### Parallel processing
```bash
python run.py --input_dir ./TVPhotos --output_dir ./Output --workers 4
```

### AI compositing (nano-banana-2/edit)
```bash
# Let fal.ai nano-banana-2/edit handle the screen replacement (no geometric warp)
# --overlay is required (or defaults to ./Input/input.png) — it is uploaded as the
# reference image the model uses to fill the TV screen
python run.py --input_dir ./TVPhotos --output_dir ./Output \
  --overlay ./Input/input.png --ai_edit

# With a custom prompt
python run.py --input_dir ./TVPhotos --output_dir ./Output \
  --overlay ./Input/input.png --ai_edit \
  --ai_prompt "Replace the TV screen with the image shown in the second photo, preserving reflections"
```

### Options
| Flag | Description | Default |
|------|-------------|---------|
| `--input_dir` | Local directory with input images | — |
| `--s3_uri` | S3 URI to download images from | — |
| `--output_dir` | Directory for processed output | `./Output` |
| `--overlay` | Image to place on TV screens | `./Input/input.png` |
| `--detector` | Detection backend: `gemini` or `moondream` | `gemini` |
| `--ai_edit` | Use fal-ai/nano-banana-2/edit for compositing (AI warp + blend) | off |
| `--ai_prompt` | Custom prompt for nano-banana-2/edit (used with `--ai_edit`) | auto |
| `--feather` | Edge feathering in pixels (0=off) | `5` |
| `--no_brightness_match` | Disable ambient brightness adjustment | off |
| `--bezel` | Bezel inset fraction (0.05 = 5%) | `0.05` |
| `--max_s3_files` | Limit S3 downloads | all |
| `--resume` | Skip already-processed images (checkpoint-based) | off |
| `--workers` | Number of parallel workers | `1` |
| `--no_cache` | Disable detection result caching | off |
| `-v` / `--verbose` | Debug logging | off |

## Compositing Backends

### Default: OpenCV geometric warp
Corner extraction (SAM2 or OpenCV fallback) → `cv2.getPerspectiveTransform` + `cv2.warpPerspective`. Fast, deterministic, works offline after detection.

### AI: fal-ai/nano-banana-2/edit (`--ai_edit`)
Uploads the scene photo and overlay image to fal.ai and instructs the model to replace the TV screen content. The model handles perspective correction, lighting integration, and reflections without explicit corner coordinates.

- **When to use**: scenes where corner detection is imprecise (highly reflective screens, severe glare, unusual angles) or when you want the most photorealistic blending.
- **Trade-off**: two fal.ai uploads per image + one generation call; non-deterministic; output resolution may differ from input.
- **Prompt**: auto-generated from detected TV positions. Override with `--ai_prompt`.

## Key Decisions

- **Gemini 3.5 Flash as primary detector** — Vastly superior scene understanding vs. smaller models. A single call handles TV detection, mirror identification, and foreground occlusion — eliminating misdetections of artwork and missed edge-case screens.
- **SAM2 for pixel-precise corners** — SAM2's box-prompted segmentation gives accurate screen quads even for angled, partially occluded, or low-contrast screens. Mask decoding uses recursive URI discovery (not a fixed key list) so it is robust to API schema changes.
- **Graceful fallback chain** — If SAM2 returns no quad, `refine_corners()` (OpenCV) runs instead. The OpenCV fallback uses hull-based diagonal-extreme corner extraction (`_corners_from_hull`) so perspective-distorted (trapezoidal) TV screens get correctly warped quads rather than a flat axis-aligned rectangle. If Gemini fails entirely, the moondream pipeline remains available via `--detector moondream`.
- **Coordinate scale normalisation** — `_parse_bbox` auto-detects Gemini's occasional 0–1000 scale output and normalises it before bbox validation, preventing silent rejection of valid detections.
- **Unified cache** — Both Gemini+SAM2 results and Moondream3 results are cached to `.cache/` keyed by image SHA-256 + prompt hash. Re-runs skip all API calls.
- **Back-of-TV filtering** — The Gemini detection prompt explicitly excludes rear-panel hardware (cable ports, HDMI connectors, ventilation slots). Moondream3 pipeline uses a dedicated refinement call with `_PROMPT_BACK_OF_TV`.
- **Mirror awareness** — Mirror regions detected alongside TVs; reflected TVs receive a horizontally flipped overlay.
- **Foreground occlusion masking** — Objects in front of TVs are detected and preserved; original pixels restored after compositing.
- **Checkpointing & resume** — Progress saved after each image; `--resume` continues interrupted runs.
- **Parallel processing** — `--workers N` processes multiple images concurrently via `ProcessPoolExecutor`.

## API Call Budget Per Image

### Gemini detector (default)
| Call | Purpose | Count |
|------|---------|-------|
| Gemini 3.5 Flash | Detect TVs + mirrors + foreground | 1 |
| SAM2 | Segment screen per TV | N per TV |
| **Total** | | **1 + N** |

With 1 TV: **2 calls**. All results cached — re-runs are free.

### Moondream3 detector (legacy)
| Call | Purpose | Count |
|------|---------|-------|
| Initial detection | Locate TV candidates | 1 |
| Screen refinement | Tighten bounds + filter back-of-TV | N per TV |
| Mirror scan | Detect mirror regions | 1 |
| Foreground detection | Find objects in front of TV | N per TV |
| **Total** | | **2 + 2N** |

With 1 TV: **4 calls**.

## Project Structure

```
run.py              # CLI entry point
src/
  detect_gemini.py  # Gemini 3.5 Flash + SAM2 detection pipeline (default)
  detect.py         # Moondream3 TV detection (legacy --detector moondream)
  corners_sam2.py   # SAM2 mask → 4-corner quad extraction
  corners.py        # OpenCV corner refinement (SAM2 fallback)
  composite.py      # Perspective warp + blending + occlusion masking (default)
  composite_ai.py   # fal-ai/nano-banana-2/edit AI compositing (--ai_edit)
  cache.py          # Disk-based detection result caching (image+prompt keyed)
  s3_loader.py      # Optional S3 download support
  utils.py          # Image I/O, logging helpers
Input/input.png     # Overlay image (beach scene)
TVPhotos/           # Input real estate photos
Output/             # Processed results
.cache/             # Auto-generated detection cache (gitignored)
```
