# TV Screen Replacement Pipeline

Automatically detects TV screens in real estate photos and composites a provided image onto each detected screen with perspective-correct warping.

## Approach

1. **TV Detection** — Uses [fal.ai Moondream3](https://fal.ai/models/fal-ai/moondream3-preview/detect) vision model to detect TV/monitor bounding boxes in each photo with a single combined prompt. Detection results are cached to disk to avoid redundant API calls on re-runs.

2. **Corner Refinement** — Moondream3 returns axis-aligned bounding boxes, not screen corner points. Within each detected bounding box, OpenCV edge detection and contour analysis find the precise 4-corner quadrilateral of the screen. This handles angled and perspective views. Falls back to the bounding box rectangle if no quad is found.

3. **Perspective Compositing** — The overlay image is warped onto each TV screen using `cv2.getPerspectiveTransform` + `cv2.warpPerspective`. Edge feathering (Gaussian-blurred mask) and ambient brightness matching produce natural-looking results.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
# Copy .env.example to .env and set your fal.ai API key
cp .env.example .env
# Edit .env with your FAL_KEY
```

## Usage

### Local images
```bash
python run.py --input_dir ./TVPhotos --output_dir ./Output --overlay ./Input/input.png
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

### Options
| Flag | Description | Default |
|------|-------------|---------|
| `--input_dir` | Local directory with input images | — |
| `--s3_uri` | S3 URI to download images from | — |
| `--output_dir` | Directory for processed output | `./Output` |
| `--overlay` | Image to place on TV screens | `./Input/input.png` |
| `--feather` | Edge feathering in pixels (0=off) | `5` |
| `--no_brightness_match` | Disable ambient brightness adjustment | off |
| `--bezel` | Bezel inset fraction (0.05 = 5%) | `0.05` |
| `--max_s3_files` | Limit S3 downloads | all |
| `--resume` | Skip already-processed images (checkpoint-based) | off |
| `--workers` | Number of parallel workers | `1` |
| `--no_cache` | Disable detection result caching | off |
| `-v` / `--verbose` | Debug logging | off |

## Key Decisions

- **fal.ai Moondream3 over classical CV** — Vision LLM provides robust detection across diverse real estate photos (varying lighting, angles, TV sizes, wall colors).
- **Hybrid detection pipeline** — LLM for coarse detection + OpenCV for precise geometry gives the best quality for perspective warping.
- **Single optimized prompt** — A combined "television or monitor screen" prompt reduces API calls from 3 to 1 per image.
- **Detection caching** — Results are cached to disk (keyed by image SHA-256) so re-runs skip the API entirely.
- **Checkpointing & resume** — Progress is saved after each image; use `--resume` to continue interrupted runs.
- **Parallel processing** — `--workers N` processes multiple images concurrently via `ProcessPoolExecutor`.
- **Automatic retries** — Transient failures are retried with exponential backoff at both API and image-processing levels.
- **Quality-first compositing** — Edge feathering and brightness matching ensure the overlay looks natural on-screen.

## Project Structure

```
run.py              # CLI entry point
src/
  detect.py         # fal.ai Moondream3 TV detection
  corners.py        # OpenCV corner refinement
  composite.py      # Perspective warp + blending
  cache.py          # Disk-based detection result caching
  s3_loader.py      # Optional S3 download support
  utils.py          # Image I/O, logging helpers
Input/input.png     # Overlay image (beach scene)
TVPhotos/           # Input real estate photos
Output/             # Processed results
.cache/             # Auto-generated detection cache (gitignored)
```
