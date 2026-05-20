# TV Screen Replacement Pipeline - Code Flow Overview

## Project Purpose
Automatically detects TV screens in real estate photos and composites a provided image onto each detected screen with perspective-correct warping. The system uses a hybrid approach combining AI vision models with classical computer vision techniques.

## Tech Stack
- **Python 3.x**
- **fal-client** - API client for fal.ai Moondream3 vision model
- **OpenCV (cv2)** - Image processing, corner detection, perspective warping
- **Pillow (PIL)** - Image I/O
- **NumPy** - Numerical operations
- **boto3** - AWS S3 integration (optional)
- **python-dotenv** - Environment variable management
- **tqdm** - Progress bars
- **httpx** - HTTP client for API requests

## Project Structure
```
run.py              # CLI entry point and orchestration
src/
  detect.py         # TV detection using fal.ai Moondream3
  corners.py        # Corner refinement using OpenCV
  composite.py      # Perspective warp + blending
  cache.py          # Disk-based detection result caching
  s3_loader.py      # S3 download support
  utils.py          # Image I/O, logging, data URI helpers
Input/input.png     # Overlay image (beach scene)
TVPhotos/           # Input real estate photos
Output/             # Processed results
.cache/             # Auto-generated detection cache (gitignored)
.env                # API keys (FAL_KEY required)
```

## High-Level Pipeline Flow

### 1. Entry Point (`run.py`)
The `main()` function:
- Parses CLI arguments (`--input_dir` or `--s3_uri`, `--output_dir`, `--overlay`, etc.)
- Validates inputs and loads the overlay image
- If S3 URI provided, downloads images to local temp directory via `s3_loader.py`
- Loads checkpoint if `--resume` flag is set
- Collects all image paths from input directory
- Filters out already-processed images if resuming
- Processes images with progress bar (sequential or parallel via `--workers`)
- Saves checkpoint after each successful image
- Prints summary statistics

### 2. Image Processing (`process_image()` in `run.py`)
For each input image:
1. Load the scene image using `utils.load_image()`
2. Call `detect_tvs()` to find TV bounding boxes (with caching support)
3. If no TVs detected, skip the image
4. For each bounding box, call `refine_corners()` to get precise 4-corner quadrilaterals
5. Call `composite_overlay()` to warp and blend overlay onto all detected TVs
6. Save the result using `utils.save_image()`
7. Return status (success/no_tv/error) with timing metrics
8. Includes retry logic with exponential backoff for transient failures

### 3. TV Detection (`src/detect.py`)
`detect_tvs(image_path)` uses fal.ai Moondream3 vision model:
- Checks cache via `cache.get_cached_detections()` if caching enabled
- Converts image to base64 data URI via `utils.image_to_data_uri()`
- Uses a single optimized prompt: "television or monitor screen"
- Sends HTTP POST to fal.ai API with exponential backoff (max 3 retries)
- Deduplicates boxes using IoU (Intersection over Union) threshold of 0.5
- Saves results to cache via `cache.save_cached_detections()` if caching enabled
- Returns list of unique `BoundingBox` objects (normalized coordinates 0-1)

**Key functions:**
- `_get_http_client()` - Creates httpx client with FAL_KEY authorization
- `_iou()` - Calculates Intersection over Union between two bounding boxes
- `_deduplicate_boxes()` - Removes duplicate detections using IoU threshold

### 4. Corner Refinement (`src/corners.py`)
`refine_corners(image, bbox)` converts axis-aligned bounding boxes to precise 4-corner quadrilaterals:
- Converts normalized bbox to pixel coordinates
- Adds 10% padding around the bbox to capture screen edges
- Crops the padded region
- Calls `_find_screen_quad()` with three fallback strategies:
  1. **Edge detection**: Canny edge detection + contour finding
  2. **Adaptive threshold**: Adaptive binary thresholding + contours
  3. **Color segmentation**: HSV low-saturation mask + contours
- For each strategy, finds contours and selects the best 4-sided polygon
- Orders corners as [top-left, top-right, bottom-right, bottom-left]
- Validates quad is roughly rectangular (not too skewed, reasonable aspect ratio)
- If no valid quad found, falls back to bbox rectangle
- Transforms crop-relative corners back to full-image coordinates

**Key functions:**
- `_bbox_to_quad()` - Fallback: converts bbox to rectangle quad
- `_try_edge_detection()` - Canny-based contour detection
- `_try_adaptive_threshold()` - Adaptive thresholding for varying lighting
- `_try_color_segmentation()` - HSV-based dark/low-saturation region detection
- `_best_quad_from_contours()` - Selects best 4-sided contour from candidates
- `_order_corners()` - Sorts corners into consistent TL-TR-BR-BL order
- `_is_reasonable_quad()` - Validates quad geometry (aspect ratio, parallelism)

### 5. Perspective Compositing (`src/composite.py`)
`composite_overlay(scene, overlay, quads)` warps and blends overlay onto each TV:
- For each detected TV quad:
  - Shrinks quad by bezel_pct (default 5%) to simulate TV bezel
  - Creates perspective transform matrix from overlay corners to TV quad
  - Optionally matches overlay brightness to ambient surroundings
  - Warps overlay using `cv2.warpPerspective()`
  - Creates mask for the TV region
  - Applies Gaussian feathering to mask edges (default 5px)
  - Blends warped overlay with scene using alpha compositing
- Returns composited image

**Key functions:**
- `_shrink_quad()` - Inset quad toward centroid by bezel percentage
- `_composite_single()` - Warps and blends overlay for one TV
- `_match_brightness()` - Adjusts overlay brightness to match ambient light around TV
  - Samples pixels in 30px border around TV
  - Calculates ratio of ambient to overlay brightness
  - Clamps ratio to 0.4-1.8 range
  - Applies brightness adjustment using `cv2.convertScaleAbs()`

### 6. Detection Caching (`src/cache.py`)
Disk-based caching of detection results:
- `get_cached_detections()` - Returns cached bounding boxes for an image (or None)
- `save_cached_detections()` - Saves bounding boxes to disk cache
- Uses SHA-256 hash of image file as cache key
- Cache stored in `.cache/` directory (gitignored)
- Handles corrupt cache files gracefully

### 7. Utilities (`src/utils.py`)
Helper functions:
- `setup_logging()` - Configures logging (INFO or DEBUG based on verbose flag)
- `collect_image_paths()` - Scans directory for image files (jpg, jpeg, png, webp, bmp, tiff)
- `load_image()` - Loads image using OpenCV (BGR format)
- `save_image()` - Saves image using OpenCV, creates directories if needed
- `image_to_data_uri()` - Converts image file to base64 data URI for API

### 8. S3 Loader (`src/s3_loader.py`)
Optional S3 download support:
- `parse_s3_uri()` - Parses `s3://bucket/prefix` into bucket and prefix
- `download_from_s3()` - Downloads images from S3 to local directory
  - Uses boto3 S3 client with pagination
  - Uses ThreadPoolExecutor for parallel downloads (max 8 workers)
  - Uses TransferConfig with multipart downloads for large files
  - Filters by image extensions
  - Respects max_files limit
  - Skips existing files
  - Returns local directory path

## Data Flow Summary

```
Input Image
    ↓
[utils.load_image] → Load as numpy array (BGR)
    ↓
[detect.detect_tvs] → fal.ai API → Bounding boxes (normalized)
    ↓
[corners.refine_corners] → OpenCV contour analysis → 4-corner quads (pixel coords)
    ↓
[composite.composite_overlay] → Perspective warp + blend → Composited image
    ↓
[utils.save_image] → Write to output directory
```

## Key Design Decisions

1. **Hybrid AI + CV approach**: Vision LLM (Moondream3) for robust detection across diverse real estate photos, classical CV for precise geometry needed for perspective warping.

2. **Single optimized prompt**: Uses "television or monitor screen" to maximize detection recall while minimizing API calls (1 call per image instead of 3).

3. **Fallback strategy**: Corner refinement tries 3 methods (edge detection, adaptive threshold, color segmentation) before falling back to bbox rectangle.

4. **Quality-first compositing**: Edge feathering and brightness matching ensure natural-looking overlays.

5. **Modular design**: Each component (detect, corners, composite, cache) is independent and testable.

6. **Parallel processing**: `--workers N` enables parallel image processing via ProcessPoolExecutor for multi-core systems.

7. **Checkpointing & resume**: Progress saved after each image; use `--resume` to continue interrupted runs.

8. **Error recovery**: Automatic retries with exponential backoff for transient failures at both API and image-processing levels.

## Configuration

Required environment variables (`.env`):
- `FAL_KEY` - fal.ai API key (required)

Optional for S3 support:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

## CLI Options

- `--input_dir` - Local directory with input images
- `--s3_uri` - S3 URI to download images from
- `--output_dir` - Output directory (default: `./Output`)
- `--overlay` - Overlay image path (default: `./Input/input.png`)
- `--feather` - Edge feathering pixels (default: 5)
- `--no_brightness_match` - Disable brightness adjustment
- `--bezel` - Bezel inset fraction (default: 0.05 = 5%)
- `--max_s3_files` - Limit S3 downloads
- `--resume` - Skip already-processed images (checkpoint-based)
- `--workers` - Number of parallel workers (default: 1)
- `--no_cache` - Disable detection result caching
- `-v, --verbose` - Debug logging
