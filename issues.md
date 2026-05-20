## 1. ~~Duplicate API Calls~~ ✅ **RESOLVED**

**Location:** `src/detect.py`

**Status:** The system now uses a single optimized prompt "television or monitor screen" instead of 3 separate prompts, reducing API calls from 3 to 1 per image.

**Implementation:**
```python
prompt = "television or monitor screen"
```

**Impact:**
- For 100 images: 100 API calls instead of 300
- 67% reduction in API costs
- Faster processing time

## 2. ~~No Result Caching~~ ✅ **RESOLVED**

**Location:** `src/cache.py`

**Status:** Disk-based caching of detection results has been implemented using SHA-256 hash of image files as cache keys.

**Implementation:**
- Cache stored in `.cache/` directory (gitignored)
- `get_cached_detections()` returns cached results or None
- `save_cached_detections()` persists detection results
- Handles corrupt cache files gracefully
- Can be disabled with `--no_cache` flag

**Impact:**
- Re-runs skip API calls entirely for cached images
- Faster iteration during development

## 3. ~~Sequential Image Processing~~ ✅ **RESOLVED**

**Location:** `run.py`

**Status:** Parallel processing has been implemented using `ProcessPoolExecutor` with configurable worker count via `--workers` flag.

**Implementation:**
```python
with ProcessPoolExecutor(max_workers=args.workers) as executor:
    futures = [executor.submit(process_image, ...) for img_path in to_process]
```

**Impact:**
- CPU-bound operations now utilize multi-core systems
- Faster overall processing for large batches
- Configurable parallelism (default: 1, user-specified via `--workers`)

## 4. ~~Redundant Data Transmission~~ ✅ **RESOLVED**

**Location:** `src/detect.py`

**Status:** With the single optimized prompt implementation, the data URI is now sent only once per image instead of 3 times.

**Impact:**
- 67% reduction in bandwidth usage
- Faster upload times for large images
- Eliminated redundant network overhead

## 5. No Batch API Support

**Problem:** The fal.ai API is called per-image rather than supporting batch detection of multiple images in a single request.

**Impact:**
- Increased HTTP overhead
- Higher latency
- More API calls than necessary

**Potential Solutions:**
- Check if fal.ai supports batch inference
- Implement local batching with retry logic
- Use a different model that supports batch processing

## 6. Corner Refinement Strategy Overhead

**Location:** `src/corners.py`

**Problem:** For each bounding box, corner refinement tries 3 different strategies sequentially (edge detection, adaptive threshold, color segmentation).

```python
quad = _try_edge_detection(gray, min_area)
if quad is not None:
    return quad

quad = _try_adaptive_threshold(gray, min_area)
if quad is not None:
    return quad

quad = _try_color_segmentation(crop, min_area)
```

**Impact:**
- Additional computation time for each TV
- Multiple OpenCV operations per detection
- Potential for redundant processing

**Note:** This is actually a reasonable fallback strategy, but could be optimized.

**Potential Solutions:**
- Use machine learning to predict which strategy will work best
- Implement early termination with confidence scoring
- Cache successful strategy per image type/lighting condition

## 7. ~~S3 Download Inefficiency~~ ✅ **RESOLVED**

**Location:** `src/s3_loader.py`

**Status:** S3 downloads now use `ThreadPoolExecutor` with parallel transfers and `TransferConfig` with multipart downloads for large files.

**Implementation:**
```python
transfer_config = boto3.s3.transfer.TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    max_concurrency=10,
    multipart_chunksize=8 * 1024 * 1024,
)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_download_one, item) for item in to_download]
```

**Impact:**
- Faster download times for many files
- Utilizes S3's parallel transfer capabilities
- Optimized for large image files (>8MB multipart threshold)

## 8. ~~Lack of Progress Persistence~~ ✅ **RESOLVED**

**Location:** `run.py`

**Status:** Checkpointing has been implemented with progress saved after each successfully processed image. Resume capability via `--resume` flag.

**Implementation:**
- Checkpoint file: `.pipeline_checkpoint.json` in output directory
- `_load_checkpoint()` loads set of completed filenames
- `_save_checkpoint()` persists completed filenames
- `--resume` flag skips already-processed images

**Impact:**
- Long-running jobs can be resumed after interruption
- No need to re-process all images from scratch
- Better user experience for large batches

## 9. No Telemetry or Monitoring

**Problem:** No metrics collection for performance monitoring, API usage tracking, or error rate analysis.

**Impact:**
- Difficult to identify bottlenecks
- No visibility into API costs
- Hard to optimize based on real usage data

**Potential Solutions:**
- Add timing metrics for each stage
- Track API call counts and costs
- Log success/error rates
- Implement structured logging for analysis

## 10. ~~Limited Error Recovery~~ ✅ **RESOLVED**

**Location:** `run.py`, `src/detect.py`

**Status:** Comprehensive retry logic with exponential backoff has been implemented at both API and image-processing levels.

**Implementation:**
- `process_image()` has retry loop with exponential backoff (max 2 retries)
- `detect_tvs()` has retry loop with exponential backoff (max 3 retries)
- Errors are logged with retry attempt information

**Impact:**
- Automatic recovery from transient failures
- Higher overall success rate
- No manual re-run required for temporary issues

## Summary of Optimizations

| Operation | Before | After | Status |
|-----------|--------|-------|--------|
| API calls per image | 3 (3 prompts) | 1 (combined prompt) | ✅ Optimized |
| Result caching | None | Disk-based SHA-256 cache | ✅ Implemented |
| Image processing | Sequential | Parallel (configurable workers) | ✅ Implemented |
| S3 downloads | Sequential | Parallel + multipart | ✅ Optimized |
| Progress persistence | None | Checkpointing + resume | ✅ Implemented |
| Error recovery | Basic | Retry with exponential backoff | ✅ Enhanced |
| Corner refinement strategies | Up to 3 per TV | Up to 3 per TV | ⚠️ Could optimize |
| Telemetry/monitoring | None | None | ⚠️ Not implemented |

## Remaining Optimization Opportunities

**Medium Priority:**
1. **Corner refinement strategy selection** - Currently tries 3 strategies sequentially. Could add heuristics to predict which strategy will work best for given lighting conditions.
2. **Telemetry and monitoring** - Add structured metrics collection for performance monitoring, API usage tracking, and error rate analysis to identify bottlenecks.

**Low Priority:**
3. **Batch API support** - Check if fal.ai supports batch inference for multiple images in a single request to further reduce HTTP overhead.
4. **Brightness matching caching** - Cache ambient lighting calculations per image to avoid recomputation when multiple TVs are detected in the same scene.

## Completed Optimizations

✅ **Result caching** - Disk-based SHA-256 cache eliminates redundant API calls on re-runs
✅ **Single prompt detection** - Reduced API calls from 3 to 1 per image via optimized prompting
✅ **Checkpointing/resume** - Progress persistence enables resuming interrupted long-running jobs
✅ **Parallel processing** - Configurable worker count via `--workers` flag for multi-core utilization
✅ **Optimized S3 downloads** - ThreadPoolExecutor with TransferConfig for parallel multipart transfers
✅ **Enhanced error recovery** - Retry logic with exponential backoff at API and processing levels
