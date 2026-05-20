#!/usr/bin/env python3
"""
TV Screen Replacement Pipeline
-------------------------------
Detects TV screens in real estate photos using fal.ai Moondream3 and
composites a provided overlay image onto each detected screen.

Usage:
    python run.py --input_dir ./TVPhotos --output_dir ./Output --overlay ./Input/input.png
    python run.py --s3_uri s3://bucket/prefix --output_dir ./Output --overlay ./Input/input.png
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import setup_logging, collect_image_paths, load_image, save_image
from src.detect import detect_tvs
from src.corners import refine_corners
from src.composite import composite_overlay

load_dotenv()
logger = logging.getLogger(__name__)


CHECKPOINT_FILE = ".pipeline_checkpoint.json"


def _load_checkpoint(output_dir: str) -> set:
    """Load set of already-processed filenames from checkpoint."""
    cp_path = os.path.join(output_dir, CHECKPOINT_FILE)
    if not os.path.exists(cp_path):
        return set()
    try:
        with open(cp_path, "r") as f:
            return set(json.load(f).get("completed", []))
    except (json.JSONDecodeError, IOError):
        return set()


def _save_checkpoint(output_dir: str, completed: set) -> None:
    """Persist set of completed filenames."""
    cp_path = os.path.join(output_dir, CHECKPOINT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    with open(cp_path, "w") as f:
        json.dump({"completed": sorted(completed)}, f)


def process_image(
    image_path: str,
    overlay: "np.ndarray",
    output_path: str,
    feather_px: int = 5,
    brightness_match: bool = True,
    bezel_pct: float = 0.05,
    max_retries: int = 2,
) -> dict:
    """Process a single image: detect TVs, refine corners, composite overlay."""
    import numpy as np

    result = {
        "file": os.path.basename(image_path),
        "tvs_found": 0,
        "status": "skipped",
        "error": None,
        "timings": {},
    }

    for retry in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            scene = load_image(image_path)
            result["timings"]["load"] = round(time.perf_counter() - t0, 3)
            logger.info(f"Processing: {result['file']}")

            t0 = time.perf_counter()
            bboxes = detect_tvs(image_path)
            result["timings"]["detect"] = round(time.perf_counter() - t0, 3)

            if not bboxes:
                logger.info(f"  No TVs detected, skipping.")
                result["status"] = "no_tv"
                return result

            t0 = time.perf_counter()
            quads = []
            for bbox in bboxes:
                quad = refine_corners(scene, bbox)
                quads.append(quad)
            result["timings"]["corners"] = round(time.perf_counter() - t0, 3)

            t0 = time.perf_counter()
            composited = composite_overlay(
                scene, overlay, quads,
                feather_px=feather_px,
                brightness_match=brightness_match,
                bezel_pct=bezel_pct,
            )
            result["timings"]["composite"] = round(time.perf_counter() - t0, 3)

            save_image(composited, output_path)
            result["tvs_found"] = len(quads)
            result["status"] = "success"
            logger.info(f"  Done — {len(quads)} TV(s) composited → {output_path}")
            logger.debug(f"  Timings: {result['timings']}")
            return result

        except Exception as e:
            if retry < max_retries:
                logger.warning(f"  Retry {retry + 1}/{max_retries} for {result['file']}: {e}")
                time.sleep(2 ** retry)
            else:
                result["status"] = "error"
                result["error"] = str(e)
                logger.error(f"  Error processing {result['file']}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect TV screens in photos and overlay an image onto them."
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing input images (local path)"
    )
    parser.add_argument(
        "--s3_uri", type=str, default=None,
        help="S3 URI to download images from (e.g., s3://bucket/prefix)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./Output",
        help="Directory to save processed images"
    )
    parser.add_argument(
        "--overlay", type=str, default="./Input/input.png",
        help="Path to the overlay image to place on TV screens"
    )
    parser.add_argument(
        "--feather", type=int, default=5,
        help="Edge feathering size in pixels (0 to disable)"
    )
    parser.add_argument(
        "--no_brightness_match", action="store_true",
        help="Disable brightness matching"
    )
    parser.add_argument(
        "--bezel", type=float, default=0.05,
        help="Bezel inset as fraction of screen size (0.05 = 5%% inset, 0 = no bezel)"
    )
    parser.add_argument(
        "--max_s3_files", type=int, default=None,
        help="Max number of files to download from S3"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip images already processed (checkpointed) in the output directory"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1 = sequential)"
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Disable detection result caching"
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if not args.input_dir and not args.s3_uri:
        parser.error("Either --input_dir or --s3_uri must be specified.")

    if not os.path.isfile(args.overlay):
        parser.error(f"Overlay image not found: {args.overlay}")

    overlay = load_image(args.overlay)
    logger.info(f"Overlay image loaded: {args.overlay} ({overlay.shape[1]}x{overlay.shape[0]})")

    input_dir = args.input_dir
    if args.s3_uri:
        from src.s3_loader import download_from_s3
        input_dir = download_from_s3(args.s3_uri, local_dir="./TVPhotos", max_files=args.max_s3_files)

    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(image_paths)} image(s) in {input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    completed = _load_checkpoint(args.output_dir) if args.resume else set()
    if args.resume and completed:
        logger.info(f"Resuming: {len(completed)} image(s) already processed, skipping them.")

    to_process = [p for p in image_paths if p.name not in completed]
    if not to_process:
        logger.info("All images already processed. Nothing to do.")
        sys.exit(0)

    pipeline_start = time.perf_counter()
    results = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for img_path in to_process:
                output_path = os.path.join(args.output_dir, img_path.name)
                fut = executor.submit(
                    process_image,
                    str(img_path),
                    overlay,
                    output_path,
                    feather_px=args.feather,
                    brightness_match=not args.no_brightness_match,
                    bezel_pct=args.bezel,
                )
                futures[fut] = img_path

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                res = fut.result()
                results.append(res)
                if res["status"] in ("success", "no_tv"):
                    completed.add(res["file"])
                    _save_checkpoint(args.output_dir, completed)
    else:
        for img_path in tqdm(to_process, desc="Processing images"):
            output_path = os.path.join(args.output_dir, img_path.name)
            res = process_image(
                str(img_path),
                overlay,
                output_path,
                feather_px=args.feather,
                brightness_match=not args.no_brightness_match,
                bezel_pct=args.bezel,
            )
            results.append(res)
            if res["status"] in ("success", "no_tv"):
                completed.add(res["file"])
                _save_checkpoint(args.output_dir, completed)

    pipeline_elapsed = round(time.perf_counter() - pipeline_start, 2)

    success = sum(1 for r in results if r["status"] == "success")
    no_tv = sum(1 for r in results if r["status"] == "no_tv")
    errors = sum(1 for r in results if r["status"] == "error")
    total_tvs = sum(r.get("tvs_found", 0) for r in results)

    print("\n" + "=" * 50)
    print(f"RESULTS SUMMARY")
    print(f"  Total images:  {len(results)}")
    print(f"  Success:       {success}")
    print(f"  No TV found:   {no_tv}")
    print(f"  Errors:        {errors}")
    print(f"  TVs composited:{total_tvs}")
    print(f"  Elapsed:       {pipeline_elapsed}s")
    print(f"  Output dir:    {os.path.abspath(args.output_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
