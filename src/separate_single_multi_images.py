"""
Single vs Multiple Parcel Image Separation Module.
Categorizes images and YOLO annotations based on parcel count:
- Single Parcel: Exactly 1 parcel/package in the frame.
- Multiple Parcels: 2 or more parcels/packages in the frame.
Populates sample_images/single_parcel/ and sample_images/multiple_parcels/
and structured processed datasets for targeted training & validation.
"""

import os
import sys
import shutil
import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    PROCESSED_DATA_DIR,
    SAMPLE_IMAGES_DIR,
    REPORTS_DIR,
    save_json,
    save_text,
    get_logger,
)

logger = get_logger("ParcelSeparator")

SINGLE_SAMPLE_DIR = SAMPLE_IMAGES_DIR / "single_parcel"
MULTI_SAMPLE_DIR = SAMPLE_IMAGES_DIR / "multiple_parcels"


def separate_dataset_by_parcel_count(
    dataset_root: Path,
    output_base_dir: Path,
) -> Dict[str, Any]:
    """
    Separates YOLO formatted images into Single Parcel and Multiple Parcels splits.
    """
    single_dir = output_base_dir / "single_parcel"
    multi_dir = output_base_dir / "multiple_parcels"

    for split in ["train", "val", "test"]:
        (single_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (single_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (multi_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (multi_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    SINGLE_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "single_parcel": {"train": 0, "val": 0, "test": 0, "total": 0},
        "multiple_parcels": {"train": 0, "val": 0, "test": 0, "total": 0},
        "empty_or_unlabeled": 0,
        "sample_images_saved": {
            "single_parcel": [],
            "multiple_parcels": [],
        },
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Process all image splits in dataset_root
    for split in ["train", "val", "test", "valid"]:
        norm_split = "val" if split == "valid" else split
        split_img_dir = dataset_root / split / "images" if (dataset_root / split / "images").exists() else dataset_root / "images" / split
        split_lbl_dir = dataset_root / split / "labels" if (dataset_root / split / "labels").exists() else dataset_root / "labels" / split

        if not split_img_dir.exists():
            continue

        for img_path in split_img_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue

            lbl_path = split_lbl_dir / f"{img_path.stem}.txt" if split_lbl_dir.exists() else None
            box_count = 0

            if lbl_path and lbl_path.exists():
                lines = [line.strip() for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
                box_count = len(lines)

            if box_count == 1:
                category = "single_parcel"
                dest_img = single_dir / "images" / norm_split / img_path.name
                dest_lbl = single_dir / "labels" / norm_split / f"{img_path.stem}.txt"
                stats["single_parcel"][norm_split] += 1
                stats["single_parcel"]["total"] += 1
            elif box_count > 1:
                category = "multiple_parcels"
                dest_img = multi_dir / "images" / norm_split / img_path.name
                dest_lbl = multi_dir / "labels" / norm_split / f"{img_path.stem}.txt"
                stats["multiple_parcels"][norm_split] += 1
                stats["multiple_parcels"]["total"] += 1
            else:
                stats["empty_or_unlabeled"] += 1
                continue

            # Copy image and label
            shutil.copy2(img_path, dest_img)
            if lbl_path and lbl_path.exists():
                shutil.copy2(lbl_path, dest_lbl)

            # Copy representative samples to sample_images/
            if category == "single_parcel" and len(stats["sample_images_saved"]["single_parcel"]) < 10:
                sample_dest = SINGLE_SAMPLE_DIR / f"single_parcel_{len(stats['sample_images_saved']['single_parcel']) + 1:02d}.jpg"
                shutil.copy2(img_path, sample_dest)
                stats["sample_images_saved"]["single_parcel"].append(str(sample_dest.name))
            elif category == "multiple_parcels" and len(stats["sample_images_saved"]["multiple_parcels"]) < 10:
                sample_dest = MULTI_SAMPLE_DIR / f"multi_parcel_{len(stats['sample_images_saved']['multiple_parcels']) + 1:02d}.jpg"
                shutil.copy2(img_path, sample_dest)
                stats["sample_images_saved"]["multiple_parcels"].append(str(sample_dest.name))

    # Also make sure existing sample_images are properly categorized
    for sf in SAMPLE_IMAGES_DIR.glob("*.jpg"):
        if "single" in sf.name.lower() or sf.name == "package.jpg":
            shutil.copy2(sf, SINGLE_SAMPLE_DIR / sf.name)
        elif "multi" in sf.name.lower():
            shutil.copy2(sf, MULTI_SAMPLE_DIR / sf.name)

    logger.info(f"Dataset Separation Complete:")
    logger.info(f"  - Single Parcel Images   : {stats['single_parcel']['total']}")
    logger.info(f"  - Multiple Parcels Images: {stats['multiple_parcels']['total']}")

    return stats


def generate_separation_reports(stats: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Saves structured JSON and TXT reports for Single vs Multiple Parcel separation.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "single_multi_parcel_separation_report.json"
    txt_path = REPORTS_DIR / "single_multi_parcel_separation_report.txt"

    save_json(stats, json_path)

    lines = [
        "=" * 75,
        "     SINGLE PARCEL VS MULTIPLE PARCELS DATASET SEPARATION REPORT",
        "=" * 75,
        "1. SINGLE PARCEL CATEGORY (Exactly 1 Package/Parcel per scene):",
        f"   - Total Images   : {stats['single_parcel']['total']}",
        f"   - Train Split    : {stats['single_parcel']['train']}",
        f"   - Val Split      : {stats['single_parcel']['val']}",
        f"   - Test Split     : {stats['single_parcel']['test']}",
        f"   - Sample Folder  : sample_images/single_parcel/",
        "",
        "2. MULTIPLE PARCELS CATEGORY (2 or more Packages/Parcels per scene):",
        f"   - Total Images   : {stats['multiple_parcels']['total']}",
        f"   - Train Split    : {stats['multiple_parcels']['train']}",
        f"   - Val Split      : {stats['multiple_parcels']['val']}",
        f"   - Test Split     : {stats['multiple_parcels']['test']}",
        f"   - Sample Folder  : sample_images/multiple_parcels/",
        "",
        "3. PIPELINE USAGE:",
        "   - Single Parcel Mode : Direct Damage Detection + Severity + Internal Risk",
        "   - Multi-Parcel Mode  : Stage 1 Package Detector -> Crop -> Stage 2 Damage Detector",
        "=" * 75,
    ]
    save_text("\n".join(lines), txt_path)
    logger.info(f"Saved separation reports to {json_path} and {txt_path}")
    return json_path, txt_path


if __name__ == "__main__":
    pkg_dataset_root = PROJECT_ROOT / "data" / "extracted_package_detector"
    output_dir = PROJECT_ROOT / "data" / "processed"

    if pkg_dataset_root.exists():
        logger.info(f"Separating dataset at {pkg_dataset_root}...")
        report_stats = separate_dataset_by_parcel_count(pkg_dataset_root, output_dir)
        generate_separation_reports(report_stats)
    else:
        logger.error(f"Dataset root not found: {pkg_dataset_root}")
