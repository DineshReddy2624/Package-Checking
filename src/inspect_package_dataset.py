"""
Dataset Inspection & Validation Module for Stage 1 Package Detector.
Validates package detection images and labels, audits splits, maps classes to unified 'package',
and generates structured JSON and TXT dataset reports.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from collections import Counter

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    save_json,
    save_text,
    get_logger,
)

logger = get_logger("InspectPackageDataset")
PACKAGE_EXTRACTED_DIR = DATA_DIR / "extracted_package_detector"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_yaml(search_root: Path) -> Optional[Path]:
    """Locates data.yaml or equivalent in the package detector dataset."""
    yaml_files = list(search_root.rglob("*.yaml")) + list(search_root.rglob("*.yml"))
    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and ("names" in data or "nc" in data or "train" in data):
                    return yf
        except Exception:
            continue
    return None


def inspect_and_validate_package_dataset(
    extracted_dir: Path = PACKAGE_EXTRACTED_DIR,
) -> Dict[str, Any]:
    """
    Performs full dataset inspection, validation, and report generation for Package Detector.
    """
    logger.info(f"Auditing Package Detector dataset at {extracted_dir}...")

    if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
        raise FileNotFoundError(f"Package dataset directory {extracted_dir} is empty or missing.")

    data_yaml_path = find_data_yaml(extracted_dir)
    raw_class_names = []
    num_classes = 0

    if data_yaml_path:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
            if "names" in yaml_cfg:
                if isinstance(yaml_cfg["names"], list):
                    raw_class_names = yaml_cfg["names"]
                elif isinstance(yaml_cfg["names"], dict):
                    raw_class_names = [yaml_cfg["names"][k] for k in sorted(yaml_cfg["names"].keys())]
            if "nc" in yaml_cfg:
                num_classes = int(yaml_cfg["nc"])
            else:
                num_classes = len(raw_class_names)

    # Class Mapping Choice:
    # We unify box/parcel/cardboard classes into single unified 'package' class
    # as specified in requirements: "Prefer a single unified: package class if the actual
    # project requirement is to detect any individual shipment package regardless of package subtype."
    unified_class_names = ["package"]
    class_mapping_doc = (
        f"Original raw classes: {raw_class_names}. "
        f"Mapped to unified class: {unified_class_names} for generalized individual package detection."
    )
    logger.info(class_mapping_doc)

    # Find train/valid/test splits
    train_imgs = list((extracted_dir / "train").rglob("*.jpg")) + list((extracted_dir / "train").rglob("*.png"))
    val_imgs = (
        list((extracted_dir / "valid").rglob("*.jpg"))
        + list((extracted_dir / "val").rglob("*.jpg"))
        + list((extracted_dir / "valid").rglob("*.png"))
        + list((extracted_dir / "val").rglob("*.png"))
    )
    test_imgs = list((extracted_dir / "test").rglob("*.jpg")) + list((extracted_dir / "test").rglob("*.png"))

    all_imgs = train_imgs + val_imgs + test_imgs
    if not all_imgs:
        all_imgs = [p for p in extracted_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        n_val = max(1, int(len(all_imgs) * 0.15))
        n_test = max(1, int(len(all_imgs) * 0.10))
        test_imgs = all_imgs[:n_test]
        val_imgs = all_imgs[n_test : n_test + n_val]
        train_imgs = all_imgs[n_test + n_val :]

    total_images = len(all_imgs)
    total_labels = 0
    corrupted_images = []
    missing_labels = []
    empty_labels = []
    invalid_labels = []
    class_counter = Counter()

    for img_p in all_imgs:
        # Check image integrity
        try:
            with Image.open(img_p) as im:
                im.verify()
        except Exception as e:
            corrupted_images.append({"image": str(img_p), "error": str(e)})
            continue

        # Look for corresponding label
        lbl_p = img_p.with_suffix(".txt")
        if not lbl_p.exists():
            path_str = str(img_p)
            if "images" in path_str:
                lbl_p = Path(path_str.replace("images", "labels")).with_suffix(".txt")

        if not lbl_p.exists():
            missing_labels.append(str(img_p))
            continue

        try:
            content = lbl_p.read_text(encoding="utf-8").strip()
            if not content:
                empty_labels.append(str(lbl_p))
            else:
                lines = content.splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counter[cls_id] += 1
                        total_labels += 1
                    else:
                        invalid_labels.append({"label": str(lbl_p), "line": line})
        except Exception as e:
            invalid_labels.append({"label": str(lbl_p), "error": str(e)})

    # Format class distribution
    class_distribution = {unified_class_names[0]: total_labels}

    report_data = {
        "dataset_name": "Stage 1 Individual Package Detector Dataset",
        "dataset_root": str(extracted_dir),
        "data_yaml_found": str(data_yaml_path) if data_yaml_path else "Inferred",
        "class_mapping_policy": class_mapping_doc,
        "total_images": total_images,
        "splits": {
            "train_images": len(train_imgs),
            "validation_images": len(val_imgs),
            "test_images": len(test_imgs),
        },
        "annotations": {
            "total_annotations": total_labels,
            "number_of_classes": len(unified_class_names),
            "class_names": unified_class_names,
            "raw_class_names": raw_class_names,
            "class_distribution": class_distribution,
        },
        "data_quality": {
            "corrupted_images_count": len(corrupted_images),
            "missing_labels_count": len(missing_labels),
            "empty_labels_count": len(empty_labels),
            "invalid_labels_count": len(invalid_labels),
        },
    }

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "package_detector_dataset_report.json"
    txt_path = REPORTS_DIR / "package_detector_dataset_report.txt"

    save_json(report_data, json_path)

    txt_lines = [
        "=" * 70,
        "PACKAGE DETECTOR DATASET INSPECTION & VALIDATION REPORT",
        "=" * 70,
        f"Dataset Root          : {extracted_dir}",
        f"Config File (YAML)    : {data_yaml_path if data_yaml_path else 'Inferred'}",
        f"Class Mapping Policy  : {class_mapping_doc}",
        f"Total Images          : {total_images}",
        f"  - Train Split       : {len(train_imgs)}",
        f"  - Validation Split  : {len(val_imgs)}",
        f"  - Test Split        : {len(test_imgs)}",
        "-" * 70,
        f"Number of Classes     : {len(unified_class_names)}",
        f"Class Names           : {', '.join(unified_class_names)}",
        f"Total Annotations     : {total_labels}",
        f"Class Distribution    : {class_distribution}",
        "-" * 70,
        "Data Quality Summary  :",
        f"  - Corrupted Images  : {len(corrupted_images)}",
        f"  - Missing Labels    : {len(missing_labels)}",
        f"  - Empty Labels      : {len(empty_labels)}",
        f"  - Invalid Labels    : {len(invalid_labels)}",
        "=" * 70,
    ]
    save_text("\n".join(txt_lines), txt_path)
    logger.info(f"Dataset report saved to {json_path} and {txt_path}")

    # Generate normalized package_detector_dataset.yaml
    create_package_yolo_yaml(extracted_dir, train_imgs, val_imgs, test_imgs, unified_class_names)

    return report_data


def create_package_yolo_yaml(
    extracted_dir: Path,
    train_imgs: List[Path],
    val_imgs: List[Path],
    test_imgs: List[Path],
    class_names: List[str],
) -> Path:
    """
    Creates data/processed/package_detector_dataset.yaml for YOLOv8 training.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_yaml = PROCESSED_DATA_DIR / "package_detector_dataset.yaml"

    # Write text lists for unambiguous paths
    train_txt = PROCESSED_DATA_DIR / "pkg_det_train.txt"
    val_txt = PROCESSED_DATA_DIR / "pkg_det_val.txt"
    test_txt = PROCESSED_DATA_DIR / "pkg_det_test.txt"

    train_txt.write_text("\n".join(str(p).replace("\\", "/") for p in train_imgs), encoding="utf-8")
    val_txt.write_text("\n".join(str(p).replace("\\", "/") for p in val_imgs), encoding="utf-8")
    test_txt.write_text("\n".join(str(p).replace("\\", "/") for p in test_imgs), encoding="utf-8")

    config = {
        "path": str(extracted_dir).replace("\\", "/"),
        "train": str(train_txt).replace("\\", "/"),
        "val": str(val_txt).replace("\\", "/"),
        "test": str(test_txt).replace("\\", "/"),
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, width=1000)

    logger.info(f"Generated normalized Package Detector config at {out_yaml}")
    return out_yaml


if __name__ == "__main__":
    report = inspect_and_validate_package_dataset()
    print("Package detector dataset inspection complete.")
