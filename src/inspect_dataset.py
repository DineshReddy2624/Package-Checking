"""
Dataset Inspection and Validation Module.
Recursively inspects the extracted dataset, checks images, labels, bounding boxes,
and generates structured JSON/TXT validation reports.
"""

import os
import sys
import yaml
import json
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
    EXTRACTED_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    save_json,
    save_text,
    get_logger,
)

logger = get_logger("InspectDataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_yaml(search_root: Path) -> Optional[Path]:
    """Finds data.yaml or equivalent YAML configuration file in the extracted dataset."""
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


def locate_dataset_splits(extracted_dir: Path) -> Dict[str, List[Path]]:
    """
    Locates image files and groups them into train, val, test splits dynamically.
    """
    splits = {"train": [], "val": [], "test": [], "unsplit": []}
    all_images = [
        p for p in extracted_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for img_path in all_images:
        path_str = str(img_path).lower().replace("\\", "/")
        if "/train/" in path_str or "/train" in path_str:
            splits["train"].append(img_path)
        elif "/val/" in path_str or "/valid/" in path_str or "/validation/" in path_str:
            splits["val"].append(img_path)
        elif "/test/" in path_str:
            splits["test"].append(img_path)
        else:
            splits["unsplit"].append(img_path)

    # If all images are unsplit or val is empty, split them systematically
    if len(splits["val"]) == 0 and len(splits["train"]) == 0 and len(splits["unsplit"]) > 0:
        logger.info("Dataset has no explicit train/val folders. Creating systematic 80/20 train/val split...")
        unsplit = sorted(splits["unsplit"])
        n_val = max(1, int(len(unsplit) * 0.2))
        splits["val"] = unsplit[:n_val]
        splits["train"] = unsplit[n_val:]
        splits["unsplit"] = []
    elif len(splits["val"]) == 0 and len(splits["train"]) > 0:
        # Split some train into val
        logger.info("Validation split was empty. Allocating 20% of train to validation split...")
        train_imgs = sorted(splits["train"])
        n_val = max(1, int(len(train_imgs) * 0.2))
        splits["val"] = train_imgs[:n_val]
        splits["train"] = train_imgs[n_val:]

    return splits


def validate_single_annotation(
    label_path: Path, num_classes: int
) -> Tuple[bool, List[int], List[str]]:
    """
    Validates a single YOLO label file.
    Returns (is_valid, class_ids, error_messages).
    """
    if not label_path.exists():
        return False, [], ["Label file does not exist"]

    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, [], [f"Cannot read label file: {e}"]

    if not content:
        return True, [], []  # Empty label is valid (background / negative image)

    lines = content.splitlines()
    class_ids = []
    errors = []

    for line_idx, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) < 5:
            errors.append(f"Line {line_idx}: Expected at least 5 values, got {len(parts)}")
            continue

        try:
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            if num_classes > 0 and (cls_id < 0 or cls_id >= num_classes):
                errors.append(f"Line {line_idx}: Class ID {cls_id} out of range [0, {num_classes-1}]")

            # Check bbox bounding ranges (with minor float tolerance)
            if not (-0.05 <= x_center <= 1.05 and -0.05 <= y_center <= 1.05):
                errors.append(f"Line {line_idx}: Center coords ({x_center}, {y_center}) outside valid [0, 1] range")
            if width <= 0 or height <= 0 or width > 1.2 or height > 1.2:
                errors.append(f"Line {line_idx}: Dimensions ({width}, {height}) invalid")

            class_ids.append(cls_id)
        except ValueError as e:
            errors.append(f"Line {line_idx}: Non-numeric value in annotation ({e})")

    is_valid = len(errors) == 0
    return is_valid, class_ids, errors


def find_corresponding_label(img_path: Path) -> Optional[Path]:
    """Finds the corresponding .txt label for an image path."""
    txt_name = img_path.stem + ".txt"

    # 1. Look in adjacent /labels/ directory if in /images/
    path_str = str(img_path)
    if "images" in path_str:
        label_str = path_str.replace("images", "labels")
        candidate = Path(label_str).with_suffix(".txt")
        if candidate.exists():
            return candidate

    # 2. Look in the same directory as image
    candidate = img_path.with_suffix(".txt")
    if candidate.exists():
        return candidate

    # 3. Look anywhere in parent hierarchy under labels/
    parent_dir = img_path.parent
    for p in [parent_dir, parent_dir.parent]:
        lbl_dir = p / "labels"
        if lbl_dir.exists():
            candidate = lbl_dir / txt_name
            if candidate.exists():
                return candidate

    return None


def inspect_and_validate_dataset(extracted_dir: Path = EXTRACTED_DATA_DIR) -> Dict[str, Any]:
    """
    Performs full dataset inspection, validation, and report generation.
    """
    logger.info(f"Starting dataset inspection in {extracted_dir}...")

    if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
        raise FileNotFoundError(f"Extracted dataset directory {extracted_dir} is empty or missing.")

    # 1. Locate data.yaml or configuration
    data_yaml_path = find_data_yaml(extracted_dir)
    class_names = []
    num_classes = 0

    if data_yaml_path:
        logger.info(f"Found dataset configuration: {data_yaml_path}")
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
            if "names" in yaml_cfg:
                if isinstance(yaml_cfg["names"], list):
                    class_names = yaml_cfg["names"]
                elif isinstance(yaml_cfg["names"], dict):
                    class_names = [yaml_cfg["names"][k] for k in sorted(yaml_cfg["names"].keys())]
            if "nc" in yaml_cfg:
                num_classes = int(yaml_cfg["nc"])
            else:
                num_classes = len(class_names)
    else:
        logger.warning("No data.yaml found in extracted files. Will infer classes from annotations.")

    # 2. Group images by split
    splits = locate_dataset_splits(extracted_dir)
    total_images = sum(len(imgs) for imgs in splits.values())
    logger.info(f"Found {total_images} total images (Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})")

    # 3. Validate images, labels, and bounding boxes
    corrupted_images = []
    missing_labels = []
    empty_labels = []
    invalid_labels = []
    class_counter = Counter()
    total_annotations = 0

    all_images = splits["train"] + splits["val"] + splits["test"] + splits["unsplit"]

    for img_path in all_images:
        # Check image integrity
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            corrupted_images.append({"image": str(img_path), "error": str(e)})
            continue

        # Check label
        lbl_path = find_corresponding_label(img_path)
        if lbl_path is None:
            missing_labels.append(str(img_path))
            continue

        is_valid, cls_ids, errors = validate_single_annotation(lbl_path, num_classes)
        if not cls_ids and is_valid:
            empty_labels.append(str(lbl_path))
        elif not is_valid:
            invalid_labels.append({"label": str(lbl_path), "errors": errors})

        for cid in cls_ids:
            class_counter[cid] += 1
            total_annotations += 1

    # Inferred class names if missing from data.yaml
    if not class_names and class_counter:
        max_cid = max(class_counter.keys())
        num_classes = max_cid + 1
        class_names = [f"damage_class_{i}" for i in range(num_classes)]

    # Format class distribution with names
    class_distribution = {}
    for cid, count in sorted(class_counter.items()):
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        class_distribution[name] = count

    # 4. Generate structured dataset report
    report_data = {
        "dataset_root": str(extracted_dir),
        "data_yaml_found": str(data_yaml_path) if data_yaml_path else "None",
        "total_images": total_images,
        "splits": {
            "train_images": len(splits["train"]),
            "validation_images": len(splits["val"]),
            "test_images": len(splits["test"]),
            "unsplit_images": len(splits["unsplit"]),
        },
        "annotations": {
            "total_annotations": total_annotations,
            "number_of_classes": num_classes,
            "class_names": class_names,
            "class_distribution": class_distribution,
        },
        "data_quality": {
            "corrupted_images_count": len(corrupted_images),
            "missing_labels_count": len(missing_labels),
            "empty_labels_count": len(empty_labels),
            "invalid_labels_count": len(invalid_labels),
            "corrupted_images_samples": corrupted_images[:5],
            "invalid_labels_samples": invalid_labels[:5],
        },
    }

    # Save JSON report
    json_path = REPORTS_DIR / "dataset_report.json"
    save_json(report_data, json_path)

    # Save human-readable TXT report
    txt_lines = [
        "=" * 65,
        "DATASET INSPECTION & VALIDATION REPORT",
        "=" * 65,
        f"Dataset Root          : {extracted_dir}",
        f"Config File (YAML)    : {data_yaml_path if data_yaml_path else 'Inferred'}",
        f"Total Images          : {total_images}",
        f"  - Train Split       : {len(splits['train'])}",
        f"  - Validation Split  : {len(splits['val'])}",
        f"  - Test Split        : {len(splits['test'])}",
        "-" * 65,
        f"Number of Classes     : {num_classes}",
        f"Class Names           : {', '.join(class_names) if class_names else 'None'}",
        f"Total Annotations     : {total_annotations}",
        "Class Distribution    :",
    ]
    for cname, count in class_distribution.items():
        txt_lines.append(f"  - {cname:20s}: {count}")

    txt_lines.extend([
        "-" * 65,
        "Data Quality Summary  :",
        f"  - Corrupted Images  : {len(corrupted_images)}",
        f"  - Missing Labels    : {len(missing_labels)}",
        f"  - Empty Labels      : {len(empty_labels)}",
        f"  - Invalid Labels    : {len(invalid_labels)}",
        "=" * 65,
    ])
    txt_path = REPORTS_DIR / "dataset_report.txt"
    save_text("\n".join(txt_lines), txt_path)

    logger.info(f"Dataset report saved to {json_path} and {txt_path}")

    # 5. Create normalized dataset.yaml in PROCESSED_DATA_DIR for YOLO training
    create_yolo_dataset_yaml(extracted_dir, data_yaml_path, splits, class_names)

    return report_data


def create_yolo_dataset_yaml(
    extracted_dir: Path,
    data_yaml_path: Optional[Path],
    splits: Dict[str, List[Path]],
    class_names: List[str],
) -> Path:
    """
    Creates a verified, clean dataset.yaml for YOLOv8 in data/processed/dataset.yaml.
    Uses proper paths discovered during inspection.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_yaml = PROCESSED_DATA_DIR / "dataset.yaml"

    # Determine train/val directory roots
    if data_yaml_path and data_yaml_path.parent:
        base_dir = data_yaml_path.parent
    else:
        base_dir = extracted_dir

    # Detect if standard images/train or train/images exist
    train_path = base_dir / "images" / "train"
    val_path = base_dir / "images" / "val"
    test_path = base_dir / "images" / "test"

    if not train_path.exists():
        train_path = base_dir / "train" / "images"
    if not val_path.exists():
        val_path = base_dir / "val" / "images"
        if not val_path.exists():
            val_path = base_dir / "valid" / "images"
    if not test_path.exists():
        test_path = base_dir / "test" / "images"

    # Fallback if standard paths don't exist: write image file list or parent
    if not train_path.exists() and splits["train"]:
        train_path = splits["train"][0].parent
    if not val_path.exists() and splits["val"]:
        val_path = splits["val"][0].parent

    config = {
        "path": str(base_dir).replace("\\", "/"),
        "train": str(train_path.relative_to(base_dir) if train_path.is_relative_to(base_dir) else train_path).replace("\\", "/"),
        "val": str(val_path.relative_to(base_dir) if val_path.is_relative_to(base_dir) else val_path).replace("\\", "/"),
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    if test_path.exists():
        config["test"] = str(test_path.relative_to(base_dir) if test_path.is_relative_to(base_dir) else test_path).replace("\\", "/")

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, width=1000)

    logger.info(f"Generated normalized YOLO dataset config at {out_yaml}")

    # Generate specialized single package dataset configuration
    create_single_package_dataset(base_dir, class_names)
    return out_yaml


def create_single_package_dataset(base_dir: Path, class_names: List[str]) -> Optional[Path]:
    """
    Creates a dedicated dataset configuration for Single Package Damage Detection
    by isolating images containing 1 or 0 boxes from manifest.csv.
    """
    manifest_path = base_dir / "manifest.csv"
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}. Single package split skipped.")
        return None

    import pandas as pd
    try:
        df = pd.read_csv(manifest_path)
        single_df = df[df["box_count"] <= 1]
        logger.info(f"Filtered {len(single_df)} single-package images from manifest.csv.")

        train_txt = PROCESSED_DATA_DIR / "single_train.txt"
        val_txt = PROCESSED_DATA_DIR / "single_val.txt"
        test_txt = PROCESSED_DATA_DIR / "single_test.txt"

        splits_map = {
            "train": train_txt,
            "valid": val_txt,
            "val": val_txt,
            "test": test_txt,
        }

        train_paths, val_paths, test_paths = [], [], []

        for _, row in single_df.iterrows():
            img_rel = row["image"]
            split_name = str(row["split"]).lower()
            abs_img_path = (base_dir / img_rel).resolve()
            if abs_img_path.exists():
                posix_path = str(abs_img_path).replace("\\", "/")
                if split_name in ("train",):
                    train_paths.append(posix_path)
                elif split_name in ("valid", "val"):
                    val_paths.append(posix_path)
                elif split_name in ("test",):
                    test_paths.append(posix_path)

        train_txt.write_text("\n".join(train_paths), encoding="utf-8")
        val_txt.write_text("\n".join(val_paths), encoding="utf-8")
        test_txt.write_text("\n".join(test_paths), encoding="utf-8")

        single_yaml = PROCESSED_DATA_DIR / "single_package_dataset.yaml"
        single_cfg = {
            "path": str(base_dir).replace("\\", "/"),
            "train": str(train_txt).replace("\\", "/"),
            "val": str(val_txt).replace("\\", "/"),
            "test": str(test_txt).replace("\\", "/"),
            "names": {i: name for i, name in enumerate(class_names)},
            "nc": len(class_names),
        }

        with open(single_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(single_cfg, f, sort_keys=False, width=1000)

        logger.info(
            f"Generated Single Package Dataset at {single_yaml} "
            f"(Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)})"
        )
        return single_yaml
    except Exception as e:
        logger.error(f"Failed to create single package dataset: {e}")
        return None


if __name__ == "__main__":
    report = inspect_and_validate_dataset()
    print("Inspection complete.")
