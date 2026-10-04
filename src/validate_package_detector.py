"""
Validation & Evaluation Module for Stage 1 Package Detector.
Evaluates package_detector_best.pt on validation split and saves verified metrics.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    WEIGHTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    OUTPUTS_DIR,
    save_json,
    save_text,
    get_logger,
)

logger = get_logger("ValidatePackageDetector")
PACKAGE_OUTPUTS_DIR = OUTPUTS_DIR / "package_detector"


def validate_package_detector(
    model_path: Optional[Path] = None,
    data_yaml_path: Optional[Path] = None,
    imgsz: int = 640,
    split: str = "val",
) -> Dict[str, Any]:
    """
    Evaluates Stage 1 Package Detector and compiles metric reports.
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = WEIGHTS_DIR / "package_detector_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Package detector model weights not found at {model_path}.")

    if data_yaml_path is None:
        data_yaml_path = PROCESSED_DATA_DIR / "package_detector_dataset.yaml"

    logger.info(f"Validating package detector '{model_path}' on split='{split}'...")
    model = YOLO(str(model_path))

    metrics_obj = model.val(
        data=str(data_yaml_path),
        imgsz=imgsz,
        split=split,
        verbose=True,
        save_json=True,
    )

    results_dict = metrics_obj.results_dict if hasattr(metrics_obj, "results_dict") else {}
    box_metrics = metrics_obj.box if hasattr(metrics_obj, "box") else None

    # Precision
    if box_metrics and hasattr(box_metrics, "mp") and box_metrics.mp is not None:
        precision_val = float(box_metrics.mp)
    elif "metrics/precision(B)" in results_dict:
        precision_val = float(results_dict["metrics/precision(B)"])
    else:
        precision_val = "Unavailable (Metric not returned by validator)"

    # Recall
    if box_metrics and hasattr(box_metrics, "mr") and box_metrics.mr is not None:
        recall_val = float(box_metrics.mr)
    elif "metrics/recall(B)" in results_dict:
        recall_val = float(results_dict["metrics/recall(B)"])
    else:
        recall_val = "Unavailable (Metric not returned by validator)"

    # mAP@0.5
    if box_metrics and hasattr(box_metrics, "map50") and box_metrics.map50 is not None:
        map50_val = float(box_metrics.map50)
    elif "metrics/mAP50(B)" in results_dict:
        map50_val = float(results_dict["metrics/mAP50(B)"])
    else:
        map50_val = "Unavailable (Metric not returned by validator)"

    # mAP@0.5:0.95
    if box_metrics and hasattr(box_metrics, "map") and box_metrics.map is not None:
        map50_95_val = float(box_metrics.map)
    elif "metrics/mAP50-95(B)" in results_dict:
        map50_95_val = float(results_dict["metrics/mAP50-95(B)"])
    else:
        map50_95_val = "Unavailable (Metric not returned by validator)"

    speed_info = metrics_obj.speed if hasattr(metrics_obj, "speed") else {}

    report_payload = {
        "model_name": "Stage 1 Individual Package Detector",
        "model_path": str(model_path),
        "dataset_config": str(data_yaml_path),
        "evaluation_split": split,
        "metrics": {
            "precision": round(precision_val, 4) if isinstance(precision_val, float) else precision_val,
            "recall": round(recall_val, 4) if isinstance(recall_val, float) else recall_val,
            "mAP_50": round(map50_val, 4) if isinstance(map50_val, float) else map50_val,
            "mAP_50_95": round(map50_95_val, 4) if isinstance(map50_95_val, float) else map50_95_val,
        },
        "inference_speed_ms": speed_info,
    }

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "package_detector_metrics.json"
    txt_path = REPORTS_DIR / "package_detector_metrics.txt"

    save_json(report_payload, json_path)

    txt_lines = [
        "=" * 70,
        "STAGE 1 INDIVIDUAL PACKAGE DETECTOR VALIDATION REPORT",
        "=" * 70,
        f"Model Weight Path    : {model_path}",
        f"Evaluation Split     : {split}",
        "-" * 70,
        "ACTUAL PERFORMANCE METRICS:",
        f"  - Precision (B)    : {report_payload['metrics']['precision']}",
        f"  - Recall (B)       : {report_payload['metrics']['recall']}",
        f"  - mAP@0.50 (B)     : {report_payload['metrics']['mAP_50']}",
        f"  - mAP@0.50:0.95 (B): {report_payload['metrics']['mAP_50_95']}",
        "-" * 70,
        f"Inference Latency    : {speed_info.get('inference', 'N/A')} ms / image",
        "=" * 70,
    ]
    save_text("\n".join(txt_lines), txt_path)
    logger.info(f"Saved package detector metrics to {json_path} and {txt_path}")

    # Copy plots to outputs/package_detector/
    if hasattr(metrics_obj, "save_dir") and metrics_obj.save_dir:
        src_val_dir = Path(metrics_obj.save_dir)
        for f in src_val_dir.glob("*.png"):
            shutil.copy2(f, PACKAGE_OUTPUTS_DIR / f.name)
        for f in src_val_dir.glob("*.jpg"):
            shutil.copy2(f, PACKAGE_OUTPUTS_DIR / f.name)

    return report_payload


if __name__ == "__main__":
    validate_package_detector()
