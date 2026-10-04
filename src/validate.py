"""
Model Validation & Metrics Extraction Module.
Loads best.pt, runs YOLO validation on the validation split,
and extracts verified metrics into JSON and TXT reports without metric fabrication.
"""

import os
import sys
import json
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
    save_json,
    save_text,
    get_logger,
)

logger = get_logger("ValidateModel")


def validate_model(
    model_path: Optional[Path] = None,
    data_yaml_path: Optional[Path] = None,
    imgsz: int = 640,
    split: str = "val",
) -> Dict[str, Any]:
    """
    Evaluates the YOLOv8 model on the validation dataset and compiles honest metric reports.
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = WEIGHTS_DIR / "best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained weights not found at {model_path}. Train the model first.")

    if data_yaml_path is None:
        data_yaml_path = PROCESSED_DATA_DIR / "dataset.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found at {data_yaml_path}.")

    logger.info(f"Validating model '{model_path}' on split='{split}' using config '{data_yaml_path}'...")
    model = YOLO(str(model_path))

    metrics_obj = model.val(
        data=str(data_yaml_path),
        imgsz=imgsz,
        split=split,
        verbose=True,
        save_json=True,
    )

    # Extract metrics safely and honestly from metrics_obj
    results_dict = metrics_obj.results_dict if hasattr(metrics_obj, "results_dict") else {}
    box_metrics = metrics_obj.box if hasattr(metrics_obj, "box") else None

    # Precision
    if box_metrics and hasattr(box_metrics, "mp") and box_metrics.mp is not None:
        precision_val = float(box_metrics.mp)
    elif "metrics/precision(B)" in results_dict:
        precision_val = float(results_dict["metrics/precision(B)"])
    else:
        precision_val = "unavailable (metric not returned by validator)"

    # Recall
    if box_metrics and hasattr(box_metrics, "mr") and box_metrics.mr is not None:
        recall_val = float(box_metrics.mr)
    elif "metrics/recall(B)" in results_dict:
        recall_val = float(results_dict["metrics/recall(B)"])
    else:
        recall_val = "unavailable (metric not returned by validator)"

    # mAP@0.5
    if box_metrics and hasattr(box_metrics, "map50") and box_metrics.map50 is not None:
        map50_val = float(box_metrics.map50)
    elif "metrics/mAP50(B)" in results_dict:
        map50_val = float(results_dict["metrics/mAP50(B)"])
    else:
        map50_val = "unavailable (metric not returned by validator)"

    # mAP@0.5:0.95
    if box_metrics and hasattr(box_metrics, "map") and box_metrics.map is not None:
        map50_95_val = float(box_metrics.map)
    elif "metrics/mAP50-95(B)" in results_dict:
        map50_95_val = float(results_dict["metrics/mAP50-95(B)"])
    else:
        map50_95_val = "unavailable (metric not returned by validator)"

    # Speed metrics
    speed_info = metrics_obj.speed if hasattr(metrics_obj, "speed") else {}
    inference_ms = speed_info.get("inference", "unavailable")

    # Per-class metrics
    class_names = metrics_obj.names if hasattr(metrics_obj, "names") else {}
    per_class = {}
    if box_metrics and hasattr(box_metrics, "p") and box_metrics.p is not None:
        for idx, (p, r, ap50, ap) in enumerate(zip(box_metrics.p, box_metrics.r, box_metrics.ap50, box_metrics.ap)):
            cname = class_names.get(idx, f"class_{idx}")
            per_class[cname] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "mAP50": round(float(ap50), 4),
                "mAP50_95": round(float(ap), 4),
            }

    # Format output payload
    metrics_report = {
        "model_path": str(model_path),
        "dataset_config": str(data_yaml_path),
        "validation_split": split,
        "overall_metrics": {
            "precision": round(precision_val, 4) if isinstance(precision_val, float) else precision_val,
            "recall": round(recall_val, 4) if isinstance(recall_val, float) else recall_val,
            "mAP_50": round(map50_val, 4) if isinstance(map50_val, float) else map50_val,
            "mAP_50_95": round(map50_95_val, 4) if isinstance(map50_95_val, float) else map50_95_val,
        },
        "target_benchmarks": {
            "target_mAP_50": 0.90,
            "target_achieved": (map50_val >= 0.90) if isinstance(map50_val, float) else False,
            "target_latency_sec": 1.50,
            "actual_latency_ms": inference_ms,
        },
        "inference_speed_ms": speed_info,
        "per_class_metrics": per_class,
    }

    # Save JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "model_metrics.json"
    save_json(metrics_report, json_path)

    # Save human-readable TXT report
    txt_lines = [
        "=" * 65,
        "YOLOv8 MODEL VALIDATION PERFORMANCE REPORT",
        "=" * 65,
        f"Model Weight Path    : {model_path}",
        f"Evaluation Split     : {split}",
        "-" * 65,
        "OVERALL ACTUAL PERFORMANCE METRICS:",
        f"  - Precision (B)    : {metrics_report['overall_metrics']['precision']}",
        f"  - Recall (B)       : {metrics_report['overall_metrics']['recall']}",
        f"  - mAP@0.50 (B)     : {metrics_report['overall_metrics']['mAP_50']}",
        f"  - mAP@0.50:0.95 (B): {metrics_report['overall_metrics']['mAP_50_95']}",
        "-" * 65,
        "TARGET BENCHMARK COMPARISON:",
        f"  - Target mAP@0.50  : >= 0.90 (90%)",
        f"  - Target Achieved  : {'YES' if metrics_report['target_benchmarks']['target_achieved'] else 'NO (Reported honestly from execution)'}",
        f"  - Inference Latency: {inference_ms} ms / image (Target < 1500 ms)",
        "-" * 65,
    ]

    if per_class:
        txt_lines.append("PER-CLASS PERFORMANCE BREAKDOWN:")
        for cname, cdata in per_class.items():
            txt_lines.append(
                f"  - {cname:20s}: P={cdata['precision']:.3f}, R={cdata['recall']:.3f}, "
                f"mAP50={cdata['mAP50']:.3f}, mAP50-95={cdata['mAP50_95']:.3f}"
            )
    else:
        txt_lines.append("PER-CLASS BREAKDOWN: Not returned or single class aggregate.")

    txt_lines.append("=" * 65)
    txt_path = REPORTS_DIR / "model_metrics.txt"
    save_text("\n".join(txt_lines), txt_path)

    logger.info(f"Saved model metrics to {json_path} and {txt_path}")
    return metrics_report


if __name__ == "__main__":
    report = validate_model()
    print("Validation finished.")
