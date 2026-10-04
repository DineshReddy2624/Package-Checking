"""
Training script for Stage 1 Package Detector.
Trains YOLOv8n on individual multi-package dataset.
Saves validation metrics, curves, and best weights to models/weights/package_detector_best.pt.
"""

import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import WEIGHTS_DIR, OUTPUTS_DIR, get_logger

logger = get_logger("TrainPackageDetector")

DATASET_YAML = project_root / "data" / "processed" / "package_detector_dataset.yaml"
OUTPUT_DIR = OUTPUTS_DIR / "package_detector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_package_detector(epochs: int = 50, imgsz: int = 640, patience: int = 15, batch_size: int = 16):
    logger.info("Initializing YOLOv8n model for Stage 1 Package Detection...")
    model = YOLO("yolov8n.pt")

    logger.info(f"Starting training on {DATASET_YAML} for {epochs} epochs (imgsz={imgsz}, patience={patience})...")
    
    # Train model
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,
        batch=batch_size,
        device="cpu",
        project="runs/detect",
        name="package_detector",
        exist_ok=True,
        workers=2,
        plots=True,
        save=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs/detect/package_detector")
    best_weights = run_dir / "weights" / "best.pt"

    target_weights = WEIGHTS_DIR / "package_detector_best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        logger.info(f"Saved best package detector weights to {target_weights}")

    # Copy curves and validation plots to outputs/package_detector
    plot_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "results.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    for pfile in plot_files:
        src_file = run_dir / pfile
        if src_file.exists():
            dst_file = OUTPUT_DIR / pfile
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied {pfile} to {dst_file}")

    # Validate model
    logger.info("Running validation on validation set...")
    val_model = YOLO(str(target_weights))
    val_results = val_model.val(
        data=str(DATASET_YAML),
        imgsz=imgsz,
        device="cpu",
        plots=True,
    )

    precision = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    recall = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    logger.info("=" * 60)
    logger.info("STAGE 1 PACKAGE DETECTOR VALIDATION METRICS:")
    logger.info(f"  • Precision     : {precision:.4f} ({precision*100:.2f}%)")
    logger.info(f"  • Recall        : {recall:.4f} ({recall*100:.2f}%)")
    logger.info(f"  • mAP@0.5       : {map50:.4f} ({map50*100:.2f}%)")
    logger.info(f"  • mAP@0.5:0.95  : {map50_95:.4f} ({map50_95*100:.2f}%)")
    logger.info("=" * 60)

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    train_package_detector()
