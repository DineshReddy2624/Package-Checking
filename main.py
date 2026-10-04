"""
Master Orchestration Script for Smart AI-Based Delivery Package Damage Detection
and Internal Damage Risk Prediction System.
Supports Full Dataset and Specialized Single Package Inspection workflows.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    ensure_directories,
    print_hardware_summary,
    detect_hardware,
    WEIGHTS_DIR,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    OUTPUT_REPORTS_DIR,
    PREDICTIONS_DIR,
    SAMPLE_IMAGES_DIR,
    PROCESSED_DATA_DIR,
    EXTRACTED_DATA_DIR,
    get_logger,
)
from src.download_dataset import download_and_extract_dataset
from src.inspect_dataset import inspect_and_validate_dataset
from src.train import train_damage_detector
from src.validate import validate_model
from src.inference import PackageDamagePredictor, run_validation_predictions

logger = get_logger("MainPipeline")


def run_pipeline(
    force_train: bool = False,
    single_package_mode: bool = False,
    model_size: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = -1,
    conf_thresh: float = 0.30,
):
    """
    Executes the entire end-to-end pipeline:
    Environment -> Download -> Inspect -> Train -> Validate -> Inference -> Reporting.
    """
    print("\n" + "=" * 80)
    print("  SMART AI-BASED DELIVERY PACKAGE DAMAGE DETECTION & INTERNAL DAMAGE RISK SYSTEM  ")
    print(f"  Mode: {'SINGLE PACKAGE INSPECTION' if single_package_mode else 'FULL PACKAGE DAMAGE DATASET'} | Model: {model_size}")
    print("=" * 80 + "\n")

    # Step 1: Ensure directories and print hardware
    ensure_directories()
    print_hardware_summary()

    # Step 2: Download dataset
    logger.info(">>> STEP 1/6: Dataset Download & Extraction <<<")
    download_success, extracted_dir = download_and_extract_dataset()
    if not download_success:
        logger.error("Dataset download failed or credentials missing. Stopping pipeline.")
        sys.exit(1)

    # Step 3: Inspect & Validate Dataset
    logger.info(">>> STEP 2/6: Dataset Inspection & Validation <<<")
    dataset_report = inspect_and_validate_dataset(extracted_dir)

    if single_package_mode:
        data_yaml_path = PROCESSED_DATA_DIR / "single_package_dataset.yaml"
        weight_name = "best_single_package.pt"
        run_name = "single_package_run"
    else:
        data_yaml_path = PROCESSED_DATA_DIR / "dataset.yaml"
        weight_name = "best.pt"
        run_name = "package_damage_run"

    if not data_yaml_path.exists():
        logger.error(f"Dataset config could not be generated at {data_yaml_path}. Aborting.")
        sys.exit(1)

    # Step 4: YOLOv8 Training
    best_model_path = WEIGHTS_DIR / weight_name
    logger.info(">>> STEP 3/6: YOLOv8 Model Training <<<")
    if best_model_path.exists() and not force_train:
        logger.info(f"Existing model found at {best_model_path}. Skipping retraining.")
    else:
        logger.info(f"Training {model_size} on {data_yaml_path.name} for {epochs} epochs...")
        best_model_path = train_damage_detector(
            data_yaml_path=data_yaml_path,
            model_size=model_size,
            epochs=epochs,
            initial_batch=batch_size,
            run_name=run_name,
            output_weight_name=weight_name,
        )

    # Step 5: Model Validation & Honest Metric Extraction
    logger.info(">>> STEP 4/6: Model Validation & Actual Metrics Extraction <<<")
    metrics_report = validate_model(
        model_path=best_model_path,
        data_yaml_path=data_yaml_path,
        split="val",
    )

    # Step 6: Validation Predictions & Inference Demonstration
    logger.info(">>> STEP 5/6: Validation Predictions & Inference <<<")
    run_validation_predictions(model_path=best_model_path, num_samples=6)

    # Step 7: Sample Image Inference & Inspection Report
    logger.info(">>> STEP 6/6: Sample Inspection Report Generation <<<")
    sample_images = [
        SAMPLE_IMAGES_DIR / "single_damaged_package.jpg",
        SAMPLE_IMAGES_DIR / "single_crushed_package.jpg",
        SAMPLE_IMAGES_DIR / "package.jpg",
    ]

    existing_samples = [p for p in sample_images if p.exists()]
    if existing_samples:
        test_img = existing_samples[0]
        predictor = PackageDamagePredictor(model_path=best_model_path, conf_threshold=conf_thresh)
        sample_res = predictor.predict_image(
            image_input=test_img,
            output_dir=PREDICTIONS_DIR,
            save_annotated=True,
            save_report=True,
            package_id="PKG-SINGLE-TEST-001",
        )
        logger.info(f"Sample single package evaluated: {test_img.name}")
        logger.info(f"Severity: {sample_res['severity_info']['severity_level']}, Risk: {sample_res['risk_info']['risk_score']}/100, Decision: {sample_res['risk_info']['delivery_decision']}")

    # Final Summary
    print_final_summary(dataset_report, metrics_report, best_model_path)


def print_final_summary(dataset_report: dict, metrics_report: dict, best_model_path: Path):
    """
    Prints the final summary based on actual execution results.
    """
    hw = detect_hardware()
    overall = metrics_report.get("overall_metrics", {})
    splits = dataset_report.get("splits", {})
    ann = dataset_report.get("annotations", {})

    print("\n" + "=" * 75)
    print("                     FINAL SYSTEM EXECUTION SUMMARY                     ")
    print("=" * 75)
    print("## DATASET")
    print(f"Dataset downloaded   : YES")
    print(f"Dataset location     : {dataset_report.get('dataset_root', 'N/A')}")
    print(f"Number of classes    : {ann.get('number_of_classes', 'N/A')}")
    print(f"Class names          : {', '.join(ann.get('class_names', []))}")
    print(f"Train images         : {splits.get('train_images', 0)}")
    print(f"Validation images    : {splits.get('validation_images', 0)}")
    print(f"Test images          : {splits.get('test_images', 0)}")

    print("\n## HARDWARE")
    print(f"CUDA available       : {hw['cuda_available']}")
    print(f"GPU                  : {hw['gpu_name']}")
    print(f"Device used          : {'GPU' if hw['cuda_available'] else 'CPU'}")

    print("\n## TRAINING")
    print(f"Best model           : {best_model_path}")
    print(f"Training completed   : YES")

    print("\n## ACTUAL MODEL RESULTS")
    print(f"Precision            : {overall.get('precision', 'N/A')}")
    print(f"Recall               : {overall.get('recall', 'N/A')}")
    print(f"mAP@0.5              : {overall.get('mAP_50', 'N/A')}")
    print(f"mAP@0.5:0.95         : {overall.get('mAP_50_95', 'N/A')}")

    print("\n## OUTPUTS & ARTIFACTS")
    print(f"Best model path      : {best_model_path}")
    print(f"Metrics report (JSON): {REPORTS_DIR / 'model_metrics.json'}")
    print(f"Metrics report (TXT) : {REPORTS_DIR / 'model_metrics.txt'}")
    print(f"Dataset report (JSON): {REPORTS_DIR / 'dataset_report.json'}")
    print(f"Visualizations       : {VISUALIZATIONS_DIR}")
    print(f"Validation preds     : {PREDICTIONS_DIR}")
    print(f"Inspection report    : {OUTPUT_REPORTS_DIR / 'inspection_report.json'}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Damage Detection & Risk Prediction")
    parser.add_argument("--force-train", action="store_true", help="Force retrain even if weights exist")
    parser.add_argument("--single-package", action="store_true", help="Train / validate on Single Package dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 architecture (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (default: auto)")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold for inference (default: 0.30)")
    args = parser.parse_args()

    run_pipeline(
        force_train=args.force_train,
        single_package_mode=args.single_package,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        conf_thresh=args.conf,
    )
