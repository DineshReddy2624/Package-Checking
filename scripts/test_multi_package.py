"""
Multi-Package Test Script.
Loads a multi-package warehouse/pallet scene, detects each individual package separately,
crops them, runs damage inspection, and generates full visual and structured reports.
"""

import os
import sys
import argparse
from typing import Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    WEIGHTS_DIR,
    SAMPLE_IMAGES_DIR,
    OUTPUTS_DIR,
    get_logger,
)
from src.multi_package_pipeline import TwoStagePackageInspector

logger = get_logger("TestMultiPackage")


def run_multi_package_test(image_path: Optional[Path] = None, pkg_conf: float = 0.25, dmg_conf: float = 0.25):
    """
    Executes the critical multi-package detection and damage assessment test.
    """
    print("\n" + "=" * 75)
    print("       TWO-STAGE MULTI-PACKAGE DETECTION & DAMAGE ASSESSMENT TEST       ")
    print("=" * 75 + "\n")

    # Locate sample test image
    if image_path is None:
        # Check sample_images
        candidates = [
            SAMPLE_IMAGES_DIR / "multi_package_warehouse.jpg",
            SAMPLE_IMAGES_DIR / "multi_package_pallet.jpg",
            PROJECT_ROOT / "data/extracted/box_defect_dataset/box_defect_dataset/real/images/test/test_bulto_00021_a2e5618c15.jpg",
            PROJECT_ROOT / "data/extracted/box_defect_dataset/box_defect_dataset/real/images/test/test_bulto_00033_64d9a9c3b3.jpg",
            SAMPLE_IMAGES_DIR / "package.jpg",
        ]
        for c in candidates:
            if c.exists():
                image_path = c
                break

    if image_path is None or not image_path.exists():
        logger.error(f"Test image not found at {image_path}. Please provide a valid image path.")
        sys.exit(1)

    logger.info(f"Loading test multi-package scene image: {image_path}")

    # Copy image to sample_images/multi_package_warehouse.jpg if not present
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sample_multi_target = SAMPLE_IMAGES_DIR / "multi_package_warehouse.jpg"
    if not sample_multi_target.exists():
        import shutil
        shutil.copy2(image_path, sample_multi_target)

    # Initialize Two-Stage Inspector
    inspector = TwoStagePackageInspector(
        package_detector_path=WEIGHTS_DIR / "package_detector_best.pt",
        damage_detector_path=WEIGHTS_DIR / "damage_detector_best.pt",
        package_conf=pkg_conf,
        damage_conf=dmg_conf,
    )

    # Run inspection
    results = inspector.inspect_scene(image_input=image_path, crop_margin_pct=0.05)

    # Print results summary
    print("\n" + "-" * 75)
    print(f"MULTI-PACKAGE DETECTION RESULTS FOR: {image_path.name}")
    print("-" * 75)
    print(f"Detected packages: {results['total_packages']}")
    for p in results["package_evaluations"]:
        print(
            f"  - {p['package_id']:12s} | Conf: {p['detector_confidence']:.2f} | "
            f"Box: {p['original_bbox']} | Crop: {Path(p['crop_path']).name}"
        )
        sev = p["severity_assessment"]["severity_level"]
        risk = p["risk_prediction"]["risk_score"]
        dec = p["risk_prediction"]["delivery_decision"]
        dmgs = ", ".join(p["damage_classes_found"]) if p["damage_classes_found"] else "None"
        print(f"    -> Damage: {dmgs} | Severity: {sev} | Risk: {risk}/100 | Decision: {dec}")

    print("\n" + "=" * 75)
    print("DELIVERY DECISION SUMMARY:")
    print(f"  Safe to Deliver        : {results['safe_count']}")
    print(f"  Inspect Before Delivery: {results['inspect_count']}")
    print(f"  Replace Package        : {results['replace_count']}")
    print(f"Stage 1 Package Detection Image : {results['stage1_image_path']}")
    print(f"Final Annotated Inspection Image: {results['final_image_path']}")
    print(f"Structured JSON Report          : {results['json_report_path']}")
    print(f"Human-Readable TXT Report       : {results['txt_report_path']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Package Two-Stage Test Script")
    parser.add_argument("--image", type=str, default=None, help="Path to multi-package scene image")
    parser.add_argument("--pkg-conf", type=float, default=0.25, help="Package detector confidence threshold")
    parser.add_argument("--dmg-conf", type=float, default=0.25, help="Damage detector confidence threshold")
    args = parser.parse_args()

    img_p = Path(args.image) if args.image else None
    run_multi_package_test(image_path=img_p, pkg_conf=args.pkg_conf, dmg_conf=args.dmg_conf)
