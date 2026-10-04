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
        candidates = [
            SAMPLE_IMAGES_DIR / "multi_package_warehouse.jpg",
            SAMPLE_IMAGES_DIR / "multiple_parcels" / "multi_parcel_01.jpg",
            SAMPLE_IMAGES_DIR / "multiple_parcels" / "multi_parcel_06.jpg",
            SAMPLE_IMAGES_DIR / "single_damaged_package.jpg",
        ]
        for c in candidates:
            if c.exists():
                image_path = c
                break

    if image_path is None or not image_path.exists():
        logger.error(f"Test image not found at {image_path}. Please provide a valid image path.")
        sys.exit(1)

    logger.info(f"Loading test scene image: {image_path}")

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
    print("\n" + "=" * 80)
    print(f"AI-BASED EXTERNAL CONDITION & FIVE-STAGE INSPECTION RESULTS: {image_path.name}")
    print("=" * 80)
    print(f"Total Individual Packages Detected: {results['total_packages']}\n")

    for p in results["package_evaluations"]:
        s1 = p["stage1_package_detection"]
        s2 = p["stage2_damage_detection"]
        s3 = p["stage3_damage_severity"]
        s4 = p["stage4_internal_damage_risk"]
        s5 = p["stage5_delivery_decision"]

        print(f"📦 [{p['package_id']}]")
        print(f"   STAGE 1 — PACKAGE DETECTION")
        print(f"     • Detection Confidence : {s1['package_detection_confidence']:.2f}")
        print(f"     • Bounding Box          : {s1['bounding_box']} (Area Ratio: {s1['bounding_box_area_ratio_pct']})")
        if s1.get("is_suspicious_crop") and s1.get("warning"):
            print(f"     • ⚠️ {s1['warning']}")
        print(f"     • Crop File            : {Path(p['crop_path']).name}")
        print(f"   STAGE 2 — DAMAGE DETECTION")
        print(f"     • Result               : {s2['damage_detection_result']}")
        print(f"     • Damage Confidence    : {s2['damage_confidence']}")
        print(f"   STAGE 3 — DAMAGE SEVERITY")
        print(f"     • Severity Level       : {s3['severity_level']} (Coverage: {s3['damage_area_coverage']})")
        print(f"     • Rationale            : {s3['rationale']}")
        print(f"   STAGE 4 — INTERNAL DAMAGE RISK")
        print(f"     • Heuristic Risk Score : {s4['heuristic_risk_score']:.0f} / 100 ({s4['score_nature']})")
        print(f"   STAGE 5 — DELIVERY DECISION")
        print(f"     • Recommended Decision : >>> {s5['recommended_delivery_decision']} <<< ({s5['decision_type']})")
        print(f"     • Action Required      : {s5['action_required']}\n")

    print("=" * 80)
    print("LOGISTICS SUMMARY (STAGE 1 QUALITY GATE & MODEL RECOMMENDATIONS):")
    print(f"  📦 Total Candidate Packages    : {results['total_packages']}")
    print(f"  ✅ Valid Individual Packages   : {results['valid_count']}")
    print(f"  ❌ Invalid / Oversized Regions : {results['invalid_count']}")
    print(f"  🟢 Recommended Safe to Deliver : {results['safe_count']}")
    print(f"  🟡 Recommended Inspect Before  : {results['inspect_count']}")
    print(f"  🔴 Recommended Replace Package : {results['replace_count']}")
    print(f"Stage 1 Package Detection Image  : {results['stage1_image_path']}")
    print(f"Final Annotated Inspection Image : {results['final_image_path']}")
    print(f"Structured JSON Report           : {results['json_report_path']}")
    print(f"Human-Readable TXT Report        : {results['txt_report_path']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Package Two-Stage Test Script")
    parser.add_argument("--image", type=str, default=None, help="Path to multi-package scene image")
    parser.add_argument("--pkg-conf", type=float, default=0.25, help="Package detector confidence threshold")
    parser.add_argument("--dmg-conf", type=float, default=0.25, help="Damage detector confidence threshold")
    args = parser.parse_args()

    img_p = Path(args.image) if args.image else None
    run_multi_package_test(image_path=img_p, pkg_conf=args.pkg_conf, dmg_conf=args.dmg_conf)
