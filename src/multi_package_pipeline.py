"""
Two-Stage Multi-Package Computer Vision & Damage Risk Pipeline.
Stage 1: Detects and isolates each individual package from a multi-package scene.
Stage 2: Runs damage detection, severity assessment, and risk prediction on each package crop.
Generates multi-package reports and final annotated visual outputs.
"""

import os
import sys
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    WEIGHTS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    save_json,
    save_text,
    get_logger,
)
from src.crop_packages import crop_individual_packages
from src.severity import (
    extract_defect_features,
    calculate_total_damage_coverage,
    assess_damage_severity,
)
from src.risk import calculate_internal_damage_risk
from src.report import generate_inspection_report

logger = get_logger("TwoStagePipeline")

PACKAGE_DETECTOR_WEIGHTS = WEIGHTS_DIR / "package_detector_best.pt"
DAMAGE_DETECTOR_WEIGHTS = WEIGHTS_DIR / "damage_detector_best.pt"

FINAL_OUTPUT_DIR = OUTPUTS_DIR / "final"
PACKAGE_DETECTOR_DIR = OUTPUTS_DIR / "package_detector"
PACKAGE_CROPS_DIR = OUTPUTS_DIR / "package_crops"
REPORTS_OUTPUT_DIR = OUTPUTS_DIR / "reports"


class TwoStagePackageInspector:
    """
    Two-Stage Inspection Engine:
    Stage 1: Individual Package Isolation
    Stage 2: Package Damage Detection & Risk Scoring
    """

    def __init__(
        self,
        package_detector_path: Optional[Path] = None,
        damage_detector_path: Optional[Path] = None,
        package_conf: float = 0.25,
        damage_conf: float = 0.25,
    ):
        from ultralytics import YOLO

        self.pkg_model_path = Path(package_detector_path or PACKAGE_DETECTOR_WEIGHTS)
        self.dmg_model_path = Path(damage_detector_path or DAMAGE_DETECTOR_WEIGHTS)
        self.package_conf = package_conf
        self.damage_conf = damage_conf

        if not self.pkg_model_path.exists():
            # Fallback to general best.pt if package detector is not yet trained
            fallback_pt = WEIGHTS_DIR / "best.pt"
            if fallback_pt.exists():
                self.pkg_model_path = fallback_pt
            else:
                raise FileNotFoundError(f"Package detector weights not found at {self.pkg_model_path}")

        if not self.dmg_model_path.exists():
            fallback_pt = WEIGHTS_DIR / "best.pt"
            if fallback_pt.exists():
                self.dmg_model_path = fallback_pt
            else:
                raise FileNotFoundError(f"Damage detector weights not found at {self.dmg_model_path}")

        logger.info(f"Loading Stage 1 Package Detector: {self.pkg_model_path.name}")
        self.pkg_model = YOLO(str(self.pkg_model_path))

        logger.info(f"Loading Stage 2 Damage Detector: {self.dmg_model_path.name}")
        self.dmg_model = YOLO(str(self.dmg_model_path))
        self.dmg_class_names = self.dmg_model.names

    def inspect_scene(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_prefix: str = "multi_package",
        crop_margin_pct: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Executes complete Two-Stage inspection on a single or multi-package scene image.
        """
        FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PACKAGE_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
        PACKAGE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            image_name = img_path.name
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError(f"Failed to read image: {img_path}")
        else:
            img_bgr = image_input
            image_name = f"{output_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

        img_h, img_w = img_bgr.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ==========================================
        # STAGE 1: INDIVIDUAL PACKAGE DETECTION
        # ==========================================
        logger.info(">>> STAGE 1: Detecting individual packages in scene...")
        pkg_results = self.pkg_model.predict(
            source=img_bgr,
            conf=self.package_conf,
            verbose=False,
        )[0]

        pkg_boxes = []
        pkg_confs = []
        if pkg_results.boxes is not None and len(pkg_results.boxes) > 0:
            for box in pkg_results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].item())
                pkg_boxes.append(xyxy)
                pkg_confs.append(conf)

        logger.info(f"Detected {len(pkg_boxes)} individual packages in scene.")

        # If no package bounding boxes detected (e.g. tight single package crop), treat whole image as Package 1
        if not pkg_boxes:
            logger.info("No individual package bounding boxes separated; inspecting full frame as Package 1.")
            pkg_boxes = [[0.0, 0.0, float(img_w), float(img_h)]]
            pkg_confs = [1.0]

        # Draw Stage 1 Package Detection image
        stage1_img = img_bgr.copy()
        for i, (pbox, pconf) in enumerate(zip(pkg_boxes, pkg_confs)):
            px1, py1, px2, py2 = [int(v) for v in pbox]
            cv2.rectangle(stage1_img, (px1, py1), (px2, py2), (255, 140, 0), 2)
            label = f"Package {i+1} ({pconf:.2f})"
            cv2.putText(
                stage1_img,
                label,
                (px1 + 4, max(20, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 140, 0),
                2,
                cv2.LINE_AA,
            )

        stage1_out_path = PACKAGE_DETECTOR_DIR / "multi_package_detection.jpg"
        cv2.imwrite(str(stage1_out_path), stage1_img)
        logger.info(f"Saved Stage 1 detection visualization to {stage1_out_path}")

        # Crop each individual package
        crops = crop_individual_packages(
            image_input=img_bgr,
            package_boxes=pkg_boxes,
            confidences=pkg_confs,
            margin_pct=crop_margin_pct,
            output_dir=PACKAGE_CROPS_DIR,
            save_to_disk=True,
        )

        # ==========================================
        # STAGE 2: DAMAGE DETECTION ON EACH CROP
        # ==========================================
        logger.info(">>> STAGE 2: Running damage detection on each package crop...")
        package_evaluations = []
        safe_count = 0
        inspect_count = 0
        replace_count = 0

        for crop_data in crops:
            crop_bgr = crop_data["crop_bgr"]
            ch, cw = crop_bgr.shape[:2]
            pkg_id = crop_data["package_id"]

            # Run damage detector
            dmg_results = self.dmg_model.predict(
                source=crop_bgr,
                conf=self.damage_conf,
                verbose=False,
            )[0]

            dmg_boxes = []
            dmg_classes = []
            dmg_confs = []

            if dmg_results.boxes is not None and len(dmg_results.boxes) > 0:
                for dbox in dmg_results.boxes:
                    d_xyxy = dbox.xyxy[0].cpu().numpy().tolist()
                    d_cid = int(dbox.cls[0].item())
                    d_cname = self.dmg_class_names.get(d_cid, f"damage_{d_cid}")
                    d_conf = float(dbox.conf[0].item())

                    dmg_boxes.append(d_xyxy)
                    dmg_classes.append(d_cname)
                    dmg_confs.append(d_conf)

            # Feature extraction on crop
            defect_features = extract_defect_features(
                boxes=dmg_boxes,
                classes=dmg_classes,
                confidences=dmg_confs,
                image_shape=(ch, cw),
            )

            total_coverage = calculate_total_damage_coverage(dmg_boxes, (ch, cw))
            severity_info = assess_damage_severity(defect_features, total_coverage)
            risk_info = calculate_internal_damage_risk(severity_info, defect_features)

            decision = risk_info["delivery_decision"]
            if decision == "SAFE TO DELIVER":
                safe_count += 1
            elif decision == "INSPECT BEFORE DELIVERY":
                inspect_count += 1
            else:
                replace_count += 1

            pkg_eval = {
                "package_id": pkg_id,
                "detector_confidence": crop_data["detector_confidence"],
                "original_bbox": crop_data["original_bbox"],
                "crop_filename": crop_data["crop_filename"],
                "crop_path": crop_data["crop_path"],
                "defects_detected_count": len(defect_features),
                "damage_classes_found": list({f["damage_class"] for f in defect_features}),
                "defects": defect_features,
                "severity_assessment": {
                    "severity_level": severity_info["severity_level"],
                    "total_coverage_percentage": f"{total_coverage * 100:.2f}%",
                    "rationale": severity_info["rationale"],
                },
                "risk_prediction": {
                    "risk_score": risk_info["risk_score"],
                    "delivery_decision": decision,
                    "action_required": risk_info["decision_action"],
                },
            }
            package_evaluations.append(pkg_eval)

        # ==========================================
        # STAGE 3: FINAL MULTI-PACKAGE ANNOTATED IMAGE
        # ==========================================
        final_annotated_img = self.draw_final_multi_package_overlay(
            img_bgr=img_bgr.copy(),
            package_evaluations=package_evaluations,
            safe_count=safe_count,
            inspect_count=inspect_count,
            replace_count=replace_count,
        )

        final_img_path = FINAL_OUTPUT_DIR / "multi_package_final_result.jpg"
        cv2.imwrite(str(final_img_path), final_annotated_img)
        logger.info(f"Saved final multi-package annotated result to {final_img_path}")

        # ==========================================
        # STAGE 4: STRUCTURED FINAL REPORTS
        # ==========================================
        full_report_data = {
            "header": {
                "system_name": "Smart Two-Stage Multi-Package Damage & Risk Inspection System",
                "timestamp": timestamp,
                "source_image": image_name,
                "final_annotated_image": str(final_img_path),
                "stage1_detection_image": str(stage1_out_path),
            },
            "summary": {
                "total_packages_detected": len(package_evaluations),
                "safe_to_deliver_count": safe_count,
                "inspect_before_delivery_count": inspect_count,
                "replace_package_count": replace_count,
            },
            "package_results": package_evaluations,
            "models_used": {
                "package_detector": self.pkg_model_path.name,
                "damage_detector": self.dmg_model_path.name,
            },
        }

        # Save JSON Report
        json_report_path = REPORTS_OUTPUT_DIR / "multi_package_inspection.json"
        save_json(full_report_data, json_report_path)

        # Save TXT Report
        txt_lines = [
            "=" * 75,
            "           MULTI-PACKAGE DAMAGE INSPECTION & RISK REPORT",
            "=" * 75,
            f"Image Analyzed        : {image_name}",
            f"Inspection Date/Time  : {timestamp}",
            f"Total Packages Found  : {len(package_evaluations)}",
            f"  - Safe to Deliver   : {safe_count}",
            f"  - Inspect Before Del: {inspect_count}",
            f"  - Replace Package   : {replace_count}",
            "-" * 75,
            "INDIVIDUAL PACKAGE BREAKDOWN:",
            "-" * 75,
        ]

        for p in package_evaluations:
            sev = p["severity_assessment"]["severity_level"]
            risk = p["risk_prediction"]["risk_score"]
            dec = p["risk_prediction"]["delivery_decision"]
            dmgs = ", ".join(p["damage_classes_found"]) if p["damage_classes_found"] else "None"

            txt_lines.extend([
                f"[{p['package_id']}]",
                f"  Bounding Box    : {p['original_bbox']} (Conf: {p['detector_confidence']:.2f})",
                f"  Crop Image Path : {p['crop_path']}",
                f"  Damage Detected : {dmgs} ({p['defects_detected_count']} defects)",
                f"  Damage Severity : {sev} ({p['severity_assessment']['total_coverage_percentage']} coverage)",
                f"  Internal Risk   : {risk} / 100",
                f"  DELIVERY DECISION: >>> {dec} <<<",
                f"  Action Required : {p['risk_prediction']['action_required']}",
                "",
            ])

        txt_lines.append("=" * 75)
        txt_report_path = REPORTS_OUTPUT_DIR / "multi_package_inspection.txt"
        save_text("\n".join(txt_lines), txt_report_path)

        logger.info(f"Saved inspection reports to {json_report_path} and {txt_report_path}")

        return {
            "total_packages": len(package_evaluations),
            "safe_count": safe_count,
            "inspect_count": inspect_count,
            "replace_count": replace_count,
            "final_image_path": str(final_img_path),
            "stage1_image_path": str(stage1_out_path),
            "json_report_path": str(json_report_path),
            "txt_report_path": str(txt_report_path),
            "package_evaluations": package_evaluations,
        }

    def draw_final_multi_package_overlay(
        self,
        img_bgr: np.ndarray,
        package_evaluations: List[Dict[str, Any]],
        safe_count: int,
        inspect_count: int,
        replace_count: int,
    ) -> np.ndarray:
        """
        Draws clear, uncluttered bounding boxes and status badges on the multi-package scene image.
        """
        canvas = img_bgr.copy()
        img_h, img_w = canvas.shape[:2]

        badge_colors = {
            "SAFE TO DELIVER": (0, 180, 0),        # Green
            "INSPECT BEFORE DELIVERY": (0, 165, 255),  # Amber
            "REPLACE PACKAGE": (0, 0, 220),       # Red
        }
        badge_tags = {
            "SAFE TO DELIVER": "SAFE",
            "INSPECT BEFORE DELIVERY": "INSPECT",
            "REPLACE PACKAGE": "REPLACE",
        }

        # Draw package boxes
        for p in package_evaluations:
            x1, y1, x2, y2 = [int(v) for v in p["original_bbox"]]
            decision = p["risk_prediction"]["delivery_decision"]
            color = badge_colors.get(decision, (0, 180, 0))
            tag = badge_tags.get(decision, "SAFE")
            pkg_id = p["package_id"]
            risk = p["risk_prediction"]["risk_score"]
            dmgs = p["damage_classes_found"]

            # Package bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Badge banner text
            if dmgs:
                banner_text = f"{pkg_id}: {tag} [{', '.join(dmgs)}] ({risk:.0f}pts)"
            else:
                banner_text = f"{pkg_id}: {tag} (Risk {risk:.0f})"

            font_scale = max(0.42, min(0.65, img_w / 1400))
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # Badge background
            by1 = max(0, y1 - th - 6)
            by2 = max(0, y1)
            cv2.rectangle(canvas, (x1, by1), (x1 + tw + 6, by2), color, -1)
            cv2.putText(
                canvas,
                banner_text,
                (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Top Header Telemetry
        header_h = max(50, int(img_h * 0.08))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, header_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        header_line = (
            f"AI MULTI-PACKAGE INSPECTOR | Detected: {len(package_evaluations)} | "
            f"Safe: {safe_count} | Inspect: {inspect_count} | Replace: {replace_count}"
        )
        font_scale = max(0.45, min(0.70, img_w / 1100))
        cv2.putText(canvas, header_line, (15, int(header_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (230, 230, 230), 2, cv2.LINE_AA)

        return canvas
