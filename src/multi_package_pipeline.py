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
        max_package_area_ratio: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Executes complete Two-Stage inspection on a single or multi-package scene image.
        Applies a strict Stage 1 Quality Gate (MAX_PACKAGE_AREA_RATIO = 0.80 default).
        Invalid/suspicious detections are quarantined and bypassed from Stage 2 damage analysis.
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

        logger.info(f"Stage 1 detected {len(pkg_boxes)} candidate package region(s).")

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

        # Crop each individual package and evaluate Stage 1 Quality Gate
        crops = crop_individual_packages(
            image_input=img_bgr,
            package_boxes=pkg_boxes,
            confidences=pkg_confs,
            margin_pct=crop_margin_pct,
            max_package_area_ratio=max_package_area_ratio,
            output_dir=PACKAGE_CROPS_DIR,
            save_to_disk=True,
        )

        # ==========================================
        # STAGE 2: DAMAGE DETECTION (STAGE 1 GATE ENFORCED)
        # ==========================================
        logger.info(">>> STAGE 2: Running damage detection on valid package crops...")
        package_evaluations = []
        valid_count = 0
        invalid_count = 0
        safe_count = 0
        inspect_count = 0
        replace_count = 0

        for crop_data in crops:
            crop_bgr = crop_data["crop_bgr"]
            ch, cw = crop_bgr.shape[:2]
            pkg_id = crop_data["package_id"]
            is_valid = crop_data["is_valid_package"]
            area_ratio = crop_data.get("box_area_ratio", 0.0)
            area_ratio_pct = crop_data.get("box_area_ratio_pct", "0.00%")
            detector_conf = crop_data["detector_confidence"]

            if is_valid:
                valid_count += 1
                # Run damage detector ONLY on valid crops
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

                dmg_conf_display = "N/A" if len(defect_features) == 0 else [f"{d['damage_class']}: {d['confidence']:.2f}" for d in defect_features]

                pkg_eval = {
                    "package_id": pkg_id,
                    "is_valid_package": True,
                    "detection_status": "VALID_PACKAGE",
                    "delivery_decision": decision,
                    "risk_score": risk_info["risk_score"],
                    "risk_score_display": f"{risk_info['risk_score']:.0f} / 100",
                    # STAGE 1: PACKAGE DETECTION
                    "stage1_package_detection": {
                        "package_id": pkg_id,
                        "status": "VALID_PACKAGE",
                        "package_detection_confidence": detector_conf,
                        "bounding_box": crop_data["original_bbox"],
                        "bounding_box_area_ratio": area_ratio,
                        "bounding_box_area_ratio_pct": area_ratio_pct,
                        "is_suspicious_crop": False,
                        "warning": None,
                    },
                    # STAGE 2: DAMAGE DETECTION
                    "stage2_damage_detection": {
                        "status": "EVALUATED",
                        "damage_detected": len(defect_features) > 0,
                        "damage_detection_result": f"{len(defect_features)} defect(s) detected: {', '.join(list({f['damage_class'] for f in defect_features}))}" if defect_features else "No damage detected",
                        "damage_classes_found": list({f["damage_class"] for f in defect_features}),
                        "damage_confidence": dmg_conf_display,
                        "defects_detected_count": len(defect_features),
                        "defects": defect_features,
                    },
                    # STAGE 3: DAMAGE SEVERITY
                    "stage3_damage_severity": {
                        "status": "EVALUATED",
                        "severity_level": severity_info["severity_level"],
                        "damage_area_coverage": f"{total_coverage * 100:.2f}%",
                        "total_coverage_ratio": total_coverage,
                        "severity_score_normalized": severity_info.get("severity_score_normalized", 0.0),
                        "rationale": severity_info["rationale"],
                    },
                    # STAGE 4: INTERNAL DAMAGE RISK
                    "stage4_internal_damage_risk": {
                        "status": "EVALUATED",
                        "risk_metric_name": "Heuristic Estimated Internal Damage Risk Score",
                        "heuristic_risk_score": risk_info["risk_score"],
                        "risk_score_display": f"{risk_info['risk_score']:.0f} / 100",
                        "score_scale": "0 - 100",
                        "is_probability": False,
                        "score_nature": "Heuristic estimated score based on external visual features, not calibrated probability.",
                    },
                    # STAGE 5: DELIVERY DECISION
                    "stage5_delivery_decision": {
                        "status": "RECOMMENDED",
                        "recommended_delivery_decision": decision,
                        "decision_type": "Model Recommendation",
                        "action_required": risk_info["decision_action"],
                        "justification": "Based on detected external condition and configured heuristic risk threshold.",
                    },
                    # Convenience aliases
                    "detector_confidence": detector_conf,
                    "original_bbox": crop_data["original_bbox"],
                    "box_area_ratio_pct": area_ratio_pct,
                    "is_suspicious_crop": False,
                    "warning": None,
                    "crop_filename": crop_data["crop_filename"],
                    "crop_path": crop_data["crop_path"],
                    "defects_detected_count": len(defect_features),
                    "damage_classes_found": list({f["damage_class"] for f in defect_features}),
                    "damage_confidence": dmg_conf_display,
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
            else:
                # STAGE 1 QUALITY GATE REJECTION: Bypass Stage 2-5 completely
                invalid_count += 1
                rejection_reason = crop_data.get("rejection_reason", "Detected region is too large to confidently represent an individual package.")
                rejection_action = crop_data.get("rejection_action", "Capture a closer image containing the individual package or improve the package detection model.")
                decision = "PACKAGE DETECTION REQUIRES REVIEW"

                pkg_eval = {
                    "package_id": pkg_id,
                    "is_valid_package": False,
                    "detection_status": "INVALID PACKAGE DETECTION",
                    "delivery_decision": decision,
                    "risk_score": None,
                    "risk_score_display": "N/A",
                    # STAGE 1: PACKAGE DETECTION
                    "stage1_package_detection": {
                        "package_id": pkg_id,
                        "status": "INVALID_PACKAGE_DETECTION",
                        "package_detection_confidence": detector_conf,
                        "bounding_box": crop_data["original_bbox"],
                        "bounding_box_area_ratio": area_ratio,
                        "bounding_box_area_ratio_pct": area_ratio_pct,
                        "is_suspicious_crop": True,
                        "warning": crop_data.get("suspicious_warning"),
                        "rejection_reason": rejection_reason,
                        "rejection_action": rejection_action,
                    },
                    # STAGE 2: DAMAGE DETECTION (BYPASSED)
                    "stage2_damage_detection": {
                        "status": "SKIPPED_QUALITY_GATE",
                        "damage_detected": None,
                        "damage_detection_result": "Damage analysis was not performed because the package boundary could not be reliably established.",
                        "damage_confidence": "N/A",
                        "defects_detected_count": 0,
                        "defects": [],
                    },
                    # STAGE 3: DAMAGE SEVERITY (BYPASSED)
                    "stage3_damage_severity": {
                        "status": "SKIPPED_QUALITY_GATE",
                        "severity_level": "Not Evaluated",
                        "damage_area_coverage": "N/A",
                        "total_coverage_ratio": 0.0,
                        "severity_score_normalized": 0.0,
                        "rationale": "Damage severity evaluation skipped because Stage 1 package detection quality gate failed.",
                    },
                    # STAGE 4: INTERNAL DAMAGE RISK (BYPASSED)
                    "stage4_internal_damage_risk": {
                        "status": "SKIPPED_QUALITY_GATE",
                        "risk_metric_name": "Heuristic Estimated Internal Damage Risk Score",
                        "heuristic_risk_score": None,
                        "risk_score_display": "N/A",
                        "score_scale": "0 - 100",
                        "is_probability": False,
                        "score_nature": "Not evaluated (Stage 1 Quality Gate rejected).",
                    },
                    # STAGE 5: DELIVERY DECISION (BYPASSED)
                    "stage5_delivery_decision": {
                        "status": "QUALITY_GATE_REJECTED",
                        "recommended_delivery_decision": decision,
                        "decision_type": "Stage 1 Quality Gate Rejection",
                        "action_required": rejection_action,
                        "justification": "Stage 1 could not reliably isolate an individual package from the input image.",
                    },
                    # Convenience aliases
                    "detector_confidence": detector_conf,
                    "original_bbox": crop_data["original_bbox"],
                    "box_area_ratio_pct": area_ratio_pct,
                    "is_suspicious_crop": True,
                    "warning": crop_data.get("suspicious_warning"),
                    "crop_filename": crop_data["crop_filename"],
                    "crop_path": crop_data["crop_path"],
                    "defects_detected_count": 0,
                    "damage_classes_found": [],
                    "damage_confidence": "N/A",
                    "defects": [],
                    "severity_assessment": {
                        "severity_level": "Not Evaluated",
                        "total_coverage_percentage": "N/A",
                        "rationale": "Damage analysis skipped due to Stage 1 Quality Gate rejection.",
                    },
                    "risk_prediction": {
                        "risk_score": None,
                        "delivery_decision": decision,
                        "action_required": rejection_action,
                    },
                }

            package_evaluations.append(pkg_eval)

        # ==========================================
        # STAGE 3: FINAL MULTI-PACKAGE ANNOTATED IMAGE
        # ==========================================
        final_annotated_img = self.draw_final_multi_package_overlay(
            img_bgr=img_bgr.copy(),
            package_evaluations=package_evaluations,
            valid_count=valid_count,
            invalid_count=invalid_count,
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
                "system_name": "Smart AI-Based Delivery Package Damage Detection and Internal Damage Risk Prediction System",
                "inspection_type": "AI-Based External Condition Assessment & Heuristic Risk Prediction",
                "timestamp": timestamp,
                "source_image": image_name,
                "final_annotated_image": str(final_img_path),
                "stage1_detection_image": str(stage1_out_path),
            },
            "summary": {
                "total_packages_detected": len(package_evaluations),
                "valid_packages_count": valid_count,
                "invalid_packages_count": invalid_count,
                "safe_to_deliver_count": safe_count,
                "inspect_before_delivery_count": inspect_count,
                "replace_package_count": replace_count,
                "quality_gate_threshold_max_area_ratio": f"{max_package_area_ratio*100:.0f}%",
                "decision_nature": "Model Recommendation based on observable external visual features",
            },
            "package_results": package_evaluations,
            "models_used": {
                "stage1_package_detector": self.pkg_model_path.name,
                "stage2_damage_detector": self.dmg_model_path.name,
            },
            "disclaimer": (
                "The delivery decision is a model recommendation derived from camera-observable external visual features "
                "and heuristic risk scoring (0-100). The system does not inspect internal package cushioning directly."
            ),
        }

        # Save JSON Report
        json_report_path = REPORTS_OUTPUT_DIR / "multi_package_inspection.json"
        save_json(full_report_data, json_report_path)

        # Save TXT Report
        txt_lines = [
            "=" * 75,
            "     AI-BASED EXTERNAL CONDITION & MULTI-PACKAGE DAMAGE INSPECTION REPORT",
            "=" * 75,
            f"Image Analyzed        : {image_name}",
            f"Inspection Date/Time  : {timestamp}",
            f"Total Packages Found  : {len(package_evaluations)}",
            f"  - Valid Packages    : {valid_count}",
            f"  - Invalid Detections: {invalid_count}",
            f"  - Recommended Safe  : {safe_count}",
            f"  - Recom. Inspection : {inspect_count}",
            f"  - Recom. Replacement: {replace_count}",
            "-" * 75,
            "INDIVIDUAL PACKAGE FIVE-STAGE BREAKDOWN:",
            "-" * 75,
        ]

        for p in package_evaluations:
            s1 = p["stage1_package_detection"]
            s2 = p["stage2_damage_detection"]
            s3 = p["stage3_damage_severity"]
            s4 = p["stage4_internal_damage_risk"]
            s5 = p["stage5_delivery_decision"]

            txt_lines.extend([
                f"[{p['package_id']}] Status: {p['detection_status']}",
                f"  STAGE 1 - PACKAGE DETECTION:",
                f"    Package Detection Confidence : {s1['package_detection_confidence']:.2f}",
                f"    Bounding Box                 : {s1['bounding_box']} (Area Ratio: {s1['bounding_box_area_ratio_pct']})",
                f"    Quality Gate Status          : {'VALID' if p['is_valid_package'] else 'REJECTED (Oversized Detection)'}",
                f"    Crop Image Path              : {p['crop_path']}",
            ])

            if p["is_valid_package"]:
                txt_lines.extend([
                    f"  STAGE 2 - DAMAGE DETECTION:",
                    f"    Damage Detection Result      : {s2['damage_detection_result']}",
                    f"    Damage Confidence            : {s2['damage_confidence']}",
                    f"  STAGE 3 - DAMAGE SEVERITY:",
                    f"    Severity Level               : {s3['severity_level']} (Damage Coverage: {s3['damage_area_coverage']})",
                    f"    Severity Rationale           : {s3['rationale']}",
                    f"  STAGE 4 - INTERNAL DAMAGE RISK:",
                    f"    Heuristic Risk Score         : {s4['heuristic_risk_score']} / 100 ({s4['score_nature']})",
                    f"  STAGE 5 - DELIVERY DECISION:",
                    f"    Recommended Delivery Decision: >>> {s5['recommended_delivery_decision']} <<<",
                    f"    Action Required              : {s5['action_required']}",
                ])
            else:
                txt_lines.extend([
                    f"  STAGE 2 - DAMAGE DETECTION:",
                    f"    Damage Detection Result      : {s2['damage_detection_result']}",
                    f"  STAGE 3 - DAMAGE SEVERITY:",
                    f"    Severity Level               : {s3['severity_level']}",
                    f"  STAGE 4 - INTERNAL DAMAGE RISK:",
                    f"    Heuristic Risk Score         : {s4['risk_score_display']}",
                    f"  STAGE 5 - DELIVERY DECISION:",
                    f"    Recommended Action           : >>> {s5['recommended_delivery_decision']} <<<",
                    f"    Action Required              : {s5['action_required']}",
                ])
            txt_lines.append("")

        txt_lines.append("=" * 75)
        txt_report_path = REPORTS_OUTPUT_DIR / "multi_package_inspection.txt"
        save_text("\n".join(txt_lines), txt_report_path)

        logger.info(f"Saved inspection reports to {json_report_path} and {txt_report_path}")

        return {
            "total_packages": len(package_evaluations),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
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
        valid_count: int,
        invalid_count: int,
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
            "PACKAGE DETECTION REQUIRES REVIEW": (0, 140, 255),  # Orange
        }
        badge_tags = {
            "SAFE TO DELIVER": "SAFE",
            "INSPECT BEFORE DELIVERY": "INSPECT",
            "REPLACE PACKAGE": "REPLACE",
            "PACKAGE DETECTION REQUIRES REVIEW": "REVIEW",
        }

        # Draw package boxes
        for p in package_evaluations:
            x1, y1, x2, y2 = [int(v) for v in p["original_bbox"]]
            is_valid = p.get("is_valid_package", True)
            decision = p["delivery_decision"]
            color = badge_colors.get(decision, (0, 140, 255) if not is_valid else (0, 180, 0))
            tag = badge_tags.get(decision, "REVIEW" if not is_valid else "SAFE")
            pkg_id = p["package_id"]
            dmgs = p.get("damage_classes_found", [])
            area_pct = p.get("box_area_ratio_pct", "0.00%")

            # Package bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Badge banner text
            if not is_valid:
                banner_text = f"{pkg_id}: INVALID / OVERSIZED ({area_pct})"
            elif dmgs:
                risk = p["risk_prediction"]["risk_score"]
                banner_text = f"{pkg_id}: {tag} [{', '.join(dmgs)}] ({risk:.0f}pts)"
            else:
                risk = p["risk_prediction"]["risk_score"]
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
            f"AI MULTI-PACKAGE INSPECTOR | Found: {len(package_evaluations)} | "
            f"Valid: {valid_count} | Invalid: {invalid_count} | "
            f"Safe: {safe_count} | Inspect: {inspect_count} | Replace: {replace_count}"
        )
        font_scale = max(0.45, min(0.70, img_w / 1100))
        cv2.putText(canvas, header_line, (15, int(header_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (230, 230, 230), 2, cv2.LINE_AA)

        return canvas
