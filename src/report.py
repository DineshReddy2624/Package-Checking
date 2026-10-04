"""
Inspection Report & Firebase Logging Module.
Generates structured JSON and human-readable text inspection reports with optional Firestore sync.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    OUTPUT_REPORTS_DIR,
    save_json,
    save_text,
    get_logger,
)

logger = get_logger("ReportGenerator")


def generate_inspection_report(
    image_name: str,
    defect_features: list,
    severity_info: Dict[str, Any],
    risk_info: Dict[str, Any],
    annotated_image_path: Optional[str] = None,
    package_id: Optional[str] = None,
    save_to_disk: bool = True,
) -> Dict[str, Any]:
    """
    Assembles a comprehensive inspection report and saves it to JSON and TXT.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not package_id:
        # Generate deterministic or random ID based on image name
        clean_stem = Path(image_name).stem.replace(" ", "_")
        package_id = f"PKG-{clean_stem}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    report_data = {
        "header": {
            "system_name": "Smart AI-Based Delivery Package Damage Detection & Risk Prediction",
            "report_type": "Automated Package Quality & Delivery Inspection Report",
            "package_id": package_id,
            "timestamp": timestamp,
            "image_filename": image_name,
            "annotated_image": annotated_image_path or "N/A",
        },
        "detection_summary": {
            "total_defects_detected": len(defect_features),
            "damage_classes_found": list({f["damage_class"] for f in defect_features}),
            "total_damage_coverage_ratio": severity_info.get("total_coverage_ratio", 0.0),
            "total_damage_coverage_percentage": f"{severity_info.get('total_coverage_ratio', 0.0) * 100:.2f}%",
        },
        "detected_defects": defect_features,
        "severity_assessment": {
            "method": severity_info.get("assessment_method", "Heuristic Damage Severity Assessment"),
            "severity_level": severity_info.get("severity_level", "No Damage"),
            "severity_score_normalized": severity_info.get("severity_score_normalized", 0.0),
            "rationale": severity_info.get("rationale", ""),
        },
        "risk_prediction": {
            "metric_name": risk_info.get("risk_metric_name", "Estimated Internal Damage Risk Score"),
            "risk_score": risk_info.get("risk_score", 0.0),
            "score_scale": risk_info.get("score_range", "0 - 100"),
            "delivery_decision": risk_info.get("delivery_decision", "SAFE TO DELIVER"),
            "recommended_action": risk_info.get("decision_action", ""),
            "decision_rule_reference": risk_info.get("decision_thresholds", {}),
        },
        "disclaimer": (
            "NOTICE: The severity level and internal damage risk score are computed via algorithmic "
            "visual heuristics from external package surface features. They are not statistically calibrated "
            "physical stress simulations."
        ),
    }

    if save_to_disk:
        OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_file = OUTPUT_REPORTS_DIR / "inspection_report.json"
        txt_file = OUTPUT_REPORTS_DIR / "inspection_report.txt"

        save_json(report_data, json_file)

        # Build formatted text report
        txt_lines = [
            "=" * 70,
            "     PACKAGE DAMAGE DETECTION & RISK PREDICTION INSPECTION REPORT",
            "=" * 70,
            f"Package ID         : {package_id}",
            f"Date / Time        : {timestamp}",
            f"Image Analyzed     : {image_name}",
            f"Annotated Result   : {annotated_image_path or 'N/A'}",
            "-" * 70,
            "1. DAMAGE DETECTION SUMMARY",
            f"   Total Defects   : {len(defect_features)}",
            f"   Damage Types    : {', '.join(report_data['detection_summary']['damage_classes_found']) if defect_features else 'None'}",
            f"   Damage Coverage : {report_data['detection_summary']['total_damage_coverage_percentage']}",
            "",
        ]

        if defect_features:
            txt_lines.append("2. DEFECT BREAKDOWN:")
            for d in defect_features:
                txt_lines.append(
                    f"   - Defect #{d['defect_id']}: {d['damage_class'].upper()} | "
                    f"Conf: {d['confidence']:.2f} | Loc: {d['location_combined']} | "
                    f"Area: {d['damage_area_percentage']}% | Box: {d['bbox']}"
                )
            txt_lines.append("")
        else:
            txt_lines.append("2. DEFECT BREAKDOWN: No external damage detected.\n")

        txt_lines.extend([
            "3. DAMAGE SEVERITY ASSESSMENT (Heuristic)",
            f"   Severity Level  : {report_data['severity_assessment']['severity_level']}",
            f"   Rationale       : {report_data['severity_assessment']['rationale']}",
            "",
            "4. INTERNAL DAMAGE RISK & DELIVERY DECISION",
            f"   Risk Metric     : {report_data['risk_prediction']['metric_name']}",
            f"   Risk Score      : {report_data['risk_prediction']['risk_score']} / 100",
            f"   DECISION        : >>> {report_data['risk_prediction']['delivery_decision']} <<<",
            f"   Action Required : {report_data['risk_prediction']['recommended_action']}",
            "-" * 70,
            "Threshold Reference: <25: Safe | 25-65: Inspect Before Delivery | >65: Replace Package",
            "=" * 70,
        ])

        save_text("\n".join(txt_lines), txt_file)
        logger.info(f"Inspection reports saved to {json_file} and {txt_file}")

    # Optional Firebase Logging
    log_to_firebase_if_configured(report_data)

    return report_data


def log_to_firebase_if_configured(report_data: Dict[str, Any]) -> bool:
    """
    Attempts to log inspection records to Google Cloud Firebase Firestore if credentials exist.
    Continues gracefully without failing if Firebase is unconfigured.
    """
    service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("FIREBASE_CREDENTIALS_PATH")
    
    if not service_account_path or not Path(service_account_path).exists():
        logger.info("Firebase logging skipped because Firebase credentials are not configured.")
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        doc_ref = db.collection("package_inspections").document(report_data["header"]["package_id"])
        doc_ref.set(report_data)
        logger.info(f"Successfully logged inspection report to Firestore collection 'package_inspections'.")
        return True
    except Exception as e:
        logger.warning(f"Firebase logging attempted but encountered error: {e}. Continuing pipeline normally.")
        return False
