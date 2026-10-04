"""
Inference & Inspection Pipeline Module.
Performs end-to-end detection, feature extraction, severity calculation,
risk scoring, and annotated prediction image generation for custom and validation images.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    WEIGHTS_DIR,
    PREDICTIONS_DIR,
    OUTPUT_REPORTS_DIR,
    SAMPLE_IMAGES_DIR,
    get_logger,
)
from src.severity import (
    extract_defect_features,
    calculate_total_damage_coverage,
    assess_damage_severity,
)
from src.risk import calculate_internal_damage_risk
from src.report import generate_inspection_report

logger = get_logger("InferencePipeline")


class PackageDamagePredictor:
    """
    End-to-End Package Damage Inference and Risk Engine.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, conf_threshold: float = 0.25):
        from ultralytics import YOLO

        if model_path is None:
            model_path = WEIGHTS_DIR / "best.pt"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}. Train the model first.")

        logger.info(f"Loading YOLOv8 model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_dir: Optional[Path] = None,
        save_annotated: bool = True,
        save_report: bool = True,
        package_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs full inference pipeline on a single image.
        """
        if output_dir is None:
            output_dir = PREDICTIONS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            image_name = img_path.name
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError(f"Failed to read image file: {img_path}")
        else:
            img_bgr = image_input
            image_name = f"image_{np.random.randint(1000, 9999)}.jpg"

        img_h, img_w = img_bgr.shape[:2]

        # 2. Run YOLOv8 detection
        results = self.model.predict(
            source=img_bgr,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        # 3. Parse detections
        boxes_list = []
        classes_list = []
        confs_list = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())

                boxes_list.append(xyxy)
                classes_list.append(cls_name)
                confs_list.append(conf)

        # 4. Extract features
        defect_features = extract_defect_features(
            boxes=boxes_list,
            classes=classes_list,
            confidences=confs_list,
            image_shape=(img_h, img_w),
        )

        # 5. Calculate damage coverage & severity
        total_coverage = calculate_total_damage_coverage(boxes_list, (img_h, img_w))
        severity_info = assess_damage_severity(defect_features, total_coverage)

        # 6. Calculate internal damage risk & delivery decision
        risk_info = calculate_internal_damage_risk(severity_info, defect_features)

        # 7. Draw rich annotated image
        annotated_path_str = None
        if save_annotated:
            annotated_img = self.draw_inspection_overlay(
                img_bgr=img_bgr.copy(),
                defect_features=defect_features,
                severity_info=severity_info,
                risk_info=risk_info,
            )
            out_filename = f"pred_{Path(image_name).stem}.jpg"
            out_file_path = output_dir / out_filename
            cv2.imwrite(str(out_file_path), annotated_img)
            annotated_path_str = str(out_file_path)
            logger.info(f"Annotated prediction image saved to {out_file_path}")

        # 8. Generate inspection report
        report_data = generate_inspection_report(
            image_name=image_name,
            defect_features=defect_features,
            severity_info=severity_info,
            risk_info=risk_info,
            annotated_image_path=annotated_path_str,
            package_id=package_id,
            save_to_disk=save_report,
        )

        return {
            "image_name": image_name,
            "annotated_image_path": annotated_path_str,
            "defect_features": defect_features,
            "severity_info": severity_info,
            "risk_info": risk_info,
            "inspection_report": report_data,
        }

    def draw_inspection_overlay(
        self,
        img_bgr: np.ndarray,
        defect_features: List[Dict[str, Any]],
        severity_info: Dict[str, Any],
        risk_info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Draws professional bounding boxes and inspection telemetry overlay onto the image.
        """
        img_h, img_w = img_bgr.shape[:2]
        canvas = img_bgr.copy()

        # Severity Colors (BGR)
        color_map = {
            "SAFE TO DELIVER": (0, 180, 0),        # Vibrant Green
            "INSPECT BEFORE DELIVERY": (0, 165, 255),  # Orange/Amber
            "REPLACE PACKAGE": (0, 0, 220),       # Bright Red
        }
        decision = risk_info.get("delivery_decision", "SAFE TO DELIVER")
        theme_color = color_map.get(decision, (0, 180, 0))

        # 1. Draw Bounding Boxes
        for d in defect_features:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cls_name = d["damage_class"]
            conf = d["confidence"]

            # Box outline
            cv2.rectangle(canvas, (x1, y1), (x2, y2), theme_color, 2)

            # Label banner
            label_text = f"{cls_name.upper()} {conf:.2f} ({d['location_combined']})"
            font_scale = max(0.45, min(0.7, img_w / 1200))
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Label background
            cv2.rectangle(
                canvas,
                (x1, max(0, y1 - th - 8)),
                (x1 + tw + 6, max(0, y1)),
                theme_color,
                -1,
            )
            cv2.putText(
                canvas,
                label_text,
                (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # 2. Header Telemetry Banner
        header_h = max(55, int(img_h * 0.10))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, header_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # Telemetry text
        risk_score = risk_info.get("risk_score", 0.0)
        sev_level = severity_info.get("severity_level", "No Damage")
        coverage_pct = severity_info.get("total_coverage_ratio", 0.0) * 100

        line1 = f"AI DAMAGE INSPECTOR | Severity: {sev_level} | Damage Area: {coverage_pct:.1f}%"
        line2 = f"Risk Score: {risk_score}/100 | DECISION: {decision}"

        font_scale = max(0.40, min(0.65, img_w / 1000))
        cv2.putText(canvas, line1, (12, int(header_h * 0.40)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (12, int(header_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, theme_color, 2, cv2.LINE_AA)

        return canvas


def run_validation_predictions(
    model_path: Optional[Path] = None,
    val_images_dir: Optional[Path] = None,
    num_samples: int = 8,
) -> List[Dict[str, Any]]:
    """
    Runs inference on a collection of validation images and stores outputs in outputs/predictions/.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictor = PackageDamagePredictor(model_path=model_path)

    # Locate validation images
    if val_images_dir is None or not Path(val_images_dir).exists():
        from src.utils import EXTRACTED_DATA_DIR
        val_imgs = list(EXTRACTED_DATA_DIR.rglob("*.jpg")) + list(EXTRACTED_DATA_DIR.rglob("*.png"))
    else:
        val_imgs = list(Path(val_images_dir).glob("*.jpg")) + list(Path(val_images_dir).glob("*.png"))

    if not val_imgs:
        logger.warning("No validation images found for batch prediction visualization.")
        return []

    sample_imgs = val_imgs[:num_samples]
    logger.info(f"Running validation predictions on {len(sample_imgs)} sample images...")

    results = []
    for img_p in sample_imgs:
        res = predictor.predict_image(img_p, output_dir=PREDICTIONS_DIR, save_annotated=True, save_report=False)
        results.append(res)

    logger.info(f"Generated {len(results)} validation prediction images in {PREDICTIONS_DIR}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Package Damage Detection & Risk Inspection Engine")
    parser.add_argument("--image", type=str, default="sample_images/single_damaged_package.jpg", help="Path to package image")
    parser.add_argument("--model", type=str, default=str(WEIGHTS_DIR / "best.pt"), help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--package-id", type=str, default="PKG-INSPECT-001", help="Package tracking ID")
    args = parser.parse_args()

    predictor = PackageDamagePredictor(model_path=args.model, conf_threshold=args.conf)
    if Path(args.image).exists():
        res = predictor.predict_image(args.image, package_id=args.package_id)
        print("\n" + "=" * 65)
        print(f"Image Tested      : {args.image}")
        print(f"Defects Detected  : {len(res['defect_features'])}")
        print(f"Severity Level    : {res['severity_info']['severity_level']}")
        print(f"Risk Score        : {res['risk_info']['risk_score']}/100")
        print(f"DELIVERY DECISION : {res['risk_info']['delivery_decision']}")
        print(f"Action Required   : {res['risk_info']['decision_action']}")
        print(f"Annotated Image   : {res['annotated_image_path']}")
        print("=" * 65)
    else:
        print(f"Image file not found: {args.image}. Please provide a valid image path.")
