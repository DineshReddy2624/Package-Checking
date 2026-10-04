"""
Image-Based Feature Extraction & Heuristic Damage Severity Assessment Module.
Extracts geometric and spatial features from YOLO bounding boxes and assesses damage severity.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def determine_spatial_location(
    x_center: float, y_center: float, img_w: int, img_h: int
) -> Dict[str, str]:
    """
    Determines qualitative horizontal and vertical spatial location of a defect.
    """
    rel_x = x_center / max(1, img_w)
    rel_y = y_center / max(1, img_h)

    # Horizontal
    if rel_x < 0.33:
        h_loc = "Left"
    elif rel_x > 0.66:
        h_loc = "Right"
    else:
        h_loc = "Center"

    # Vertical
    if rel_y < 0.33:
        v_loc = "Top"
    elif rel_y > 0.66:
        v_loc = "Bottom"
    else:
        v_loc = "Middle"

    return {
        "horizontal": h_loc,
        "vertical": v_loc,
        "combined": f"{v_loc}-{h_loc}",
    }


def extract_defect_features(
    boxes: List[List[float]],
    classes: List[str],
    confidences: List[float],
    image_shape: Tuple[int, int],  # (height, width)
) -> List[Dict[str, Any]]:
    """
    Extracts geometric, spatial, and probabilistic features from detected bounding boxes.
    """
    img_h, img_w = image_shape
    img_area = float(img_w * img_h)
    features = []

    for i, (box, cls_name, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        bbox_area = float(bw * bh)
        area_ratio = bbox_area / max(1.0, img_area)

        x_center = x1 + bw / 2.0
        y_center = y1 + bh / 2.0

        loc = determine_spatial_location(x_center, y_center, img_w, img_h)

        feat = {
            "defect_id": i + 1,
            "damage_class": cls_name,
            "confidence": round(float(conf), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_width": round(bw, 1),
            "bbox_height": round(bh, 1),
            "bbox_area": round(bbox_area, 1),
            "image_width": int(img_w),
            "image_height": int(img_h),
            "damage_area_ratio": round(area_ratio, 4),
            "damage_area_percentage": round(area_ratio * 100, 2),
            "location_horizontal": loc["horizontal"],
            "location_vertical": loc["vertical"],
            "location_combined": loc["combined"],
        }
        features.append(feat)

    return features


def calculate_total_damage_coverage(
    boxes: List[List[float]], image_shape: Tuple[int, int]
) -> float:
    """
    Calculates non-overlapping damage coverage ratio across the image surface using a binary grid.
    """
    if not boxes:
        return 0.0

    img_h, img_w = image_shape
    grid_h, grid_w = min(100, img_h), min(100, img_w)
    mask = np.zeros((grid_h, grid_w), dtype=bool)

    for box in boxes:
        x1, y1, x2, y2 = box
        gx1 = int(np.clip((x1 / img_w) * grid_w, 0, grid_w))
        gy1 = int(np.clip((y1 / img_h) * grid_h, 0, grid_h))
        gx2 = int(np.clip(np.ceil((x2 / img_w) * grid_w), 0, grid_w))
        gy2 = int(np.clip(np.ceil((y2 / img_h) * grid_h), 0, grid_h))
        mask[gy1:gy2, gx1:gx2] = True

    coverage_ratio = float(np.count_nonzero(mask)) / float(grid_h * grid_w)
    return round(coverage_ratio, 4)


def assess_damage_severity(
    extracted_features: List[Dict[str, Any]],
    total_coverage_ratio: float,
) -> Dict[str, Any]:
    """
    Heuristic Damage Severity Assessment.
    Derived transparently from observable defect features:
    - Number of defects
    - Total damage coverage ratio
    - Maximum confidence
    - Defect classes (crushed/puncture/tear vs surface mark/scratch)

    Returns severity classification ('No Damage', 'Minor', 'Moderate', 'Severe')
    with clear heuristic rationale.
    """
    num_defects = len(extracted_features)

    if num_defects == 0:
        return {
            "assessment_method": "Heuristic Damage Severity Assessment",
            "severity_level": "No Damage",
            "num_defects": 0,
            "total_coverage_ratio": 0.0,
            "severity_score_normalized": 0.0,
            "rationale": "No structural or surface defects detected on external package packaging.",
        }

    max_conf = max(f["confidence"] for f in extracted_features)
    max_single_area = max(f["damage_area_ratio"] for f in extracted_features)

    # Calculate structural severity weighting based on class keywords
    structural_weight = 1.0
    critical_classes = {"crush", "crushed", "puncture", "hole", "break", "broken", "tear", "dent", "defective", "damage"}
    has_critical_type = any(
        any(k in f["damage_class"].lower() for k in critical_classes)
        for f in extracted_features
    )
    if has_critical_type:
        structural_weight = 1.25

    # Heuristic Severity score (0.0 to 1.0)
    # 40% based on coverage, 30% based on defect count, 20% on max single defect, 10% on confidence
    count_factor = min(1.0, num_defects / 4.0)
    coverage_factor = min(1.0, total_coverage_ratio / 0.25)
    single_area_factor = min(1.0, max_single_area / 0.15)
    conf_factor = max_conf

    raw_score = (
        (0.40 * coverage_factor)
        + (0.30 * count_factor)
        + (0.20 * single_area_factor)
        + (0.10 * conf_factor)
    ) * structural_weight

    severity_score = min(1.0, raw_score)

    # Classification boundaries
    if severity_score < 0.25 and total_coverage_ratio < 0.04 and num_defects == 1:
        severity_level = "Minor"
        rationale = f"Minor surface flaw ({num_defects} defect, {total_coverage_ratio*100:.1f}% area coverage)."
    elif severity_score < 0.60:
        severity_level = "Moderate"
        rationale = f"Moderate damage detected ({num_defects} defects, {total_coverage_ratio*100:.1f}% area coverage)."
    else:
        severity_level = "Severe"
        rationale = f"Severe external damage ({num_defects} defects, {total_coverage_ratio*100:.1f}% area coverage, structural impact observed)."

    return {
        "assessment_method": "Heuristic Damage Severity Assessment",
        "severity_level": severity_level,
        "num_defects": num_defects,
        "total_coverage_ratio": total_coverage_ratio,
        "severity_score_normalized": round(severity_score, 4),
        "rationale": rationale,
    }
