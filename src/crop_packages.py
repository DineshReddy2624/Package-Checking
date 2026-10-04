"""
Package Cropping Module.
Crops each individual detected package bounding box from the scene
with a configurable margin, clamped to image dimensions.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import OUTPUTS_DIR, get_logger

logger = get_logger("PackageCropper")
PACKAGE_CROPS_DIR = OUTPUTS_DIR / "package_crops"


def crop_individual_packages(
    image_input: Union[str, Path, np.ndarray],
    package_boxes: List[List[float]],
    confidences: Optional[List[float]] = None,
    margin_pct: float = 0.05,
    output_dir: Optional[Path] = None,
    save_to_disk: bool = True,
) -> List[Dict[str, Any]]:
    """
    Crops every detected package separately with a margin, clamped to image dimensions.
    
    Returns list of dicts with:
    - package_id: 'Package 1', 'Package 2', ...
    - crop_filename: 'package_001.jpg', ...
    - crop_path: absolute path on disk
    - crop_bgr: numpy ndarray
    - original_bbox: [x1, y1, x2, y2]
    - crop_bbox: [cx1, cy1, cx2, cy2] (including margin)
    - confidence: float
    """
    if output_dir is None:
        output_dir = PACKAGE_CROPS_DIR

    if save_to_disk:
        output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_input, (str, Path)):
        img_p = Path(image_input)
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_p}")
    else:
        img_bgr = image_input

    img_h, img_w = img_bgr.shape[:2]
    crops_info = []

    for i, box in enumerate(package_boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1

        # Calculate margin in pixels
        mx = bw * margin_pct
        my = bh * margin_pct

        # Clamped coordinates
        cx1 = max(0, int(np.floor(x1 - mx)))
        cy1 = max(0, int(np.floor(y1 - my)))
        cx2 = min(img_w, int(np.ceil(x2 + mx)))
        cy2 = min(img_h, int(np.ceil(y2 + my)))

        # Ensure valid crop area
        if cx2 <= cx1 or cy2 <= cy1:
            logger.warning(f"Invalid crop dimensions for box {box}: ({cx1}, {cy1}, {cx2}, {cy2})")
            continue

        crop_bgr = img_bgr[cy1:cy2, cx1:cx2].copy()
        pkg_num = i + 1
        pkg_id = f"Package {pkg_num}"
        crop_filename = f"package_{pkg_num:03d}.jpg"
        crop_path = output_dir / crop_filename

        if save_to_disk:
            cv2.imwrite(str(crop_path), crop_bgr)

        conf = confidences[i] if confidences and i < len(confidences) else 1.0

        crops_info.append({
            "package_id": pkg_id,
            "package_index": pkg_num,
            "crop_filename": crop_filename,
            "crop_path": str(crop_path) if save_to_disk else None,
            "crop_bgr": crop_bgr,
            "original_bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "crop_bbox_clamped": [cx1, cy1, cx2, cy2],
            "crop_width": cx2 - cx1,
            "crop_height": cy2 - cy1,
            "detector_confidence": round(float(conf), 4),
        })

    logger.info(f"Cropped {len(crops_info)} individual packages into {output_dir}")
    return crops_info
