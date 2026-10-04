"""
Stage 1 Package Isolation & Crop Generation Test Script.
Runs Package Detector model, applies Quality Gate validation,
generates individual package crops, and creates visual contact sheet preview.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

test_img_path = Path("sample_images/multi_package_warehouse.jpg")
output_crops_dir = Path("outputs/package_crops")
output_pkg_dir = Path("outputs/package_detector")

output_crops_dir.mkdir(parents=True, exist_ok=True)
output_pkg_dir.mkdir(parents=True, exist_ok=True)

model = YOLO("models/weights/package_detector_best.pt")
img = cv2.imread(str(test_img_path))
h, w = img.shape[:2]
img_area = w * h

results = model(str(test_img_path), conf=0.25, iou=0.45)[0]

print("=" * 60)
print(f"MULTI-PACKAGE DETECTION TEST ON: {test_img_path.name} ({w}x{h})")
print(f"Total Raw Detections: {len(results.boxes)}")

crops = []
valid_detections = []
invalid_detections = []

vis_img = img.copy()

for i, box in enumerate(results.boxes):
    xyxy = box.xyxy[0].cpu().numpy().tolist()
    conf = float(box.conf[0].cpu().numpy())
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    
    # Clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    bw = x2 - x1
    bh = y2 - y1
    box_area = bw * bh
    area_ratio = box_area / img_area
    area_ratio_pct = area_ratio * 100
    
    is_valid = (area_ratio <= 0.80) and (bw > 10) and (bh > 10)
    status_str = "VALID" if is_valid else "INVALID"
    
    pkg_id = f"Package {i+1}"
    print(f"[{pkg_id}] Bounding Box: [{x1}, {y1}, {x2}, {y2}] | Area: {area_ratio_pct:.2f}% | Conf: {conf:.2f} | Status: {status_str}")
    
    # Crop
    crop = img[y1:y2, x1:x2]
    crop_path = output_crops_dir / f"package_{i+1:03d}.jpg"
    cv2.imwrite(str(crop_path), crop)
    crops.append(crop)
    
    if is_valid:
        valid_detections.append(pkg_id)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        lbl = f"{pkg_id} ({conf:.2f}, {area_ratio_pct:.1f}%)"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, max(th + 4, y1)), (0, 200, 0), -1)
        cv2.putText(vis_img, lbl, (x1 + 2, max(th + 2, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        invalid_detections.append(pkg_id)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Save annotated prediction preview
pred_preview_path = output_pkg_dir / "multi_package_prediction_preview.jpg"
cv2.imwrite(str(pred_preview_path), vis_img)

# Create Individual Package Crops Contact Sheet
resized_crops = []
for idx, c in enumerate(crops):
    ch, cw = c.shape[:2]
    scale = min(220 / cw, 220 / ch)
    nw, nh = int(cw * scale), int(ch * scale)
    scaled = cv2.resize(c, (nw, nh))
    
    canvas = np.zeros((260, 260, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.rectangle(canvas, (0, 0), (259, 259), (70, 70, 70), 1)
    
    ox = (260 - nw) // 2
    oy = (260 - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = scaled
    
    cv2.putText(canvas, f"Package {idx+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
    resized_crops.append(canvas)

crops_sheet = np.hstack(resized_crops)
crops_preview_path = output_pkg_dir / "individual_package_crops_preview.jpg"
cv2.imwrite(str(crops_preview_path), crops_sheet)

print("=" * 60)
print(f"Valid Individual Package Detections   : {len(valid_detections)}")
print(f"Invalid / Oversized Detections        : {len(invalid_detections)}")
print(f"Saved prediction overlay to           : {pred_preview_path}")
print(f"Saved crops contact sheet to          : {crops_preview_path}")
print("=" * 60)
