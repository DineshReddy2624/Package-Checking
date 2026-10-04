"""
Stage 2 Damage Detector Validation Script.
Inspects damage detector model architecture, weights, class names,
and runs damage detection on all isolated package crops.
"""

import sys
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.severity import (
    extract_defect_features,
    calculate_total_damage_coverage,
    assess_damage_severity,
)
from src.risk import calculate_internal_damage_risk

dmg_model_path = project_root / "models" / "weights" / "damage_detector_best.pt"
print("=" * 70)
print(f"STAGE 2 MODEL INSPECTION: {dmg_model_path}")
print("=" * 70)

model = YOLO(str(dmg_model_path))
print(f"Task        : {model.task}")
print(f"Class Names : {model.names}")
print(f"Total Params: {sum(p.numel() for p in model.model.parameters()):,}")

crops = sorted(glob.glob("outputs/package_crops/package_*.jpg"))
print("\n" + "=" * 70)
print(f"RUNNING STAGE 2 DAMAGE DETECTION ON {len(crops)} CROPS:")
print("=" * 70)

for cp in crops:
    cpath = Path(cp)
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    
    res = model(str(cpath), conf=0.25, verbose=False)[0]
    
    boxes = []
    classes = []
    confs = []
    
    for b in res.boxes:
        xyxy = [float(v) for v in b.xyxy[0].tolist()]
        cid = int(b.cls[0])
        cname = model.names.get(cid, f"class_{cid}")
        conf = float(b.conf[0])
        boxes.append(xyxy)
        classes.append(cname)
        confs.append(conf)
        
    defect_features = extract_defect_features(boxes, classes, confs, (h, w))
    coverage = calculate_total_damage_coverage(boxes, (h, w))
    sev_info = assess_damage_severity(defect_features, coverage)
    risk_info = calculate_internal_damage_risk(sev_info, defect_features)
    
    print(f"\n📦 [{cpath.stem.upper()}] ({w}x{h} px)")
    print(f"   • Defects Detected : {len(boxes)}")
    if len(boxes) == 0:
        print("   • Damage Class     : None (No externally visible damage detected)")
        print("   • Damage Confidence: N/A")
        print("   • Bounding Box     : None")
        print(f"   • Damage Area      : 0.00%")
    else:
        for df in defect_features:
            print(f"   • Damage Class     : {df['damage_class']}")
            print(f"   • Damage Confidence: {df['confidence']:.4f}")
            print(f"   • Bounding Box     : {df['bbox']}")
            print(f"   • Damage Area      : {df['damage_area_percentage']:.2f}%")
            
    print(f"   • Severity Level   : {sev_info['severity_level']}")
    print(f"   • Heuristic Risk   : {risk_info['risk_score']} / 100")
    print(f"   • Decision         : {risk_info['delivery_decision']}")
print("=" * 70)
