# Smart AI-Based Delivery Package Damage Detection and Internal Damage Risk Prediction System

An end-to-end computer vision and intelligent risk assessment pipeline designed for automated logistics inspection. The system processes package images, detects physical package damage types and bounding boxes using YOLOv8, extracts spatial/geometric visual features, computes a heuristic damage severity score, evaluates an Estimated Internal Damage Risk Score (0–100), and outputs actionable delivery decisions alongside structured inspection reports and optional cloud synchronization.

---

## Table of Contents
1. [Project Overview & Pipeline Architecture](#project-overview--pipeline-architecture)
2. [Project Structure](#project-structure)
3. [Environment & Hardware Detection](#environment--hardware-detection)
4. [Installation & Setup](#installation--setup)
5. [Kaggle Authentication & Dataset Download](#kaggle-authentication--dataset-download)
6. [Dataset Inspection & Validation](#dataset-inspection--validation)
7. [YOLOv8 Model Training](#yolov8-model-training)
8. [Model Validation & Actual Metrics](#model-validation--actual-metrics)
9. [Image-Based Feature Extraction](#image-based-feature-extraction)
10. [Damage Severity Assessment (Heuristic)](#damage-severity-assessment-heuristic)
11. [Estimated Internal Damage Risk & Delivery Decisions](#estimated-internal-damage-risk--delivery-decisions)
12. [Inspection Reports & Cloud Logging](#inspection-reports--cloud-logging)
13. [CLI & Custom Image Inference Guide](#cli--custom-image-inference-guide)
14. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview & Pipeline Architecture

```
Package Image (Camera / Stream / Upload)
      │
      ▼
Image Preprocessing & Normalization
      │
      ▼
YOLOv8 Damage Detection (yolov8n.pt)
      │
      ▼
Defect Bounding Boxes + Damage Classes + Confidence Scores
      │
      ▼
Image-Based Geometric Feature Extraction (Area Ratio, Aspect Ratio, Spatial Coordinates)
      │
      ▼
Heuristic Damage Severity Assessment (No Damage / Minor / Moderate / Severe)
      │
      ▼
Estimated Internal Damage Risk Score (0 – 100 Index)
      │
      ▼
Automated Delivery Decision (SAFE TO DELIVER / INSPECT BEFORE DELIVERY / REPLACE PACKAGE)
      │
      ▼
Human-Readable & JSON Inspection Reports (`outputs/reports/`)
      │
      ▼
Optional Firebase Firestore Cloud Logging
```

---

## 2. Project Structure

```
.
├── data/
│   ├── raw/                  # Downloaded raw zip archives
│   ├── extracted/            # Extracted dataset files (train, val, test, yaml)
│   └── processed/            # Normalized YOLO dataset configuration (dataset.yaml)
│
├── models/
│   └── weights/              # Best trained YOLOv8 model weights (best.pt)
│
├── runs/
│   └── train/                # Raw Ultralytics training runs, logs, and loss curves
│
├── reports/
│   ├── dataset_report.json   # Verified dataset split counts and data quality metrics
│   ├── dataset_report.txt    # Human-readable dataset quality report
│   ├── model_metrics.json    # Verified precision, recall, mAP@0.5, mAP@0.5:0.95
│   └── model_metrics.txt     # Human-readable validation summary
│
├── outputs/
│   ├── predictions/          # Annotated inspection images with telemetry overlay
│   ├── reports/              # Inspection reports (JSON and TXT)
│   └── visualizations/       # Training loss curves, confusion matrices, PR/F1 curves
│
├── src/
│   ├── utils.py              # Hardware detection, path constants, logging, serialization
│   ├── download_dataset.py   # Kaggle authentication, automated download and extraction
│   ├── inspect_dataset.py    # Recursive dataset audit, label verification, YAML generation
│   ├── train.py              # YOLOv8n training pipeline with OOM auto-recovery
│   ├── validate.py           # Model evaluation and metric extraction without fabrication
│   ├── severity.py           # Bounding-box spatial feature extraction & severity heuristic
│   ├── risk.py               # Estimated Internal Damage Risk scoring & delivery rules
│   ├── report.py             # Inspection report generation & Firestore cloud logging
│   └── inference.py          # End-to-end prediction engine for custom & batch images
│
├── sample_images/            # Directory for sample test images
├── main.py                   # Master orchestration script
├── requirements.txt          # Python dependencies
└── README.md                 # Complete system documentation
```

---

## 3. Environment & Hardware Detection

The system dynamically detects whether an NVIDIA CUDA GPU or CPU is available:
- **PyTorch Version**: Verified at runtime.
- **CUDA Device**: Selected automatically if available (`device='0'`); falls back cleanly to CPU (`device='cpu'`) if unavailable.
- **Out-of-Memory (OOM) Recovery**: Automatically adjusts batch sizes (`auto` -> `16` -> `8` -> `4`) if GPU VRAM limits are reached.

---

## 4. Installation & Setup

Ensure Python 3.10+ is installed. Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- `ultralytics>=8.0.0` (YOLOv8 framework)
- `torch`, `torchvision` (Deep learning engine)
- `opencv-python`, `Pillow` (Image processing and telemetry overlay)
- `numpy`, `pandas`, `scikit-learn` (Numerical and evaluation utilities)
- `matplotlib`, `seaborn` (Visualizations)
- `PyYAML` (Configuration management)
- `kaggle` (Automated dataset ingestion)

---

## 5. Kaggle Authentication & Dataset Download

The automated dataset downloader requires Kaggle API credentials.

### Setup Instructions:
1. Navigate to [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Click **Create New Token** to download `kaggle.json`.
3. Place `kaggle.json` in your home directory:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
4. *Alternatively*, set system environment variables:
   ```bash
   # Windows (cmd/powershell)
   set KAGGLE_USERNAME=your_username
   set KAGGLE_KEY=your_api_key
   ```

### Download Execution:
```bash
python src/download_dataset.py
```
- Downloads the dataset into `data/raw/`.
- Extracts all folders into `data/extracted/`.
- Automatically handles primary and verified fallback package damage datasets.

---

## 6. Dataset Inspection & Validation

Audits all image files, corresponding YOLO `.txt` labels, bounding-box coordinate ranges, and class distributions:

```bash
python src/inspect_dataset.py
```

Outputs:
- `reports/dataset_report.json`
- `reports/dataset_report.txt`
- `data/processed/dataset.yaml`

---

## 7. YOLOv8 Model Training

Trains a lightweight `YOLOv8n` model for 50 epochs:

```bash
python src/train.py --epochs 50 --batch -1
```

Features:
- Auto-downloads `yolov8n.pt` pretrained weights.
- Early stopping with `patience=15`.
- Automatic batch sizing with CUDA OOM retry mechanism.
- Saves the final optimal weights to `models/weights/best.pt`.
- Copies training losses, confusion matrix, and PR curves to `outputs/visualizations/`.

---

## 8. Model Validation & Actual Metrics

Evaluates `models/weights/best.pt` against the validation split:

```bash
python src/validate.py
```

Outputs:
- `reports/model_metrics.json`
- `reports/model_metrics.txt`

Metrics reported:
- **Precision (B)**
- **Recall (B)**
- **mAP@0.50 (B)**
- **mAP@0.50:0.95 (B)**
- **Inference Latency (ms)**
- **Per-Class Breakdown**

---

## 9. Image-Based Feature Extraction

For every detected damage instance, the pipeline extracts:
1. **Damage Class & Confidence**: Model prediction probabilities.
2. **Bounding Box Coordinates**: `[xmin, ymin, xmax, ymax]`.
3. **Dimensions & Area**: Box width, height, and area in pixels.
4. **Damage Area Ratio**: `bbox_area / image_area` and total non-overlapping package coverage ratio.
5. **Spatial Placement**: Qualitative coordinates (`Top-Left`, `Center`, `Bottom-Right`, etc.).

---

## 10. Damage Severity Assessment (Heuristic)

> **Notice**: Severity classification is derived via transparent geometric and class heuristics from external observable features.

Levels:
- **No Damage**: No defects detected.
- **Minor**: Small surface flaws (<4% coverage, single defect).
- **Moderate**: Moderate tears, dents, or multiple defects (4% - 15% coverage).
- **Severe**: Significant structural collapse, punctures, or large-area crushing (>15% coverage).

---

## 11. Estimated Internal Damage Risk & Delivery Decisions

The **Estimated Internal Damage Risk Score (0–100)** is an engineered index reflecting external risk indicators:

$$\text{Risk Score} = \text{Base Severity Pts} + \text{Area Coverage Pts} + \text{Defect Count Pts} + \text{Vulnerability Pts}$$

### Delivery Decision Rules:
| Risk Score | Delivery Decision | Recommended Action |
| :--- | :--- | :--- |
| **< 25** | **SAFE TO DELIVER** | Proceed with direct recipient delivery. |
| **25 – 65** | **INSPECT BEFORE DELIVERY** | Flag for visual verification by courier/warehouse before handover. |
| **> 65** | **REPLACE PACKAGE** | Halt delivery; return to hub for item replacement or internal check. |

---

## 12. Inspection Reports & Cloud Logging

Reports are generated in JSON and TXT formats in `outputs/reports/inspection_report.json`.

### Optional Firebase Firestore Integration:
If `GOOGLE_APPLICATION_CREDENTIALS` or `FIREBASE_CREDENTIALS_PATH` environment variable points to a valid service account JSON, inspection records are logged to Firestore under collection `package_inspections`. If not configured, the system skips logging gracefully without failing.

---

## 13. CLI & Custom Image Inference Guide

### Full Pipeline Run (Standard Full Dataset):
```bash
python main.py
```

### Single Package Inspection Mode (Specialized Dataset & Targeting):
To train and validate specifically on isolated single-package delivery images:
```bash
# Train on curated single-package subset (695 images) with model scaling
python main.py --single-package --model yolov8s.pt --epochs 50 --batch 16
```

### Model Scaling Options:
You can choose from multiple YOLOv8 architectures depending on your compute and accuracy requirements:
* `yolov8n.pt` (Nano - 3.2M params, fastest inference ~43ms)
* `yolov8s.pt` (Small - 11.2M params, higher feature capacity)
* `yolov8m.pt` (Medium - 25.9M params, deep feature extraction)

```bash
# Train with YOLOv8 Small on Single Package dataset
python src/train.py --single-package --model yolov8s.pt --epochs 50 --batch 16
```

### Single Package Image Inference CLI:
```bash
# Test on a single crushed package
python src/inference.py --image sample_images/single_crushed_package.jpg --conf 0.30

# Test on a single damaged parcel
python src/inference.py --image sample_images/single_damaged_package.jpg --conf 0.25
```

Or programmatically in Python:
```python
from src.inference import PackageDamagePredictor

predictor = PackageDamagePredictor(model_path="models/weights/best.pt", conf_threshold=0.30)
result = predictor.predict_image(
    image_input="sample_images/single_damaged_package.jpg",
    save_annotated=True,
    save_report=True,
    package_id="PKG-SINGLE-001"
)
print("Severity:", result["severity_info"]["severity_level"])
print("Risk Score:", result["risk_info"]["risk_score"])
print("Delivery Decision:", result["risk_info"]["delivery_decision"])
```

---

## 14. Troubleshooting

1. **Kaggle 403 Forbidden / Missing Credentials**:
   - Verify `kaggle.json` exists in `~/.kaggle/` and permissions are valid.
2. **CUDA Out of Memory**:
   - The training script automatically retries with smaller batch sizes (`16`, `8`, `4`).
3. **No GPU Detected**:
   - The pipeline runs seamlessly on CPU.
