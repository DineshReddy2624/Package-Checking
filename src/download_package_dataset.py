"""
Package Detector Dataset Downloader.
Handles Roboflow Universe dataset 'abl-okoo2/parcels-and-boxes-dv7sg' and Kaggle box detection datasets.
Checks Roboflow API credentials and downloads/prepares package detection data cleanly.
"""

import os
import sys
import json
import zipfile
import requests
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    DATA_DIR,
    RAW_DATA_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger("DownloadPackageDataset")

ROBOFLOW_WORKSPACE = "abl-okoo2"
ROBOFLOW_PROJECT = "parcels-and-boxes-dv7sg"
ROBOFLOW_VERSION = 1

PACKAGE_EXTRACTED_DIR = DATA_DIR / "extracted_package_detector"


def check_roboflow_credentials() -> Tuple[bool, Optional[str]]:
    """
    Checks if ROBOFLOW_API_KEY is configured in the environment.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key and len(api_key.strip()) > 5:
        return True, api_key.strip()

    help_msg = (
        "\n" + "=" * 75 + "\n"
        "NOTICE: Roboflow API Key not found in environment (ROBOFLOW_API_KEY).\n"
        "To download directly from Roboflow Universe ('abl-okoo2/parcels-and-boxes-dv7sg'):\n"
        "  1. Go to https://universe.roboflow.com/abl-okoo2/parcels-and-boxes-dv7sg\n"
        "  2. Sign in and get your Roboflow API key from Settings > Roboflow API\n"
        "  3. Set environment variable: set ROBOFLOW_API_KEY=your_key\n\n"
        "Utilizing verified public package and parcel detection dataset from Kaggle...\n"
        "=" * 75
    )
    logger.info(help_msg)
    return False, None


def download_package_dataset() -> Tuple[bool, Path]:
    """
    Downloads and extracts dataset for Stage 1 Package Detector.
    """
    ensure_directories()
    PACKAGE_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    yaml_candidates = list(PACKAGE_EXTRACTED_DIR.rglob("*.yaml")) + list(PACKAGE_EXTRACTED_DIR.rglob("*.yml"))
    if yaml_candidates and len(list(PACKAGE_EXTRACTED_DIR.rglob("*.jpg"))) > 50:
        logger.info(f"Package detector dataset already exists in {PACKAGE_EXTRACTED_DIR}. Skipping download.")
        return True, PACKAGE_EXTRACTED_DIR

    # 1. Try Roboflow if key is present
    has_rf_key, rf_key = check_roboflow_credentials()
    if has_rf_key:
        try:
            logger.info(f"Downloading Roboflow dataset '{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}' v{ROBOFLOW_VERSION}...")
            from roboflow import Roboflow
            rf = Roboflow(api_key=rf_key)
            project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
            version = project.version(ROBOFLOW_VERSION)
            dataset = version.download("yolov8", location=str(PACKAGE_EXTRACTED_DIR))
            logger.info(f"Roboflow dataset downloaded successfully to {PACKAGE_EXTRACTED_DIR}")
            return True, PACKAGE_EXTRACTED_DIR
        except Exception as e:
            logger.warning(f"Roboflow download encountered error: {e}. Proceeding to verified package dataset.")

    # 2. Utilize Kaggle cardboard/parcel box detection dataset
    cardboard_zip = RAW_DATA_DIR / "cardboard_single_boxes.zip"
    if not cardboard_zip.exists():
        logger.info("Downloading cardboard/package detection archive from Kaggle...")
        home_dir = Path.home()
        kaggle_json = home_dir / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            cfg = json.load(open(kaggle_json, "r", encoding="utf-8"))
            r = requests.get(
                "https://www.kaggle.com/api/v1/datasets/download/sakshi12345/cardboard-box-images-with-annotations-for-yolov5",
                auth=(cfg["username"], cfg["key"]),
                stream=True,
                timeout=60,
            )
            r.raise_for_status()
            with open(cardboard_zip, "wb") as f:
                f.write(r.content)

    if cardboard_zip.exists():
        logger.info(f"Extracting package dataset to {PACKAGE_EXTRACTED_DIR}...")
        with zipfile.ZipFile(cardboard_zip, "r") as z:
            z.extractall(PACKAGE_EXTRACTED_DIR)
        logger.info(f"Package detector dataset extracted to {PACKAGE_EXTRACTED_DIR}")
        return True, PACKAGE_EXTRACTED_DIR

    logger.error("No package detection dataset could be downloaded or located.")
    return False, PACKAGE_EXTRACTED_DIR


if __name__ == "__main__":
    ok, path = download_package_dataset()
    print(f"Package detector dataset status: {ok}, path: {path}")
