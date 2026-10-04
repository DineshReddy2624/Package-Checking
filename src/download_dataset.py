"""
Dataset Download Module for Package Damage Detection.
Downloads and extracts the dataset from Kaggle using official API/credentials.
Attempts primary handle first, with fallback to verified public YOLO dataset if primary returns 403/404.
Streams download directly with robust chunking and extraction.
"""

import os
import sys
import json
import zipfile
import requests
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    RAW_DATA_DIR,
    EXTRACTED_DATA_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger("DownloadDataset")

PRIMARY_DATASET_HANDLE = "subhashish/package-damage-detection"
FALLBACK_DATASET_HANDLE = "ivannuke/defective-box-detection-real-vs-synthetic"


def get_kaggle_auth() -> Optional[Tuple[str, str]]:
    """
    Retrieves Kaggle (username, key) tuple from env or ~/.kaggle/kaggle.json.
    """
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return (env_user, env_key)

    home_dir = Path.home()
    kaggle_json_path = home_dir / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return (data.get("username"), data.get("key"))
        except Exception:
            pass

    appdata = os.environ.get("APPDATA")
    if appdata:
        appdata_kaggle = Path(appdata) / "kaggle" / "kaggle.json"
        if appdata_kaggle.exists():
            try:
                with open(appdata_kaggle, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return (data.get("username"), data.get("key"))
            except Exception:
                pass

    return None


def check_kaggle_credentials() -> Tuple[bool, str]:
    """
    Checks if Kaggle credentials exist.
    """
    auth = get_kaggle_auth()
    if auth and auth[0] and auth[1]:
        return True, f"Found valid Kaggle credentials for user '{auth[0]}'"

    help_msg = (
        "\n" + "=" * 70 + "\n"
        "ERROR: Kaggle credentials not found!\n"
        "To download the dataset automatically, configure your Kaggle credentials using either:\n\n"
        "Option A: kaggle.json file\n"
        "  1. Go to https://www.kaggle.com/settings\n"
        "  2. Click 'Create New Token' to download kaggle.json\n"
        f"  3. Place it at: {Path.home() / '.kaggle' / 'kaggle.json'}\n\n"
        "Option B: Environment Variables\n"
        "  Set KAGGLE_USERNAME and KAGGLE_KEY in your system environment:\n"
        "  set KAGGLE_USERNAME=your_username\n"
        "  set KAGGLE_KEY=your_api_key\n"
        "=" * 70
    )
    return False, help_msg


def download_stream(url: str, auth: Tuple[str, str], dest_path: Path) -> bool:
    """
    Streams download directly to destination path with progress bar.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to Kaggle download stream...")
    
    with requests.get(url, auth=auth, stream=True, timeout=60) as r:
        if r.status_code == 403 or r.status_code == 404:
            logger.warning(f"HTTP {r.status_code}: Dataset not accessible or private.")
            return False
        r.raise_for_status()

        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 2 * 1024 * 1024  # 2MB chunks

        logger.info(f"Downloading archive ({total_size / (1024*1024):.2f} MB) to {dest_path.name}...")
        with open(dest_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    return True


def download_and_extract_dataset(force_download: bool = False) -> Tuple[bool, Path]:
    """
    Downloads the dataset from Kaggle into data/raw/ and extracts it into data/extracted/.
    Returns (success_status, extracted_directory_path).
    """
    ensure_directories()

    # Check if already extracted
    extracted_contents = list(EXTRACTED_DATA_DIR.glob("*"))
    if extracted_contents and not force_download:
        logger.info(f"Extracted dataset already exists in {EXTRACTED_DATA_DIR}. Skipping download.")
        return True, EXTRACTED_DATA_DIR

    # Check credentials
    has_creds, msg = check_kaggle_credentials()
    if not has_creds:
        logger.error(msg)
        return False, EXTRACTED_DATA_DIR

    auth = get_kaggle_auth()
    logger.info(f"Kaggle authentication verified: {msg}")

    # 1. Try primary dataset
    primary_url = f"https://www.kaggle.com/api/v1/datasets/download/{PRIMARY_DATASET_HANDLE}"
    primary_zip = RAW_DATA_DIR / f"{PRIMARY_DATASET_HANDLE.split('/')[-1]}.zip"
    
    logger.info(f"Attempting download for primary dataset '{PRIMARY_DATASET_HANDLE}'...")
    download_ok = download_stream(primary_url, auth, primary_zip)
    active_zip = primary_zip

    # 2. Fallback if primary fails
    if not download_ok or not primary_zip.exists() or primary_zip.stat().st_size == 0:
        logger.warning(
            f"Primary dataset '{PRIMARY_DATASET_HANDLE}' was inaccessible. "
            f"Falling back to verified public YOLO package damage dataset '{FALLBACK_DATASET_HANDLE}'..."
        )
        fallback_url = f"https://www.kaggle.com/api/v1/datasets/download/{FALLBACK_DATASET_HANDLE}"
        fallback_zip = RAW_DATA_DIR / f"{FALLBACK_DATASET_HANDLE.split('/')[-1]}.zip"
        
        fb_ok = download_stream(fallback_url, auth, fallback_zip)
        if not fb_ok or not fallback_zip.exists() or fallback_zip.stat().st_size == 0:
            logger.error("Failed to download fallback dataset.")
            return False, EXTRACTED_DATA_DIR
        active_zip = fallback_zip

    logger.info(f"Extracting {active_zip.name} ({active_zip.stat().st_size / (1024*1024):.2f} MB) to {EXTRACTED_DATA_DIR}...")
    with zipfile.ZipFile(active_zip, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_DATA_DIR)

    logger.info(f"Dataset successfully extracted to {EXTRACTED_DATA_DIR}")
    return True, EXTRACTED_DATA_DIR


if __name__ == "__main__":
    success, path = download_and_extract_dataset()
    if not success:
        sys.exit(1)
    else:
        print(f"Dataset ready at: {path}")
