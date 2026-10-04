"""
Utility functions for hardware detection, logging, directory management, and report generation.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Define Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OUTPUT_REPORTS_DIR = OUTPUTS_DIR / "reports"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"


def ensure_directories():
    """Create all required project directories if they do not exist."""
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        EXTRACTED_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        WEIGHTS_DIR,
        RUNS_DIR,
        REPORTS_DIR,
        OUTPUTS_DIR,
        PREDICTIONS_DIR,
        OUTPUT_REPORTS_DIR,
        VISUALIZATIONS_DIR,
        SAMPLE_IMAGES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "PackageDamageDetection") -> logging.Logger:
    """Configures and returns a consistent logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def detect_hardware() -> Dict[str, Any]:
    """
    Detects hardware capabilities (CPU/GPU) honestly without fabricating availability.
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA GPU detected)"
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    
    device_str = "0" if cuda_available else "cpu"
    
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpu_name": device_name,
        "cuda_version": cuda_version,
        "recommended_device": device_str,
    }
    return info


def print_hardware_summary():
    """Prints a clean summary of detected hardware."""
    hw = detect_hardware()
    print("=" * 60)
    print("HARDWARE & ENVIRONMENT DETECTION")
    print("=" * 60)
    print(f"PyTorch Version   : {hw['pytorch_version']}")
    print(f"CUDA Available    : {hw['cuda_available']}")
    print(f"CUDA Version      : {hw['cuda_version']}")
    print(f"GPU Name          : {hw['gpu_name']}")
    print(f"Device Count      : {hw['device_count']}")
    print(f"Device Selected   : {'GPU (' + hw['gpu_name'] + ')' if hw['cuda_available'] else 'CPU'}")
    print("=" * 60)


def save_json(data: Any, filepath: Path) -> None:
    """Safely saves data to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(text: str, filepath: Path) -> None:
    """Safely saves text to a file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    ensure_directories()
    print_hardware_summary()
