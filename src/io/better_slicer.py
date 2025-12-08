import shutil, yaml
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

def _get_data_config():
    ROOT = Path(__file__).resolve().parents[2]
    DATA_CONFIG = ROOT / "configs" / "data.yaml"
    with open(DATA_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

# Global constants
INPUT_PATH = _get_data_config()["path"]
OUTPUT_PATH = _get_data_config()["path_sliced_resized"]
MTSD_TYPES = ("train_full", "train_partial", "val")
TARGET_SIZE = 1280
OVERLAP = 0.2
MIN_ABS_AREA = 100
MIN_AREA_RATIO = 0.6

