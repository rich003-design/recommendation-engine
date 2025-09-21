import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default settings
DEFAULT_PROJECT_ID = "recommendationengine-472815"
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "config" / "secrets.json"
DEFAULT_START_DATE = "20201101"
DEFAULT_END_DATE = "20251231"
