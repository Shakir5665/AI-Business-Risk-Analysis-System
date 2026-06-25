from pathlib import Path

# ======================================================
# Google Drive Root
# ======================================================

GOOGLE_DRIVE = Path("/content/drive/MyDrive")

PROJECT_STORAGE = GOOGLE_DRIVE / "AI-Business-Risk-Analysis-System"

# ======================================================
# Dataset
# ======================================================

DATASET_DIR = PROJECT_STORAGE / "Dataset"

RAW_DATA_DIR = DATASET_DIR / "raw"

PROCESSED_DATA_DIR = DATASET_DIR / "processed"

FINAL_DATA_DIR = DATASET_DIR / "final"

# ======================================================
# Models
# ======================================================

MODEL_DIR = PROJECT_STORAGE / "Models"

# ======================================================
# Outputs
# ======================================================

OUTPUT_DIR = PROJECT_STORAGE / "Outputs"

LOG_DIR = OUTPUT_DIR / "logs"

REPORT_DIR = OUTPUT_DIR / "reports"

FIGURE_DIR = OUTPUT_DIR / "figures"