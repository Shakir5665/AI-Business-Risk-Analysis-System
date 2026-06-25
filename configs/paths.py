from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FINAL_DATA_DIR = DATA_DIR / "final"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

LOG_DIR = OUTPUT_DIR / "logs"

REPORT_DIR = OUTPUT_DIR / "reports"

FIGURE_DIR = OUTPUT_DIR / "figures"