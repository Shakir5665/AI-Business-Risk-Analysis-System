"""
Google Drive Checkpoint & History Sync Utility

Synchronizes training artifacts (`best_model.pt`, `latest_checkpoint.pt`, `history.json`)
from the local Google Colab runtime to persistent Google Drive storage.

Project:
AI-Powered Business Risk Analysis
and Recommendation System

Usage:
    # From Terminal or Colab Notebook:
    python drive_update.py
    
    # Or in Python code:
    from drive_update import sync_to_gdrive
    sync_to_gdrive()
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ==================================================
# Default Configuration & Fallbacks
# ==================================================

try:
    # If run within the repo, import config constants
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configs.training_config import (
        CHECKPOINT_DIR as CFG_LOCAL_CKPT,
        BEST_MODEL_NAME as CFG_BEST_NAME,
        LATEST_CHECKPOINT_NAME as CFG_LATEST_NAME,
        GDRIVE_CHECKPOINT_DIR as CFG_GDRIVE_CKPT,
        GDRIVE_REPORTS_DIR as CFG_GDRIVE_REPORTS
    )
except Exception:
    CFG_LOCAL_CKPT = "checkpoints"
    CFG_BEST_NAME = "best_model.pt"
    CFG_LATEST_NAME = "latest_checkpoint.pt"
    CFG_GDRIVE_CKPT = "/content/drive/MyDrive/AI-Business-Risk-Analysis-System/checkpoints"
    CFG_GDRIVE_REPORTS = "/content/drive/MyDrive/AI-Business-Risk-Analysis-System/Outputs/reports"


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:3.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def is_running_in_colab() -> bool:
    """Check if the environment is Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def ensure_drive_mounted(mount_point: str = "/content/drive") -> bool:
    """
    Ensure Google Drive is mounted if running in Google Colab.
    Returns True if mounted successfully or already mounted, False otherwise.
    """
    if not is_running_in_colab():
        print("[INFO] Not running in Google Colab environment. Skipping drive.mount().")
        return True

    drive_root = Path(mount_point) / "MyDrive"
    if drive_root.exists():
        print(f"[INFO] Google Drive is already mounted at '{mount_point}'.")
        return True

    try:
        print("[INFO] Mounting Google Drive...")
        from google.colab import drive
        drive.mount(mount_point)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to mount Google Drive: {e}")
        return False


def find_source_file(filename: str, search_dirs: list) -> Path:
    """Search for a file across multiple candidate directories."""
    for d in search_dirs:
        candidate = Path(d) / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def copy_file_if_modified(src: Path, dst: Path, dry_run: bool = False) -> tuple:
    """
    Copy src to dst if dst does not exist or src is newer/different in size.
    Returns: (status: str, message: str)
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_size = src.stat().st_size
    src_mtime = src.stat().st_mtime

    if dst.exists() and dst.is_file():
        dst_size = dst.stat().st_size
        dst_mtime = dst.stat().st_mtime

        # Check if file has not changed
        if src_size == dst_size and src_mtime <= dst_mtime:
            return ("SKIPPED", f"Up-to-date ({format_size(src_size)})")

    if dry_run:
        return ("DRY_RUN", f"Would copy {format_size(src_size)}")

    try:
        shutil.copy2(src, dst)
        return ("COPIED", f"Successfully synced ({format_size(src_size)})")
    except Exception as e:
        return ("FAILED", f"Error: {e}")


def sync_to_gdrive(
    local_checkpoint_dir: str = CFG_LOCAL_CKPT,
    gdrive_checkpoint_dir: str = CFG_GDRIVE_CKPT,
    gdrive_reports_dir: str = CFG_GDRIVE_REPORTS,
    dry_run: bool = False,
    sync_all_files: bool = True
) -> dict:
    """
    Synchronizes best_model.pt, latest_checkpoint.pt, and history.json
    (plus any additional checkpoints) from Colab to Google Drive.

    Parameters:
        local_checkpoint_dir: Path to local checkpoints folder (e.g. 'checkpoints')
        gdrive_checkpoint_dir: Target Google Drive checkpoints path
        gdrive_reports_dir: Target Google Drive reports path
        dry_run: If True, only previews changes without copying
        sync_all_files: If True, also syncs any additional checkpoint files found

    Returns:
        Summary dictionary with transfer results.
    """
    print("=" * 65)
    print(" 🚀 Google Drive Checkpoint & History Sync")
    print("=" * 65)
    print(f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Local Directory  : {Path(local_checkpoint_dir).resolve()}")
    print(f"Drive Destination: {gdrive_checkpoint_dir}")
    print(f"Dry Run Mode     : {dry_run}")
    print("-" * 65)

    # 1. Mount Google Drive if in Colab
    ensure_drive_mounted()

    results = {}
    search_dirs = [
        Path(local_checkpoint_dir),
        Path("outputs/reports"),
        Path("outputs"),
        Path(".")
    ]

    # Target key files to synchronize
    target_files = [
        (CFG_BEST_NAME, Path(gdrive_checkpoint_dir) / CFG_BEST_NAME),
        (CFG_LATEST_NAME, Path(gdrive_checkpoint_dir) / CFG_LATEST_NAME),
        ("history.json", Path(gdrive_checkpoint_dir) / "history.json"),
    ]

    # Optionally sync history.json to reports dir as well if configured
    if gdrive_reports_dir and gdrive_reports_dir != gdrive_checkpoint_dir:
        target_files.append(("history.json", Path(gdrive_reports_dir) / "history.json"))

    # Also collect any extra files present in the local checkpoint folder
    local_dir_path = Path(local_checkpoint_dir)
    if local_dir_path.exists() and sync_all_files:
        for extra_file in local_dir_path.glob("*"):
            if extra_file.is_file() and extra_file.name not in [t[0] for t in target_files]:
                target_files.append((extra_file.name, Path(gdrive_checkpoint_dir) / extra_file.name))

    print(f"{'Target File':<24} | {'Status':<10} | {'Details'}")
    print("-" * 65)

    for filename, dst_path in target_files:
        src_path = find_source_file(filename, search_dirs)

        if src_path is None or not src_path.exists():
            status = "NOT FOUND"
            message = "Local file does not exist yet"
        else:
            status, message = copy_file_if_modified(src_path, dst_path, dry_run=dry_run)

        results[f"{filename} -> {dst_path}"] = {"status": status, "message": message}
        display_name = f"{filename} ({dst_path.parent.name}/)"
        print(f"{display_name:<24} | {status:<10} | {message}")

    print("=" * 65)
    return results


# ==================================================
# CLI Execution
# ==================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync best_model.pt, latest_checkpoint.pt, and history.json to Google Drive."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=CFG_LOCAL_CKPT,
        help="Local checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=CFG_GDRIVE_CKPT,
        help="Google Drive destination directory"
    )
    parser.add_argument(
        "--reports-dest",
        type=str,
        default=CFG_GDRIVE_REPORTS,
        help="Google Drive reports destination directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry-run without copying files"
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Sync only the 3 primary files without scanning for extra files"
    )

    args = parser.parse_args()

    sync_to_gdrive(
        local_checkpoint_dir=args.source,
        gdrive_checkpoint_dir=args.dest,
        gdrive_reports_dir=args.reports_dest,
        dry_run=args.dry_run,
        sync_all_files=not args.target_only
    )
