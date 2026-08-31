"""
Google Drive Checkpoint & History Sync Utility

Synchronizes training artifacts (`best_model.pt`, `latest_checkpoint.pt`, `history.json`)
strictly from the local `checkpoints/` folder to persistent Google Drive storage.

Project:
AI-Powered Business Risk Analysis
and Recommendation System

Usage:
    # From Terminal or Colab Notebook:
    python drive_update.py
    
    # Or force overwrite:
    python drive_update.py --force
    
    # Or in Python code:
    from drive_update import sync_to_gdrive
    sync_to_gdrive(force=True)
"""

import os
import sys
import shutil
import hashlib
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
        GDRIVE_CHECKPOINT_DIR as CFG_GDRIVE_CKPT
    )
except Exception:
    CFG_LOCAL_CKPT = "checkpoints"
    CFG_BEST_NAME = "best_model.pt"
    CFG_LATEST_NAME = "latest_checkpoint.pt"
    CFG_GDRIVE_CKPT = "/content/drive/MyDrive/AI-Business-Risk-Analysis-System/checkpoints"


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:3.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def compute_md5(file_path: Path, max_bytes: int = 20 * 1024 * 1024) -> str:
    """
    Compute MD5 hash for a file.
    For small files (<= 20MB, e.g. history.json), reads the entire file.
    For large files (> 20MB, e.g. model checkpoints), samples head, tail, and size.
    """
    hasher = hashlib.md5()
    file_size = file_path.stat().st_size

    if file_size <= max_bytes:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    else:
        with open(file_path, "rb") as f:
            hasher.update(f.read(10 * 1024 * 1024))
            f.seek(max(0, file_size - 10 * 1024 * 1024))
            hasher.update(f.read(10 * 1024 * 1024))
            hasher.update(str(file_size).encode())

    return hasher.hexdigest()


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


def copy_file(src: Path, dst: Path, force: bool = False, dry_run: bool = False) -> tuple:
    """
    Copies src to dst.
    Uses MD5/content checksum and file size to accurately detect real differences.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_size = src.stat().st_size

    if dst.exists() and dst.is_file() and not force:
        dst_size = dst.stat().st_size

        # For files like history.json, full MD5 content comparison
        if src_size < 10 * 1024 * 1024:
            src_md5 = compute_md5(src)
            dst_md5 = compute_md5(dst)
            if src_md5 == dst_md5:
                return ("SKIPPED", f"Up-to-date ({format_size(src_size)}, content matches)")
        else:
            # For large checkpoints
            if src_size == dst_size:
                src_md5 = compute_md5(src)
                dst_md5 = compute_md5(dst)
                if src_md5 == dst_md5:
                    return ("SKIPPED", f"Up-to-date ({format_size(src_size)})")

    if dry_run:
        return ("DRY_RUN", f"Would copy {format_size(src_size)} from {src}")

    try:
        # Overwrite destination directly
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                pass

        shutil.copy2(src, dst)

        # Flush file cache (essential for Google Colab FUSE filesystem)
        if hasattr(os, "sync"):
            os.sync()

        return ("COPIED", f"Successfully synced ({format_size(src_size)}) from {src}")
    except Exception as e:
        return ("FAILED", f"Error: {e}")


def sync_to_gdrive(
    local_checkpoint_dir: str = CFG_LOCAL_CKPT,
    gdrive_checkpoint_dir: str = CFG_GDRIVE_CKPT,
    force: bool = False,
    dry_run: bool = False,
    sync_all_files: bool = True
) -> dict:
    """
    Synchronizes best_model.pt, latest_checkpoint.pt, and history.json
    (plus any additional files) strictly from local checkpoints/ to Google Drive.

    Parameters:
        local_checkpoint_dir: Path to local checkpoints folder (default: 'checkpoints')
        gdrive_checkpoint_dir: Target Google Drive checkpoints path
        force: If True, forces overwrite regardless of timestamps/hash
        dry_run: If True, only previews changes without copying
        sync_all_files: If True, also syncs any additional checkpoint files found

    Returns:
        Summary dictionary with transfer results.
    """
    print("=" * 72)
    print(" 🚀 Google Drive Checkpoint & History Sync")
    print("=" * 72)
    print(f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source Directory : {Path(local_checkpoint_dir).resolve()} (ONLY)")
    print(f"Drive Destination: {gdrive_checkpoint_dir}")
    print(f"Force Overwrite  : {force}")
    print(f"Dry Run Mode     : {dry_run}")
    print("-" * 72)

    # 1. Mount Google Drive if in Colab
    ensure_drive_mounted()

    results = {}
    local_dir = Path(local_checkpoint_dir)

    # Explicit list of primary targets sourced strictly from local_checkpoint_dir
    primary_filenames = [
        CFG_BEST_NAME,
        CFG_LATEST_NAME,
        "history.json"
    ]

    files_to_sync = []
    for fname in primary_filenames:
        src = local_dir / fname
        dst = Path(gdrive_checkpoint_dir) / fname
        files_to_sync.append((fname, src, dst))

    # Also collect any extra files present in the local checkpoint folder
    if local_dir.exists() and sync_all_files:
        for extra_file in local_dir.glob("*"):
            if extra_file.is_file() and extra_file.name not in primary_filenames:
                dst = Path(gdrive_checkpoint_dir) / extra_file.name
                files_to_sync.append((extra_file.name, extra_file, dst))

    print(f"{'File Name':<28} | {'Status':<10} | {'Details'}")
    print("-" * 72)

    for filename, src_path, dst_path in files_to_sync:
        if not src_path.exists():
            status = "NOT FOUND"
            message = f"Missing in {local_checkpoint_dir}/"
        else:
            status, message = copy_file(src_path, dst_path, force=force, dry_run=dry_run)

        results[f"{filename} -> {dst_path}"] = {"status": status, "message": message}
        print(f"{filename:<28} | {status:<10} | {message}")

    print("=" * 72)
    return results


# ==================================================
# CLI Execution
# ==================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync best_model.pt, latest_checkpoint.pt, and history.json strictly from checkpoints/ to Google Drive."
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
        "--force", "-f",
        action="store_true",
        help="Force overwrite all files regardless of checksums"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry-run without copying files"
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Sync only best_model.pt, latest_checkpoint.pt, and history.json"
    )

    args = parser.parse_args()

    sync_to_gdrive(
        local_checkpoint_dir=args.source,
        gdrive_checkpoint_dir=args.dest,
        force=args.force,
        dry_run=args.dry_run,
        sync_all_files=not args.target_only
    )
