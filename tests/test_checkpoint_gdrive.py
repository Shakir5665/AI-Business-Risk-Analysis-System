import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import shutil

# Mock torch
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

# Mock src.utils.logger
mock_logger = MagicMock()
sys.modules['src.utils.logger'] = mock_logger
sys.modules['src.utils.logger.logger'] = mock_logger.logger

# Now import CheckpointManager
from src.training.checkpoint import CheckpointManager
from configs.training_config import (
    CHECKPOINT_DIR,
    LATEST_CHECKPOINT_NAME,
    GDRIVE_CHECKPOINT_DIR
)

class TestCheckpointGDrive(unittest.TestCase):
    def setUp(self):
        # We use temporary test paths
        self.local_dir = Path("test_checkpoints")
        self.gdrive_dir = Path("test_gdrive_checkpoints")
        
        if self.local_dir.exists():
            shutil.rmtree(self.local_dir)
        if self.gdrive_dir.exists():
            shutil.rmtree(self.gdrive_dir)
            
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.gdrive_dir.mkdir(parents=True, exist_ok=True)
        
        # Patch the CheckpointManager attributes
        self.manager = CheckpointManager()
        self.manager.checkpoint_dir = self.local_dir
        
        # Patch the GDRIVE_CHECKPOINT_DIR inside checkpoint module
        self.patcher = patch('src.training.checkpoint.GDRIVE_CHECKPOINT_DIR', str(self.gdrive_dir))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if self.local_dir.exists():
            shutil.rmtree(self.local_dir)
        if self.gdrive_dir.exists():
            shutil.rmtree(self.gdrive_dir)

    def test_save_latest_copies_to_gdrive(self):
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Call save_latest
        self.manager.save_latest(model, optimizer, scheduler, epoch=1)
        
        # Verify torch.save was called
        mock_torch.save.assert_called()
        
        # We manually create the local file because torch.save is mocked and didn't write it
        local_file = self.local_dir / LATEST_CHECKPOINT_NAME
        local_file.write_text("dummy checkpoint content")
        
        # Call save_latest again to test shutil.copy2
        self.manager.save_latest(model, optimizer, scheduler, epoch=1)
        
        # Check that file exists on local and gdrive
        gdrive_file = self.gdrive_dir / LATEST_CHECKPOINT_NAME
        self.assertTrue(gdrive_file.exists())
        self.assertEqual(gdrive_file.read_text(), "dummy checkpoint content")

    def test_load_latest_copies_from_gdrive_when_local_missing(self):
        model = MagicMock()
        
        # Save dummy file in Google Drive
        gdrive_file = self.gdrive_dir / LATEST_CHECKPOINT_NAME
        gdrive_file.write_text("gdrive checkpoint content")
        
        # Verify local file does not exist
        local_file = self.local_dir / LATEST_CHECKPOINT_NAME
        self.assertFalse(local_file.exists())
        
        # Call load_latest
        self.manager.load_latest(model)
        
        # Check that the local file has been copied from Google Drive
        self.assertTrue(local_file.exists())
        self.assertEqual(local_file.read_text(), "gdrive checkpoint content")

if __name__ == '__main__':
    unittest.main()
