"""
Checkpoint Manager

Handles saving and loading training checkpoints.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path

import torch

from configs.training_config import (
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    LATEST_CHECKPOINT_NAME
)

from src.utils.logger import logger


class CheckpointManager:

    """
    Saves and loads checkpoints.
    """

    def __init__(self):

        self.checkpoint_dir = Path(CHECKPOINT_DIR)

        self.checkpoint_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        logger.info(
            "CheckpointManager initialized."
        )

    # --------------------------------------------------

    def save_best(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        validation_loss
    ):

        path = self.checkpoint_dir / BEST_MODEL_NAME

        torch.save(

            {

                "epoch": epoch,

                "validation_loss": validation_loss,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "scheduler_state_dict": scheduler.state_dict()

            },

            path

        )

        logger.info(
            f"Best model saved: {path}"
        )

    # --------------------------------------------------

    def save_latest(
        self,
        model,
        optimizer,
        scheduler,
        epoch
    ):

        path = self.checkpoint_dir / LATEST_CHECKPOINT_NAME

        torch.save(

            {

                "epoch": epoch,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "scheduler_state_dict": scheduler.state_dict()

            },

            path

        )

        logger.info(
            f"Latest checkpoint saved: {path}"
        )

    # --------------------------------------------------

    def load(
        self,
        path,
        model,
        optimizer=None,
        scheduler=None
    ):

        checkpoint = torch.load(

            path,

            map_location="cpu"

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        if optimizer is not None:

            optimizer.load_state_dict(

                checkpoint["optimizer_state_dict"]

            )

        if scheduler is not None:

            scheduler.load_state_dict(

                checkpoint["scheduler_state_dict"]

            )

        logger.info(
            f"Checkpoint loaded: {path}"
        )

        return checkpoint