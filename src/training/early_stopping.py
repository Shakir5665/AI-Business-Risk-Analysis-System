"""
Early Stopping

Stops training when validation loss
has not improved for several epochs.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from configs.training_config import (
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA
)

from src.utils.logger import logger


class EarlyStopping:
    """
    Monitors validation loss and
    determines when training should stop.
    """

    def __init__(self):

        self.patience = EARLY_STOPPING_PATIENCE

        self.min_delta = MIN_DELTA

        self.best_loss = float("inf")

        self.counter = 0

        logger.info("EarlyStopping initialized.")

    # --------------------------------------------------

    def should_stop(
        self,
        validation_loss: float
    ) -> bool:

        # ------------------------------------------
        # Validation loss improved
        # ------------------------------------------

        if validation_loss < (self.best_loss - self.min_delta):

            self.best_loss = validation_loss

            self.counter = 0

            return False

        # ------------------------------------------
        # No improvement
        # ------------------------------------------

        self.counter += 1

        logger.info(

            f"EarlyStopping Counter: "

            f"{self.counter}/{self.patience}"

        )

        if self.counter >= self.patience:

            logger.info(

                "Early stopping triggered."

            )

            return True

        return False

    # --------------------------------------------------

    def reset(self):

        self.best_loss = float("inf")

        self.counter = 0

        logger.info("EarlyStopping reset.")