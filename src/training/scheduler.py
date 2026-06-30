"""
Scheduler Factory

Creates learning rate scheduler.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from transformers import get_linear_schedule_with_warmup

from configs.training_config import (
    WARMUP_RATIO
)

from src.utils.logger import logger


class SchedulerFactory:
    """
    Creates learning rate scheduler.
    """

    @staticmethod
    def create(
        optimizer,
        total_training_steps: int
    ):

        warmup_steps = int(
            total_training_steps * WARMUP_RATIO
        )

        scheduler = get_linear_schedule_with_warmup(

            optimizer=optimizer,

            num_warmup_steps=warmup_steps,

            num_training_steps=total_training_steps

        )

        logger.info(
            f"Scheduler created "
            f"(Warmup Steps: {warmup_steps}, "
            f"Training Steps: {total_training_steps})"
        )

        return scheduler