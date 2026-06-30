"""
Optimizer Factory

Creates optimizer for training.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

import torch.optim as optim

from configs.training_config import (
    LEARNING_RATE,
    WEIGHT_DECAY
)

from src.utils.logger import logger


class OptimizerFactory:
    """
    Creates optimizer for model training.
    """

    @staticmethod
    def create(model):

        optimizer = optim.AdamW(

            filter(

                lambda parameter: parameter.requires_grad,

                model.parameters()

            ),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY

        )

        logger.info("AdamW optimizer created.")

        return optimizer