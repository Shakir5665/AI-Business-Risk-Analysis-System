"""
Loss Functions

Computes all losses for the
Business Risk Model.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from typing import Dict

import torch
import torch.nn as nn

from src.utils.logger import logger


class LossComputer:
    """
    Computes:

    - Sentiment Loss
    - Aspect Loss
    - Total Loss
    """

    def __init__(self):

        self.sentiment_loss = nn.CrossEntropyLoss()

        self.aspect_loss = nn.BCEWithLogitsLoss()

        logger.info("LossComputer initialized.")

    # --------------------------------------------------

    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        sentiment_targets: torch.Tensor,
        aspect_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        sentiment_loss = self.sentiment_loss(

            outputs["sentiment_logits"],

            sentiment_targets

        )

        aspect_loss = self.aspect_loss(

            outputs["aspect_logits"],

            aspect_targets

        )

        total_loss = sentiment_loss + aspect_loss

        return {

            "sentiment_loss": sentiment_loss,

            "aspect_loss": aspect_loss,

            "total_loss": total_loss

        }