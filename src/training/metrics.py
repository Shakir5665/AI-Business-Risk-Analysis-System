"""
Metrics

Computes evaluation metrics
for Business Risk Model.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from typing import Dict

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)

from src.utils.logger import logger


class MetricsComputer:

    """
    Computes metrics for

    • Sentiment Classification

    • Aspect Classification
    """

    def __init__(self):

        logger.info(
            "MetricsComputer initialized."
        )

    # --------------------------------------------------

    def compute_sentiment(

        self,

        logits: torch.Tensor,

        targets: torch.Tensor

    ) -> Dict:

        predictions = torch.argmax(

            logits,

            dim=1

        ).cpu().numpy()

        targets = targets.cpu().numpy()

        accuracy = accuracy_score(

            targets,

            predictions

        )

        precision, recall, f1, _ = (

            precision_recall_fscore_support(

                targets,

                predictions,

                average="weighted",

                zero_division=0

            )

        )

        return {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1

        }

    # --------------------------------------------------

    def compute_aspect(

        self,

        logits: torch.Tensor,

        targets: torch.Tensor,

        threshold: float = 0.5

    ) -> Dict:

        probabilities = torch.sigmoid(

            logits

        )

        predictions = (

            probabilities >= threshold

        ).int().cpu().numpy()

        targets = targets.cpu().numpy()

        micro_f1 = f1_score(

            targets,

            predictions,

            average="micro",

            zero_division=0

        )

        macro_f1 = f1_score(

            targets,

            predictions,

            average="macro",

            zero_division=0

        )

        return {

            "micro_f1": micro_f1,

            "macro_f1": macro_f1

        }