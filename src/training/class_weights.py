"""
Class Weight Calculator

Computes class weights for:

- Sentiment Classification
- Aspect Classification

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from collections import Counter

import torch

from src.utils.logger import logger


class ClassWeightCalculator:
    """
    Computes class weights automatically
    from the training dataset.
    """

    def __init__(self):

        logger.info(
            "ClassWeightCalculator initialized."
        )

    # --------------------------------------------------

    def sentiment_weights(
        self,
        dataset
    ) -> torch.Tensor:
        """
        Computes weights for CrossEntropyLoss.
        """

        labels = []

        for sample in dataset:

            labels.append(
                int(sample["sentiment"])
            )

        counts = Counter(labels)

        total = len(labels)

        num_classes = len(counts)

        weights = []

        for i in range(num_classes):

            weight = total / (

                num_classes *

                counts[i]

            )

            weights.append(weight)

        weights = torch.tensor(

            weights,

            dtype=torch.float

        )

        logger.info(

            f"Sentiment Weights: {weights.tolist()}"

        )

        return weights

    # --------------------------------------------------

    def aspect_pos_weights(
        self,
        dataset
    ) -> torch.Tensor:
        """
        Computes pos_weight for BCEWithLogitsLoss.
        """

        aspect_matrix = []

        for sample in dataset:

            aspect_matrix.append(

                sample["aspects"]

            )

        aspect_matrix = torch.tensor(

            aspect_matrix,

            dtype=torch.float

        )

        positives = aspect_matrix.sum(dim=0)

        negatives = len(dataset) - positives

        pos_weight = negatives / positives

        logger.info(

            f"Aspect Pos Weights: {pos_weight.tolist()}"

        )

        return pos_weight