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
from ast import literal_eval

import pandas as pd
import torch

from src.utils.logger import logger


class ClassWeightCalculator:
    """
    Computes class weights directly
    from the training DataFrame.
    """

    def __init__(self):

        logger.info(
            "ClassWeightCalculator initialized."
        )

    # --------------------------------------------------

    def sentiment_weights(
        self,
        dataframe: pd.DataFrame
    ) -> torch.Tensor:
        """
        Computes weights for CrossEntropyLoss.
        """

        labels = dataframe["sentiment"].tolist()

        counts = Counter(labels)

        sentiment_order = [

            "Negative",
            "Neutral",
            "Positive"

        ]

        total = len(labels)

        num_classes = len(sentiment_order)

        weights = []

        for label in sentiment_order:

            weight = total / (

                num_classes *

                counts[label]

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
        dataframe: pd.DataFrame
    ) -> torch.Tensor:
        """
        Computes pos_weight for BCEWithLogitsLoss.
        """

        aspect_order = [

            "Delivery",
            "Quality",
            "Trust"

        ]

        positives = torch.zeros(

            len(aspect_order),

            dtype=torch.float

        )

        total_samples = len(dataframe)

        for aspects in dataframe["aspects"]:

            # Handles JSON strings if needed
            if isinstance(aspects, str):

                aspects = literal_eval(aspects)

            for i, aspect in enumerate(aspect_order):

                if aspect in aspects:

                    positives[i] += 1

        negatives = total_samples - positives

        pos_weight = negatives / positives

        logger.info(

            f"Aspect Pos Weights: {pos_weight.tolist()}"

        )

        return pos_weight