"""
Validator

Runs validation on the validation dataset.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from typing import Dict

import torch

from src.training.losses import LossComputer
from src.training.metrics import MetricsComputer

from src.utils.logger import logger


class Validator:

    """
    Runs model validation.
    """

    def __init__(self):

        self.loss_computer = LossComputer()

        self.metrics_computer = MetricsComputer()

        logger.info("Validator initialized.")

    # --------------------------------------------------

    def validate(
        self,
        model,
        dataloader,
        device
    ) -> Dict:

        model.eval()

        total_sentiment_loss = 0.0
        total_aspect_loss = 0.0
        total_loss = 0.0

        sentiment_logits = []
        sentiment_targets = []

        aspect_logits = []
        aspect_targets = []

        with torch.no_grad():

            for batch in dataloader:

                input_ids = batch["input_ids"].to(device)

                attention_mask = batch["attention_mask"].to(device)

                sentiment = batch["sentiment"].to(device)

                aspects = batch["aspects"].to(device)

                outputs = model(

                    input_ids=input_ids,

                    attention_mask=attention_mask

                )

                losses = self.loss_computer.compute(

                    outputs,

                    sentiment,

                    aspects

                )

                total_sentiment_loss += losses["sentiment_loss"].item()

                total_aspect_loss += losses["aspect_loss"].item()

                total_loss += losses["total_loss"].item()

                sentiment_logits.append(

                    outputs["sentiment_logits"].cpu()

                )

                sentiment_targets.append(

                    sentiment.cpu()

                )

                aspect_logits.append(

                    outputs["aspect_logits"].cpu()

                )

                aspect_targets.append(

                    aspects.cpu()

                )

        sentiment_logits = torch.cat(sentiment_logits)

        sentiment_targets = torch.cat(sentiment_targets)

        aspect_logits = torch.cat(aspect_logits)

        aspect_targets = torch.cat(aspect_targets)

        sentiment_metrics = self.metrics_computer.compute_sentiment(

            sentiment_logits,

            sentiment_targets

        )

        aspect_metrics = self.metrics_computer.compute_aspect(

            aspect_logits,

            aspect_targets

        )

        batches = len(dataloader)

        return {

            "loss": {

                "sentiment": total_sentiment_loss / batches,

                "aspect": total_aspect_loss / batches,

                "total": total_loss / batches

            },

            "sentiment": sentiment_metrics,

            "aspect": aspect_metrics

        }