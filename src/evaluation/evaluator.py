"""
Evaluator

Evaluates the trained model on the test dataset.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from typing import Dict

import torch

from src.training.losses import LossComputer
from src.training.metrics import MetricsComputer
import torch.nn.functional as F

from src.utils.logger import logger


class Evaluator:

    """
    Evaluates the model on the test dataset.
    """

    def __init__(self):

        self.loss_computer = LossComputer()

        self.metrics_computer = MetricsComputer()

        logger.info("Evaluator initialized.")

    # --------------------------------------------------

    def evaluate(
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

        # -----------------------------
        # Predictions
        # -----------------------------

        sentiment_predictions = []

        aspect_predictions = []

        # -----------------------------
        # Probabilities
        # -----------------------------

        sentiment_probabilities = []

        aspect_probabilities = []

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

                # -----------------------------
                # Sentiment
                # -----------------------------

                sentiment_probs = F.softmax(

                    outputs["sentiment_logits"],

                    dim=1

                )

                sentiment_preds = torch.argmax(

                    sentiment_probs,

                    dim=1

                )

                # -----------------------------
                # Aspect
                # -----------------------------

                aspect_probs = torch.sigmoid(

                    outputs["aspect_logits"]

                )

                aspect_preds = (

                    aspect_probs >= 0.5

                ).int()


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

                # -----------------------------
                # Store Predictions
                # -----------------------------

                sentiment_predictions.append(

                    sentiment_preds.cpu()

                )

                aspect_predictions.append(

                    aspect_preds.cpu()

                )

                # -----------------------------
                # Store Probabilities
                # -----------------------------

                sentiment_probabilities.append(

                    sentiment_probs.cpu()

                )

                aspect_probabilities.append(

                    aspect_probs.cpu()

                )

        sentiment_logits = torch.cat(sentiment_logits)

        sentiment_targets = torch.cat(sentiment_targets)

        aspect_logits = torch.cat(aspect_logits)

        aspect_targets = torch.cat(aspect_targets)

        sentiment_predictions = torch.cat(

            sentiment_predictions

        )

        aspect_predictions = torch.cat(

            aspect_predictions

        )

        sentiment_probabilities = torch.cat(

            sentiment_probabilities

        )

        aspect_probabilities = torch.cat(

            aspect_probabilities

        )

        sentiment_metrics = self.metrics_computer.compute_sentiment(

            sentiment_logits,

            sentiment_targets

        )

        aspect_metrics = self.metrics_computer.compute_aspect(

            aspect_logits,

            aspect_targets

        )

        model.train()

        num_batches = len(dataloader)

        return {

        "loss": {

            "sentiment": total_sentiment_loss / num_batches,

            "aspect": total_aspect_loss / num_batches,

            "total": total_loss / num_batches

        },

        "sentiment": sentiment_metrics,

        "aspect": aspect_metrics,

        "predictions": {

            "sentiment": sentiment_predictions,

            "aspect": aspect_predictions

        },

        "targets": {

            "sentiment": sentiment_targets,

            "aspect": aspect_targets

        },

        "probabilities": {

            "sentiment": sentiment_probabilities,

            "aspect": aspect_probabilities

        }

}