"""
Trainer

Handles the complete training pipeline.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

import torch
import torch.nn as nn

from configs.training_config import (
    NUM_EPOCHS,
    GRADIENT_CLIP
)

from src.training.losses import LossComputer
from src.training.validator import Validator
from src.training.checkpoint import CheckpointManager

from src.utils.logger import logger


class Trainer:
    """
    Trainer for BusinessRiskModel.
    """

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        optimizer,
        scheduler,
        device
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device

        self.loss_computer = LossComputer()

        self.validator = Validator()

        self.checkpoint_manager = CheckpointManager()

        self.num_epochs = NUM_EPOCHS

        self.gradient_clip = GRADIENT_CLIP

        self.best_validation_loss = float("inf")

        logger.info("Trainer initialized.")
    
        # --------------------------------------------------

    def train_epoch(self):

        self.model.train()

        total_sentiment_loss = 0.0

        total_aspect_loss = 0.0

        total_loss = 0.0

        for batch in self.train_loader:

            input_ids = batch["input_ids"].to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)

            sentiment = batch["sentiment"].to(self.device)

            aspects = batch["aspects"].to(self.device)

            # -----------------------------
            # Forward Pass
            # -----------------------------

            outputs = self.model(

                input_ids=input_ids,

                attention_mask=attention_mask

            )

            # -----------------------------
            # Compute Loss
            # -----------------------------

            losses = self.loss_computer.compute(

                outputs,

                sentiment,

                aspects

            )

            # -----------------------------
            # Backpropagation
            # -----------------------------

            self.optimizer.zero_grad()

            losses["total_loss"].backward()

            # -----------------------------
            # Gradient Clipping
            # -----------------------------

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.gradient_clip

            )

            # -----------------------------
            # Update Parameters
            # -----------------------------

            self.optimizer.step()

            self.scheduler.step()

            # -----------------------------
            # Statistics
            # -----------------------------

            total_sentiment_loss += losses["sentiment_loss"].item()

            total_aspect_loss += losses["aspect_loss"].item()

            total_loss += losses["total_loss"].item()

        num_batches = len(self.train_loader)

        return {

            "sentiment": total_sentiment_loss / num_batches,

            "aspect": total_aspect_loss / num_batches,

            "total": total_loss / num_batches

        }
    
        # --------------------------------------------------

    def train(self):

        logger.info("Training started.")

        for epoch in range(1, self.num_epochs + 1):

            print()

            print("=" * 60)

            print(f"Epoch {epoch}/{self.num_epochs}")

            print("=" * 60)

            # -----------------------------
            # Training
            # -----------------------------

            train_loss = self.train_epoch()

            # -----------------------------
            # Validation
            # -----------------------------

            validation_results = self.validator.validate(

                model=self.model,

                dataloader=self.validation_loader,

                device=self.device

            )

            validation_loss = validation_results["loss"]["total"]

            # -----------------------------
            # Save Latest Checkpoint
            # -----------------------------

            self.checkpoint_manager.save_latest(

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                epoch=epoch

            )

            # -----------------------------
            # Save Best Model
            # -----------------------------

            if validation_loss < self.best_validation_loss:

                self.best_validation_loss = validation_loss

                self.checkpoint_manager.save_best(

                    model=self.model,

                    optimizer=self.optimizer,

                    scheduler=self.scheduler,

                    epoch=epoch,

                    validation_loss=validation_loss

                )

                best_saved = "YES"

            else:

                best_saved = "NO"

            # -----------------------------
            # Epoch Summary
            # -----------------------------

            print()

            print("Training Loss")

            print(f"  Sentiment : {train_loss['sentiment']:.4f}")

            print(f"  Aspect    : {train_loss['aspect']:.4f}")

            print(f"  Total     : {train_loss['total']:.4f}")

            print()

            print("Validation Loss")

            print(f"  Sentiment : {validation_results['loss']['sentiment']:.4f}")

            print(f"  Aspect    : {validation_results['loss']['aspect']:.4f}")

            print(f"  Total     : {validation_results['loss']['total']:.4f}")

            print()

            print("Sentiment Metrics")

            print(f"  Accuracy  : {validation_results['sentiment']['accuracy']:.4f}")

            print(f"  Precision : {validation_results['sentiment']['precision']:.4f}")

            print(f"  Recall    : {validation_results['sentiment']['recall']:.4f}")

            print(f"  F1 Score  : {validation_results['sentiment']['f1']:.4f}")

            print()

            print("Aspect Metrics")

            print(f"  Micro F1  : {validation_results['aspect']['micro_f1']:.4f}")

            print(f"  Macro F1  : {validation_results['aspect']['macro_f1']:.4f}")

            print()

            print(f"Best Model Saved : {best_saved}")

        logger.info("Training completed.")
    
    