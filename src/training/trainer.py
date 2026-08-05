"""
Trainer

Handles the complete training pipeline.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from configs.training_config import (
    NUM_EPOCHS,
    GRADIENT_CLIP,
    EARLY_STOPPING
)

from src.training.early_stopping import EarlyStopping

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
        loss_computer,
        device
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device

        self.loss_computer = loss_computer

        self.validator = Validator()

        self.checkpoint_manager = CheckpointManager()

        self.start_epoch = 1

        # -----------------------------
        # Early Stopping
        # -----------------------------

        self.early_stopping = None

        if EARLY_STOPPING:

            self.early_stopping = EarlyStopping()

        self.num_epochs = NUM_EPOCHS

        self.gradient_clip = GRADIENT_CLIP

        self.best_validation_loss = float("inf")

        self.history = {
            "epochs": [],
            "train_loss_sentiment": [],
            "train_loss_aspect": [],
            "train_loss_total": [],
            "val_loss_sentiment": [],
            "val_loss_aspect": [],
            "val_loss_total": [],
            "val_sentiment_acc": [],
            "val_sentiment_f1": [],
            "val_aspect_micro_f1": [],
            "val_aspect_macro_f1": []
        }

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

        for epoch in range(self.start_epoch, self.num_epochs + 1):

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
            # Early Stopping
            # -----------------------------

            if self.early_stopping is not None:

                if self.early_stopping.should_stop(validation_loss):

                    print()

                    print("=" * 60)

                    print("Early Stopping Triggered")

                    print(f"Training stopped at Epoch {epoch}")

                    print("=" * 60)

                    break

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

            # -----------------------------
            # Save Training History
            # -----------------------------
            self.history["epochs"].append(epoch)
            self.history["train_loss_sentiment"].append(train_loss['sentiment'])
            self.history["train_loss_aspect"].append(train_loss['aspect'])
            self.history["train_loss_total"].append(train_loss['total'])
            self.history["val_loss_sentiment"].append(validation_results['loss']['sentiment'])
            self.history["val_loss_aspect"].append(validation_results['loss']['aspect'])
            self.history["val_loss_total"].append(validation_results['loss']['total'])
            self.history["val_sentiment_acc"].append(validation_results['sentiment']['accuracy'])
            self.history["val_sentiment_f1"].append(validation_results['sentiment']['f1'])
            self.history["val_aspect_micro_f1"].append(validation_results['aspect']['micro_f1'])
            self.history["val_aspect_macro_f1"].append(validation_results['aspect']['macro_f1'])

            for target_path in ["outputs/reports/history.json", "checkpoints/history.json"]:
                p = Path(target_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w") as f:
                    json.dump(self.history, f, indent=2)

        logger.info("Training completed.")
    
    