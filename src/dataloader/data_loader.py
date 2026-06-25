"""
DataLoader Factory

Creates PyTorch DataLoaders.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from torch.utils.data import DataLoader

from configs.training_config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SHUFFLE_TRAIN,
    SHUFFLE_VALIDATION,
    SHUFFLE_TEST
)

from src.utils.logger import logger


class BusinessDataLoader:

    def __init__(self):

        logger.info(
            "BusinessDataLoader initialized."
        )

    # --------------------------------------------------

    def create_train_loader(
        self,
        dataset
    ):

        return DataLoader(

            dataset=dataset,

            batch_size=BATCH_SIZE,

            shuffle=SHUFFLE_TRAIN,

            num_workers=NUM_WORKERS,

            pin_memory=PIN_MEMORY

        )

    # --------------------------------------------------

    def create_validation_loader(
        self,
        dataset
    ):

        return DataLoader(

            dataset=dataset,

            batch_size=BATCH_SIZE,

            shuffle=SHUFFLE_VALIDATION,

            num_workers=NUM_WORKERS,

            pin_memory=PIN_MEMORY

        )

    # --------------------------------------------------

    def create_test_loader(
        self,
        dataset
    ):

        return DataLoader(

            dataset=dataset,

            batch_size=BATCH_SIZE,

            shuffle=SHUFFLE_TEST,

            num_workers=NUM_WORKERS,

            pin_memory=PIN_MEMORY

        )