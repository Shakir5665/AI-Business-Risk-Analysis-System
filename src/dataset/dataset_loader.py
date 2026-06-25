"""
Dataset Loader

Loads processed datasets from JSONL files.

Responsibilities:
    - Load training dataset
    - Load validation dataset
    - Load testing dataset
    - Return pandas DataFrames

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import logger


class DatasetLoader:

    """
    Loads processed datasets.
    """

    def __init__(self):

        logger.info("DatasetLoader initialized.")

    # ---------------------------------------------------------

    def load(self, dataset_path: str) -> pd.DataFrame:

        path = Path(dataset_path)

        if not path.exists():

            raise FileNotFoundError(

                f"Dataset not found: {path}"

            )

        dataframe = pd.read_json(

            path,

            lines=True

        )

        logger.info(

            f"Loaded {len(dataframe)} samples from {path.name}"

        )

        return dataframe

    # ---------------------------------------------------------

    def load_train(self):

        return self.load(

            "data/processed/train.jsonl"

        )

    # ---------------------------------------------------------

    def load_validation(self):

        return self.load(

            "data/processed/validation.jsonl"

        )

    # ---------------------------------------------------------

    def load_test(self):

        return self.load(

            "data/processed/test.jsonl"

        )