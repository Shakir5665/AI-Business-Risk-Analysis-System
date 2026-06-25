"""
Dataset Splitter

Splits the processed dataset into
Train / Validation / Test datasets.

Uses stratified splitting based on
sentiment labels.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path
import json

import pandas as pd

from sklearn.model_selection import train_test_split

from configs.training_config import (
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE,
    RANDOM_STATE
)

from src.utils.logger import logger


class DatasetSplitter:

    def __init__(self):

        logger.info(
            "DatasetSplitter initialized."
        )

    # --------------------------------------------------

    def load_dataset(

        self,

        dataset_path: str

    ) -> pd.DataFrame:

        dataframe = pd.read_json(

            dataset_path,

            lines=True

        )

        logger.info(

            f"Loaded {len(dataframe)} reviews."

        )

        return dataframe

    # --------------------------------------------------

    def split(

        self,

        dataframe: pd.DataFrame

    ):

        train_df, temp_df = train_test_split(

            dataframe,

            train_size=TRAIN_SIZE,

            random_state=RANDOM_STATE,

            stratify=dataframe["sentiment"]

        )

        validation_ratio = (

            VALIDATION_SIZE /

            (

                VALIDATION_SIZE +

                TEST_SIZE

            )

        )

        validation_df, test_df = train_test_split(

            temp_df,

            train_size=validation_ratio,

            random_state=RANDOM_STATE,

            stratify=temp_df["sentiment"]

        )

        logger.info(

            f"Train : {len(train_df)}"

        )

        logger.info(

            f"Validation : {len(validation_df)}"

        )

        logger.info(

            f"Test : {len(test_df)}"

        )

        return (

            train_df,

            validation_df,

            test_df

        )

    # --------------------------------------------------

    def save(

        self,

        dataframe: pd.DataFrame,

        output_path: str

    ):

        path = Path(output_path)

        path.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        dataframe.to_json(

            path,

            orient="records",

            lines=True,

            force_ascii=False

        )

        logger.info(

            f"Saved -> {path}"

        )

    # --------------------------------------------------

    def print_distribution(

        self,

        dataframe: pd.DataFrame,

        title: str

    ):

        print()

        print("=" * 50)

        print(title)

        print("=" * 50)

        print()

        print(

            dataframe["sentiment"]

            .value_counts()

        )

        print()

        print(

            dataframe["sentiment"]

            .value_counts(

                normalize=True

            )

        )

    # --------------------------------------------------

    def execute(

        self,

        dataset_path: str,

        output_directory: str

    ):

        dataframe = self.load_dataset(

            dataset_path

        )

        train_df, validation_df, test_df = self.split(

            dataframe

        )

        self.save(

            train_df,

            f"{output_directory}/train.jsonl"

        )

        self.save(

            validation_df,

            f"{output_directory}/validation.jsonl"

        )

        self.save(

            test_df,

            f"{output_directory}/test.jsonl"

        )

        self.print_distribution(

            train_df,

            "Training Set"

        )

        self.print_distribution(

            validation_df,

            "Validation Set"

        )

        self.print_distribution(

            test_df,

            "Test Set"

        )

        logger.info(

            "Dataset splitting completed."
        )