"""
Dataset Preprocessor

Reads the raw dataset, preprocesses every review,
and saves a processed dataset.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

import json

from configs.paths import (
    RAW_DATASET_PATH,
    PROCESSED_DATASET_PATH
)

from src.preprocessing.preprocessor import ReviewPreprocessor

from src.utils.logger import logger


class DatasetPreprocessor:
    """
    Generates a fully preprocessed dataset.
    """

    def __init__(self):

        self.preprocessor = ReviewPreprocessor()

        logger.info("DatasetPreprocessor initialized.")

    # --------------------------------------------------

    def process(self):

        logger.info("Loading raw dataset...")

        total_reviews = 0

        with open(
            RAW_DATASET_PATH,
            "r",
            encoding="utf-8"
        ) as infile, open(
            PROCESSED_DATASET_PATH,
            "w",
            encoding="utf-8"
        ) as outfile:

            for line in infile:

                record = json.loads(line)

                record["text"] = self.preprocessor.preprocess(

                    record["text"]

                )

                outfile.write(

                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"

                )

                total_reviews += 1

        logger.info(
            f"Preprocessed {total_reviews} reviews."
        )

        logger.info(
            f"Saved dataset to: {PROCESSED_DATASET_PATH}"
        )