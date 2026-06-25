"""
Dataset Validator

Validates dataset before preprocessing and training.
"""

from collections import Counter
from src.utils.logger import logger


class DatasetValidator:

    VALID_SENTIMENTS = {
        "Positive",
        "Negative",
        "Neutral"
    }

    REQUIRED_FIELDS = {
        "text",
        "tokens",
        "bio_labels",
        "sentiment"
    }

    def validate(self, dataset):

        logger.info("Starting dataset validation...")

        errors = []

        duplicate_counter = Counter()

        for index, sample in enumerate(dataset):

            # -----------------------------
            # Required Fields
            # -----------------------------

            missing = self.REQUIRED_FIELDS - sample.keys()

            if missing:
                errors.append(
                    f"Sample {index}: Missing fields -> {missing}"
                )

            # -----------------------------
            # Empty Text
            # -----------------------------

            text = sample.get("text", "").strip()

            if not text:
                errors.append(
                    f"Sample {index}: Empty text"
                )

            # -----------------------------
            # Tokens
            # -----------------------------

            tokens = sample.get("tokens", [])

            if len(tokens) == 0:
                errors.append(
                    f"Sample {index}: Empty token list"
                )

            # -----------------------------
            # BIO Labels
            # -----------------------------

            bio = sample.get("bio_labels", [])

            if len(tokens) != len(bio):
                errors.append(
                    f"Sample {index}: Token/BIO length mismatch"
                )

            # -----------------------------
            # Sentiment
            # -----------------------------

            sentiment = sample.get("sentiment", "")

            if sentiment not in self.VALID_SENTIMENTS:
                errors.append(
                    f"Sample {index}: Invalid sentiment -> {sentiment}"
                )

            # -----------------------------
            # Duplicate Review
            # -----------------------------

            duplicate_counter[text] += 1

        duplicates = sum(
            1 for value in duplicate_counter.values()
            if value > 1
        )

        logger.info(f"Duplicate Reviews : {duplicates}")

        if len(errors) == 0:

            logger.info("Dataset validation completed successfully.")

        else:

            logger.warning(
                f"Validation completed with {len(errors)} issues."
            )

        return errors