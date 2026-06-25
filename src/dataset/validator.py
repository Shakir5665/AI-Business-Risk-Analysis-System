"""
Dataset Validator

Validates the dataset before preprocessing and training.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

from collections import Counter
from src.utils.logger import logger


class DatasetValidator:

    VALID_SENTIMENTS = {
        "positive",
        "negative",
        "neutral"
    }

    VALID_ASPECTS = {
        "delivery",
        "quality",
        "trust"
    }

    REQUIRED_FIELDS = {
        "review_id",
        "text",
        "sentiment",
        "aspects"
    }

    def validate(self, dataset):

        logger.info("Starting dataset validation...")

        errors = []

        duplicate_counter = Counter()

        for index, sample in enumerate(dataset):

            # =====================================================
            # Required Fields
            # =====================================================

            missing = self.REQUIRED_FIELDS - sample.keys()

            if missing:
                errors.append(
                    f"Sample {index}: Missing fields -> {missing}"
                )
                continue

            # =====================================================
            # Empty Review
            # =====================================================

            text = sample["text"].strip()

            if text == "":
                errors.append(
                    f"Sample {index}: Empty review"
                )

            # =====================================================
            # Sentiment
            # =====================================================

            sentiment = sample["sentiment"].strip().lower()

            if sentiment not in self.VALID_SENTIMENTS:

                errors.append(
                    f"Sample {index}: Invalid sentiment -> {sentiment}"
                )

            # =====================================================
            # Aspect Validation
            # =====================================================

            aspects = sample["aspects"]

            if not isinstance(aspects, list):

                errors.append(
                    f"Sample {index}: Aspects must be a list"
                )

            elif len(aspects) == 0:

                errors.append(
                    f"Sample {index}: Empty aspect list"
                )

            else:

                for aspect in aspects:

                    if aspect.lower() not in self.VALID_ASPECTS:

                        errors.append(
                            f"Sample {index}: Invalid aspect -> {aspect}"
                        )

            # =====================================================
            # Duplicate Review
            # =====================================================

            duplicate_counter[text] += 1

        duplicates = sum(
            count - 1
            for count in duplicate_counter.values()
            if count > 1
        )

        logger.info(f"Duplicate Reviews : {duplicates}")

        if errors:

            logger.warning(
                f"Validation completed with {len(errors)} issues."
            )

        else:

            logger.info("Dataset validation successful.")

        return errors