"""
Dataset Statistics Engine

Computes statistical information about the dataset.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

from collections import Counter
from statistics import mean, median, stdev


class DatasetStatistics:

    def __init__(self, dataset):
        self.dataset = dataset

    def compute(self):
        """
        Compute dataset statistics.

        Returns:
            dict
        """

        sentiment_counter = Counter()
        aspect_counter = Counter()

        review_lengths = []

        unique_reviews = set()

        duplicate_count = 0

        for sample in self.dataset:

            # ----------------------------
            # Sentiment
            # ----------------------------

            sentiment_counter[
                sample["sentiment"].lower()
            ] += 1

            # ----------------------------
            # Aspects
            # ----------------------------

            for aspect in sample["aspects"]:

                aspect_counter[
                    aspect.lower()
                ] += 1

            # ----------------------------
            # Review Length
            # ----------------------------

            review_lengths.append(
                len(sample["text"].split())
            )

            # ----------------------------
            # Duplicate Detection
            # ----------------------------

            text = sample["text"].strip()

            if text in unique_reviews:
                duplicate_count += 1
            else:
                unique_reviews.add(text)

        statistics = {

            "total_reviews": len(self.dataset),

            "unique_reviews": len(unique_reviews),

            "duplicate_reviews": duplicate_count,

            "sentiment_distribution": dict(sentiment_counter),

            "aspect_distribution": dict(aspect_counter),

            "review_length": {

                "average": round(mean(review_lengths), 2),

                "minimum": min(review_lengths),

                "maximum": max(review_lengths),

                "median": median(review_lengths),

                "std": round(
                    stdev(review_lengths), 2
                ) if len(review_lengths) > 1 else 0
            }

        }

        return statistics