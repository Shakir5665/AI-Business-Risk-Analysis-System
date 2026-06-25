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

        sentiment_counter = Counter()
        aspect_counter = Counter()

        review_lengths = []

        unique_reviews = set()

        duplicate_reviews = 0

        for sample in self.dataset:

            # -------------------------
            # Sentiment
            # -------------------------

            sentiment = sample["sentiment"].strip().lower()

            sentiment_counter[sentiment] += 1

            # -------------------------
            # Aspects
            # -------------------------

            for aspect in sample["aspects"]:

                aspect_counter[aspect.lower()] += 1

            # -------------------------
            # Review Length
            # -------------------------

            length = len(sample["text"].split())

            review_lengths.append(length)

            # -------------------------
            # Duplicate Detection
            # -------------------------

            text = sample["text"].strip()

            if text in unique_reviews:
                duplicate_reviews += 1
            else:
                unique_reviews.add(text)

        return {

            "total_reviews": len(self.dataset),

            "unique_reviews": len(unique_reviews),

            "duplicate_reviews": duplicate_reviews,

            "sentiment_distribution": dict(sentiment_counter),

            "aspect_distribution": dict(aspect_counter),

            # Raw lengths for plotting
            "review_lengths": review_lengths,

            # Summary statistics
            "review_length_statistics": {

                "average": round(mean(review_lengths), 2),

                "minimum": min(review_lengths),

                "maximum": max(review_lengths),

                "median": median(review_lengths),

                "std": round(
                    stdev(review_lengths), 2
                ) if len(review_lengths) > 1 else 0

            }

        }