"""
Dataset Analyzer

Generates statistics and reports for the dataset.
"""

from collections import Counter
from statistics import mean, median
import json

from src.utils.logger import logger
from configs.storage import REPORT_DIR


class DatasetAnalyzer:

    def analyze(self, dataset):

        logger.info("Starting dataset analysis...")

        sentiments = Counter()
        aspects = Counter()

        review_lengths = []

        for sample in dataset:

            sentiments[sample["sentiment"]] += 1

            for aspect in sample["aspects"]:
                aspects[aspect] += 1

            review_lengths.append(
                len(sample["text"].split())
            )

        report = {

            "total_reviews": len(dataset),

            "sentiment_distribution": dict(sentiments),

            "aspect_distribution": dict(aspects),

            "review_length": {

                "average": round(mean(review_lengths),2),

                "minimum": min(review_lengths),

                "maximum": max(review_lengths),

                "median": median(review_lengths)

            }

        }

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        report_file = REPORT_DIR / "dataset_report.json"

        with open(report_file,"w") as f:

            json.dump(report,f,indent=4)

        logger.info("Dataset report saved.")

        return report