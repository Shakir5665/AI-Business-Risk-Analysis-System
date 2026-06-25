"""
Visualization Engine

Generates publication-quality figures
for dataset analysis.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

from pathlib import Path

import matplotlib.pyplot as plt

from configs.storage import FIGURE_DIR
from src.utils.logger import logger


class DatasetVisualizer:

    def __init__(self):

        FIGURE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==========================================================
    # Sentiment Distribution
    # ==========================================================

    def plot_sentiment_distribution(
        self,
        sentiment_distribution
    ):

        labels = list(sentiment_distribution.keys())
        values = list(sentiment_distribution.values())

        plt.figure(figsize=(7,5))

        plt.bar(labels, values)

        plt.title("Sentiment Distribution")

        plt.xlabel("Sentiment")

        plt.ylabel("Number of Reviews")

        plt.tight_layout()

        output = FIGURE_DIR / "sentiment_distribution.png"

        plt.savefig(output, dpi=300)

        plt.close()

        logger.info(
            f"Saved {output}"
        )

    # ==========================================================
    # Aspect Distribution
    # ==========================================================

    def plot_aspect_distribution(
        self,
        aspect_distribution
    ):

        labels = list(aspect_distribution.keys())
        values = list(aspect_distribution.values())

        plt.figure(figsize=(7,5))

        plt.bar(labels, values)

        plt.title("Aspect Distribution")

        plt.xlabel("Aspect")

        plt.ylabel("Occurrences")

        plt.tight_layout()

        output = FIGURE_DIR / "aspect_distribution.png"

        plt.savefig(output, dpi=300)

        plt.close()

        logger.info(
            f"Saved {output}"
        )

    # ==========================================================
    # Review Length Histogram
    # ==========================================================

    def plot_review_length_distribution(
        self,
        review_lengths
    ):

        plt.figure(figsize=(8,5))

        plt.hist(
            review_lengths,
            bins=30
        )

        plt.title("Review Length Distribution")

        plt.xlabel("Number of Words")

        plt.ylabel("Number of Reviews")

        plt.tight_layout()

        output = FIGURE_DIR / "review_length_distribution.png"

        plt.savefig(output, dpi=300)

        plt.close()

        logger.info(
            f"Saved {output}"
        )