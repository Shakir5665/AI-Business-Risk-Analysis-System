"""
Visualization Engine

Creates publication-quality charts for dataset analysis.

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

    # =========================================================
    # Common Save Function
    # =========================================================

    def _save_plot(self, filename):

        output_path = FIGURE_DIR / filename

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(f"Saved figure -> {output_path}")

    # =========================================================
    # Sentiment Distribution
    # =========================================================

    def plot_sentiment_distribution(self, sentiments):

        plt.figure(figsize=(7, 5))

        plt.bar(
            sentiments.keys(),
            sentiments.values()
        )

        plt.title("Sentiment Distribution")

        plt.xlabel("Sentiment")

        plt.ylabel("Number of Reviews")

        self._save_plot("sentiment_distribution.png")

    # =========================================================
    # Aspect Distribution
    # =========================================================

    def plot_aspect_distribution(self, aspects):

        plt.figure(figsize=(7, 5))

        plt.bar(
            aspects.keys(),
            aspects.values()
        )

        plt.title("Aspect Distribution")

        plt.xlabel("Aspect")

        plt.ylabel("Occurrences")

        self._save_plot("aspect_distribution.png")

    # =========================================================
    # Review Length Histogram
    # =========================================================

    def plot_review_length_distribution(self, review_lengths):

        plt.figure(figsize=(8, 5))

        plt.hist(
            review_lengths,
            bins=30
        )

        plt.title("Review Length Distribution")

        plt.xlabel("Words per Review")

        plt.ylabel("Frequency")

        self._save_plot("review_length_distribution.png")
    
    def plot_language_distribution(
    self,
    language_distribution
    ):

        plt.figure(figsize=(7,5))

        plt.bar(
            language_distribution.keys(),
            language_distribution.values()
        )

        plt.title("Language Distribution")

        plt.xlabel("Language")

        plt.ylabel("Reviews")

        self._save_plot(
            "language_distribution.png"
        )
    
    