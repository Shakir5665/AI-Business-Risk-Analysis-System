"""
Confusion Matrix Generator

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from src.utils.logger import logger


class ConfusionMatrixGenerator:

    """
    Generates and saves confusion matrices.
    """

    def __init__(self):

        logger.info(
            "ConfusionMatrixGenerator initialized."
        )

    # --------------------------------------------------

    def generate(
        self,
        true_labels,
        predicted_labels,
        class_names,
        save_path
    ):

        cm = confusion_matrix(

            true_labels,

            predicted_labels

        )

        display = ConfusionMatrixDisplay(

            confusion_matrix=cm,

            display_labels=class_names

        )

        fig, ax = plt.subplots(figsize=(7, 6))

        display.plot(

            cmap="Blues",

            ax=ax,

            colorbar=False

        )

        plt.title("Sentiment Confusion Matrix")

        Path(save_path).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

        logger.info(

            f"Confusion matrix saved: {save_path}"

        )