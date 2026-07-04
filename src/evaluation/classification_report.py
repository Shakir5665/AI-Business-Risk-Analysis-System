"""
Classification Report Generator

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path

from sklearn.metrics import classification_report

from src.utils.logger import logger


class ClassificationReportGenerator:
    """
    Generates a classification report and saves it.
    """

    def __init__(self):

        logger.info(
            "ClassificationReportGenerator initialized."
        )

    # --------------------------------------------------

    def generate(
        self,
        true_labels,
        predicted_labels,
        class_names,
        save_path
    ):

        report = classification_report(

            true_labels,

            predicted_labels,

            target_names=class_names,

            digits=4

        )

        Path(save_path).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        with open(save_path, "w") as file:

            file.write(report)

        logger.info(

            f"Classification report saved: {save_path}"

        )

        return report