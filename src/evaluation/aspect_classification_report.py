"""
Aspect Classification Report Generator

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path

from sklearn.metrics import classification_report

from src.utils.logger import logger


class AspectClassificationReportGenerator:

    def __init__(self):

        logger.info(
            "AspectClassificationReportGenerator initialized."
        )

    # --------------------------------------------------

    def generate(
        self,
        true_labels,
        predicted_labels,
        target_names,
        save_path
    ):

        report = classification_report(

            true_labels,

            predicted_labels,

            target_names=target_names,

            digits=4,

            zero_division=0

        )

        Path(save_path).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        with open(save_path, "w") as file:

            file.write(report)

        logger.info(

            f"Aspect classification report saved: {save_path}"

        )

        return report