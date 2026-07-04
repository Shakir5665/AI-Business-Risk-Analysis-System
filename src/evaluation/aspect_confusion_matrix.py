"""
Aspect Confusion Matrix Generator

Generates one confusion matrix
for each aspect.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.logger import logger


class AspectConfusionMatrixGenerator:

    """
    Generates binary confusion matrices
    for each aspect.
    """

    def __init__(self):

        logger.info(
            "AspectConfusionMatrixGenerator initialized."
        )

    # --------------------------------------------------

    def generate(

        self,

        true_labels,

        predicted_labels,

        aspect_names,

        output_directory

    ):

        Path(output_directory).mkdir(

            parents=True,

            exist_ok=True

        )

        for i, aspect in enumerate(aspect_names):

            cm = confusion_matrix(

                true_labels[:, i],

                predicted_labels[:, i]

            )

            display = ConfusionMatrixDisplay(

                confusion_matrix=cm,

                display_labels=["No", "Yes"]

            )

            fig, ax = plt.subplots(figsize=(5,5))

            display.plot(

                cmap="Blues",

                colorbar=False,

                ax=ax

            )

            plt.title(f"{aspect} Confusion Matrix")

            plt.savefig(

                f"{output_directory}/{aspect.lower()}_confusion_matrix.png",

                dpi=300,

                bbox_inches="tight"

            )

            plt.close()

        logger.info(

            "Aspect confusion matrices generated successfully."

        )