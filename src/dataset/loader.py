"""
Dataset Loader

Loads JSONL datasets safely with logging and validation.

Author: Mohamed Shakir
Project:
AI-Powered Business Risk Analysis System
"""

import json
from pathlib import Path
from typing import List, Dict

from src.utils.logger import logger


class DatasetLoader:
    """
    Loads datasets stored in JSONL format.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)

    def load(self) -> List[Dict]:
        """
        Load JSONL dataset.

        Returns
        -------
        List[Dict]
            List of dataset samples.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        dataset = []

        logger.info(f"Loading dataset from: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    dataset.append(sample)

                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid JSON at line {line_number}: {e}"
                    )
                    raise

        logger.info(f"Dataset loaded successfully.")
        logger.info(f"Total samples: {len(dataset)}")

        return dataset