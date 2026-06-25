"""
XLM-R Tokenizer

Tokenizes multilingual reviews for XLM-RoBERTa.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer

from configs.model_config import (
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH
)

from src.utils.logger import logger


class ReviewTokenizer:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        logger.info(
            f"Loaded tokenizer: {MODEL_NAME}"
        )

    # --------------------------------------------------

    def tokenize(
        self,
        review: str
    ) -> Dict:

        return self.tokenizer(

            review,

            padding="max_length",

            truncation=True,

            max_length=MAX_SEQUENCE_LENGTH,

            return_attention_mask=True,

            return_tensors="pt"

        )

    # --------------------------------------------------

    def tokenize_batch(
        self,
        reviews: List[str]
    ) -> Dict:

        return self.tokenizer(

            reviews,

            padding=True,

            truncation=True,

            max_length=MAX_SEQUENCE_LENGTH,

            return_attention_mask=True,

            return_tensors="pt"

        )

    # --------------------------------------------------

    def decode(
        self,
        input_ids
    ) -> str:

        return self.tokenizer.decode(

            input_ids,

            skip_special_tokens=True

        )

    # --------------------------------------------------

    def save(
        self,
        path: str
    ):

        Path(path).mkdir(
            parents=True,
            exist_ok=True
        )

        self.tokenizer.save_pretrained(path)

        logger.info(
            f"Tokenizer saved to {path}"
        )

    # --------------------------------------------------

    def load(
        self,
        path: str
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            path
        )

        logger.info(
            f"Tokenizer loaded from {path}"
        )