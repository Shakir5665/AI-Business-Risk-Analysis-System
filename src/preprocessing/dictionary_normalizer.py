"""
Dictionary Normalizer

Normalizes domain-specific words across
Sinhala, Singlish and English.

Unknown words are preserved because
XLM-R understands multilingual text.

Project:
AI-Powered Business Risk Analysis and Recommendation System
"""

import json
import re
from pathlib import Path

from src.utils.logger import logger


class DictionaryNormalizer:

    def __init__(self):

        dictionary_path = Path(
            "resources/domain_dictionary.json"
        )

        if not dictionary_path.exists():

            raise FileNotFoundError(
                f"Dictionary not found: {dictionary_path}"
            )

        with open(
            dictionary_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.dictionary = json.load(f)

        self.sinhala_pattern = re.compile(
            r'[\u0D80-\u0DFF]'
        )

        logger.info(
            f"Loaded {len(self.dictionary)} domain mappings."
        )

    def detect_sinhala(self, text):

        return bool(
            self.sinhala_pattern.search(text)
        )

    def normalize(self, text):

        if not text:

            return ""

        words = text.split()

        normalized = []

        for word in words:

            normalized.append(

                self.dictionary.get(

                    word,

                    word

                )

            )

        return " ".join(normalized)

    def normalize_batch(self, texts):

        return [

            self.normalize(text)

            for text in texts

        ]